"""Tests that exercise validate_script against the mock OpenAI server.

These tests start an in-process mock server, point the gatekeeper at it,
and call validate_script through the MCP client -- the same way Goose or
Claude would call it. The full path runs for real: MCP client → validate_script
tool → gatekeeper prompt building → HTTP request → mock server → HTTP response
→ response parsing → tool result. The only fake is the LLM itself.
"""

from importlib import import_module
from typing import Any

import httpx
import pytest

from fastmcp.exceptions import ToolError

from linux_mcp_server.config import CONFIG
from linux_mcp_server.config import GatekeeperProvider
from linux_mcp_server.config import OpenAIGatekeeperConfig
from linux_mcp_server.config import Toolset
from linux_mcp_server.tools.run_script import ScriptStore
from tests.mock_openai_provider.server import DEFAULT_RULES
from tests.mock_openai_provider.server import start_mock_server


run_script_mod = import_module("linux_mcp_server.tools.run_script")
http_utils_mod = import_module("linux_mcp_server.gatekeeper.http_utils")


@pytest.fixture
async def client(setup_client):
    yield await setup_client(toolset=Toolset.RUN_SCRIPT)


@pytest.fixture
def script_store_fresh(monkeypatch) -> ScriptStore:
    """Isolate script_store so tests do not share global script IDs."""
    store = ScriptStore()
    monkeypatch.setattr(run_script_mod, "script_store", store)
    return store


@pytest.fixture
async def mock_gatekeeper(monkeypatch):
    """Start the mock server and configure the gatekeeper to use it."""
    mock, runner, base_url = await start_mock_server(rules=DEFAULT_RULES)

    monkeypatch.setattr(CONFIG.gatekeeper, "provider", GatekeeperProvider.OPENAI)
    monkeypatch.setattr(CONFIG.gatekeeper, "model", "mock-gatekeeper")
    monkeypatch.setattr(CONFIG.gatekeeper, "structured_output", False)
    monkeypatch.setattr(CONFIG.gatekeeper, "openai", OpenAIGatekeeperConfig(base_url=base_url))
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    http_client = httpx.AsyncClient()
    monkeypatch.setattr(http_utils_mod, "HTTP_CLIENT", http_client)

    try:
        yield mock
    finally:
        await http_client.aclose()
        await runner.cleanup()


class TestValidateScriptWithMockGatekeeper:
    """Call validate_script through the MCP client with a mock gatekeeper."""

    async def test_safe_script_approved(
        self, client: Any, script_store_fresh: ScriptStore, mock_gatekeeper: Any
    ):
        """A safe script passes validation and gets a token."""
        result = await client.call_tool(
            "validate_script",
            {
                "description": "Check disk usage",
                "script_type": "bash",
                "script": "df -h",
                "readonly": True,
            },
        )
        assert result.structured_content["needs_confirmation"] is False
        token = result.structured_content["token"]
        assert script_store_fresh.get_script_details(token).state == "waiting-approval"

    async def test_dangerous_script_rejected(
        self, client: Any, script_store_fresh: ScriptStore, mock_gatekeeper: Any
    ):
        """A destructive rm -rf should be blocked."""
        with pytest.raises(ToolError, match="Dangerous"):
            await client.call_tool(
                "validate_script",
                {
                    "description": "Clean up files",
                    "script_type": "bash",
                    "script": "rm -rf /",
                    "readonly": False,
                },
            )

    async def test_malicious_script_rejected(
        self, client: Any, script_store_fresh: ScriptStore, mock_gatekeeper: Any
    ):
        """A malicious script is rejected by the gatekeeper."""
        with pytest.raises(ToolError, match="malicious"):
            await client.call_tool(
                "validate_script",
                {
                    "description": "Run a test",
                    "script_type": "bash",
                    "script": ":(){ :|:& };:",
                    "readonly": False,
                },
            )

    async def test_mkfs_rejected(
        self, client: Any, script_store_fresh: ScriptStore, mock_gatekeeper: Any
    ):
        """Formatting a filesystem is rejected."""
        with pytest.raises(ToolError, match="Dangerous"):
            await client.call_tool(
                "validate_script",
                {
                    "description": "Format disk",
                    "script_type": "bash",
                    "script": "mkfs.ext4 /dev/sda1",
                    "readonly": False,
                },
            )

    async def test_safe_systemctl_approved(
        self, client: Any, script_store_fresh: ScriptStore, mock_gatekeeper: Any
    ):
        """A systemctl status check passes validation."""
        result = await client.call_tool(
            "validate_script",
            {
                "description": "Check chronyd status",
                "script_type": "bash",
                "script": "systemctl status chronyd",
                "readonly": True,
            },
        )
        assert result.structured_content["needs_confirmation"] is False

    async def test_readonly_script_no_confirmation_needed(
        self, client: Any, script_store_fresh: ScriptStore, mock_gatekeeper: Any
    ):
        """A readonly script does not need confirmation."""
        result = await client.call_tool(
            "validate_script",
            {
                "description": "List files",
                "script_type": "bash",
                "script": "ls -la /tmp",
                "readonly": True,
            },
        )
        assert result.structured_content["needs_confirmation"] is False

    async def test_readwrite_script_needs_confirmation(
        self, client: Any, script_store_fresh: ScriptStore, mock_gatekeeper: Any
    ):
        """A non-readonly script needs confirmation."""
        result = await client.call_tool(
            "validate_script",
            {
                "description": "Restart chronyd",
                "script_type": "bash",
                "script": "systemctl restart chronyd",
                "readonly": False,
            },
        )
        assert result.structured_content["needs_confirmation"] is True

    async def test_mock_logs_request(
        self, client: Any, script_store_fresh: ScriptStore, mock_gatekeeper: Any
    ):
        """The mock server logs the request for inspection."""
        await client.call_tool(
            "validate_script",
            {
                "description": "List files",
                "script_type": "bash",
                "script": "ls -la /tmp",
                "readonly": True,
            },
        )
        assert len(mock_gatekeeper.requests) == 1
        body = mock_gatekeeper.requests[0]
        assert body["model"] == "mock-gatekeeper"
        assert "ls -la /tmp" in body["input"]

    async def test_custom_rules(
        self, client: Any, script_store_fresh: ScriptStore, monkeypatch: Any
    ):
        """Custom rules can test specific scenarios."""
        custom_rules = [
            ("apt install", '{"status": "POLICY", "detail": "no package installs"}'),
            ("", '{"status": "OK", "detail": ""}'),
        ]
        mock, runner, base_url = await start_mock_server(rules=custom_rules)
        monkeypatch.setattr(CONFIG.gatekeeper, "provider", GatekeeperProvider.OPENAI)
        monkeypatch.setattr(CONFIG.gatekeeper, "model", "mock-gatekeeper")
        monkeypatch.setattr(CONFIG.gatekeeper, "structured_output", False)
        monkeypatch.setattr(CONFIG.gatekeeper, "openai", OpenAIGatekeeperConfig(base_url=base_url))
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        http_client = httpx.AsyncClient()
        monkeypatch.setattr(http_utils_mod, "HTTP_CLIENT", http_client)

        try:
            with pytest.raises(ToolError, match="Policy violation"):
                await client.call_tool(
                    "validate_script",
                    {
                        "description": "Install nginx",
                        "script_type": "bash",
                        "script": "apt install nginx",
                        "readonly": False,
                    },
                )
        finally:
            await http_client.aclose()
            await runner.cleanup()
