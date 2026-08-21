"""Tests for the chat completions endpoint (/v1/chat/completions).

These tests hit the mock server directly with httpx and verify the SSE
stream returns the correct scenario steps -- tool calls or text -- based
on how many tool results are in the conversation.
"""

import json

from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.mock_openai_provider.server import load_scenario
from tests.mock_openai_provider.server import start_mock_server


SCENARIO_DIR = Path(__file__).parent / "scenarios"


async def _collect_sse_chunks(url: str, body: dict[str, Any]) -> list[dict[str, Any]]:
    """Send a chat completions request and collect all SSE chunks."""
    chunks = []
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, json=body) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk = json.loads(line[len("data: "):])
                    chunks.append(chunk)
    return chunks


def _make_request(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a minimal chat completions request body."""
    return {
        "model": "mock",
        "messages": messages,
        "stream": True,
    }


@pytest.fixture
async def chat_server():
    """Start a mock server loaded with the restart-chronyd scenario."""
    scenario = load_scenario(SCENARIO_DIR / "restart-chronyd.yaml")
    mock, runner, base_url = await start_mock_server(scenario=scenario)
    try:
        yield mock, f"{base_url}/chat/completions"
    finally:
        await runner.cleanup()


class TestChatCompletions:
    """Test the /v1/chat/completions endpoint with scenario playback."""

    async def test_step_0_returns_validate_script_tool_call(self, chat_server):
        """First request (no tool results) returns validate_script tool call."""
        mock, url = chat_server
        body = _make_request([
            {"role": "user", "content": "restart chronyd on 192.168.64.115"},
        ])

        chunks = await _collect_sse_chunks(url, body)

        first_chunk = chunks[0]
        delta = first_chunk["choices"][0]["delta"]
        assert delta["role"] == "assistant"
        tool_call = delta["tool_calls"][0]
        assert tool_call["function"]["name"] == "linuxmcpserverarbitrary__validate_script"

        last_chunk = chunks[-1]
        assert last_chunk["choices"][0]["finish_reason"] == "tool_calls"

    async def test_step_1_returns_run_script_tool_call(self, chat_server):
        """Second request (1 tool result) returns run_script_interactive tool call."""
        mock, url = chat_server
        body = _make_request([
            {"role": "user", "content": "restart chronyd on 192.168.64.115"},
            {"role": "assistant", "tool_calls": [{"id": "call_1", "function": {"name": "validate_script"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "Script approved, token=abc123"},
        ])

        chunks = await _collect_sse_chunks(url, body)

        first_chunk = chunks[0]
        tool_call = first_chunk["choices"][0]["delta"]["tool_calls"][0]
        assert tool_call["function"]["name"] == "linuxmcpserverarbitrary__run_script_interactive"

        last_chunk = chunks[-1]
        assert last_chunk["choices"][0]["finish_reason"] == "tool_calls"

    async def test_step_2_returns_final_text(self, chat_server):
        """Third request (2 tool results) returns final text response."""
        mock, url = chat_server
        body = _make_request([
            {"role": "user", "content": "restart chronyd on 192.168.64.115"},
            {"role": "assistant", "tool_calls": [{"id": "call_1", "function": {"name": "validate_script"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "Script approved"},
            {"role": "assistant", "tool_calls": [{"id": "call_2", "function": {"name": "run_script"}}]},
            {"role": "tool", "tool_call_id": "call_2", "content": "chronyd restarted"},
        ])

        chunks = await _collect_sse_chunks(url, body)

        content_chunk = chunks[1]
        text = content_chunk["choices"][0]["delta"]["content"]
        assert "chronyd" in text
        assert "192.168.64.115" in text

        last_chunk = chunks[-1]
        assert last_chunk["choices"][0]["finish_reason"] == "stop"

    async def test_exhausted_scenario_returns_default_text(self, chat_server):
        """Requests beyond the scenario steps return a default message."""
        mock, url = chat_server
        body = _make_request([
            {"role": "user", "content": "restart chronyd"},
            {"role": "tool", "tool_call_id": "c1", "content": "result 1"},
            {"role": "tool", "tool_call_id": "c2", "content": "result 2"},
            {"role": "tool", "tool_call_id": "c3", "content": "result 3"},
        ])

        chunks = await _collect_sse_chunks(url, body)

        content_chunk = chunks[1]
        text = content_chunk["choices"][0]["delta"]["content"]
        assert "Mock scenario complete" in text

    async def test_sse_format(self, chat_server):
        """Verify the raw SSE stream format."""
        mock, url = chat_server
        body = _make_request([
            {"role": "user", "content": "restart chronyd on 192.168.64.115"},
        ])

        lines = []
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, json=body) as response:
                assert response.headers["content-type"] == "text/event-stream"
                async for line in response.aiter_lines():
                    if line:
                        lines.append(line)

        assert all(line.startswith("data: ") for line in lines)
        assert lines[-1] == "data: [DONE]"

    async def test_arguments_are_json_string(self, chat_server):
        """Tool call arguments should be a JSON string, not a dict."""
        mock, url = chat_server
        body = _make_request([
            {"role": "user", "content": "restart chronyd on 192.168.64.115"},
        ])

        chunks = await _collect_sse_chunks(url, body)

        args_chunk = chunks[1]
        arguments = args_chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(arguments, str)
        parsed = json.loads(arguments)
        assert parsed["script_type"] == "bash"
        assert "systemctl restart chronyd" in parsed["script"]

    async def test_requests_are_logged(self, chat_server):
        """The mock logs all chat requests for inspection."""
        mock, url = chat_server
        body = _make_request([
            {"role": "user", "content": "restart chronyd on 192.168.64.115"},
        ])

        await _collect_sse_chunks(url, body)

        assert len(mock.chat_requests) == 1
        assert mock.chat_requests[0]["model"] == "mock"
