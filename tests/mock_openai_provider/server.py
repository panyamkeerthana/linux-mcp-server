"""Mock OpenAI server for deterministic gatekeeper and chat model testing.

A lightweight HTTP server that implements:
- POST /v1/responses -- for the gatekeeper (substring matching, returns text)
- POST /v1/chat/completions -- for the chat model (scripted scenarios, returns
  tool calls or text as SSE streams)

No real inference -- just pattern matching and scripted sequences for
predictable, repeatable test results.

Usage as a standalone server:
    python -m tests.mock_openai_provider.server --port 8080

Usage in tests:
    See the start_mock_server() helper at the bottom of this file.
"""

import argparse
import asyncio
import json
import logging

from pathlib import Path
from typing import Any

import yaml

from aiohttp import web


logger = logging.getLogger(__name__)


# Gatekeeper rules


Rule = tuple[str, str]

DEFAULT_RULES: list[Rule] = [
    ("rm -rf /", '{"status": "DANGEROUS", "detail": "destructive command"}'),
    ("rm -rf", '{"status": "DANGEROUS", "detail": "destructive command"}'),
    ("mkfs", '{"status": "DANGEROUS", "detail": "filesystem format"}'),
    ("dd if=", '{"status": "DANGEROUS", "detail": "raw disk write"}'),
    (":(){ :|:& };:", '{"status": "MALICIOUS", "detail": "fork bomb"}'),
    ("", '{"status": "OK", "detail": ""}'),
]


def _match_rule(text: str, rules: list[Rule]) -> str:
    """Find the first rule whose substring appears in text. Return the response."""
    for substring, response in rules:
        if substring in text:
            return response
    return '{"status": "OK", "detail": ""}'


def _build_response(output_text: str) -> dict[str, Any]:
    """Build a minimal OpenAI Responses API response.

    Matches the format expected by OpenAIResponse in openai_client.py:
    output[] contains message items with content[] containing output_text items.
    """
    return {
        "id": "mock-response",
        "object": "response",
        "output": [
            {
                "type": "message",
                "id": "msg-mock",
                "content": [
                    {
                        "type": "output_text",
                        "text": output_text,
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
        },
    }


# Scenario loading


VALID_STEP_TYPES = {"tool_call", "text"}
REQUIRED_FIELDS = {
    "tool_call": ["tool_name", "arguments"],
    "text": ["content"],
}


def load_scenario(path: Path) -> list[dict[str, Any]]:
    """Load a scenario from a YAML file. Returns a list of validated step dicts."""
    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level, got {type(data).__name__}")

    steps = data.get("steps")
    if not isinstance(steps, list):
        raise ValueError(f"{path}: 'steps' must be a list, got {type(steps).__name__}")

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"{path}: step {i} is not a dict")

        step_type = step.get("type")
        if step_type not in VALID_STEP_TYPES:
            raise ValueError(
                f"{path}: step {i} has invalid type '{step_type}', expected one of {VALID_STEP_TYPES}"
            )

        for field in REQUIRED_FIELDS[step_type]:
            if field not in step:
                raise ValueError(f"{path}: step {i} (type '{step_type}') is missing required field '{field}'")

    return steps


def _build_tool_call_chunks(step: dict[str, Any]) -> list[str]:
    """Build SSE chunks for a tool call step."""
    tool_name = step["tool_name"]
    call_id = step.get("call_id", "call_mock_001")
    arguments = json.dumps(step["arguments"])

    chunks = [
        # First chunk: role + tool call with name
        json.dumps({
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [{
                        "index": 0,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": tool_name, "arguments": ""},
                    }],
                },
            }],
        }),
        # Second chunk: the arguments
        json.dumps({
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": arguments},
                    }],
                },
            }],
        }),
        # Final chunk: finish reason
        json.dumps({
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }),
    ]
    return chunks


def _build_text_chunks(step: dict[str, Any]) -> list[str]:
    """Build SSE chunks for a text step."""
    content = step["content"]

    chunks = [
        # First chunk: role
        json.dumps({
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
            }],
        }),
        # Second chunk: the content
        json.dumps({
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {"content": content},
            }],
        }),
        # Final chunk: finish reason
        json.dumps({
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }),
    ]
    return chunks


# Server


class MockOpenAIServer:
    """Mock server that handles gatekeeper and chat model requests.

    - POST /v1/responses: gatekeeper (substring matching, returns text)
    - POST /v1/chat/completions: chat model (scripted scenario, returns SSE stream)
    """

    def __init__(
        self,
        rules: list[Rule] | None = None,
        scenario: list[dict[str, Any]] | None = None,
    ):
        self.rules = rules or DEFAULT_RULES
        self.scenario = scenario or []
        self.requests: list[dict[str, Any]] = []
        self.chat_requests: list[dict[str, Any]] = []

    async def handle_responses(self, request: web.Request) -> web.Response:
        """Handle POST /v1/responses (gatekeeper)."""
        body = await request.json()
        self.requests.append(body)

        prompt = body.get("input", "")
        matched_response = _match_rule(prompt, self.rules)

        logger.info("Mock gatekeeper request for model=%s", body.get("model", "unknown"))
        logger.info("Matched rule -> %s", matched_response)

        return web.json_response(_build_response(matched_response))

    async def handle_chat_completions(self, request: web.Request) -> web.StreamResponse:
        """Handle POST /v1/chat/completions (chat model).

        Figures out which step in the scenario to return by counting
        the number of tool result messages in the conversation.
        Returns the response as an SSE stream.
        """
        body = await request.json()
        self.chat_requests.append(body)

        # Count tool results to figure out which step we're on
        messages = body.get("messages", [])
        tool_result_count = sum(1 for m in messages if m.get("role") == "tool")
        step_index = tool_result_count

        if step_index >= len(self.scenario):
            logger.warning("Scenario exhausted at step %d", step_index)
            step = {"type": "text", "content": "Mock scenario complete."}
        else:
            step = self.scenario[step_index]
            logger.info("Chat completions step %d/%d", step_index + 1, len(self.scenario))

        # Build SSE chunks based on step type
        if step["type"] == "tool_call":
            chunks = _build_tool_call_chunks(step)
        else:
            chunks = _build_text_chunks(step)

        # Send as SSE stream
        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
        )
        await response.prepare(request)

        for chunk in chunks:
            await response.write(f"data: {chunk}\n\n".encode())

        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    def create_app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/v1/responses", self.handle_responses)
        app.router.add_post("/v1/chat/completions", self.handle_chat_completions)
        return app


async def start_mock_server(
    rules: list[Rule] | None = None,
    scenario: list[dict[str, Any]] | None = None,
    host: str = "127.0.0.1",
    port: int = 0,
) -> tuple[MockOpenAIServer, web.AppRunner, str]:
    """Start a mock server on the given port (0 = random free port).

    Returns (server_instance, runner, base_url).
    The caller should call `await runner.cleanup()` when done.
    """
    mock = MockOpenAIServer(rules, scenario)
    app = mock.create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()

    sockets = site._server.sockets  # type: ignore[union-attr]
    actual_port = sockets[0].getsockname()[1]
    base_url = f"http://{host}:{actual_port}/v1"

    logger.info("Mock OpenAI server running at %s", base_url)
    return mock, runner, base_url


def main():
    parser = argparse.ArgumentParser(description="Mock OpenAI server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--scenario", type=str, default=None, help="Path to a scenario YAML file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    scenario = None
    if args.scenario:
        scenario = load_scenario(Path(args.scenario))
        logger.info("Loaded scenario with %d steps from %s", len(scenario), args.scenario)

    async def run():
        _mock, runner, base_url = await start_mock_server(
            host=args.host, port=args.port, scenario=scenario,
        )
        logger.info("Mock OpenAI server listening at %s", base_url)
        try:
            await asyncio.Event().wait()
        finally:
            await runner.cleanup()

    asyncio.run(run())


if __name__ == "__main__":
    main()
