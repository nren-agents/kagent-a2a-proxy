"""
Integration tests for /v1/chat/completions and /v1/models endpoints.
Uses respx to mock the kagent A2A HTTP calls.

kagent events are wrapped in a JSON-RPC envelope and use the pre-v1.0
`kind` discriminator. The final assistant text arrives as an artifact-update,
not as a completed status-update message.
"""

import httpx
import respx
from fastapi.testclient import TestClient

from conftest import artifact_event, completed_event, sse_response, working_event
from kagent_a2a_proxy.config import settings
from kagent_a2a_proxy.main import app

client = TestClient(app)

KAGENT_URL = (
    f"{str(settings.kagent_base_url).rstrip('/')}/api/a2a"
    f"/{settings.kagent_namespace}/agent-one"
)


# ---------------------------------------------------------------------------
# /healthz/ready
# ---------------------------------------------------------------------------


def test_healthz():
    r = client.get("/healthz/ready")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------


def test_list_models():
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "list"
    model_ids = [m["id"] for m in data["data"]]
    assert "agent-one" in model_ids


# ---------------------------------------------------------------------------
# /v1/chat/completions — non-streaming returns artifact text
# ---------------------------------------------------------------------------


@respx.mock
def test_non_streaming_completion():
    events = [
        working_event("Thinking..."),
        artifact_event("Done — interface is healthy."),
        completed_event(),
    ]
    respx.post(KAGENT_URL).mock(
        return_value=httpx.Response(
            200,
            content=sse_response(events),
            headers={"content-type": "text/event-stream"},
        )
    )

    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "agent-one",
            "messages": [{"role": "user", "content": "Check interface ae-0/0/1"}],
            "stream": False,
        },
    )

    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    content = body["choices"][0]["message"]["content"]
    # Artifact text shows up as the assistant content; working text does not.
    assert content == "Done — interface is healthy."


# ---------------------------------------------------------------------------
# /v1/chat/completions — streaming
# ---------------------------------------------------------------------------


@respx.mock
def test_streaming_completion():
    events = [
        working_event("Running query..."),
        artifact_event("Result: ok"),
        completed_event(),
    ]
    respx.post(KAGENT_URL).mock(
        return_value=httpx.Response(
            200,
            content=sse_response(events),
            headers={"content-type": "text/event-stream"},
        )
    )

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "agent-one",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        lines = [line for line in r.iter_lines() if line.startswith("data:")]

    body = "\n".join(lines)
    assert '"role":"assistant"' in body or '"role": "assistant"' in body
    assert "[DONE]" in body
    # The artifact text reaches the client as a content delta.
    assert "Result: ok" in body
    # The working text reaches the client as a reasoning_content delta (LibreChat "Thinking" pane).
    assert "Running query" in body
    assert "reasoning_content" in body


# ---------------------------------------------------------------------------
# /v1/chat/completions — kagent 503 error
# ---------------------------------------------------------------------------


@respx.mock
def test_kagent_error_surfaced_in_stream():
    respx.post(KAGENT_URL).mock(
        return_value=httpx.Response(503, content=b"unavailable")
    )

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "agent-one",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        lines = "".join(r.iter_lines())

    assert "Error" in lines or "503" in lines
    assert "[DONE]" in lines
