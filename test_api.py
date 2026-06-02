"""
Integration tests for /v1/chat/completions and /v1/models endpoints.
Uses respx to mock the kagent A2A HTTP calls.

kagent events are wrapped in a JSON-RPC envelope and use the pre-v1.0
`kind` discriminator. The agent's answer is streamed as `working` text
partials (→ content); the same text is re-sent as a non-partial copy and as
an artifact-update, both of which the proxy de-duplicates.
"""

import json

import httpx
import respx
from fastapi.testclient import TestClient

from conftest import (
    artifact_event,
    completed_event,
    failed_event,
    sse_response,
    working_event,
)
from kagent_a2a_proxy import hitl
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
# /v1/chat/completions — non-streaming reconstructs the streamed answer
# ---------------------------------------------------------------------------


@respx.mock
def test_non_streaming_completion():
    events = [
        working_event("Done — ", partial=True),
        working_event("interface is healthy.", partial=True),
        working_event("Done — interface is healthy.", partial=False),  # aggregate
        artifact_event("Done — interface is healthy."),  # duplicate
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
    # The streamed partials reconstruct the answer; the aggregate copy and the
    # artifact are de-duplicated, so it appears exactly once.
    assert content == "Done — interface is healthy."


# ---------------------------------------------------------------------------
# /v1/chat/completions — streaming: answer → content, no duplication
# ---------------------------------------------------------------------------


@respx.mock
def test_streaming_answer_goes_to_content_without_duplication():
    events = [
        working_event("Result ", partial=True),
        working_event("okandgo", partial=True),
        working_event("Result okandgo", partial=False),  # aggregate, skipped
        artifact_event("Result okandgo"),  # duplicate, dropped
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
        body = "".join(line for line in r.iter_lines() if line.startswith("data:"))

    assert '"role":"assistant"' in body or '"role": "assistant"' in body
    assert "[DONE]" in body
    # Answer streams as content deltas, once (aggregate + artifact deduped).
    assert '"content":"okandgo"' in body
    assert body.count("okandgo") == 1
    # No populated reasoning channel for a pure answer (only null keys).
    assert '"reasoning_content":"' not in body


@respx.mock
def test_streaming_thought_goes_to_reasoning():
    events = [
        working_event("let me think", thought=True, partial=True),
        working_event("the answer", partial=True),
        artifact_event("the answer"),
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
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        body = "".join(line for line in r.iter_lines() if line.startswith("data:"))

    assert '"reasoning_content":"' in body
    assert "let me think" in body
    assert '"content":"the answer"' in body
    assert body.count("the answer") == 1


@respx.mock
def test_streaming_failed_state_surfaces_error():
    events = [
        working_event("trying", partial=True),
        failed_event("tool 'get_telemetry' timed out"),
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
            "messages": [{"role": "user", "content": "check"}],
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        body = "".join(line for line in r.iter_lines() if line.startswith("data:"))

    assert "Agent run failed" in body
    assert "timed out" in body
    assert "[DONE]" in body


# ---------------------------------------------------------------------------
# /v1/chat/completions — human-in-the-loop approve / deny resume
# ---------------------------------------------------------------------------


def _approval_history(marker: str, reply: str) -> list[dict]:
    return [
        {"role": "user", "content": "restart the router"},
        {"role": "assistant", "content": "⚠️ Approval required" + marker},
        {"role": "user", "content": reply},
    ]


@respx.mock
def test_approve_reply_resumes_paused_task(monkeypatch):
    monkeypatch.setattr(settings, "hitl_secret", "s3cr3t")
    marker = hitl.encode_marker("task-7", "ctx-7", "s3cr3t")
    route = respx.post(KAGENT_URL).mock(
        return_value=httpx.Response(
            200,
            content=sse_response(
                [
                    working_event("Restarted ", partial=True),
                    working_event("the router.", partial=True),
                    completed_event(),
                ]
            ),
            headers={"content-type": "text/event-stream"},
        )
    )

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "agent-one",
            "messages": _approval_history(marker, "approve"),
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        body = "".join(line for line in r.iter_lines() if line.startswith("data:"))

    # The proxy sent a decision DataPart routed by taskId + contextId.
    assert route.called
    sent = json.loads(route.calls.last.request.content)
    message = sent["params"]["message"]
    assert message["taskId"] == "task-7"
    assert message["contextId"] == "ctx-7"
    assert message["parts"][0]["data"]["decision_type"] == "approve"
    # The resumed answer streams back to the reply.
    assert "Restarted" in body
    assert "the router." in body


@respx.mock
def test_ambiguous_reply_reprompts_without_calling_kagent(monkeypatch):
    monkeypatch.setattr(settings, "hitl_secret", "s3cr3t")
    marker = hitl.encode_marker("task-7", "ctx-7", "s3cr3t")
    route = respx.post(KAGENT_URL).mock(
        return_value=httpx.Response(
            200, content=b"", headers={"content-type": "text/event-stream"}
        )
    )

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "agent-one",
            "messages": _approval_history(marker, "what will it do?"),
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        body = "".join(line for line in r.iter_lines() if line.startswith("data:"))

    assert not route.called  # no kagent call for an ambiguous reply
    assert "pending approval" in body
    assert "kagent-hitl" in body  # marker re-embedded so state survives


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


# ---------------------------------------------------------------------------
# /v1/chat/completions — SSE read timeout disabled
# ---------------------------------------------------------------------------


@respx.mock
def test_streaming_disables_read_timeout():
    # kagent can go silent for a long time between SSE events (long-running
    # tools, human-in-the-loop approval); a read timeout would kill the stream.
    route = respx.post(KAGENT_URL).mock(
        return_value=httpx.Response(
            200,
            content=sse_response([artifact_event("ok"), completed_event()]),
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
        list(r.iter_lines())

    timeout = route.calls.last.request.extensions["timeout"]
    assert timeout["read"] is None
    assert timeout["connect"] == settings.request_timeout
    assert timeout["write"] == settings.request_timeout
    assert timeout["pool"] == settings.request_timeout
