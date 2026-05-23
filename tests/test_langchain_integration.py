"""M4 — integration tests against real LangChain primitives.

These exercise PhionyxCallbackHandler the way downstream users will:
wired into a real ``RunnableLambda`` / tool / fake LLM and invoked
through the public LangChain API. The goal is to catch callback-API
signature drift between langchain-core minor versions before the
PyPI alpha ships.

Differences from ``test_envelope_chain.py``:
- Unit tests there call ``handler.on_chain_start(...)`` directly with
  synthetic run_ids.
- These tests let LangChain itself drive the callbacks. We assert on
  the resulting envelope chain rather than on direct method calls.

If LangChain renames a callback method or changes its argument order
in a future release, these tests catch it; the unit tests would not.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

# Suppress LangChain's deprecation warnings — we're exercising the
# public API contract, not a particular implementation pattern.
warnings.filterwarnings("ignore", category=DeprecationWarning)


def _make_handler(tmp_path: Path, trace_id: str | None = None):
    from phionyx_langchain_langgraph import FilesystemEnvelopeStore, PhionyxCallbackHandler

    return PhionyxCallbackHandler(
        trace_id=trace_id,
        store=FilesystemEnvelopeStore(root=tmp_path),
    )


# ---------------------------------------------------------------------------
# RunnableLambda — the simplest real chain
# ---------------------------------------------------------------------------


def test_runnable_lambda_emits_chain_events(tmp_path: Path) -> None:
    """A bare RunnableLambda invocation emits at least one chain_start + chain_end."""
    from langchain_core.runnables import RunnableLambda

    handler = _make_handler(tmp_path)

    runnable = RunnableLambda(lambda x: x.upper())
    result = runnable.invoke("hello", config={"callbacks": [handler]})

    assert result == "HELLO"

    event_types = [e.event_type for e in handler.events]
    assert "chain_start" in event_types
    assert "chain_end" in event_types

    # Every event also produced an envelope on the chain.
    assert len(handler.envelopes) == len(handler.events)


def test_runnable_lambda_envelope_chain_verifies(tmp_path: Path) -> None:
    """End-to-end: real LangChain run → chain on disk → verify_chain() ok."""
    from langchain_core.runnables import RunnableLambda

    handler = _make_handler(tmp_path, trace_id="integration-runnable")

    runnable = RunnableLambda(lambda x: f"processed: {x}")
    runnable.invoke("input-1", config={"callbacks": [handler]})

    report = handler.verify_chain()
    assert report["ok"] is True
    assert report["envelope_count"] == len(handler.envelopes)
    assert report["errors"] == []


def test_chained_runnables_emit_nested_events(tmp_path: Path) -> None:
    """A | B chain emits events for both runnables with parent_run_id linkage."""
    from langchain_core.runnables import RunnableLambda

    handler = _make_handler(tmp_path)

    upper = RunnableLambda(lambda x: x.upper())
    exclaim = RunnableLambda(lambda x: f"{x}!")
    pipeline = upper | exclaim

    result = pipeline.invoke("hello", config={"callbacks": [handler]})
    assert result == "HELLO!"

    event_types = [e.event_type for e in handler.events]
    # At least 2 chain_start + 2 chain_end pairs from the composition.
    assert event_types.count("chain_start") >= 2
    assert event_types.count("chain_end") >= 2

    # parent_run_id links exist somewhere in the chain (at least one
    # child event has a parent_run_id pointing at an earlier event).
    has_parented_event = any(e.parent_run_id is not None for e in handler.events)
    assert has_parented_event, "expected at least one child event with a parent_run_id"


# ---------------------------------------------------------------------------
# Tool — exercises on_tool_start / on_tool_end
# ---------------------------------------------------------------------------


def test_tool_invocation_emits_tool_events(tmp_path: Path) -> None:
    """A real LangChain @tool-decorated function emits tool_start + tool_end."""
    from langchain_core.tools import tool

    handler = _make_handler(tmp_path)

    @tool
    def shout(text: str) -> str:
        """Return the input in caps."""
        return text.upper()

    result = shout.invoke({"text": "hello"}, config={"callbacks": [handler]})
    assert result == "HELLO"

    event_types = [e.event_type for e in handler.events]
    assert "tool_start" in event_types
    assert "tool_end" in event_types


def test_tool_error_emits_tool_error_event(tmp_path: Path) -> None:
    """When the tool raises, on_tool_error fires + envelope reflects it."""
    from langchain_core.tools import tool

    handler = _make_handler(tmp_path)

    @tool
    def boom(text: str) -> str:
        """Always raise."""
        raise RuntimeError(f"boom: {text}")

    with pytest.raises(Exception):
        boom.invoke({"text": "x"}, config={"callbacks": [handler]})

    event_types = [e.event_type for e in handler.events]
    assert "tool_error" in event_types

    # The error envelope carries the error_type + error_message we recorded.
    error_envelopes = [
        e for e in handler.envelopes if e["subject"]["event_type"] == "tool_error"
    ]
    assert len(error_envelopes) >= 1
    err_data = error_envelopes[0]["message"]["payload"]["data"]
    assert err_data["error_type"] == "RuntimeError"
    assert "boom: x" in err_data["error_message"]


# ---------------------------------------------------------------------------
# FakeListLLM — exercises on_llm_start / on_llm_end without API keys
# ---------------------------------------------------------------------------


def test_fake_llm_emits_llm_events(tmp_path: Path) -> None:
    from langchain_core.language_models.fake import FakeListLLM

    handler = _make_handler(tmp_path)
    llm = FakeListLLM(responses=["mocked answer"])

    result = llm.invoke("any prompt", config={"callbacks": [handler]})
    assert result == "mocked answer"

    event_types = [e.event_type for e in handler.events]
    # Either llm_* or chat_model_* surface — FakeListLLM uses llm_*.
    assert any("llm" in t for t in event_types), f"got: {event_types}"


def test_fake_llm_chained_with_lambda(tmp_path: Path) -> None:
    """LLM | post-process — exercises mixed event types in one chain."""
    from langchain_core.language_models.fake import FakeListLLM
    from langchain_core.runnables import RunnableLambda

    handler = _make_handler(tmp_path)
    llm = FakeListLLM(responses=["TWELVE"])
    parse_lower = RunnableLambda(lambda x: x.lower())
    pipeline = llm | parse_lower

    result = pipeline.invoke("how many?", config={"callbacks": [handler]})
    assert result == "twelve"

    event_types = set(e.event_type for e in handler.events)
    assert "chain_start" in event_types
    assert "chain_end" in event_types

    # Chain still verifies after a mixed-event run.
    report = handler.verify_chain()
    assert report["ok"] is True


# ---------------------------------------------------------------------------
# Persistence + JSONL export round-trip from a real run
# ---------------------------------------------------------------------------


def test_real_run_persists_and_exports_jsonl(tmp_path: Path) -> None:
    """End-to-end: real LangChain run → persisted chain → JSONL → re-verify."""
    from langchain_core.runnables import RunnableLambda

    from phionyx_langchain_langgraph import verify_chain as module_verify_chain

    handler = _make_handler(tmp_path, trace_id="integration-export")

    upper = RunnableLambda(lambda x: x.upper())
    exclaim = RunnableLambda(lambda x: f"{x}!")
    (upper | exclaim).invoke("phionyx", config={"callbacks": [handler]})

    out = tmp_path / "exported.jsonl"
    count = handler.export_envelopes(out)
    assert count == len(handler.envelopes)

    # Re-load from JSONL and re-verify with the module-level helper.
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    reloaded = [json.loads(line) for line in lines]
    report = module_verify_chain(reloaded)
    assert report["ok"] is True
    assert report["envelope_count"] == count


# ---------------------------------------------------------------------------
# Supervisor + LangChain handler in one real run — F5 prerequisite
# ---------------------------------------------------------------------------


def test_supervisor_plus_real_langchain_child(tmp_path: Path) -> None:
    """Supervisor registers a child; child runs a real LangChain pipeline; both chains verify."""
    from langchain_core.runnables import RunnableLambda

    from phionyx_langchain_langgraph import (
        FilesystemEnvelopeStore,
        PhionyxCallbackHandler,
        PhionyxLangGraphSupervisor,
    )

    store = FilesystemEnvelopeStore(root=tmp_path)
    sup = PhionyxLangGraphSupervisor(parent_trace_id="integration-sup", store=store)

    researcher_trace = sup.register(child_node="researcher")
    sup.handoff(from_node="supervisor", to_node="researcher", payload={"task": "uppercase 'phionyx'"})

    child = PhionyxCallbackHandler(trace_id=researcher_trace, store=store)
    runnable = RunnableLambda(lambda x: x.upper())
    result = runnable.invoke("phionyx", config={"callbacks": [child]})
    assert result == "PHIONYX"

    # Both chains verify independently.
    assert sup.verify_chain()["ok"] is True
    assert child.verify_chain()["ok"] is True

    # Supervisor chain: register + handoff (2 envelopes).
    assert len(sup.envelopes) == 2
    # Child chain: at least chain_start + chain_end from the real run.
    child_event_types = [e.event_type for e in child.events]
    assert "chain_start" in child_event_types
    assert "chain_end" in child_event_types

    # Sibling directories under one store root.
    assert (tmp_path / "integration-sup").is_dir()
    assert (tmp_path / "integration-sup:child:researcher").is_dir()
