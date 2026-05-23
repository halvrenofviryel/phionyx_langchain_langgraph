"""M1 smoke tests for phionyx-langchain-langgraph.

These tests verify the package-level contract that survives M1 → M2:
- The package is importable.
- PhionyxCallbackHandler subclasses LangChain's BaseCallbackHandler.
- The handler records events when LangChain callback methods fire.
- The LangGraph supervisor surface raises NotImplementedError until M3.

M2-specific behavior (envelope emission, hash chain, signature
verification, JSONL export) lives in ``test_envelope_chain.py``.
"""

from __future__ import annotations

import uuid

import pytest

# These tests reuse a temporary store so they don't pollute the user's
# ~/.phionyx/langchain_audit dir.


def _make_handler(tmp_path):
    from phionyx_langchain_langgraph import FilesystemEnvelopeStore, PhionyxCallbackHandler

    store = FilesystemEnvelopeStore(root=tmp_path)
    return PhionyxCallbackHandler(store=store)


# ---------------------------------------------------------------------------
# Module-level smoke
# ---------------------------------------------------------------------------


def test_package_imports() -> None:
    import phionyx_langchain_langgraph as pkg

    assert pkg.__version__.startswith("0.1.0a1")
    assert hasattr(pkg, "PhionyxCallbackHandler")
    assert hasattr(pkg, "PhionyxLangGraphSupervisor")
    # M2 additions:
    assert hasattr(pkg, "FilesystemEnvelopeStore")
    assert hasattr(pkg, "HmacSigner")
    assert hasattr(pkg, "build_envelope")
    assert hasattr(pkg, "verify_chain")


def test_handler_subclasses_langchain_base() -> None:
    from langchain_core.callbacks import BaseCallbackHandler

    from phionyx_langchain_langgraph import PhionyxCallbackHandler

    assert issubclass(PhionyxCallbackHandler, BaseCallbackHandler)


# ---------------------------------------------------------------------------
# PhionyxCallbackHandler — base behavior preserved from M1
# ---------------------------------------------------------------------------


def test_handler_generates_trace_id_when_omitted(tmp_path) -> None:
    handler = _make_handler(tmp_path)
    assert handler.trace_id.startswith("phionyx-langchain-")
    assert len(handler.trace_id) > len("phionyx-langchain-")


def test_handler_accepts_explicit_trace_id(tmp_path) -> None:
    from phionyx_langchain_langgraph import FilesystemEnvelopeStore, PhionyxCallbackHandler

    handler = PhionyxCallbackHandler(
        trace_id="custom-trace-123",
        store=FilesystemEnvelopeStore(root=tmp_path),
    )
    assert handler.trace_id == "custom-trace-123"


def test_handler_records_chain_start_and_end(tmp_path) -> None:
    handler = _make_handler(tmp_path)
    run_id = uuid.uuid4()

    handler.on_chain_start(
        serialized={"name": "MyChain"},
        inputs={"question": "What is Phionyx?"},
        run_id=run_id,
    )
    handler.on_chain_end(
        outputs={"answer": "Runtime evidence layer."},
        run_id=run_id,
    )

    assert len(handler.events) == 2
    assert handler.events[0].event_type == "chain_start"
    assert handler.events[0].run_id == str(run_id)
    assert handler.events[0].parent_run_id is None
    assert handler.events[0].payload["inputs"]["question"] == "What is Phionyx?"
    assert handler.events[1].event_type == "chain_end"


def test_handler_records_tool_lifecycle(tmp_path) -> None:
    handler = _make_handler(tmp_path)
    run_id = uuid.uuid4()
    parent_run_id = uuid.uuid4()

    handler.on_tool_start(
        serialized={"name": "search"},
        input_str="what is phionyx",
        run_id=run_id,
        parent_run_id=parent_run_id,
    )
    handler.on_tool_end(
        output="Phionyx is a runtime evidence layer.",
        run_id=run_id,
        parent_run_id=parent_run_id,
    )

    assert len(handler.events) == 2
    assert handler.events[0].parent_run_id == str(parent_run_id)


def test_handler_records_llm_lifecycle(tmp_path) -> None:
    handler = _make_handler(tmp_path)
    run_id = uuid.uuid4()

    handler.on_llm_start(
        serialized={"name": "ChatOpenAI"},
        prompts=["Hello?"],
        run_id=run_id,
    )
    handler.on_llm_end(response="LLMResult(...)", run_id=run_id)

    assert len(handler.events) == 2
    assert handler.events[0].payload["prompts"] == ["Hello?"]


def test_handler_records_error_events(tmp_path) -> None:
    handler = _make_handler(tmp_path)
    handler.on_chain_error(error=ValueError("boom"), run_id=uuid.uuid4())

    assert len(handler.events) == 1
    assert handler.events[0].payload["error_type"] == "ValueError"
    assert handler.events[0].payload["error_message"] == "boom"


def test_handler_events_property_returns_copy(tmp_path) -> None:
    """The .events property returns a new list — mutations don't leak."""
    handler = _make_handler(tmp_path)
    handler.on_chain_start(serialized={}, inputs={}, run_id=uuid.uuid4())

    snapshot = handler.events
    snapshot.clear()
    assert len(handler.events) == 1


# ---------------------------------------------------------------------------
# LangGraph supervisor — constructor / accessor smoke (M3 behavioral tests
# live in test_supervisor_chain.py)
# ---------------------------------------------------------------------------


def test_supervisor_default_parent_trace_id_generated() -> None:
    from phionyx_langchain_langgraph import PhionyxLangGraphSupervisor

    sup = PhionyxLangGraphSupervisor()
    assert sup.parent_trace_id.startswith("phionyx-langgraph-")


def test_supervisor_accepts_explicit_parent_trace_id() -> None:
    from phionyx_langchain_langgraph import PhionyxLangGraphSupervisor

    sup = PhionyxLangGraphSupervisor(parent_trace_id="custom-parent")
    assert sup.parent_trace_id == "custom-parent"
