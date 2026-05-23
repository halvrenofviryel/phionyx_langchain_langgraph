"""M3 tests for the LangGraph supervisor envelope chain.

Verifies:
- ``register`` returns a derived child_trace_id following the
  documented convention.
- Re-registering an already-registered child returns the same trace_id
  and still emits an envelope (audit-faithful).
- ``handoff`` emits a ``supervisor_handoff`` envelope with both nodes
  and their child trace_ids embedded.
- The supervisor chain uses the same audit-chain infra as the
  callback handler: signed, hash-chained, verifiable, exportable,
  tamper-detectable.
- The supervisor and a child callback handler can share a single
  ``FilesystemEnvelopeStore`` root — their chains live under
  separate directories named by trace_id.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest


def _make_supervisor(tmp_path: Path, parent_trace_id: str | None = None):
    from phionyx_langchain_langgraph import FilesystemEnvelopeStore, PhionyxLangGraphSupervisor

    return PhionyxLangGraphSupervisor(
        parent_trace_id=parent_trace_id,
        store=FilesystemEnvelopeStore(root=tmp_path),
    )


# ---------------------------------------------------------------------------
# Derived child trace_id convention
# ---------------------------------------------------------------------------


def test_derive_child_trace_id_format() -> None:
    from phionyx_langchain_langgraph.langgraph_handler import derive_child_trace_id

    assert derive_child_trace_id("parent-123", "researcher") == "parent-123:child:researcher"


def test_register_returns_derived_child_trace_id(tmp_path: Path) -> None:
    sup = _make_supervisor(tmp_path, parent_trace_id="sup-001")
    assigned = sup.register(child_node="researcher")
    assert assigned == "sup-001:child:researcher"
    assert sup.children["researcher"] == "sup-001:child:researcher"


def test_register_accepts_explicit_child_trace_id(tmp_path: Path) -> None:
    sup = _make_supervisor(tmp_path)
    assigned = sup.register(child_node="writer", child_trace_id="custom-writer-trace")
    assert assigned == "custom-writer-trace"
    assert sup.children["writer"] == "custom-writer-trace"


def test_register_returns_same_id_on_re_register(tmp_path: Path) -> None:
    sup = _make_supervisor(tmp_path)
    first = sup.register(child_node="researcher")
    second = sup.register(child_node="researcher")
    assert first == second
    # But both calls produce envelopes — the audit reflects the duplicate.
    assert len(sup.envelopes) == 2
    assert all(e["subject"]["event_type"] == "supervisor_register" for e in sup.envelopes)


def test_register_rejects_empty_child_node(tmp_path: Path) -> None:
    sup = _make_supervisor(tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        sup.register(child_node="")


# ---------------------------------------------------------------------------
# Envelope emission
# ---------------------------------------------------------------------------


def test_register_emits_supervisor_register_envelope(tmp_path: Path) -> None:
    sup = _make_supervisor(tmp_path, parent_trace_id="sup-001")
    sup.register(child_node="researcher")

    assert len(sup.envelopes) == 1
    env = sup.envelopes[0]
    assert env["schema"] == "phionyx.langchain_event_envelope.v1"
    assert env["subject"]["event_type"] == "supervisor_register"
    assert env["subject"]["turn_index"] == 0

    inner = env["message"]
    assert inner["protocol"] == "langgraph"
    assert inner["trace_id"] == "sup-001"
    assert inner["payload"]["data"]["child_node"] == "researcher"
    assert inner["payload"]["data"]["child_trace_id"] == "sup-001:child:researcher"
    assert inner["payload"]["data"]["parent_trace_id"] == "sup-001"


def test_handoff_emits_supervisor_handoff_envelope(tmp_path: Path) -> None:
    sup = _make_supervisor(tmp_path)
    sup.register(child_node="researcher")
    sup.register(child_node="writer")
    sup.handoff(from_node="researcher", to_node="writer", payload={"sources": [1, 2]})

    # 2 register + 1 handoff = 3 envelopes total
    assert len(sup.envelopes) == 3

    handoff_env = sup.envelopes[2]
    assert handoff_env["subject"]["event_type"] == "supervisor_handoff"
    data = handoff_env["message"]["payload"]["data"]
    assert data["from_node"] == "researcher"
    assert data["to_node"] == "writer"
    assert data["from_child_trace_id"] == sup.children["researcher"]
    assert data["to_child_trace_id"] == sup.children["writer"]
    assert data["payload"] == {"sources": [1, 2]}


def test_handoff_with_unregistered_node_uses_none_for_trace_id(tmp_path: Path) -> None:
    """Supervisor itself isn't a registered child — its trace_id is None."""
    sup = _make_supervisor(tmp_path)
    sup.register(child_node="researcher")
    env = sup.handoff(from_node="supervisor", to_node="researcher", payload={"task": "go"})

    data = env["message"]["payload"]["data"]
    assert data["from_node"] == "supervisor"
    assert data["from_child_trace_id"] is None  # supervisor not registered as child
    assert data["to_child_trace_id"] == sup.children["researcher"]


def test_handoff_rejects_empty_nodes(tmp_path: Path) -> None:
    sup = _make_supervisor(tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        sup.handoff(from_node="", to_node="researcher")
    with pytest.raises(ValueError, match="non-empty"):
        sup.handoff(from_node="researcher", to_node="")


# ---------------------------------------------------------------------------
# Chain integrity (supervisor uses same audit-chain as callback handler)
# ---------------------------------------------------------------------------


def test_supervisor_chain_hash_continuity(tmp_path: Path) -> None:
    from phionyx_langchain_langgraph.audit_chain import GENESIS_HASH

    sup = _make_supervisor(tmp_path)
    sup.register(child_node="a")
    sup.register(child_node="b")
    sup.handoff(from_node="a", to_node="b")
    sup.handoff(from_node="b", to_node="a")

    envs = sup.envelopes
    assert envs[0]["integrity"]["previous"] == GENESIS_HASH
    for i in range(1, len(envs)):
        assert envs[i]["integrity"]["previous"] == envs[i - 1]["integrity"]["current"]
    assert [e["subject"]["turn_index"] for e in envs] == [0, 1, 2, 3]


def test_supervisor_verify_chain_passes(tmp_path: Path) -> None:
    sup = _make_supervisor(tmp_path)
    sup.register(child_node="a")
    sup.handoff(from_node="a", to_node="a")

    report = sup.verify_chain()
    assert report["ok"] is True
    assert report["envelope_count"] == 2


def test_supervisor_verify_chain_detects_tamper(tmp_path: Path) -> None:
    sup = _make_supervisor(tmp_path, parent_trace_id="tamper-sup")
    sup.register(child_node="a")
    sup.handoff(from_node="a", to_node="a")

    target = tmp_path / "tamper-sup" / "00000001.json"
    env = json.loads(target.read_text(encoding="utf-8"))
    env["message"]["payload"]["data"]["payload"] = {"injected": True}
    target.write_text(json.dumps(env), encoding="utf-8")

    report = sup.verify_chain()
    assert report["ok"] is False
    assert any("current_hash mismatch" in e for e in report["errors"])


def test_supervisor_export_envelopes_writes_jsonl(tmp_path: Path) -> None:
    sup = _make_supervisor(tmp_path)
    sup.register(child_node="a")
    sup.handoff(from_node="a", to_node="a", payload={"k": "v"})

    out = tmp_path / "supervisor.jsonl"
    n = sup.export_envelopes(str(out))
    assert n == 2

    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        env = json.loads(line)
        assert env["schema"] == "phionyx.langchain_event_envelope.v1"


# ---------------------------------------------------------------------------
# Supervisor + child callback handler integration — F5 prep
# ---------------------------------------------------------------------------


def test_supervisor_and_child_share_store_under_separate_trace_dirs(tmp_path: Path) -> None:
    """The supervisor chain and a child chain coexist under one store root.

    This is the F5 prerequisite — multi-agent verifiers walk parent +
    child chains together. Directory naming alone (parent vs
    parent:child:<node>) carries the relationship without side-channel
    metadata.
    """
    from phionyx_langchain_langgraph import (
        FilesystemEnvelopeStore,
        PhionyxCallbackHandler,
        PhionyxLangGraphSupervisor,
    )

    store = FilesystemEnvelopeStore(root=tmp_path)
    sup = PhionyxLangGraphSupervisor(parent_trace_id="sup-multi", store=store)

    researcher_trace = sup.register(child_node="researcher")
    sup.handoff(from_node="supervisor", to_node="researcher", payload={"task": "go"})

    # Caller wires a child handler to the derived child_trace_id.
    child = PhionyxCallbackHandler(trace_id=researcher_trace, store=store)
    child.on_chain_start(serialized={"name": "Researcher"}, inputs={"q": "?"}, run_id=uuid.uuid4())
    child.on_chain_end(outputs={"answer": "ok"}, run_id=uuid.uuid4())

    # Both chains exist on disk as siblings.
    parent_dir = tmp_path / "sup-multi"
    child_dir = tmp_path / "sup-multi:child:researcher"
    assert parent_dir.is_dir()
    assert child_dir.is_dir()

    # Each chain verifies independently.
    sup_report = sup.verify_chain()
    child_report = child.verify_chain()
    assert sup_report["ok"] is True
    assert child_report["ok"] is True
    assert sup_report["envelope_count"] == 2  # register + handoff
    assert child_report["envelope_count"] == 2  # chain_start + chain_end


def test_supervisor_envelopes_property_returns_copy(tmp_path: Path) -> None:
    sup = _make_supervisor(tmp_path)
    sup.register(child_node="a")

    snapshot = sup.envelopes
    snapshot.clear()
    assert len(sup.envelopes) == 1
