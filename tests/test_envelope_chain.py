"""M2 tests for the envelope chain.

These verify the contract that callers actually depend on:
- Every LangChain callback event emits one signed envelope.
- Envelopes form a hash chain — each previous matches the prior current.
- The chain is persisted to disk and can be re-read.
- ``verify_chain()`` detects tamper.
- ``export_envelopes()`` writes a valid JSONL file.
- AgentMessageEnvelope is the inner record (founder-confirmed schema choice).
- HmacSigner produces deterministic signatures for a given current_hash.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest


def _make_handler(tmp_path: Path, trace_id: str | None = None):
    from phionyx_langchain_langgraph import FilesystemEnvelopeStore, PhionyxCallbackHandler

    store = FilesystemEnvelopeStore(root=tmp_path)
    return PhionyxCallbackHandler(trace_id=trace_id, store=store)


# ---------------------------------------------------------------------------
# Basic emission
# ---------------------------------------------------------------------------


def test_single_event_emits_one_envelope(tmp_path: Path) -> None:
    handler = _make_handler(tmp_path)
    handler.on_chain_start(serialized={"name": "Chain"}, inputs={"q": "?"}, run_id=uuid.uuid4())

    assert len(handler.envelopes) == 1
    env = handler.envelopes[0]
    assert env["schema"] == "phionyx.langchain_event_envelope.v1"
    assert env["subject"]["event_type"] == "chain_start"
    assert env["subject"]["turn_index"] == 0
    assert env["subject"]["runtime"] == "phionyx-langchain-langgraph"
    assert "message" in env
    assert "integrity" in env


def test_chain_hash_continuity(tmp_path: Path) -> None:
    handler = _make_handler(tmp_path)
    run_id = uuid.uuid4()

    handler.on_chain_start(serialized={}, inputs={}, run_id=run_id)
    handler.on_tool_start(serialized={}, input_str="x", run_id=uuid.uuid4(), parent_run_id=run_id)
    handler.on_tool_end(output="y", run_id=uuid.uuid4(), parent_run_id=run_id)
    handler.on_chain_end(outputs={}, run_id=run_id)

    envs = handler.envelopes
    assert len(envs) == 4
    # First envelope's previous is the genesis sentinel.
    from phionyx_langchain_langgraph.audit_chain import GENESIS_HASH

    assert envs[0]["integrity"]["previous"] == GENESIS_HASH
    # Each subsequent envelope's previous matches the prior current.
    for i in range(1, len(envs)):
        assert envs[i]["integrity"]["previous"] == envs[i - 1]["integrity"]["current"]
    # Turn indices are sequential from 0.
    assert [e["subject"]["turn_index"] for e in envs] == [0, 1, 2, 3]


def test_envelopes_persisted_to_disk(tmp_path: Path) -> None:
    handler = _make_handler(tmp_path, trace_id="persist-test")
    handler.on_chain_start(serialized={}, inputs={"q": "?"}, run_id=uuid.uuid4())
    handler.on_chain_end(outputs={"answer": "ok"}, run_id=uuid.uuid4())

    trace_dir = tmp_path / "persist-test"
    assert trace_dir.is_dir()
    # 00000000.json + 00000001.json + chain.jsonl
    files = sorted(p.name for p in trace_dir.iterdir())
    assert files == ["00000000.json", "00000001.json", "chain.jsonl"]

    # chain.jsonl indexes both envelopes.
    index_lines = (trace_dir / "chain.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(index_lines) == 2
    entry0 = json.loads(index_lines[0])
    entry1 = json.loads(index_lines[1])
    assert entry0["turn_index"] == 0
    assert entry1["turn_index"] == 1
    assert entry1["previous"] == entry0["current"]


def test_verify_chain_passes_on_clean_chain(tmp_path: Path) -> None:
    handler = _make_handler(tmp_path)
    for _ in range(5):
        handler.on_chain_start(serialized={}, inputs={}, run_id=uuid.uuid4())

    report = handler.verify_chain()
    assert report["ok"] is True
    assert report["envelope_count"] == 5
    assert report["errors"] == []


def test_verify_chain_detects_tampered_payload(tmp_path: Path) -> None:
    handler = _make_handler(tmp_path, trace_id="tamper-test")
    handler.on_chain_start(serialized={}, inputs={"q": "original"}, run_id=uuid.uuid4())
    handler.on_chain_end(outputs={"a": "original"}, run_id=uuid.uuid4())

    # Tamper with the persisted envelope — change a payload field.
    target = tmp_path / "tamper-test" / "00000000.json"
    env = json.loads(target.read_text(encoding="utf-8"))
    env["message"]["payload"]["data"]["inputs"]["q"] = "tampered"
    target.write_text(json.dumps(env), encoding="utf-8")

    # verify_chain re-reads from disk and re-derives the hash.
    report = handler.verify_chain()
    assert report["ok"] is False
    assert any("current_hash mismatch" in e for e in report["errors"])


def test_verify_chain_detects_broken_link(tmp_path: Path) -> None:
    handler = _make_handler(tmp_path, trace_id="broken-link-test")
    handler.on_chain_start(serialized={}, inputs={}, run_id=uuid.uuid4())
    handler.on_chain_end(outputs={}, run_id=uuid.uuid4())

    # Tamper with envelope[1]'s previous — should no longer match envelope[0]'s current.
    target = tmp_path / "broken-link-test" / "00000001.json"
    env = json.loads(target.read_text(encoding="utf-8"))
    env["integrity"]["previous"] = "sha256:" + "1" * 64
    target.write_text(json.dumps(env), encoding="utf-8")

    report = handler.verify_chain()
    assert report["ok"] is False
    # Both a previous_hash mismatch AND a current_hash mismatch should be reported
    # (current_hash is computed over payload + previous, and previous was tampered).
    assert any("previous_hash mismatch" in e for e in report["errors"])


# ---------------------------------------------------------------------------
# AgentMessageEnvelope inner record
# ---------------------------------------------------------------------------


def test_inner_record_is_agent_message_envelope(tmp_path: Path) -> None:
    """The ``message`` block must be a valid AgentMessageEnvelope dump."""
    from phionyx_core.contracts.envelopes import AgentMessageEnvelope

    handler = _make_handler(tmp_path)
    handler.on_chain_start(
        serialized={"name": "MyChain"},
        inputs={"question": "What is Phionyx?"},
        run_id=uuid.uuid4(),
    )

    env = handler.envelopes[0]
    # Re-hydrate the inner message via Pydantic — proves the shape is correct.
    msg = AgentMessageEnvelope(**env["message"])
    assert msg.protocol == "langchain"
    assert msg.trace_id == handler.trace_id
    assert msg.turn_id == 1
    assert msg.payload["event_type"] == "chain_start"
    assert msg.payload["data"]["inputs"]["question"] == "What is Phionyx?"
    assert msg.metadata["phionyx.event_type"] == "chain_start"


def test_turn_id_is_monotonic_per_handler(tmp_path: Path) -> None:
    handler = _make_handler(tmp_path)
    for _ in range(3):
        handler.on_chain_start(serialized={}, inputs={}, run_id=uuid.uuid4())

    turn_ids = [env["message"]["turn_id"] for env in handler.envelopes]
    assert turn_ids == [1, 2, 3]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_export_envelopes_writes_valid_jsonl(tmp_path: Path) -> None:
    handler = _make_handler(tmp_path)
    handler.on_chain_start(serialized={}, inputs={"q": "?"}, run_id=uuid.uuid4())
    handler.on_tool_start(serialized={}, input_str="x", run_id=uuid.uuid4())
    handler.on_chain_end(outputs={"a": "ok"}, run_id=uuid.uuid4())

    out = tmp_path / "exported.jsonl"
    count = handler.export_envelopes(out)
    assert count == 3
    assert out.exists()

    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3
    # Each line is valid JSON with the expected schema.
    for line in lines:
        env = json.loads(line)
        assert env["schema"] == "phionyx.langchain_event_envelope.v1"
        assert "integrity" in env


def test_export_envelopes_creates_parent_dir(tmp_path: Path) -> None:
    handler = _make_handler(tmp_path)
    handler.on_chain_start(serialized={}, inputs={}, run_id=uuid.uuid4())

    out = tmp_path / "nested" / "subdir" / "evidence.jsonl"
    handler.export_envelopes(out)
    assert out.exists()


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------


def test_hmac_signer_is_deterministic() -> None:
    from phionyx_langchain_langgraph import HmacSigner

    signer = HmacSigner()
    sig1 = signer.sign("sha256:" + "a" * 64)
    sig2 = signer.sign("sha256:" + "a" * 64)
    assert sig1 == sig2
    assert sig1.startswith("hmac-sha256:")


def test_hmac_signer_different_secrets_produce_different_signatures() -> None:
    from phionyx_langchain_langgraph import HmacSigner

    s1 = HmacSigner(secret="alpha")
    s2 = HmacSigner(secret="beta")
    digest_input = "sha256:" + "f" * 64
    assert s1.sign(digest_input) != s2.sign(digest_input)


def test_custom_signer_is_invoked(tmp_path: Path) -> None:
    """Custom Signer is used for every envelope — proves the Signer protocol works."""
    from phionyx_langchain_langgraph import FilesystemEnvelopeStore, PhionyxCallbackHandler

    calls: list[str] = []

    class RecordingSigner:
        def sign(self, current_hash: str) -> str:
            calls.append(current_hash)
            return f"custom:{len(calls)}"

    handler = PhionyxCallbackHandler(
        store=FilesystemEnvelopeStore(root=tmp_path),
        signer=RecordingSigner(),
    )
    handler.on_chain_start(serialized={}, inputs={}, run_id=uuid.uuid4())
    handler.on_chain_end(outputs={}, run_id=uuid.uuid4())

    assert len(calls) == 2
    # Each envelope's signature comes from our custom signer.
    sigs = [env["integrity"]["signature"] for env in handler.envelopes]
    assert sigs == ["custom:1", "custom:2"]


# ---------------------------------------------------------------------------
# Module-level verify_chain helper (works on raw envelope lists)
# ---------------------------------------------------------------------------


def test_module_level_verify_chain_on_empty_list() -> None:
    from phionyx_langchain_langgraph import verify_chain

    report = verify_chain([])
    assert report["ok"] is True
    assert report["envelope_count"] == 0


def test_module_level_verify_chain_on_handler_envelopes(tmp_path: Path) -> None:
    from phionyx_langchain_langgraph import verify_chain

    handler = _make_handler(tmp_path)
    handler.on_chain_start(serialized={}, inputs={}, run_id=uuid.uuid4())
    handler.on_chain_end(outputs={}, run_id=uuid.uuid4())

    report = verify_chain(handler.envelopes)
    assert report["ok"] is True
    assert report["envelope_count"] == 2


# ---------------------------------------------------------------------------
# Filesystem isolation — tests must not write to the user's $HOME
# ---------------------------------------------------------------------------


def test_filesystem_store_respects_constructor_root(tmp_path: Path) -> None:
    from phionyx_langchain_langgraph import FilesystemEnvelopeStore

    store = FilesystemEnvelopeStore(root=tmp_path)
    assert store.root == tmp_path


def test_filesystem_store_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from phionyx_langchain_langgraph import FilesystemEnvelopeStore

    custom_root = tmp_path / "from_env"
    monkeypatch.setenv("PHIONYX_LANGCHAIN_AUDIT_ROOT", str(custom_root))
    store = FilesystemEnvelopeStore()
    assert store.root == custom_root
