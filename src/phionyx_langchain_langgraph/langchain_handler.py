"""LangChain BaseCallbackHandler adapter for Phionyx runtime evidence.

M2 status: envelope-emitting. Each LangChain callback event becomes:
    1. An ``AgentMessageEnvelope`` from ``phionyx_core.contracts.envelopes``
       capturing trace_id, turn_id, message_id, timestamp_utc, nonce, payload.
    2. A signed, hash-chained outer envelope (see ``audit_chain.py``) that
       links the message to its predecessor in the chain and is signed by
       the operator's Signer.

The handler emits to an :class:`EnvelopeStore` (filesystem by default).
``verify_chain()`` re-reads the persisted chain and checks integrity.
``export_envelopes(path)`` writes the chain as JSONL for sharing.

Design notes:
- The legacy M1 in-memory ``_PendingEvent`` list is retained for inspection
  and quick debugging; emission to the store is the canonical path now.
- We subclass ``langchain_core.callbacks.BaseCallbackHandler`` (preferred)
  with a fallback to the older ``langchain.callbacks.base`` import path.
- Turn indices are monotonic per handler instance, scoped by ``trace_id``.
"""

from __future__ import annotations

import itertools
import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phionyx_core.contracts.envelopes import AgentMessageEnvelope
from phionyx_core.contracts.participants import ParticipantRef, ParticipantType

from .audit_chain import (
    EnvelopeContext,
    EnvelopeStore,
    FilesystemEnvelopeStore,
    HmacSigner,
    Signer,
    build_envelope,
    canonical_json,
    verify_chain,
)

try:  # langchain-core >= 0.2.0 (preferred)
    from langchain_core.callbacks import BaseCallbackHandler  # type: ignore
except ImportError:  # pragma: no cover — fallback for older installs
    try:
        from langchain.callbacks.base import BaseCallbackHandler  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "phionyx-langchain-langgraph requires langchain-core>=0.2.0; "
            "install with `pip install langchain-core>=0.2.0`."
        ) from exc

__version__ = "0.1.0a1"

# Default participant identities. Callers may override via constructor.
_DEFAULT_SENDER = ParticipantRef(
    id="langchain.callback_handler",
    type=ParticipantType.SYSTEM,
    name="LangChain Runtime (Phionyx instrumented)",
)
_DEFAULT_RECEIVER = ParticipantRef(
    id="phionyx.evidence_chain",
    type=ParticipantType.SYSTEM,
    name="Phionyx Runtime Evidence Sink",
)


@dataclass
class _PendingEvent:
    """In-memory event record (M1 legacy surface, retained for inspection)."""

    event_type: str
    run_id: str
    parent_run_id: str | None
    timestamp_iso: str
    payload: dict[str, Any] = field(default_factory=dict)


class PhionyxCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that emits signed Phionyx envelopes per event.

    Parameters
    ----------
    trace_id:
        Logical trace ID. All events emitted by this handler share it. Auto-
        generated if omitted.
    operator_signing_key:
        Reserved for Ed25519. M2 ships with :class:`HmacSigner` (demo). Pass
        a custom :class:`Signer` via ``signer=`` to override.
    signer:
        Custom :class:`Signer` instance. Defaults to ``HmacSigner()``.
    store:
        Custom :class:`EnvelopeStore`. Defaults to :class:`FilesystemEnvelopeStore`
        rooted at ``~/.phionyx/langchain_audit`` (override via
        ``PHIONYX_LANGCHAIN_AUDIT_ROOT`` env var).
    sender / receiver:
        :class:`ParticipantRef` identities written into every envelope.
        Sensible defaults are used; override for richer attribution.
    """

    # LangChain inspects these attributes when dispatching events.
    raise_error: bool = False
    run_inline: bool = False

    def __init__(
        self,
        *,
        trace_id: str | None = None,
        operator_signing_key: bytes | str | None = None,
        signer: Signer | None = None,
        store: EnvelopeStore | None = None,
        sender: ParticipantRef | None = None,
        receiver: ParticipantRef | None = None,
    ) -> None:
        super().__init__()
        self.trace_id = trace_id or f"phionyx-langchain-{uuid.uuid4().hex[:12]}"
        self._operator_signing_key = operator_signing_key
        self._signer: Signer = signer or HmacSigner()
        self._store: EnvelopeStore = store or FilesystemEnvelopeStore()
        self._sender = sender or _DEFAULT_SENDER
        self._receiver = receiver or _DEFAULT_RECEIVER

        # Monotonic counters scoped to this handler instance. ``turn_index``
        # drives the envelope chain ordering; ``turn_id`` is the
        # AgentMessageEnvelope's monotonic id (also 1-based per spec).
        self._turn_index_iter = itertools.count(start=0)
        self._turn_id_iter = itertools.count(start=1)
        self._lock = threading.Lock()  # cross-thread safety for callback storms

        # M1 legacy in-memory event log retained for inspection / tests.
        self._events: list[_PendingEvent] = []
        # In-memory mirror of the emitted envelope chain.
        self._envelopes: list[dict[str, Any]] = []

    # --- M1 inspection surfaces -------------------------------------------------

    @property
    def events(self) -> list[_PendingEvent]:
        return list(self._events)

    @property
    def envelopes(self) -> list[dict[str, Any]]:
        return list(self._envelopes)

    @property
    def store(self) -> EnvelopeStore:
        return self._store

    # --- core emission -------------------------------------------------------

    def _record_and_emit(
        self,
        event_type: str,
        run_id: Any,
        parent_run_id: Any | None,
        payload: dict[str, Any],
    ) -> None:
        """Record the event and emit a signed envelope into the chain."""
        with self._lock:
            now_iso = datetime.now(timezone.utc).isoformat()
            self._events.append(
                _PendingEvent(
                    event_type=event_type,
                    run_id=str(run_id),
                    parent_run_id=str(parent_run_id) if parent_run_id is not None else None,
                    timestamp_iso=now_iso,
                    payload=payload,
                )
            )

            turn_id = next(self._turn_id_iter)
            turn_index = next(self._turn_index_iter)

            message = AgentMessageEnvelope.create(
                protocol="langchain",
                sender_participant_ref=self._sender,
                receiver_participant_ref=self._receiver,
                trace_id=self.trace_id,
                turn_id=turn_id,
                payload={
                    "event_type": event_type,
                    "run_id": str(run_id),
                    "parent_run_id": str(parent_run_id) if parent_run_id is not None else None,
                    "data": payload,
                },
                ttl_seconds=0,  # audit envelopes do not expire by design
                metadata={
                    "phionyx.event_type": event_type,
                    "phionyx.handler_version": __version__,
                },
            )

            ctx = EnvelopeContext(
                trace_id=self.trace_id,
                turn_index=turn_index,
                event_type=event_type,
                agent_message_payload=message.model_dump(mode="json"),
                package_version=__version__,
            )
            previous = self._store.head(self.trace_id)
            envelope = build_envelope(ctx, previous_hash=previous, signer=self._signer)
            self._store.append(self.trace_id, envelope)
            self._envelopes.append(envelope)

    # --- LangChain BaseCallbackHandler surface -------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self._record_and_emit(
            "chain_start",
            run_id,
            parent_run_id,
            {"serialized": serialized, "inputs": inputs, "extra": kwargs},
        )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self._record_and_emit(
            "chain_end",
            run_id,
            parent_run_id,
            {"outputs": outputs, "extra": kwargs},
        )

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self._record_and_emit(
            "chain_error",
            run_id,
            parent_run_id,
            {"error_type": type(error).__name__, "error_message": str(error), "extra": kwargs},
        )

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self._record_and_emit(
            "tool_start",
            run_id,
            parent_run_id,
            {"serialized": serialized, "input_str": input_str, "extra": kwargs},
        )

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self._record_and_emit(
            "tool_end",
            run_id,
            parent_run_id,
            {"output": output, "extra": kwargs},
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self._record_and_emit(
            "tool_error",
            run_id,
            parent_run_id,
            {"error_type": type(error).__name__, "error_message": str(error), "extra": kwargs},
        )

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self._record_and_emit(
            "llm_start",
            run_id,
            parent_run_id,
            {"serialized": serialized, "prompts": prompts, "extra": kwargs},
        )

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        # Best-effort serialization. If repr is unrepresentable, fall back to str.
        try:
            response_repr = repr(response)
        except Exception:  # pragma: no cover — defensive
            response_repr = str(response)
        self._record_and_emit(
            "llm_end",
            run_id,
            parent_run_id,
            {"response_repr": response_repr, "extra": kwargs},
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self._record_and_emit(
            "llm_error",
            run_id,
            parent_run_id,
            {"error_type": type(error).__name__, "error_message": str(error), "extra": kwargs},
        )

    # --- verification & export ------------------------------------------------

    def verify_chain(self) -> dict[str, Any]:
        """Re-read the persisted chain and verify integrity.

        Returns the structured report from
        :func:`phionyx_langchain_langgraph.audit_chain.verify_chain` —
        ``{"ok": bool, "envelope_count": int, "errors": [...]}``.
        """
        envelopes = list(self._store.iter_chain(self.trace_id))
        return verify_chain(envelopes)

    def export_envelopes(self, path: str | Path) -> int:
        """Export the chain to JSONL. Returns the count exported."""
        envelopes = list(self._store.iter_chain(self.trace_id))
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for env in envelopes:
                fh.write(canonical_json(env) + "\n")
        return len(envelopes)
