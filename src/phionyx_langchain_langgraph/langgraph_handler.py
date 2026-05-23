"""LangGraph supervisor adapter for Phionyx multi-agent evidence chains.

M3 status: envelope-emitting. ``register`` and ``handoff`` produce
signed, hash-chained envelopes against the same audit-chain infra used
by :class:`PhionyxCallbackHandler`. This makes Phionyx envelope chains
the trust-object substrate above LangGraph supervisor flows (and the
ingestion point for F5 multi-agent audit in v0.6.0).

The supervisor maintains its OWN trace (the parent trace_id chain).
Each ``register`` produces a derived child_trace_id of the form
``<parent_trace_id>:child:<child_node>``; downstream callers wire a
:class:`PhionyxCallbackHandler` to that derived trace_id so child
chains link back to the supervisor via the derived trace_id naming
convention.

This module does NOT import ``langgraph`` itself — the supervisor is
a coordination/audit class that callers wire into their langgraph
StateGraph. Optional ``[langgraph]`` install pulls in the framework
but is not required for the supervisor adapter to function.
"""

from __future__ import annotations

import itertools
import threading
import uuid
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

__version__ = "0.1.0a1"

_DEFAULT_SUPERVISOR_SENDER = ParticipantRef(
    id="langgraph.supervisor",
    type=ParticipantType.SYSTEM,
    name="LangGraph Supervisor (Phionyx instrumented)",
)
_DEFAULT_RECEIVER = ParticipantRef(
    id="phionyx.evidence_chain",
    type=ParticipantType.SYSTEM,
    name="Phionyx Runtime Evidence Sink",
)


def derive_child_trace_id(parent_trace_id: str, child_node: str) -> str:
    """Convention for child trace IDs spawned by a supervisor.

    Format: ``<parent_trace_id>:child:<child_node>``. A child handler
    constructed with this trace_id writes its own envelope chain to a
    sibling directory under the same store, and verifiers can trivially
    reconstruct the parent/child relationship from the directory names.
    """
    return f"{parent_trace_id}:child:{child_node}"


class PhionyxLangGraphSupervisor:
    """LangGraph supervisor adapter for Phionyx multi-agent evidence chains.

    Each ``register`` and ``handoff`` call emits one signed envelope
    into the supervisor's own chain (separate from child chains). The
    derived child_trace_id naming convention (parent:child:<node>) lets
    verifiers join parent and child chains without any side-channel
    metadata.

    Parameters
    ----------
    parent_trace_id:
        Trace ID of the parent supervisor run. Auto-generated if omitted.
    signer:
        Signer instance. Defaults to :class:`HmacSigner`.
    store:
        EnvelopeStore. Defaults to :class:`FilesystemEnvelopeStore`.
    sender / receiver:
        ParticipantRef identities written into every envelope.

    Target usage (the supervisor is wired into a LangGraph StateGraph)::

        sup = PhionyxLangGraphSupervisor()

        researcher_trace = sup.register(child_node="researcher")
        writer_trace = sup.register(child_node="writer")

        # In your supervisor function, on every dispatch:
        sup.handoff(from_node="supervisor", to_node="researcher",
                    payload={"task": "gather sources"})
        # ... later
        sup.handoff(from_node="researcher", to_node="writer",
                    payload={"sources": [...]})

        sup.verify_chain()    # returns {"ok": True, ...}
        sup.export_envelopes("supervisor.jsonl")
    """

    def __init__(
        self,
        *,
        parent_trace_id: str | None = None,
        operator_signing_key: bytes | str | None = None,
        signer: Signer | None = None,
        store: EnvelopeStore | None = None,
        sender: ParticipantRef | None = None,
        receiver: ParticipantRef | None = None,
    ) -> None:
        self.parent_trace_id = parent_trace_id or f"phionyx-langgraph-{uuid.uuid4().hex[:12]}"
        self._operator_signing_key = operator_signing_key
        self._signer: Signer = signer or HmacSigner()
        self._store: EnvelopeStore = store or FilesystemEnvelopeStore()
        self._sender = sender or _DEFAULT_SUPERVISOR_SENDER
        self._receiver = receiver or _DEFAULT_RECEIVER

        self._turn_index_iter = itertools.count(start=0)
        self._turn_id_iter = itertools.count(start=1)
        self._lock = threading.Lock()

        self._envelopes: list[dict[str, Any]] = []
        # Registered children indexed by node name → derived trace_id.
        self._children: dict[str, str] = {}

    # --- inspection surfaces -------------------------------------------------

    @property
    def envelopes(self) -> list[dict[str, Any]]:
        return list(self._envelopes)

    @property
    def store(self) -> EnvelopeStore:
        return self._store

    @property
    def children(self) -> dict[str, str]:
        """Read-only view of registered children: {node_name: child_trace_id}."""
        return dict(self._children)

    # --- envelope emission helper -------------------------------------------

    def _emit(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            turn_id = next(self._turn_id_iter)
            turn_index = next(self._turn_index_iter)

            message = AgentMessageEnvelope.create(
                protocol="langgraph",
                sender_participant_ref=self._sender,
                receiver_participant_ref=self._receiver,
                trace_id=self.parent_trace_id,
                turn_id=turn_id,
                payload={
                    "event_type": event_type,
                    "data": payload,
                },
                ttl_seconds=0,
                metadata={
                    "phionyx.event_type": event_type,
                    "phionyx.handler_version": __version__,
                    "phionyx.adapter": "langgraph_supervisor",
                },
            )

            ctx = EnvelopeContext(
                trace_id=self.parent_trace_id,
                turn_index=turn_index,
                event_type=event_type,
                agent_message_payload=message.model_dump(mode="json"),
                package_version=__version__,
            )
            previous = self._store.head(self.parent_trace_id)
            envelope = build_envelope(ctx, previous_hash=previous, signer=self._signer)
            self._store.append(self.parent_trace_id, envelope)
            self._envelopes.append(envelope)
            return envelope

    # --- public surface -----------------------------------------------------

    def register(self, *, child_node: str, child_trace_id: str | None = None) -> str:
        """Register a child node + emit a ``supervisor_register`` envelope.

        Returns the (possibly derived) ``child_trace_id``. Callers wire a
        :class:`PhionyxCallbackHandler` to this trace_id so the child's
        own chain is linkable to the supervisor's chain by name.

        Re-registering an already-registered ``child_node`` returns the
        previously-assigned trace_id and emits a ``supervisor_register``
        envelope (so the audit record reflects the duplicate call).
        """
        if not child_node:
            raise ValueError("child_node must be a non-empty string")

        if child_node in self._children:
            assigned = self._children[child_node]
        else:
            assigned = child_trace_id or derive_child_trace_id(self.parent_trace_id, child_node)
            self._children[child_node] = assigned

        self._emit(
            "supervisor_register",
            {
                "child_node": child_node,
                "child_trace_id": assigned,
                "parent_trace_id": self.parent_trace_id,
            },
        )
        return assigned

    def handoff(
        self,
        *,
        from_node: str,
        to_node: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a parent-coordinated handoff + emit a signed envelope.

        The envelope payload includes both nodes by name and their
        derived child trace_ids (if either side is a registered child).
        ``payload`` is the optional caller-supplied handoff state — it
        is serialized into the envelope's ``data.payload`` field.

        Returns the emitted envelope dict.
        """
        if not from_node or not to_node:
            raise ValueError("from_node and to_node must be non-empty")

        return self._emit(
            "supervisor_handoff",
            {
                "from_node": from_node,
                "to_node": to_node,
                "from_child_trace_id": self._children.get(from_node),
                "to_child_trace_id": self._children.get(to_node),
                "payload": payload or {},
            },
        )

    # --- verification & export ---------------------------------------------

    def verify_chain(self) -> dict[str, Any]:
        """Re-read the persisted supervisor chain and verify integrity."""
        envelopes = list(self._store.iter_chain(self.parent_trace_id))
        return verify_chain(envelopes)

    def export_envelopes(self, path: str) -> int:
        """Export the supervisor chain to JSONL."""
        from pathlib import Path

        envelopes = list(self._store.iter_chain(self.parent_trace_id))
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for env in envelopes:
                fh.write(canonical_json(env) + "\n")
        return len(envelopes)
