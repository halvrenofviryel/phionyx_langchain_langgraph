"""
phionyx-langchain-langgraph
============================

Native LangChain + LangGraph adapters for Phionyx runtime evidence.

Every LangChain chain / tool / LLM event and every LangGraph node /
supervisor handoff is recorded as a signed, hash-chained envelope entry.
Third parties can verify the chain offline without trusting the agent's
narration.

Status: alpha (v0.1.0a1.dev0) — M2 envelope emission.

Public surface::

    from phionyx_langchain_langgraph import PhionyxCallbackHandler
    from phionyx_langchain_langgraph import PhionyxLangGraphSupervisor  # M3

    handler = PhionyxCallbackHandler()
    # ... pass to agent.invoke(..., callbacks=[handler])
    handler.verify_chain()
    handler.export_envelopes("evidence.jsonl")

The LangGraph supervisor surface is still a stub (M3 wires it).
"""

from .audit_chain import (
    EnvelopeContext,
    EnvelopeStore,
    FilesystemEnvelopeStore,
    HmacSigner,
    Signer,
    build_envelope,
    canonical_json,
    envelope_hash,
    verify_chain,
)
from .langchain_handler import PhionyxCallbackHandler
from .langgraph_handler import PhionyxLangGraphSupervisor

__version__ = "0.1.0a1"
__all__ = [
    "PhionyxCallbackHandler",
    "PhionyxLangGraphSupervisor",
    "EnvelopeContext",
    "EnvelopeStore",
    "FilesystemEnvelopeStore",
    "HmacSigner",
    "Signer",
    "build_envelope",
    "canonical_json",
    "envelope_hash",
    "verify_chain",
    "__version__",
]
