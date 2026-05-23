# phionyx-langchain-langgraph

> **Status:** alpha (v0.1.0a1) — LangChain callback adapter, LangGraph supervisor adapter, and integration tests against real LangChain primitives all live. PyPI publish pending founder authorisation.

Native LangChain + LangGraph adapters for [Phionyx](https://phionyx.ai/runtime-evidence) runtime evidence.

Every LangChain `chain`, `tool`, and `llm` event — and every LangGraph supervisor handoff — is recorded as a signed, hash-chained envelope entry. Third parties can verify the chain offline without trusting the agent's narration.

## Why

LangChain ships an observability surface (LangSmith, callback handlers, tracing) optimized for *debugging*. It is not optimized for *third-party verification*: a callback log is mutable, unsigned, and the agent's own narration. Phionyx envelopes are immutable, hash-chained, and signed under the operator's Ed25519 key — they survive review even when the agent and the trace store are not trusted.

LangGraph's supervisor patterns track *flow* but do not sign parent → child handoffs. F5 (Phionyx v0.6.0) adds full multi-agent envelope chains; this adapter is the ingestion surface.

## Position vs adjacent tooling

- **vs LangSmith / Langfuse / Helicone**: observability tools record what happened; Phionyx makes what happened *signed, replayable, and third-party-verifiable*. The two layers compose — Phionyx envelope export to OTel / LangSmith is supported.
- **vs OpenTelemetry GenAI conventions**: OTel conventions remain in active development. Phionyx envelopes are OTel-compatible while preserving stronger evidence semantics (hash chain + signature).
- **vs A2A (Google Agent2Agent)**: A2A handles agent-to-agent delegation; Phionyx envelopes are designed protocol-agnostic. v1.1 adds an explicit A2A agent-card adapter.

## Install (preview, not yet on PyPI)

```bash
pip install phionyx-langchain-langgraph    # core: LangChain callback adapter
pip install "phionyx-langchain-langgraph[langgraph]"   # + LangGraph supervisor adapter
```

Until v0.1.0a1 ships to PyPI, install from source:

```bash
pip install -e tools/phionyx_langchain_langgraph
```

## 60-second usage

```python
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from phionyx_langchain_langgraph import PhionyxCallbackHandler

handler = PhionyxCallbackHandler()   # default HmacSigner + filesystem store

@tool
def count_words(text: str) -> str:
    """Return the word count of the input text."""
    return f"{len(text.split())} words"

count_words.invoke(
    {"text": "Phionyx makes runtime evidence verifiable."},
    config={"callbacks": [handler]},
)

# Inspect, verify, export.
print(f"{len(handler.envelopes)} signed envelopes")
print(f"Chain verifies: {handler.verify_chain()['ok']}")
handler.export_envelopes("evidence/run.jsonl")
```

A complete runnable example covering tool, chain, and LLM events is in
[`examples/quickstart.py`](examples/quickstart.py):

```bash
pip install -e tools/phionyx_langchain_langgraph
python tools/phionyx_langchain_langgraph/examples/quickstart.py
```

Expected output::

    Tool result : 5 words
    LLM result  : the answer is forty-two.
    Envelopes   : 8 ({'tool_start': 1, 'tool_end': 1, 'chain_start': 2, 'llm_start': 1, 'llm_end': 1, 'chain_end': 2})
    Verify chain: OK (8 envelopes; 0 errors)
    Exported    : 8 envelopes → /tmp/phionyx-demo-XXXXXXXX/quickstart-evidence.jsonl

## Multi-agent (LangGraph supervisor)

```python
from phionyx_langchain_langgraph import (
    PhionyxLangGraphSupervisor,
    PhionyxCallbackHandler,
    FilesystemEnvelopeStore,
)

store = FilesystemEnvelopeStore()
sup = PhionyxLangGraphSupervisor(parent_trace_id="run-2026-05-23", store=store)

# Register children — returns derived child trace_ids.
researcher_trace = sup.register(child_node="researcher")
writer_trace     = sup.register(child_node="writer")

# Wire a child handler to each derived trace_id.
researcher = PhionyxCallbackHandler(trace_id=researcher_trace, store=store)
writer     = PhionyxCallbackHandler(trace_id=writer_trace,     store=store)

# Record handoffs as the supervisor dispatches.
sup.handoff(from_node="supervisor", to_node="researcher", payload={"task": "..."})
# ... later
sup.handoff(from_node="researcher", to_node="writer", payload={"sources": [...]})

# Both chains verify independently.
assert sup.verify_chain()["ok"]
assert researcher.verify_chain()["ok"]
assert writer.verify_chain()["ok"]
```

Child chains live in sibling directories under one store root, indexed
by trace_id. Verifiers walk parent + child chains without any
side-channel metadata. This is the F5 multi-agent ingestion surface.

## Status — what's live in v0.1.0a1

- ✅ **PhionyxCallbackHandler** — `on_chain_*`, `on_tool_*`, `on_llm_*`
  events + error variants emit signed envelopes.
- ✅ **PhionyxLangGraphSupervisor** — `register` + `handoff` emit
  signed envelopes; derived child trace_ids; parent/child chains
  coexist under one store root.
- ✅ **AgentMessageEnvelope** as the inner record (from
  `phionyx_core.contracts.envelopes`).
- ✅ **HmacSigner** demo + **Signer** protocol for Ed25519 swap.
- ✅ **FilesystemEnvelopeStore** with env-var override.
- ✅ **`verify_chain`** — detects payload tamper + broken links.
- ✅ **`export_envelopes`** — JSONL round-trip.
- ✅ **52 tests** — unit, envelope chain, supervisor, real LangChain
  integration (RunnableLambda, @tool, FakeListLLM, chain composition,
  tool error path, JSONL round-trip).

Roadmap beyond v0.1.0a1: PyPI publish (gated on founder authorisation),
v0.1.0 stable schema lock alongside `phionyx-core` v0.5.0, full F5
multi-agent block wiring in `phionyx-core` v0.6.0.

## License

AGPL-3.0-or-later. Commercial dual-license available — contact founder@phionyx.ai.

## See also

- [phionyx.ai/runtime-evidence](https://phionyx.ai/runtime-evidence) — long-form thesis
- `phionyx-core` (PyPI) — core envelope schema + Ed25519 signing
- `phionyx-mcp-server` (PyPI) — MCP trust boundary companion
- `phionyx-pipeline-mcp` (PyPI) — agent self-claim gate companion
- `phionyx-eval-inspect` (PyPI) — UK AISI Inspect bridge
