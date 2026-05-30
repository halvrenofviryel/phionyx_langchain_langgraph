# phionyx-langchain-langgraph

> **Status:** alpha (v0.1.0a1) — live on PyPI. LangChain callback adapter, LangGraph supervisor adapter, and integration tests against real LangChain primitives all live.

Native LangChain + LangGraph adapters for [Phionyx](https://phionyx.ai) runtime evidence. This package surfaces on [phionyx.ai/narrative-coherence](https://phionyx.ai/narrative-coherence) as one of the framework adapters that turn third-party agent runs into reviewer-runnable evidence.

**Where this sits in the Phionyx portfolio:** this is a **framework adapter** with its own version line (v0.1.0a1). It is distinct from the **engine** ([`phionyx-core`](https://pypi.org/project/phionyx-core/), latest v0.7.2 — the deterministic-cognition runtime whose envelope schema this adapter emits), the **self-governance gate** ([`phionyx-pipeline-mcp`](https://github.com/halvrenofviryel/phionyx-pipeline-mcp), stable v0.2.0), and the **Evaluation Standard** ([`phionyx-evaluation-standard`](https://github.com/halvrenofviryel/phionyx-evaluation-standard)). These are separate version namespaces and must not be cross-attributed.

Every LangChain `chain`, `tool`, and `llm` event — and every LangGraph supervisor handoff — is recorded as a signed, hash-chained envelope entry. Third parties can verify the chain offline without trusting the agent's narration.

## Why

LangChain ships an observability surface (LangSmith, callback handlers, tracing) optimized for *debugging*. It is not optimized for *third-party verification*: a callback log is mutable, unsigned, and the agent's own narration. Phionyx envelopes are immutable, hash-chained, and signed under the operator's Ed25519 key — they survive review even when the agent and the trace store are not trusted.

LangGraph's supervisor patterns track *flow* but do not sign parent → child handoffs. Phionyx's multi-agent envelope schema (delivered in `phionyx-core` v0.6.0, shipped; latest engine is v0.7.2) adds full multi-agent envelope chains; this adapter is the ingestion surface today.

## Position vs adjacent tooling

- **vs LangSmith / Langfuse / Helicone**: observability tools record what happened; Phionyx makes what happened *signed, replayable, and third-party-verifiable*. The two layers compose — Phionyx envelope export to OTel / LangSmith is supported.
- **vs OpenTelemetry GenAI conventions**: OTel conventions remain in active development. Phionyx envelopes are OTel-compatible while preserving stronger evidence semantics (hash chain + signature).
- **vs A2A (Google Agent2Agent)**: A2A handles agent-to-agent delegation; Phionyx envelopes are designed protocol-agnostic. A future minor release of this adapter (the v1.1 milestone on this package's own roadmap) adds an explicit A2A agent-card adapter.

## Install

```bash
pip install phionyx-langchain-langgraph              # core: LangChain callback adapter
pip install "phionyx-langchain-langgraph[langgraph]" # + LangGraph supervisor adapter
```

Source: [github.com/halvrenofviryel/phionyx-langchain-langgraph](https://github.com/halvrenofviryel/phionyx-langchain-langgraph).

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
pip install phionyx-langchain-langgraph
python examples/quickstart.py
```

Expected output:

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
side-channel metadata. This is the multi-agent ingestion surface.

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

Roadmap beyond v0.1.0a1: a v0.1.0 stable release that locks the schema against the current `phionyx-core` engine (latest v0.7.2), building on the multi-agent envelope wiring already shipped in `phionyx-core` v0.6.0.

## License

AGPL-3.0-or-later. Commercial dual-license available — contact founder@phionyx.ai.

## See also

- [phionyx.ai/narrative-coherence](https://phionyx.ai/narrative-coherence) — entry pillar this package surfaces under
- [phionyx.ai/evidence](https://phionyx.ai/evidence) — Evidence Matrix: every load-bearing claim paired with a reviewer-runnable command
- [`phionyx-core`](https://pypi.org/project/phionyx-core/) (PyPI) — the deterministic-cognition **engine** (latest v0.7.2); core envelope schema + Ed25519 signing
- [`phionyx-evaluation-standard`](https://github.com/halvrenofviryel/phionyx-evaluation-standard) — vendor-neutral **Evaluation Standard** defining the L0–L3 (evaluation maturity), D0–D3 (determinism), and CG-L0…CG-L5 (claim-governance) scales; `phionyx-core` is the reference implementation scoring L3 + D3
- [`phionyx-mcp-server`](https://github.com/halvrenofviryel/phionyx-mcp-server) — MCP trust boundary companion
- [`phionyx-pipeline-mcp`](https://github.com/halvrenofviryel/phionyx-pipeline-mcp) — agent self-claim **gate** companion; the component rated on the CG-L0…CG-L5 claim-governance ladder (stable v0.2.0 = CG-L2)
- [`phionyx-eval-inspect`](https://github.com/halvrenofviryel/phionyx-eval-inspect) — Inspect AI bridge companion
- [`phionyx-openai-agents`](https://github.com/halvrenofviryel/phionyx-openai-agents) — OpenAI Agents SDK tracing bridge companion
