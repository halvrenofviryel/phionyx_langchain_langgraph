"""Quickstart: capture signed runtime evidence for a LangGraph supervisor flow.

This demo shows ``PhionyxLangGraphSupervisor`` coordinating two child
agents (researcher + writer). Every supervisor register, supervisor
handoff, and child-side chain event lands on its own signed envelope
chain. Parent and child chains coexist under one store root, indexed
by trace_id — the F5 multi-agent ingestion surface.

No openai-agents, no live LLM calls — just LangChain's
``RunnableLambda`` and ``FakeListLLM`` so the demo runs anywhere.

Run::

    pip install -e tools/phionyx_langchain_langgraph
    python tools/phionyx_langchain_langgraph/examples/langgraph_supervisor.py
"""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

from langchain_core.language_models.fake import FakeListLLM
from langchain_core.runnables import RunnableLambda

from phionyx_langchain_langgraph import (
    FilesystemEnvelopeStore,
    PhionyxCallbackHandler,
    PhionyxLangGraphSupervisor,
)


def main() -> None:
    audit_root = Path(tempfile.mkdtemp(prefix="phionyx-langgraph-demo-"))
    store = FilesystemEnvelopeStore(root=audit_root)

    sup = PhionyxLangGraphSupervisor(parent_trace_id="lg-demo", store=store)

    # --- 1. Register children + emit supervisor envelopes -----------------
    researcher_trace = sup.register(child_node="researcher")
    writer_trace = sup.register(child_node="writer")

    # --- 2. Researcher actually runs a LangChain pipeline -----------------
    researcher = PhionyxCallbackHandler(trace_id=researcher_trace, store=store)
    research_llm = FakeListLLM(responses=["sources: arXiv, AISI, NIST"])
    research_chain = research_llm | RunnableLambda(lambda x: x.split(": ")[1])
    sources = research_chain.invoke(
        "find sources on AI evidence",
        config={"callbacks": [researcher]},
    )

    sup.handoff(
        from_node="researcher",
        to_node="writer",
        payload={"sources": sources},
    )

    # --- 3. Writer runs its own LangChain pipeline ------------------------
    writer = PhionyxCallbackHandler(trace_id=writer_trace, store=store)
    write_llm = FakeListLLM(responses=["Phionyx makes runtime evidence verifiable."])
    write_chain = write_llm | RunnableLambda(lambda x: x.upper())
    headline = write_chain.invoke(
        f"write a headline citing {sources}",
        config={"callbacks": [writer]},
    )

    # --- 4. Inspect everything --------------------------------------------
    print(f"Sources         : {sources}")
    print(f"Headline        : {headline}")

    sup_counts = Counter(env["subject"]["event_type"] for env in sup.envelopes)
    res_counts = Counter(e.event_type for e in researcher.events)
    wri_counts = Counter(e.event_type for e in writer.events)
    print(f"Supervisor chain: {len(sup.envelopes)} envelopes {dict(sup_counts)}")
    print(f"Researcher chain: {len(researcher.envelopes)} envelopes {dict(res_counts)}")
    print(f"Writer chain    : {len(writer.envelopes)} envelopes {dict(wri_counts)}")

    # --- 5. Each chain verifies independently -----------------------------
    sup_report = sup.verify_chain()
    res_report = researcher.verify_chain()
    wri_report = writer.verify_chain()
    print(
        f"Verify          : sup={'OK' if sup_report['ok'] else 'FAIL'}  "
        f"researcher={'OK' if res_report['ok'] else 'FAIL'}  "
        f"writer={'OK' if wri_report['ok'] else 'FAIL'}"
    )

    # --- 6. Each chain exports its own JSONL ------------------------------
    exports = {
        "supervisor.jsonl": sup,
        "researcher.jsonl": researcher,
        "writer.jsonl": writer,
    }
    for name, h in exports.items():
        n = h.export_envelopes(audit_root / name)
        print(f"Exported        : {n:>3} envelopes → {audit_root / name}")

    print(f"Audit root      : {audit_root}")
    print()
    print("The 3 chains live in sibling directories named by trace_id —")
    print("verifiers can walk parent + child chains without side-channel metadata.")


if __name__ == "__main__":
    main()
