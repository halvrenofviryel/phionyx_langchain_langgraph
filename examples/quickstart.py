"""Quickstart: capture signed runtime evidence for a real LangChain run.

This 60-line demo exercises every event type the v0.1.0a1 adapter
supports — chain, tool, and LLM — and writes a verifiable evidence
chain to disk. Run it::

    pip install -e tools/phionyx_langchain_langgraph
    python tools/phionyx_langchain_langgraph/examples/quickstart.py

You will see:
- A LangChain pipeline that calls a tool and a fake LLM.
- A signed, hash-chained envelope per event, persisted to a temp dir.
- A verification report that re-reads the chain from disk and confirms
  it has not been tampered with.
- A JSONL export of the chain for sharing or auditing.

No API keys required — ``FakeListLLM`` stands in for a real provider.
"""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

from langchain_core.language_models.fake import FakeListLLM
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from phionyx_langchain_langgraph import FilesystemEnvelopeStore, PhionyxCallbackHandler


@tool
def count_words(text: str) -> str:
    """Return the word count of the input text."""
    return f"{len(text.split())} words"


def main() -> None:
    # All evidence lives under a temp dir for this demo. Production
    # deployments either rely on the default ``~/.phionyx/langchain_audit``
    # location or pass an explicit root (or override via the
    # ``PHIONYX_LANGCHAIN_AUDIT_ROOT`` env var).
    audit_root = Path(tempfile.mkdtemp(prefix="phionyx-demo-"))
    handler = PhionyxCallbackHandler(
        trace_id="quickstart-demo",
        store=FilesystemEnvelopeStore(root=audit_root),
    )

    # --- 1. Tool invocation ------------------------------------------------
    tool_result = count_words.invoke(
        {"text": "Phionyx makes runtime evidence verifiable."},
        config={"callbacks": [handler]},
    )

    # --- 2. LLM + post-process chain --------------------------------------
    llm = FakeListLLM(responses=["The answer is forty-two."])
    lower = RunnableLambda(lambda x: x.lower())
    pipeline = llm | lower
    llm_result = pipeline.invoke(
        "What is the answer?",
        config={"callbacks": [handler]},
    )

    # --- 3. Inspect the chain ---------------------------------------------
    event_counts = Counter(e.event_type for e in handler.events)
    print(f"Tool result : {tool_result}")
    print(f"LLM result  : {llm_result}")
    print(f"Envelopes   : {len(handler.envelopes)} ({dict(event_counts)})")

    # --- 4. Verify chain integrity ----------------------------------------
    report = handler.verify_chain()
    status = "OK" if report["ok"] else "FAILED"
    print(f"Verify chain: {status} ({report['envelope_count']} envelopes; "
          f"{len(report['errors'])} errors)")

    # --- 5. Export the chain ----------------------------------------------
    export_path = audit_root / "quickstart-evidence.jsonl"
    exported = handler.export_envelopes(export_path)
    print(f"Exported    : {exported} envelopes → {export_path}")
    print(f"Audit root  : {audit_root}")


if __name__ == "__main__":
    main()
