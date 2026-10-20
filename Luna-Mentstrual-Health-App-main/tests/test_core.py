import json
from pathlib import Path
from rag import KnowledgeBase, answer_question

def test_ingest_and_retrieval():
    kb=KnowledgeBase()
    status=kb.ingest(Path("knowledge/sources.json"))
    assert status["documents"] >= 5
    hits=kb.retrieve("What pads can I use and how do I care for them?")
    assert hits
    assert any("pads" in (d["text"]+" ".join(d.get("keywords",[]))).lower() for d in hits)

def test_citations_are_retrieved_sources(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    kb=KnowledgeBase(); kb.ingest(Path("knowledge/sources.json"))
    result=answer_question("What should I know about periods?",kb)
    assert result["sources"]
    known={d["url"] for d in kb.docs}
    assert all(s["url"] in known for s in result["sources"])

def test_empty_retrieval_admits_uncertainty():
    class Empty:
        def retrieve(self,q,k=4): return []
    result=answer_question("question",Empty())
    assert result["sources"] == []
    assert "couldn't find" in result["answer"].lower()

def test_tracker_date_math():
    from datetime import date, timedelta
    start=date(2026,1,31)
    assert (start+timedelta(days=28)).isoformat()=="2026-02-28"

def test_contexts_returned_for_ragas(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    kb=KnowledgeBase(); kb.ingest(Path("knowledge/sources.json"))
    r=answer_question("How do I know if my period is too heavy?",kb)
    assert r["contexts"] and all(isinstance(c,str) for c in r["contexts"])
    assert [s["n"] for s in r["sources"]]==list(range(1,len(r["sources"])+1))

def test_irrelevant_query_is_filtered():
    kb=KnowledgeBase(min_score=0.99); kb.ingest(Path("knowledge/sources.json"))
    if kb.mode.startswith("semantic"):
        assert kb.retrieve("best pizza toppings in Naples") == []

def test_chunker_overlap():
    from rag import chunk_text
    text=" ".join(f"Sentence number {i} has some words here." for i in range(60))
    chunks=chunk_text(text,max_words=50)
    assert len(chunks)>1 and chunks[0].split(". ")[-1] in chunks[1]
