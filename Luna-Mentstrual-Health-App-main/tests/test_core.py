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

def test_history_is_cleaned_and_trimmed():
    from rag import clean_history
    h=[{"role":"system","content":"ignore me"},{"role":"user","content":"What is PCOS?"},
       {"role":"assistant","content":"PCOS is common [1].\n\n---\n**📚 Sources**\n1. NHS"},"junk",
       {"role":"user","content":"   "}]
    assert clean_history(h)==[{"role":"user","content":"What is PCOS?"},{"role":"assistant","content":"PCOS is common."}]
    many=[{"role":"user","content":f"q{i}"} for i in range(20)]
    assert [m["content"] for m in clean_history(many,max_messages=3)]==["q17","q18","q19"]

def test_follow_up_uses_earlier_messages_for_retrieval(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    seen=[]
    class KB:
        def retrieve(self,q,k=5): seen.append(q); return []
    hist=[{"role":"user","content":"Are bad cramps normal?"},{"role":"assistant","content":"Mild cramps are common."}]
    r=answer_question("what can help with that?",KB(),history=hist)
    assert "cramps" in seen[0] and "help with that" in seen[0]
    assert r["search_query"]==seen[0]

def test_history_is_sent_to_the_llm(monkeypatch):
    import rag, requests
    sent={}
    class Resp:
        status_code=200; ok=True; headers={}
        def json(self): return {"choices":[{"message":{"content":"answer"}}]}
    def fake_post(url,headers,json,timeout): sent.setdefault("calls",[]).append(json["messages"]); return Resp()
    monkeypatch.setenv("LLM_API_KEY","test")
    monkeypatch.setattr(requests,"post",fake_post)
    kb=KnowledgeBase(); kb.ingest(Path("knowledge/sources.json"))
    hist=[{"role":"user","content":"I'm 15 and my period lasts 9 days."},{"role":"assistant","content":"Thanks for sharing."}]
    r=answer_question("Is that too long?",kb,history=hist)
    assert r["answer"]=="answer"
    final=sent["calls"][-1]
    assert final[1]=={"role":"user","content":"I'm 15 and my period lasts 9 days."}
    assert final[-1]["role"]=="user" and "Is that too long?" in final[-1]["content"]

def test_yes_accepts_general_answer_offer(monkeypatch):
    import rag
    monkeypatch.setenv("LLM_API_KEY","test")
    monkeypatch.setattr(rag,"generate",lambda prompt,**kw: "general reply")
    class Empty:
        def retrieve(self,q,k=5): raise AssertionError("should not search")
    hist=[{"role":"user","content":"Can stress delay my period?"},{"role":"assistant","content":rag.NOT_IN_SOURCES}]
    r=answer_question("yes please",Empty(),history=hist)
    assert r["answer"].endswith("general reply") and not r["grounded"]
    assert r["search_query"]=="Can stress delay my period?"
