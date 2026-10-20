"""Evaluate the Luna RAG pipeline.

Always computes retrieval metrics against labelled source ids (hit rate, MRR, precision@k).
If LLM_API_KEY is set, also runs the pipeline end-to-end and scores it with RAGAS:
  faithfulness, answer (response) relevancy, context precision, context recall.

Usage:  python eval/ragas_eval.py [--k 4] [--out eval/results.csv]
"""
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from rag import KnowledgeBase, answer_question

load_dotenv(ROOT / ".env", override=True)


def retrieval_metrics(kb, rows, k):
    """A hit = a retrieved chunk from the same source page (URL) as an expected source id,
    so sections ingested from the full web page count as well as the curated excerpt."""
    url_of = {d["id"]: d["url"].rstrip("/") for d in kb.docs}
    hits = rr = prec = 0.0
    for r in rows:
        got = [c["url"].rstrip("/") for c in kb.retrieve(r["question"], k=k)]
        exp = {url_of[i] for i in r["expected_ids"] if i in url_of}
        rank = next((i + 1 for i, u in enumerate(got) if u in exp), None)
        hits += rank is not None
        rr += 1 / rank if rank else 0
        prec += sum(u in exp for u in got) / k
    n = len(rows)
    return {f"hit_rate@{k}": hits / n, "mrr": rr / n, f"precision@{k}": prec / n}


def ragas_eval(kb, rows, k, out):
    from ragas import EvaluationDataset, SingleTurnSample, evaluate, RunConfig
    from ragas.metrics import (Faithfulness, ResponseRelevancy,
                               LLMContextPrecisionWithReference, LLMContextRecall)
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI
    from ragas.embeddings.base import BaseRagasEmbeddings
    from rag import get_embedder

    class OnnxEmbeddings(BaseRagasEmbeddings):  # same MiniLM model as retrieval, no PyTorch
        def __init__(self):
            super().__init__()
            self.f = get_embedder()
        def embed_documents(self, texts): return self.f(texts)
        def embed_query(self, text): return self.f([text])[0]
        async def aembed_documents(self, texts): return self.embed_documents(texts)
        async def aembed_query(self, text): return self.embed_query(text)

    samples = []
    for r in rows:
        res = answer_question(r["question"], kb, k=k)
        samples.append(SingleTurnSample(user_input=r["question"], retrieved_contexts=res["contexts"],
                                        response=res["answer"], reference=r["reference"]))
    judge = LangchainLLMWrapper(ChatOpenAI(
        model=os.getenv("RAGAS_JUDGE_MODEL") or os.getenv("LLM_MODEL") or "gpt-4o-mini",
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.environ["LLM_API_KEY"], temperature=0, max_tokens=4096),
        # Groq/Mistral reject n>1: make separate calls instead of asking for several completions at once.
        bypass_n=True)
    emb = OnnxEmbeddings()
    result = evaluate(EvaluationDataset(samples=samples),
                      metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()],
                      llm=judge, embeddings=emb,
                      # Low concurrency + retries so free-tier rate limits (e.g. Mistral, Groq) don't fail the run.
                      run_config=RunConfig(max_workers=int(os.getenv("RAGAS_MAX_WORKERS", "2")),
                                           max_retries=10, max_wait=60, timeout=180))
    df = result.to_pandas()
    df.to_csv(out, index=False)
    print(f"\nPer-question RAGAS scores written to {out}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "eval" / "results.csv"))
    a = ap.parse_args()
    rows = json.loads((ROOT / "eval" / "testset.json").read_text(encoding="utf-8"))
    kb = KnowledgeBase()
    print("Index:", kb.ingest(ROOT / "knowledge" / "sources.json"))
    print("Retrieval:", {m: round(v, 3) for m, v in retrieval_metrics(kb, rows, a.k).items()})
    if not os.getenv("LLM_API_KEY"):
        print("LLM_API_KEY not set - skipping RAGAS (needs an LLM judge).")
        return
    print("RAGAS:", ragas_eval(kb, rows, a.k, a.out))


if __name__ == "__main__":
    main()
