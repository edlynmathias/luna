# Luna — menstrual-health chatbot + period tracker

A small Flask app with a single plain HTML page (static/index.html, inline CSS + JS), a local period tracker, and source-grounded retrieval. Intended as an educational MVP, not a diagnostic service.

## Run locally (Windows / macOS / Linux)

1. Install Python 3.10–3.12 from https://www.python.org/downloads/ (on Windows, select **Add Python to PATH**).
2. In this folder, create a virtual environment:
   - Windows PowerShell: `py -m venv .venv; .\.venv\Scripts\Activate.ps1`
   - macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`
3. Install dependencies (no PyTorch needed):
   - **App:** `pip install -r requirements.txt`
   - **App + tests + RAGAS:** `pip install -r requirements-eval.txt`
4. Copy `.env.example` to `.env`. Add your own API key and adjust `LLM_BASE_URL` / `LLM_MODEL` for a compatible hosted chat-completions API. **Never commit `.env`.**
5. Run `python app.py` and open http://127.0.0.1:5000.

### Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repo (do **not** commit `.env`).
2. At https://share.streamlit.io choose **New app**, pick the repo, and set the main file to `streamlit_app.py`.
3. In **Advanced settings → Secrets**, paste:
   ```
   LLM_API_KEY = "gsk_..."
   LLM_BASE_URL = "https://api.groq.com/openai/v1"
   LLM_MODEL = "llama-3.3-70b-versatile"
   ```
Locally: `streamlit run streamlit_app.py`. (`app.py` + `static/index.html` is the Flask version; both use the same `rag.py`.)
Note: the Streamlit tracker keeps logs only for the session; the Flask page saves them in the browser.

### Internet / API requirements

- The first run downloads a small ONNX MiniLM embedding model (~80 MB); allow internet access and enough disk space. Chroma stores vectors locally under `.chroma`.
- With a working embedding install, retrieval uses semantic embeddings and Chroma. If that stack fails to initialize, retrieval falls back to a small keyword ranker.
- A hosted LLM API key and internet access are required for generated answers. Without `LLM_API_KEY`, Luna returns an explicitly labelled source-based excerpt summary, not a generated AI response. API costs and provider privacy terms apply.

## Tests

Run `pytest -q`. Tests cover retrieval, chunking, relevance filtering, RAGAS-ready `contexts`, source-registry citation integrity, uncertainty when retrieval returns no evidence, and a basic cycle-date calculation. Tracker UI logic is browser-side; no browser automation is included.

## RAG pipeline

`knowledge/sources.json` → sentence-aware chunking (overlap 1 sentence) → MiniLM embeddings (ONNX Runtime, no PyTorch) → Chroma (cosine) → top-k chunks filtered by `MIN_RELEVANCE` → grounded prompt with numbered citations → LLM. If nothing passes the threshold, Luna says it lacks evidence instead of guessing. `/api/chat` returns `answer`, `sources` (numbered to match `[n]` citations), `contexts` and `retrieval_mode`.

## Conversation memory (follow-up questions)

Luna remembers the current chat, so you can ask follow-ups like "is that normal?" or "what helps with the second one?" and refer back to things you said earlier (e.g. your age or symptoms).

- Both front-ends send the last few turns with each question (`/api/chat` accepts `{"message": ..., "history": [{"role": "user"|"assistant", "content": ...}]}`).
- For retrieval, a follow-up is first rewritten into a standalone question using the earlier turns (an extra low-cost LLM call at temperature 0). Without `LLM_API_KEY`, the previous question is combined with the new one instead. The response includes this as `search_query`.
- The earlier turns are sent to the LLM before the grounded prompt, so it can resolve "it"/"that" and remember context; facts still come only from the retrieved excerpts. Old source lists and `[n]` citation numbers are stripped from history so they don't clash with new citations.
- Replying "yes" after Luna offers a general AI answer now gives that answer.
- `HISTORY_MESSAGES` (default 8) sets how many earlier messages are used. **New chat** clears the conversation. History is kept in the page only (not saved on the server); reloading the page starts fresh.

## Knowledge base

- `knowledge/sources.json`: a hand-curated seed of 14 short excerpts. It always loads.
- `knowledge/urls.txt` → `python ingest.py` → `knowledge/web_sources.json`: the full pages from about 45 trusted pages (NHS, MedlinePlus, the U.S. Office on Women's Health, ACOG, KidsHealth), split into one entry per section. The launchers run this automatically on first start. Add URLs and re-run it any time, then restart the app.
- The vector index is cached in `.chroma` and rebuilt only when the sources change.

`LLM_TEMPERATURE` defaults to 0.2: low enough to stick closely to the sources, high enough to read naturally. Keep it between 0 and 0.3 for health answers. The RAGAS judge always uses 0.

## Evaluation (RAGAS)

`python eval/ragas_eval.py` runs `eval/testset.json` (questions, reference answers, expected source ids):

- **Always:** retrieval hit rate@k, MRR and precision@k against the labelled source ids. No API key is needed.
- **With `LLM_API_KEY`:** an end-to-end run scored by RAGAS (Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference, LLMContextRecall), with per-question scores in `eval/results.csv`. `RAGAS_JUDGE_MODEL` lets you use a stronger judge.

Add a row to `testset.json` whenever you add sources, and tune `MIN_RELEVANCE` and `k` from these scores.

## Sources and ingestion report

`knowledge/sources.json` is the source registry: six short manually curated excerpts with source titles, URLs, and retrieval keywords from NHS, MedlinePlus, the U.S. Office on Women's Health, ACOG, and CDC. On startup, the app reads these six records and attempts to embed/index them. The live `/api/status` endpoint reports the document count and whether semantic indexing initialized. This is a small hand-curated seed set, **not a full-text crawl**; verify source pages and expand/clinically review excerpts before wider use.

## Privacy

Period entries and cycle preference stay in browser local storage and are not included in chat requests. Browser-profile users can access this data; clearing site data may erase it. Use the tracker’s delete controls to remove logs. Chat questions are sent to your configured hosted API; do not include identifying details.

## Limitations / unfinished

- No clinical review, diagnostic capability, user accounts, encryption-at-rest beyond browser defaults, sync, or medical emergency triage system.
- Semantic dependencies/models may fail to install on some laptops; keyword fallback is intentionally included.
- Current RAG corpus is intentionally small. The pad answer is grounded in the supplied excerpts, but product-specific details (e.g. wings, fragrance, exact change intervals) are not all represented in the seed corpus; do not treat the MVP as a definitive product guide.
- The date prediction uses latest logged start + configured typical cycle length; it is an estimate, not contraception or fertility guidance.
- API errors are shown in the chat.
