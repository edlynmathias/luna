"""Streamlit version of Luna (for Streamlit Community Cloud). Run: streamlit run streamlit_app.py"""
import os, sys
from datetime import date, timedelta

# Streamlit Cloud's system sqlite can be too old for Chroma; use the bundled one if available.
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)  # always use the latest .env values (Streamlit reruns keep old env vars)
# On Streamlit Cloud, keys come from App settings -> Secrets (same names as .env).
try:
    for k, v in st.secrets.items():
        os.environ.setdefault(k, str(v))
except Exception:
    pass

from rag import KnowledgeBase, answer_question, general_answer

st.set_page_config(page_title="Luna · period care", page_icon="🩸", layout="centered")

# ---------- period-themed styling ----------
st.markdown("""
<style>
.stApp { background: radial-gradient(circle at 10% 0%, #ffe3ea 0%, #fff1f3 40%, #fff7f8 100%); }
.block-container { padding-top: 1.6rem; max-width: 780px; }
.luna-hero { background: linear-gradient(135deg, #8B0A1E 0%, #C8102E 45%, #E8537A 100%);
  color: #fff; border-radius: 22px; padding: 22px 24px; margin-bottom: 14px;
  box-shadow: 0 10px 30px rgba(139,10,30,.25); position: relative; overflow: hidden; }
.luna-hero h1 { color: #fff; margin: 0; font-size: 2.1rem; letter-spacing: .5px; }
.luna-hero p { margin: 4px 0 0; opacity: .92; }
.luna-hero .drops { position: absolute; right: 18px; top: 10px; font-size: 2.6rem; opacity: .35; }
.pill { display: inline-block; background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.35);
  border-radius: 999px; padding: 3px 10px; font-size: .75rem; margin: 10px 6px 0 0; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; background: #ffd9e0; padding: 5px; border-radius: 14px; }
.stTabs [data-baseweb="tab"] { border-radius: 10px; padding: 8px 18px; color: #8B0A1E; font-weight: 600; }
.stTabs [aria-selected="true"] { background: #C8102E !important; color: #fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
[data-testid="stChatMessage"] { background: #ffffff; border: 1px solid #f6c9d3; border-radius: 18px;
  box-shadow: 0 2px 10px rgba(200,16,46,.06); }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background: #ffe3ea; }
[data-testid="stChatInput"] textarea { border-radius: 14px !important; }
.card { background: #fff; border: 1px solid #f6c9d3; border-radius: 18px; padding: 16px 18px; margin: 10px 0;
  box-shadow: 0 2px 10px rgba(200,16,46,.06); }
.next { background: linear-gradient(135deg, #C8102E, #E8537A); color: #fff; text-align: center; }
.next small { letter-spacing: .12em; opacity: .85; font-weight: 700; }
.next .big { font-size: 1.9rem; font-weight: 800; margin: 4px 0; }
.phase-bar { display: flex; height: 14px; border-radius: 999px; overflow: hidden; margin: 10px 0 4px; }
.phase-bar div { height: 100%; }
.legend span { font-size: .78rem; margin-right: 12px; color: #6b2a3a; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }
.log { display: flex; justify-content: space-between; border-bottom: 1px dashed #f6c9d3; padding: 8px 0; }
.log:last-child { border-bottom: 0; }
.muted { color: #9a5a69; font-size: .82rem; }
.stButton > button { border-radius: 12px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading Luna's health library…")
def get_kb():
    kb = KnowledgeBase()
    kb.ingest("knowledge/sources.json")
    return kb


kb = get_kb()
llm_on = bool(os.getenv("LLM_API_KEY"))
st.markdown(f"""
<div class="luna-hero"><div class="drops">🩸🌙</div>
  <h1>Luna</h1>
  <p>Your gentle guide to periods, cycles and menstrual health</p>
  <span class="pill">📚 {len(kb.docs)} trusted source sections</span>
  <span class="pill">🔎 {'semantic search' if kb.mode.startswith('semantic') else 'keyword search'}</span>
  <span class="pill">✨ AI answers {'on' if llm_on else 'off'}</span>
</div>""", unsafe_allow_html=True)

chat, tracker = st.tabs(["🩸 Ask Luna", "📅 Cycle tracker"])

# ---------- chat ----------
with chat:
    if "msgs" not in st.session_state:
        st.session_state.msgs = [{"role": "assistant", "content":
            "Hi, I'm Luna 🌙 Ask me anything about periods, cramps, PCOS, products or your cycle. "
            "I answer from trusted health sources and show you where it came from."}]
    for m in st.session_state.msgs:
        with st.chat_message(m["role"], avatar="🌙" if m["role"] == "assistant" else "💗"):
            st.markdown(m["content"])
    offer = st.session_state.get("offer")
    if offer:
        a, b = st.columns([2, 1])
        if a.button("✨ Yes, answer with AI", type="primary", use_container_width=True):
            st.session_state.offer = None
            try:
                with st.spinner("Asking the AI…"):
                    st.session_state.msgs.append({"role": "assistant", "content": general_answer(offer, st.session_state.msgs[1:])})
            except Exception as e:
                st.session_state.msgs.append({"role": "assistant", "content": f"Sorry, the AI provider returned an error: {e}"})
            st.rerun()
        if b.button("No thanks", use_container_width=True):
            st.session_state.offer = None
            st.rerun()
    if len(st.session_state.msgs) == 1:
        cols = st.columns(3)
        for col, q in zip(cols, ["What is PCOS?", "Are bad cramps normal?", "Is my period too heavy?"]):
            if col.button(q, use_container_width=True):
                st.session_state.pending = q
                st.rerun()
    else:
        if st.button("🔄 New chat", help="Clear the conversation so Luna forgets earlier messages"):
            st.session_state.pop("msgs", None)
            st.session_state.offer = None
            st.rerun()
    st.caption("🩺 Educational information only — not a diagnosis. For severe pain, very heavy bleeding or "
               "feeling faint, contact a doctor or emergency services.")

q = st.chat_input("Ask about periods, symptoms, products…") or st.session_state.pop("pending", None)
if q:
    st.session_state.msgs.append({"role": "user", "content": q})
    st.session_state.offer = None
    try:
        with st.spinner("Checking trusted sources…"):
            # Earlier turns (minus the greeting and this question) give Luna the conversation's context.
            r = answer_question(q, kb, history=st.session_state.msgs[1:-1])
        text = r["answer"]
        if r.get("offer_general"):
            st.session_state.offer = r.get("search_query", q)
        if r["sources"]:
            text += "\n\n---\n**📚 Sources**\n" + "\n".join(f"{s['n']}. [{s['title']}]({s['url']})" for s in r["sources"])
    except Exception as e:
        text = ("The AI provider is rate-limiting requests (free plan limit). Wait a minute and ask again."
                if "429" in str(e) else f"Sorry, the AI provider returned an error: {e}")
    st.session_state.msgs.append({"role": "assistant", "content": text})
    st.rerun()

# ---------- tracker ----------
PHASES = [("Period", "#C8102E"), ("Follicular", "#F48FB1"), ("Ovulation (est.)", "#8E24AA"), ("Luteal", "#F8BBD0")]

with tracker:
    logs = st.session_state.setdefault("logs", [])
    c1, c2, c3 = st.columns(3)
    start = c1.date_input("Period started", value=None)
    end = c2.date_input("Period ended", value=None)
    cycle = int(c3.number_input("Cycle length (days)", 15, 60, 28))
    if st.button("🩸 Save period", type="primary"):
        if not start or not end or end < start:
            st.error("Enter a start and an end date (end on or after start).")
        else:
            logs.append((start, end))
            logs.sort(reverse=True)
            st.success("Saved for this session.")

    if logs:
        last_start, last_end = logs[0]
        period_len = (last_end - last_start).days + 1
        nxt = last_start + timedelta(days=cycle)
        day = (date.today() - last_start).days % cycle + 1
        ovu = max(cycle - 14, period_len + 1)  # ovulation ~14 days before next period
        if day <= period_len: phase = PHASES[0]
        elif day < ovu - 1: phase = PHASES[1]
        elif day <= ovu + 1: phase = PHASES[2]
        else: phase = PHASES[3]
        widths = [period_len, max(ovu - 2 - period_len, 0), 3, max(cycle - ovu - 1, 0)]
        bar = "".join(f'<div style="width:{w / cycle * 100:.1f}%;background:{c}"></div>' for w, (_, c) in zip(widths, PHASES))
        marker = f'<div class="muted" style="margin-left:{(day - 0.5) / cycle * 100:.1f}%">▲ today</div>'
        legend = "".join(f'<span><i class="dot" style="background:{c}"></i>{n}</span>' for n, c in PHASES)

        st.markdown(f"""
<div class="card next"><small>NEXT PERIOD (ESTIMATE)</small>
  <div class="big">{nxt:%d %b %Y}</div>
  <div>{max((nxt - date.today()).days, 0)} days to go · you're on day {day} of {cycle}</div></div>
<div class="card"><b>Where you are in your cycle</b>
  <span class="muted"> — now: <b style="color:{phase[1]}">{phase[0]}</b></span>
  <div class="phase-bar">{bar}</div>{marker}<div class="legend">{legend}</div></div>
""", unsafe_allow_html=True)

        rows = "".join(f'<div class="log"><span>🩸 {s:%d %b %Y}</span>'
                       f'<span class="muted">{(e - s).days + 1} days · ended {e:%d %b}</span></div>' for s, e in logs)
        st.markdown(f'<div class="card"><b>Past periods</b>{rows}</div>', unsafe_allow_html=True)
        if st.button("Delete all logs"):
            logs.clear()
            st.rerun()
    else:
        st.markdown('<div class="card muted">Log your last period to see your next estimated date and cycle phase.</div>',
                    unsafe_allow_html=True)
    st.caption("Estimates assume a regular cycle and are not contraception or fertility advice. "
               "Logs are kept for this browser session only.")
