"""Store SOP Assistant — RAG chatbot + SOP viewer popup.

Professional Streamlit app for A.J. Textile Mills Limited.
Stack: BGE-M3 embeddings · ChromaDB · Groq LLM · Structure-aware chunks.
"""
from __future__ import annotations

import os

import streamlit as st

import config as C
import ingest
import rag

# PAGE CONFIG  (must be first Streamlit call)
st.set_page_config(
    page_title=C.APP_TITLE,
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# GLOBAL CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, .stTextInput, .stButton, .stSelectbox {
    font-family: 'Inter', sans-serif !important;
}
.block-container { padding: 1.4rem 2rem 2rem 2rem; max-width: 1180px; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1a2744 100%);
    border-right: 1px solid rgba(59,130,246,.18);
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] .stTextInput input {
    background: rgba(30,41,59,.9) !important;
    border: 1px solid rgba(59,130,246,.3) !important;
    border-radius: 8px !important;
    color: #E2E8F0 !important;
}
[data-testid="stSidebar"] .stButton button {
    text-align: left; justify-content: flex-start; border-radius: 9px;
    font-size: .82rem; font-weight: 500; padding: .42rem .75rem;
    background: rgba(30,41,59,.7) !important;
    border: 1px solid rgba(59,130,246,.2) !important;
    color: #94A3B8 !important; transition: all .2s ease; width: 100%; margin-bottom: 2px;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(59,130,246,.18) !important;
    border-color: rgba(59,130,246,.55) !important;
    color: #60A5FA !important; transform: translateX(3px);
}

.hero-wrap {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 45%, #1D4ED8 100%);
    border: 1px solid rgba(59,130,246,.25); border-radius: 18px;
    padding: 1.6rem 2rem; margin-bottom: 1.4rem; position: relative; overflow: hidden;
}
.hero-wrap::before {
    content: ""; position: absolute; top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(59,130,246,.22) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-wrap::after {
    content: ""; position: absolute; bottom: -40px; left: 30%;
    width: 160px; height: 160px;
    background: radial-gradient(circle, rgba(139,92,246,.14) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title { font-size:1.75rem; font-weight:800; color:#F1F5F9; margin:0 0 .3rem 0; letter-spacing:-.02em; }
.hero-sub { font-size:.9rem; color:#94A3B8; margin:0 0 .9rem 0; }
.pill {
    display:inline-block; background:rgba(59,130,246,.15); border:1px solid rgba(59,130,246,.35);
    color:#93C5FD; padding:3px 11px; border-radius:999px; font-size:.72rem; font-weight:600;
    letter-spacing:.03em; margin-right:5px;
}
.pill-green { background:rgba(16,185,129,.12); border-color:rgba(16,185,129,.3); color:#6EE7B7; }
.pill-purple { background:rgba(139,92,246,.12); border-color:rgba(139,92,246,.3); color:#C4B5FD; }

.kpi-card {
    background: linear-gradient(135deg, #1E293B, #162032);
    border: 1px solid rgba(59,130,246,.18); border-radius: 14px;
    padding: .9rem 1.2rem; text-align: center; transition: border-color .25s, transform .25s; cursor: default;
}
.kpi-card:hover { border-color: rgba(59,130,246,.45); transform: translateY(-2px); }
.kpi-val { font-size:2rem; font-weight:800; color:#60A5FA; line-height:1.1; display:block; }
.kpi-label { font-size:.72rem; font-weight:600; color:#64748B; text-transform:uppercase; letter-spacing:.07em; margin-top:.25rem; display:block; }
.kpi-icon { font-size:1.3rem; margin-bottom:.15rem; display:block; }

.section-head {
    font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.1em;
    color:#475569; margin:1.4rem 0 .6rem 0; display:flex; align-items:center; gap:8px;
}
.section-head::after { content:""; flex:1; height:1px; background:rgba(71,85,105,.3); }

.stChatMessage { border-radius:14px !important; border:1px solid rgba(59,130,246,.1) !important; margin-bottom:.6rem !important; }
[data-testid="stChatMessageContent"] p { line-height:1.7; }

.source-header { font-size:.78rem; font-weight:700; color:#60A5FA; letter-spacing:.04em; }
.chunk-badge { display:inline-block; padding:1px 8px; border-radius:999px; font-size:.68rem; font-weight:600; }
.badge-text { background:rgba(16,185,129,.15); color:#6EE7B7; border:1px solid rgba(16,185,129,.3); }
.badge-table { background:rgba(245,158,11,.12); color:#FCD34D; border:1px solid rgba(245,158,11,.3); }

.stAlert { border-radius:12px !important; }
[data-testid="stExpander"] {
    border:1px solid rgba(59,130,246,.15) !important;
    border-radius:12px !important; background:rgba(15,23,42,.5) !important;
}
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #3B82F6, #8B5CF6) !important; border-radius:999px;
}

::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:#0F172A; }
::-webkit-scrollbar-thumb { background:#334155; border-radius:999px; }
::-webkit-scrollbar-thumb:hover { background:#3B82F6; }

.brand-logo {
    display:flex; align-items:center; gap:10px; padding:.5rem 0 1rem 0;
    border-bottom:1px solid rgba(59,130,246,.15); margin-bottom:.8rem;
}
.brand-name { font-size:1rem; font-weight:800; color:#F1F5F9 !important; letter-spacing:-.01em; }
.brand-org { font-size:.73rem; color:#64748B !important; font-weight:400; }
.status-dot {
    width:8px; height:8px; background:#10B981; border-radius:50%; display:inline-block;
    box-shadow:0 0 6px #10B981; animation:pulse-dot 2s infinite;
}
@keyframes pulse-dot {
    0%, 100% { box-shadow:0 0 6px #10B981; }
    50% { box-shadow:0 0 12px #10B981, 0 0 20px rgba(16,185,129,.3); }
}
div[data-testid="column"] .stButton button {
    border-radius:10px; font-size:.82rem; background:rgba(30,41,59,.8) !important;
    border:1px solid rgba(59,130,246,.2) !important; color:#94A3B8 !important;
    transition:all .2s; text-align:left;
}
div[data-testid="column"] .stButton button:hover {
    background:rgba(59,130,246,.15) !important; border-color:rgba(59,130,246,.5) !important; color:#E2E8F0 !important;
}
</style>
""", unsafe_allow_html=True)


# HELPERS
def groq_key() -> str | None:
    try:
        k = st.secrets.get("GROQ_API_KEY", "")
        return k if k else None
    except Exception:
        return os.getenv("GROQ_API_KEY") or None


def index_ready() -> bool:
    try:
        return C.SOPS_JSON.exists() and rag.get_collection().count() > 0
    except Exception:
        return False


def build_index(path=C.DATA_PATH):
    bar = st.progress(0.0, text="Preparing chunks…")
    stats = ingest.build_index(
        path,
        lambda d, t: bar.progress(d / t, text=f"Embedding chunks {d}/{t}…")
    )
    bar.empty()
    st.session_state.lib_version = st.session_state.get("lib_version", 0) + 1
    return stats


# BOOT
if not index_ready():
    if not C.DATA_PATH.exists():
        st.error(f"SOP file not found at `{C.DATA_PATH}`. Place your markdown SOP file there and reload.")
        st.stop()
    with st.spinner("First run — building knowledge base (BGE-M3 downloads ~2 GB once)…"):
        build_index()
    st.rerun()


# DATA
@st.cache_data(show_spinner=False)
def library(version: int) -> list[dict]:
    return ingest.load_sops()


sops      = library(st.session_state.get("lib_version", 0))
sop_by_id = {s["id"]: s for s in sops}
n_chunks  = rag.get_collection().count()
n_tables  = sum(s["n_tables"] for s in sops)


# SOP POPUP DIALOG
@st.dialog("📄  SOP Viewer", width="large")
def sop_dialog(sop_id: str, focus: dict | None = None):
    sop = sop_by_id.get(sop_id)
    if not sop:
        st.error("SOP not found.")
        return
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#1E3A5F,#1D4ED8);'
        f'border-radius:12px;padding:1rem 1.3rem;margin-bottom:1rem;">'
        f'<div style="font-size:1.15rem;font-weight:800;color:#F1F5F9;margin-bottom:.25rem;">{sop["title"]}</div>'
        f'<span style="font-size:.75rem;color:#93C5FD;">📝 {sop["n_points"]} numbered points &nbsp;·&nbsp; 📊 {sop["n_tables"]} table(s)</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if focus:
        st.markdown(f'**🔍 Highlighted passage** &nbsp; `{focus.get("section","")}`')
        with st.container(border=True):
            st.markdown(focus["body"])
        st.divider()
    st.markdown("#### Full SOP Content")
    st.markdown(sop["markdown"])


def request_open(sop_id: str, focus: dict | None = None):
    st.session_state.open_sop = (sop_id, focus)


# SIDEBAR
with st.sidebar:
    st.markdown(
        f'<div class="brand-logo">'
        f'<span style="font-size:1.7rem;">📘</span>'
        f'<div><div class="brand-name">{C.APP_TITLE}</div>'
        f'<div class="brand-org">{C.ORG_NAME}</div></div>'
        f'<span class="status-dot" title="Index active"></span></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-head">📂 SOP Library</div>', unsafe_allow_html=True)
    query = st.text_input("Search SOPs", placeholder="🔍  Search by title…", label_visibility="collapsed", key="sidebar_search")
    shown = [s for s in sops if query.lower() in s["title"].lower()]
    st.caption(f"{len(shown)} of {len(sops)} SOPs  ·  click any to open")

    with st.container(height=380, border=False):
        for s in shown:
            icon = "📊" if s["n_tables"] > 0 else "📄"
            st.button(f"{icon}  {s['title']}", key=f"lib_{s['id']}", use_container_width=True,
                      on_click=request_open, args=(s["id"],))

    st.divider()

    with st.expander("⚙️  Retrieval Settings"):
        top_k = st.slider("Passages to retrieve", 1, 10, C.TOP_K, key="top_k")
        min_score = st.slider("Min. relevance score", 0.0, 0.8, C.MIN_SCORE, 0.05, key="min_score",
                              help="Cosine similarity threshold. Passages below this are ignored.")
        temperature = st.slider("LLM temperature", 0.0, 1.0, C.LLM_TEMPERATURE, 0.05, key="temp")

    with st.expander("🗄️  Knowledge Base"):
        st.caption("Re-index with a new SOP file:")
        up = st.file_uploader("Upload SOP markdown", type=["md", "txt"], label_visibility="collapsed")
        if st.button("🔄  Re-index", use_container_width=True, disabled=(up is None and not C.DATA_PATH.exists())):
            if up is not None:
                C.DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
                C.DATA_PATH.write_bytes(up.getvalue())
            with st.spinner("Re-indexing…"):
                stats = build_index()
            st.success(f"Done! {stats['sops']} SOPs · {stats['chunks']} chunks · {stats['tables']} tables")
            st.rerun()

    st.divider()
    api_ok = bool(groq_key())
    st.markdown(
        f"**LLM:** {'🟢 Connected' if api_ok else '🔴 No API key'}  \n"
        f"**Model:** `{C.LLM_MODEL.split('-versatile')[0]}`  \n"
        f"**Embed:** `BGE-M3`"
    )
    if st.button("🗑️  Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# MAIN — HERO
st.markdown(
    f'<div class="hero-wrap">'
    f'<div class="hero-title">📘 {C.APP_TITLE}</div>'
    f'<div class="hero-sub">Ask any question about {C.ORG_NAME}\'s store procedures, or open any SOP from the library.</div>'
    f'<span class="pill">BGE-M3</span>'
    f'<span class="pill pill-green">ChromaDB</span>'
    f'<span class="pill pill-purple">Structure-aware chunks</span>'
    f'<span class="pill">Text + Tables</span>'
    f'<span class="pill pill-green">Groq LLM</span>'
    f'</div>',
    unsafe_allow_html=True,
)

# KPI CARDS
c1, c2, c3, c4 = st.columns(4)
for col, icon, val, label in (
    (c1, "📋", len(sops), "SOPs indexed"),
    (c2, "🔢", n_chunks,  "Embedded chunks"),
    (c3, "📊", n_tables,  "Tables extracted"),
    (c4, "💬", len(st.session_state.get("messages", [])) // 2, "Questions answered"),
):
    col.markdown(
        f'<div class="kpi-card"><span class="kpi-icon">{icon}</span>'
        f'<span class="kpi-val">{val}</span><span class="kpi-label">{label}</span></div>',
        unsafe_allow_html=True,
    )
st.write("")

if not groq_key():
    st.warning(
        "No `GROQ_API_KEY` found — running in **retrieval-only** mode.  \n"
        "Add your key to `.streamlit/secrets.toml` as `GROQ_API_KEY = \"gsk_...\"` and restart.",
        icon="⚠️",
    )


# SOURCE RENDERER
def render_sources(sources: list[dict], msg_idx: int):
    with st.expander(f"📚  Sources ({len(sources)})  ·  click to expand", expanded=False):
        for i, h in enumerate(sources, 1):
            badge_cls  = "badge-table" if h["chunk_type"] == "table" else "badge-text"
            badge_text = "📊 Table" if h["chunk_type"] == "table" else "📝 Text"
            score_icon = "🟢" if h["score"] >= .65 else ("🟡" if h["score"] >= .5 else "🔴")
            st.markdown(
                f'<div class="source-header">[{i}] {h["sop_title"]}</div>'
                f'<span class="chunk-badge {badge_cls}">{badge_text}</span> &nbsp;'
                f'`{h["section"]}` &nbsp; {score_icon} **{h["score"]:.2f}**',
                unsafe_allow_html=True,
            )
            body = h["body"]
            with st.container(border=True):
                st.markdown(body if h["chunk_type"] == "table" or len(body) < 700 else body[:700] + "…")
            st.button(
                "🔍  Open full SOP", key=f"src_{msg_idx}_{i}",
                on_click=request_open,
                args=(h["sop_id"], {"section": h["section"], "body": h["body"], "chunk_type": h["chunk_type"]}),
            )
            if i < len(sources):
                st.markdown('<hr style="border-color:rgba(59,130,246,.1);margin:.5rem 0">', unsafe_allow_html=True)


# CHAT HISTORY
st.session_state.setdefault("messages", [])
for idx, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            render_sources(m["sources"], idx)

# EXAMPLE QUESTIONS
if not st.session_state.messages:
    st.markdown('<div class="section-head">💡 Try asking</div>', unsafe_allow_html=True)
    examples = [
        "Who is responsible for a shortage in paper cones?",
        "What are the duties of the Gate Clerk?",
        "What is the procedure for an emergency store issue?",
        "What must the Admin Manager check on waste bales?",
        "Explain the yarn unloading procedure.",
        "What reports does the Excise Incharge send to Head Office?",
    ]
    cols = st.columns(2)
    for i, q in enumerate(examples):
        cols[i % 2].button(
            f"💬  {q}", key=f"ex_{i}", use_container_width=True,
            on_click=lambda _q=q: st.session_state.update(queued=_q),
        )

# CHAT INPUT + RESPONSE
prompt = st.chat_input("Ask anything about the SOPs…") or st.session_state.pop("queued", None)
if prompt:
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        key, hits, answer = groq_key(), [], ""
        try:
            with st.spinner("🔍  Searching SOPs…"):
                search_q = rag.condense(history, prompt, key) if key and history else prompt
                hits = rag.retrieve(search_q, top_k, min_score)
            if not hits:
                answer = (
                    "❌  I could not find relevant passages for that question.  \n"
                    "Try rephrasing, or browse the **SOP library** in the sidebar."
                )
                st.markdown(answer)
            elif not key:
                answer = (
                    "🔍  **Retrieval-only mode** — no LLM key configured.  \n"
                    "Here are the most relevant SOP passages (see Sources below)."
                )
                st.markdown(answer)
            else:
                answer = st.write_stream(rag.stream_answer(prompt, hits, history, key, temperature))
        except Exception as e:
            answer = f"Something went wrong: `{e}`"
            st.error(answer)

        slim = [{k: h[k] for k in ("sop_id", "sop_title", "section", "chunk_type", "score", "body")} for h in hits]
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": slim})
        if slim:
            render_sources(slim, len(st.session_state.messages) - 1)

# FOOTER
st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center;font-size:.72rem;color:#334155;padding:.8rem 0;">'
    '📘 <b>Store SOP Assistant</b> &nbsp;·&nbsp; '
    'Answers are grounded solely in the indexed SOPs. '
    'Verify critical decisions against the original document.</div>',
    unsafe_allow_html=True,
)

# DIALOG TRIGGER (must be last)
req = st.session_state.pop("open_sop", None)
if req:
    sop_dialog(*req)
