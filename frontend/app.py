"""
PRISM — Precision Retrieval with Intelligent Semantic Multimodal
----------------------------------------------------------------
frontend/app.py

Streamlit application entry point.

Run with:
    streamlit run frontend/app.py
"""

import httpx
import streamlit as st

# ── Page config — must be the first Streamlit call ────────────────────
st.set_page_config(
    page_title="PRISM — RAG Engine",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = "http://localhost:8000/api/v1"


# ══════════════════════════════════════════════════════════════════════
# Global CSS
# ══════════════════════════════════════════════════════════════════════

def inject_global_css() -> None:
    st.markdown("""
    <style>
    /* ── Hide default Streamlit chrome ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* ── App background ── */
    .stApp {
        background-color: #0f1117;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #1a1f2e !important;
        border-right: 1px solid #2d3748;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: #a0aec0;
        font-size: 13px;
    }

    /* ── Buttons ── */
    .stButton > button {
        background-color: #7F77DD;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.4rem 1.2rem;
        font-size: 14px;
        font-weight: 500;
        transition: background 0.2s;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #534AB7;
        color: white;
        border: none;
    }

    /* ── Secondary button ── */
    .stButton > button[kind="secondary"] {
        background-color: #1a1f2e;
        color: #a0aec0;
        border: 1px solid #2d3748;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: #2d3748;
        color: #e2e8f0;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background-color: #1a1f2e;
        border: 1px dashed #2d3748;
        border-radius: 10px;
        padding: 1rem;
    }

    /* ── Text input ── */
    .stTextInput > div > div > input {
        background-color: #1a1f2e;
        color: #e2e8f0;
        border: 1px solid #2d3748;
        border-radius: 8px;
    }

    /* ── Select box ── */
    .stSelectbox > div > div {
        background-color: #1a1f2e;
        color: #e2e8f0;
        border: 1px solid #2d3748;
        border-radius: 8px;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background-color: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 1rem;
    }
    [data-testid="stMetricValue"] {
        color: #7F77DD;
        font-size: 1.8rem;
    }
    [data-testid="stMetricLabel"] {
        color: #718096;
        font-size: 12px;
    }

    /* ── Divider ── */
    hr {
        border-color: #2d3748;
        margin: 1rem 0;
    }

    /* ── Code blocks ── */
    .stCodeBlock {
        background-color: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 8px;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background-color: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 8px;
        color: #a0aec0;
    }

    /* ── Alert / info boxes ── */
    .stAlert {
        background-color: #1a1f2e;
        border-radius: 8px;
    }

    /* ── Source chip style ── */
    .source-chip {
        display: inline-block;
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 999px;
        padding: 3px 10px;
        font-size: 11px;
        color: #718096;
        margin: 2px;
    }

    /* ── Chat message bubbles ── */
    .user-bubble {
        background: #1a2035;
        border: 1px solid #2d3748;
        border-radius: 12px 12px 2px 12px;
        padding: 12px 16px;
        color: #cbd5e0;
        font-size: 14px;
        line-height: 1.6;
        margin-bottom: 4px;
    }
    .ai-bubble {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 12px 12px 12px 2px;
        padding: 12px 16px;
        color: #e2e8f0;
        font-size: 14px;
        line-height: 1.6;
        margin-bottom: 4px;
    }

    /* ── PRISM logo text ── */
    .prism-logo {
        font-size: 22px;
        font-weight: 600;
        background: linear-gradient(135deg, #7F77DD, #1D9E75);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 2px;
    }
    .prism-sub {
        font-size: 10px;
        color: #4a5568;
        letter-spacing: 0.05em;
        margin-top: -4px;
    }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════

def render_sidebar() -> None:
    with st.sidebar:
        # Logo
        st.markdown("""
        <div style="padding: 0.5rem 0 1.2rem 0; border-bottom: 1px solid #2d3748; margin-bottom: 1rem;">
            <div class="prism-logo">PRISM</div>
            <div class="prism-sub">Precision Retrieval with Intelligent Semantic Multimodal</div>
        </div>
        """, unsafe_allow_html=True)

        # Navigation
        st.page_link("app.py", label="🏠  Home", )
        st.page_link("pages/01_upload.py", label="📤  Upload Documents")
        st.page_link("pages/02_chat.py", label="💬  Chat")
        st.page_link("pages/03_evaluation.py", label="📊  Evaluation")

        st.divider()

        # Loaded documents
        st.markdown(
            "<p style='font-size:10px;color:#4a5568;text-transform:uppercase;"
            "letter-spacing:0.08em;'>Loaded Documents</p>",
            unsafe_allow_html=True,
        )

        docs = _fetch_documents()
        if docs:
            for doc in docs:
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px;"
                    f"padding:4px 0;font-size:12px;color:#718096;'>"
                    f"<span style='width:6px;height:6px;border-radius:50%;"
                    f"background:#1D9E75;display:inline-block;flex-shrink:0'></span>"
                    f"{doc}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<p style='font-size:12px;color:#4a5568;'>No documents loaded yet.</p>",
                unsafe_allow_html=True,
            )

        st.divider()

        # Stats
        stats = _fetch_stats()
        if stats:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Chunks", f"{stats.get('chunk_count', 0):,}")
            with col2:
                st.metric("Docs", len(docs) if docs else 0)

            st.markdown(
                f"<p style='font-size:11px;color:#4a5568;margin-top:8px;'>"
                f"🔍 Hybrid Search (BM25 + Vector)<br>"
                f"🧠 {stats.get('embedding_model','').split('/')[-1]}</p>",
                unsafe_allow_html=True,
            )

        # Backend status
        st.divider()
        backend_ok = _check_backend()
        status_color = "#1D9E75" if backend_ok else "#E24B4A"
        status_text  = "Backend Online" if backend_ok else "Backend Offline"
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:6px;"
            f"font-size:11px;color:{status_color};'>"
            f"<span style='width:6px;height:6px;border-radius:50%;"
            f"background:{status_color};display:inline-block'></span>"
            f"{status_text}</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════
# Backend helpers
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=30)
def _fetch_documents() -> list[str]:
    try:
        resp = httpx.get(f"{BACKEND_URL}/ingest/documents", timeout=5)
        if resp.status_code == 200:
            return resp.json().get("sources", [])
    except Exception:
        pass
    return []


@st.cache_data(ttl=30)
def _fetch_stats() -> dict:
    try:
        resp = httpx.get(f"{BACKEND_URL}/ingest/documents", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "chunk_count": data.get("total_chunks", 0),
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            }
    except Exception:
        pass
    return {}


def _check_backend() -> bool:
    try:
        resp = httpx.get(f"{BACKEND_URL}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════
# Home page
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    inject_global_css()
    render_sidebar()

    st.markdown("""
    <div style="padding: 2rem 0 1rem 0;">
        <div class="prism-logo" style="font-size:36px;">PRISM</div>
        <p style="color:#718096;font-size:14px;margin-top:4px;">
            Precision Retrieval with Intelligent Semantic Multimodal RAG Engine
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<p style='color:#a0aec0;font-size:15px;max-width:600px;line-height:1.7'>"
        "Upload your documents, ask questions, and get grounded answers "
        "with source citations — powered by hybrid BM25 + vector search "
        "and GPT-4o-mini generation."
        "</p>",
        unsafe_allow_html=True,
    )

    st.divider()

    # Feature cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background:#1a1f2e;border:1px solid #2d3748;border-radius:10px;padding:1.2rem;">
            <div style="font-size:20px;margin-bottom:8px;">📤</div>
            <div style="font-size:14px;font-weight:500;color:#e2e8f0;margin-bottom:6px;">Upload</div>
            <div style="font-size:12px;color:#718096;line-height:1.6;">
                PDF, TXT, MD, DOCX — ingested with semantic chunking and stored in ChromaDB.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background:#1a1f2e;border:1px solid #2d3748;border-radius:10px;padding:1.2rem;">
            <div style="font-size:20px;margin-bottom:8px;">💬</div>
            <div style="font-size:14px;font-weight:500;color:#e2e8f0;margin-bottom:6px;">Chat</div>
            <div style="font-size:12px;color:#718096;line-height:1.6;">
                Ask questions and get streaming answers grounded in your documents with page citations.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background:#1a1f2e;border:1px solid #2d3748;border-radius:10px;padding:1.2rem;">
            <div style="font-size:20px;margin-bottom:8px;">📊</div>
            <div style="font-size:14px;font-weight:500;color:#e2e8f0;margin-bottom:6px;">Evaluate</div>
            <div style="font-size:12px;color:#718096;line-height:1.6;">
                Track faithfulness, relevancy and context precision with Ragas + MLflow.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown(
        "<p style='font-size:12px;color:#4a5568;'>👈 Start by uploading a document from the sidebar</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()