"""
PRISM — Precision Retrieval with Intelligent Semantic Multimodal
----------------------------------------------------------------
frontend/pages/01_upload.py

Document upload and ingestion page.
"""

import httpx
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app import inject_global_css, render_sidebar, BACKEND_URL

st.set_page_config(
    page_title="PRISM — Upload",
    page_icon="📤",
    layout="wide",
)

inject_global_css()
render_sidebar()


# ══════════════════════════════════════════════════════════════════════
# Page
# ══════════════════════════════════════════════════════════════════════

st.markdown(
    "<h1 style='color:#e2e8f0;font-size:22px;font-weight:500;margin-bottom:4px;'>"
    "📤 Upload Documents</h1>"
    "<p style='color:#718096;font-size:13px;margin-bottom:1.5rem;'>"
    "Ingest PDF, TXT, MD, or DOCX files into PRISM's vector store.</p>",
    unsafe_allow_html=True,
)

# ── Upload form ───────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Drop your document here",
        type=["pdf", "txt", "md", "docx"],
        help="Max 50MB. Supported: PDF, TXT, Markdown, DOCX",
    )

with col2:
    st.markdown(
        "<p style='font-size:12px;color:#718096;margin-bottom:8px;'>Chunking Strategy</p>",
        unsafe_allow_html=True,
    )
    strategy = st.selectbox(
        "Strategy",
        options=["recursive", "token", "semantic"],
        index=0,
        label_visibility="collapsed",
        help=(
            "recursive — best general purpose\n"
            "token — precise token counting\n"
            "semantic — topic-aware splitting (slow)"
        ),
    )

    st.markdown(
        "<p style='font-size:12px;color:#718096;margin-bottom:8px;margin-top:12px;'>Chunk Size</p>",
        unsafe_allow_html=True,
    )
    chunk_size = st.slider(
        "Chunk Size",
        min_value=128,
        max_value=1024,
        value=512,
        step=64,
        label_visibility="collapsed",
    )

    st.markdown(
        f"<p style='font-size:11px;color:#4a5568;'>{chunk_size} tokens per chunk</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='font-size:12px;color:#718096;margin-bottom:8px;margin-top:12px;'>Vision OCR (Phase 2)</p>",
        unsafe_allow_html=True,
    )
    vision = st.toggle(
        "Extract text from charts/images",
        value=False,
        help="Uses OpenCV + EasyOCR to read text from embedded images. Slower but more complete.",
    )

st.divider()

# ── Ingest button ─────────────────────────────────────────────────────
if uploaded_file:
    st.markdown(
        f"<div style='background:#1a1f2e;border:1px solid #2d3748;border-radius:8px;"
        f"padding:10px 14px;margin-bottom:1rem;font-size:13px;color:#a0aec0;'>"
        f"📄 <strong style='color:#e2e8f0'>{uploaded_file.name}</strong> — "
        f"{uploaded_file.size / 1024:.1f} KB</div>",
        unsafe_allow_html=True,
    )

    if st.button("⚡ Ingest Document", key="ingest_btn"):
        with st.spinner("Running ingestion pipeline..."):
            try:
                response = httpx.post(
                    f"{BACKEND_URL}/ingest/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                    data={
                        "chunk_strategy": strategy,
                        "chunk_size": str(chunk_size),
                        "vision": str(vision).lower(),
                    },
                    timeout=120,
                )

                if response.status_code == 201:
                    data = response.json()

                    st.success(f"✅ '{data['filename']}' ingested successfully!")

                    # Stats row
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Pages Loaded", data["pages_loaded"])
                    with c2:
                        st.metric("Chunks Created", data["chunks_created"])
                    with c3:
                        st.metric("Strategy", data["chunk_strategy"].title())
                    with c4:
                        st.metric(
                            "Total Chunks",
                            data["collection_stats"].get("chunk_count", "—"),
                        )

                    st.markdown(
                        f"<div style='background:#0a1628;border:1px solid #1D9E75;"
                        f"border-radius:8px;padding:10px 14px;margin-top:1rem;"
                        f"font-size:12px;color:#5DCAA5;'>"
                        f"🔑 Source ID: <code style='color:#9FE1CB'>{data['source_id']}</code>"
                        f"<br><span style='color:#4a5568;font-size:11px;'>"
                        f"Save this ID to delete the document later.</span></div>",
                        unsafe_allow_html=True,
                    )

                    # Clear cache so sidebar updates
                    st.cache_data.clear()

                else:
                    error = response.json()
                    st.error(f"❌ Ingestion failed: {error.get('message', response.text)}")

            except httpx.TimeoutException:
                st.error("⏱ Request timed out. Large files may take longer — try again.")
            except Exception as e:
                st.error(f"❌ Connection error: {str(e)}")
else:
    st.markdown(
        "<p style='font-size:13px;color:#4a5568;text-align:center;padding:2rem;'>"
        "Upload a file above to get started.</p>",
        unsafe_allow_html=True,
    )

st.divider()

# ── Loaded documents table ────────────────────────────────────────────
st.markdown(
    "<h2 style='color:#e2e8f0;font-size:16px;font-weight:500;margin-bottom:1rem;'>"
    "Loaded Documents</h2>",
    unsafe_allow_html=True,
)

try:
    resp = httpx.get(f"{BACKEND_URL}/ingest/documents", timeout=5)
    if resp.status_code == 200:
        data = resp.json()
        sources = data.get("sources", [])

        if sources:
            for src in sources:
                st.markdown(
                    f"<div style='display:flex;align-items:center;justify-content:space-between;"
                    f"background:#1a1f2e;border:1px solid #2d3748;border-radius:8px;"
                    f"padding:10px 14px;margin-bottom:6px;'>"
                    f"<div style='display:flex;align-items:center;gap:10px;'>"
                    f"<span style='color:#1D9E75;font-size:16px;'>📄</span>"
                    f"<span style='font-size:13px;color:#e2e8f0;'>{src}</span>"
                    f"</div>"
                    f"<span style='font-size:11px;color:#4a5568;'>indexed</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                f"<p style='font-size:11px;color:#4a5568;margin-top:8px;'>"
                f"Total chunks in store: <strong style='color:#7F77DD'>"
                f"{data.get('total_chunks', 0):,}</strong></p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<p style='font-size:13px;color:#4a5568;'>No documents ingested yet.</p>",
                unsafe_allow_html=True,
            )
except Exception:
    st.warning("Could not fetch document list — is the backend running?")