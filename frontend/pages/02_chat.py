"""
PRISM — Precision Retrieval with Intelligent Semantic Multimodal
----------------------------------------------------------------
frontend/pages/02_chat.py

Streaming RAG chat interface.
"""

import json
import httpx
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app import inject_global_css, render_sidebar, BACKEND_URL

inject_global_css()
render_sidebar()

# ── Session state ─────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources" not in st.session_state:
    st.session_state.sources = []


# ══════════════════════════════════════════════════════════════════════
# Page header
# ══════════════════════════════════════════════════════════════════════

header_col, btn_col = st.columns([4, 1])
with header_col:
    st.markdown(
        "<h1 style='color:#e2e8f0;font-size:22px;font-weight:500;margin-bottom:4px;'>"
        "💬 Chat with your Documents</h1>"
        "<p style='color:#718096;font-size:13px;'>"
        "Ask anything — answers are grounded in your uploaded documents.</p>",
        unsafe_allow_html=True,
    )
with btn_col:
    if st.button("🗑 Clear Chat", key="clear"):
        st.session_state.messages = []
        st.session_state.sources = []
        st.rerun()

st.divider()

# ── Settings row ──────────────────────────────────────────────────────
s1, s2, s3 = st.columns(3)
with s1:
    top_k = st.slider("Chunks to retrieve (k)", 1, 10, 5)
with s2:
    prompt_type = st.selectbox(
        "Prompt mode",
        ["qa", "conversational", "summary"],
        index=0,
    )
with s3:
    try:
        resp = httpx.get(f"{BACKEND_URL}/ingest/documents", timeout=3)
        sources_list = resp.json().get("sources", []) if resp.status_code == 200 else []
    except Exception:
        sources_list = []

    source_filter = st.selectbox(
        "Filter by document",
        ["All documents"] + sources_list,
        index=0,
    )
    source_filter = None if source_filter == "All documents" else source_filter

st.divider()

# ══════════════════════════════════════════════════════════════════════
# Chat history display
# ══════════════════════════════════════════════════════════════════════

for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(
                f"<div class='user-bubble'>{msg['content']}</div>",
                unsafe_allow_html=True,
            )
    else:
        with st.chat_message("assistant", avatar="🔷"):
            st.markdown(
                f"<div class='ai-bubble'>{msg['content']}</div>",
                unsafe_allow_html=True,
            )
            if msg.get("sources"):
                _render_sources(msg["sources"])


def _render_sources(sources: list[dict]) -> None:
    """Render source citation chips below an answer."""
    if not sources:
        return
    chips = ""
    for src in sources:
        chips += (
            f"<span class='source-chip'>"
            f"📄 {src.get('filename','?')} · p.{src.get('page','?')}"
            f"</span>"
        )
    st.markdown(
        f"<div style='margin-top:8px;'>{chips}</div>",
        unsafe_allow_html=True,
    )

    with st.expander("📎 View source excerpts", expanded=False):
        for i, src in enumerate(sources, 1):
            st.markdown(
                f"<div style='background:#0f1117;border:1px solid #2d3748;"
                f"border-radius:8px;padding:10px 14px;margin-bottom:8px;'>"
                f"<p style='font-size:11px;color:#7F77DD;margin-bottom:4px;'>"
                f"[{i}] {src.get('filename','?')} — Page {src.get('page','?')}</p>"
                f"<p style='font-size:12px;color:#718096;line-height:1.6;margin:0;'>"
                f"{src.get('snippet','')}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════
# Chat input + streaming response
# ══════════════════════════════════════════════════════════════════════

if question := st.chat_input("Ask a question about your documents..."):

    # Check backend
    try:
        health = httpx.get(f"{BACKEND_URL}/health", timeout=3)
        if health.status_code != 200:
            st.error("❌ Backend is not responding. Start it with: uvicorn backend.main:app --reload")
            st.stop()
    except Exception:
        st.error("❌ Cannot connect to backend. Make sure it's running on port 8000.")
        st.stop()

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user", avatar="👤"):
        st.markdown(
            f"<div class='user-bubble'>{question}</div>",
            unsafe_allow_html=True,
        )

    # Stream the response
    with st.chat_message("assistant", avatar="🔷"):
        response_placeholder = st.empty()
        full_response = ""
        sources_received = []

        try:
            with httpx.stream(
                "POST",
                f"{BACKEND_URL}/query/stream",
                json={
                    "question": question,
                    "k": top_k,
                    "prompt_type": prompt_type,
                    "source_filter": source_filter,
                },
                timeout=60,
            ) as stream:
                for line in stream.iter_lines():
                    if not line.startswith("data: "):
                        continue

                    raw = line[6:]  # strip "data: "
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("type")

                    if event_type == "token":
                        full_response += event.get("content", "")
                        response_placeholder.markdown(
                            f"<div class='ai-bubble'>{full_response}▌</div>",
                            unsafe_allow_html=True,
                        )

                    elif event_type == "sources":
                        sources_received = event.get("content", [])

                    elif event_type == "error":
                        st.error(f"❌ {event.get('content', 'Unknown error')}")
                        break

                    elif event_type == "done":
                        break

            # Final render without cursor
            response_placeholder.markdown(
                f"<div class='ai-bubble'>{full_response}</div>",
                unsafe_allow_html=True,
            )

            # Render sources
            if sources_received:
                _render_sources(sources_received)

            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources_received,
            })

        except httpx.TimeoutException:
            st.error("⏱ Response timed out. Try a simpler question or reduce k.")
        except Exception as e:
            st.error(f"❌ Streaming error: {str(e)}")