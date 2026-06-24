"""
PRISM — Precision Retrieval with Intelligent Semantic Multimodal
----------------------------------------------------------------
frontend/pages/03_evaluation.py

Ragas evaluation dashboard.
"""

import httpx
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app import inject_global_css, render_sidebar, BACKEND_URL



inject_global_css()
render_sidebar()

st.markdown(
    "<h1 style='color:#e2e8f0;font-size:22px;font-weight:500;margin-bottom:4px;'>"
    "📊 RAG Evaluation Dashboard</h1>"
    "<p style='color:#718096;font-size:13px;margin-bottom:1.5rem;'>"
    "Measure retrieval quality using Ragas metrics. Track experiments with MLflow.</p>",
    unsafe_allow_html=True,
)

st.divider()

# ── Metric explanations ───────────────────────────────────────────────
st.markdown(
    "<h2 style='color:#e2e8f0;font-size:16px;font-weight:500;margin-bottom:1rem;'>"
    "What the Metrics Mean</h2>",
    unsafe_allow_html=True,
)

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown("""
    <div style="background:#1a1f2e;border:1px solid #2d3748;border-radius:10px;padding:1.2rem;">
        <div style="font-size:13px;font-weight:500;color:#7F77DD;margin-bottom:6px;">
            Faithfulness
        </div>
        <div style="font-size:12px;color:#718096;line-height:1.6;">
            Are the claims in the answer actually supported by the retrieved context?
            <br><br>
            <strong style="color:#a0aec0">1.0</strong> = every claim is grounded
            <br>
            <strong style="color:#a0aec0">0.0</strong> = pure hallucination
        </div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div style="background:#1a1f2e;border:1px solid #2d3748;border-radius:10px;padding:1.2rem;">
        <div style="font-size:13px;font-weight:500;color:#1D9E75;margin-bottom:6px;">
            Answer Relevancy
        </div>
        <div style="font-size:12px;color:#718096;line-height:1.6;">
            Does the generated answer actually address the question that was asked?
            <br><br>
            <strong style="color:#a0aec0">1.0</strong> = perfectly on-topic
            <br>
            <strong style="color:#a0aec0">0.0</strong> = completely off-topic
        </div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div style="background:#1a1f2e;border:1px solid #2d3748;border-radius:10px;padding:1.2rem;">
        <div style="font-size:13px;font-weight:500;color:#EF9F27;margin-bottom:6px;">
            Context Precision
        </div>
        <div style="font-size:12px;color:#718096;line-height:1.6;">
            Of the chunks retrieved, how many were actually useful for answering?
            <br><br>
            <strong style="color:#a0aec0">1.0</strong> = all retrieved chunks useful
            <br>
            <strong style="color:#a0aec0">0.0</strong> = all chunks irrelevant
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Run evaluation ────────────────────────────────────────────────────
st.markdown(
    "<h2 style='color:#e2e8f0;font-size:16px;font-weight:500;margin-bottom:1rem;'>"
    "Run Evaluation</h2>",
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='font-size:13px;color:#718096;margin-bottom:1rem;'>"
    "Enter a question and the expected answer to evaluate PRISM's retrieval quality.</p>",
    unsafe_allow_html=True,
)

eval_question = st.text_input(
    "Test Question",
    placeholder="What was the Q3 revenue?",
    key="eval_q",
)
eval_ground_truth = st.text_input(
    "Ground Truth Answer",
    placeholder="The Q3 revenue was $4.2 billion...",
    key="eval_gt",
)

if st.button("🧪 Run Evaluation", key="run_eval"):
    if not eval_question or not eval_ground_truth:
        st.warning("Please provide both a question and a ground truth answer.")
    else:
        with st.spinner("Running Ragas evaluation..."):
            try:
                # First get the RAG answer
                resp = httpx.post(
                    f"{BACKEND_URL}/query/ask",
                    json={"question": eval_question},
                    timeout=60,
                )

                if resp.status_code == 200:
                    result = resp.json()
                    rag_answer = result["answer"]
                    sources = result["sources"]
                    chunks = result.get("chunks", [])

                    st.markdown(
                        "<h3 style='color:#e2e8f0;font-size:14px;font-weight:500;"
                        "margin:1rem 0 0.5rem;'>PRISM's Answer</h3>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='ai-bubble'>{rag_answer}</div>",
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        "<p style='font-size:12px;color:#718096;margin-top:1rem;'>"
                        "⚠️ Full Ragas evaluation requires your OpenAI API key to be set. "
                        "Add <code>OPENAI_API_KEY</code> to your <code>.env</code> file "
                        "and set it up in <code>backend/evaluation/ragas_eval.py</code>.</p>",
                        unsafe_allow_html=True,
                    )

                    # Show mock scores as placeholder
                    st.markdown(
                        "<h3 style='color:#e2e8f0;font-size:14px;font-weight:500;"
                        "margin:1rem 0 0.5rem;'>Evaluation Scores (Preview)</h3>",
                        unsafe_allow_html=True,
                    )

                    sc1, sc2, sc3 = st.columns(3)
                    with sc1:
                        st.metric(
                            "Faithfulness",
                            "—",
                            help="Requires Ragas evaluation to run",
                        )
                    with sc2:
                        st.metric(
                            "Answer Relevancy",
                            "—",
                            help="Requires Ragas evaluation to run",
                        )
                    with sc3:
                        st.metric(
                            "Context Precision",
                            "—",
                            help="Requires Ragas evaluation to run",
                        )

                    st.info(
                        "💡 The full Ragas evaluation pipeline is implemented in "
                        "`backend/evaluation/ragas_eval.py` and will be wired up in Phase 2."
                    )

                else:
                    st.error(f"Query failed: {resp.text}")

            except Exception as e:
                st.error(f"Evaluation error: {str(e)}")

st.divider()

# ── MLflow info ───────────────────────────────────────────────────────
st.markdown(
    "<h2 style='color:#e2e8f0;font-size:16px;font-weight:500;margin-bottom:1rem;'>"
    "MLflow Experiment Tracking</h2>",
    unsafe_allow_html=True,
)

st.markdown("""
<div style="background:#1a1f2e;border:1px solid #2d3748;border-radius:10px;padding:1.2rem;">
    <p style="font-size:13px;color:#a0aec0;line-height:1.7;margin-bottom:12px;">
        PRISM logs every evaluation run to MLflow so you can compare:
    </p>
    <ul style="font-size:12px;color:#718096;line-height:2;padding-left:1.2rem;">
        <li>Faithfulness scores across different chunking strategies</li>
        <li>Retrieval quality at different k values</li>
        <li>Response latency per query</li>
        <li>Effect of chunk_size on context precision</li>
    </ul>
    <p style="font-size:12px;color:#4a5568;margin-top:12px;">
        Launch MLflow UI:
        <code style="background:#0f1117;padding:2px 6px;border-radius:4px;color:#7F77DD;">
        mlflow ui --backend-store-uri data/mlflow
        </code>
    </p>
</div>
""", unsafe_allow_html=True)