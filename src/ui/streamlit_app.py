"""Streamlit interface for the Cascade Bank local RAG teaching demo."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.config import get_config

st.set_page_config(page_title="Cascade Bank RAG", page_icon="🔎", layout="wide")

cfg = get_config()

st.title("Cascade Bank RAG Demo")
st.caption(
    "Ask a question about the fictional Cascade Bank corpus. "
    "Answers use retrieved documents and cite them by source number."
)

with st.sidebar:
    st.header("Retrieval settings")
    top_k = st.slider("Candidates to retrieve (top_k)", min_value=1, max_value=20, value=cfg.search.top_k)
    top_n = st.slider(
        "Sources for the answer (top_n)",
        min_value=1,
        max_value=10,
        value=min(cfg.rerank.top_n, 10),
    )
    use_rerank = st.checkbox("Use cross-encoder reranking", value=True)
    doc_type = st.selectbox("Document type", ["all", "pdf", "sop", "csv"])

has_api_key = bool(cfg.openrouter_api_key)
if has_api_key:
    st.sidebar.success("OpenRouter API key found. LLM answer generation is enabled.")
else:
    st.sidebar.info(
        "No OpenRouter API key found. The app will show retrieved context "
        "without making an LLM call."
    )

with st.form("question_form"):
    question = st.text_area(
        "Your question",
        placeholder="For example: What is the bank's procedure for reporting a lost card?",
        height=100,
    )
    submitted = st.form_submit_button("Get answer", type="primary")

if submitted:
    question = question.strip()
    if not question:
        st.warning("Enter a question to continue.")
    else:
        st.session_state.pop("last_result", None)
        with st.spinner("Searching the vector store and preparing an answer..."):
            from src.rerank.cross_encoder_rerank import search_and_rerank

            result = search_and_rerank(
                question,
                top_k=top_k,
                top_n=top_n,
                doc_type=None if doc_type == "all" else doc_type,
                use_rerank=use_rerank,
            )
            sources = result["reranked"]

            if not sources:
                answer = "No relevant source documents were found for this question."
                answer_mode = "No sources found"
            elif has_api_key:
                from src.generate.answer_synthesis import synthesize_answer

                answer = synthesize_answer(question, sources)
                answer_mode = "LLM-generated, grounded in retrieved sources"
            else:
                from src.generate.answer_synthesis import answer_only_mode

                answer = answer_only_mode(question, sources)
                answer_mode = "Retrieved context only — no LLM call was made"

            st.session_state["last_result"] = {
                "question": question,
                "answer": answer,
                "answer_mode": answer_mode,
                "sources": sources,
                "used_rerank": result["used_rerank"],
            }

last_result = st.session_state.get("last_result")
if last_result:
    st.subheader("Answer")
    st.caption(last_result["answer_mode"])
    st.markdown(last_result["answer"])

    st.subheader("Sources")
    st.caption(
        f"{len(last_result['sources'])} source(s) · "
        f"cross-encoder reranking {'on' if last_result['used_rerank'] else 'off'}"
    )
    for rank, source in enumerate(last_result["sources"], start=1):
        metadata = source.get("metadata") or {}
        source_file = metadata.get("source_file", "Unknown source")
        page = metadata.get("page_number", metadata.get("page", "N/A"))
        with st.expander(f"[{rank}] {source_file} · page {page}"):
            score = source.get("score")
            if score is not None:
                st.caption(f"Semantic similarity: {score:.4f}")
            rerank_score = source.get("rerank_score")
            if rerank_score is not None:
                st.caption(f"Rerank score: {rerank_score:.4f}")
            st.text(source.get("text", ""))
