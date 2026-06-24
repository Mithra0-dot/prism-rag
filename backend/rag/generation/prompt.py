"""
PRISM — Precision Retrieval with Intelligent Semantic Multimodal
----------------------------------------------------------------
backend/rag/generation/prompt.py

Prompt templates for PRISM's RAG generation pipeline.

Prompt engineering is one of the most critical and underrated parts
of building a production RAG system. A poorly designed prompt causes:
  - Hallucination (model invents facts not in the context)
  - Ignoring the context (model uses training knowledge instead)
  - Verbose, unfocused answers
  - Missing source citations

PRISM uses three prompt variants:
  1. RAG_QA_PROMPT       — standard Q&A with strict grounding
  2. CONVERSATIONAL_PROMPT — multi-turn chat with history
  3. SUMMARY_PROMPT      — document summarisation

All prompts follow these principles:
  - Explicitly instruct the model to ONLY use provided context
  - Ask for structured output (answer + sources)
  - Instruct the model to say "I don't know" when context is insufficient
    (this is critical for reducing hallucination)

Usage:
    from backend.rag.generation.prompt import get_rag_prompt, PromptType

    prompt = get_rag_prompt(PromptType.QA)
"""

from enum import Enum
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


# ══════════════════════════════════════════════════════════════════════
# Prompt Type Enum
# ══════════════════════════════════════════════════════════════════════

class PromptType(str, Enum):
    QA             = "qa"
    CONVERSATIONAL = "conversational"
    SUMMARY        = "summary"


# ══════════════════════════════════════════════════════════════════════
# System Instruction (shared across all prompts)
# ══════════════════════════════════════════════════════════════════════

_SYSTEM_INSTRUCTION = """You are PRISM, an expert document analysis assistant.
Your job is to answer questions accurately and concisely using ONLY the context \
provided below.

STRICT RULES you must follow:
1. ONLY use information from the provided context to answer.
2. If the context does not contain enough information, say exactly: \
"I don't have enough information in the provided documents to answer this question."
3. NEVER make up facts, statistics, or details not present in the context.
4. Always cite which source and page number your answer comes from.
5. Be concise — answer in 3-5 sentences unless the question requires more detail.
6. If the question is ambiguous, answer the most likely interpretation and state \
your assumption."""


# ══════════════════════════════════════════════════════════════════════
# Prompt 1 — Standard RAG Q&A
# ══════════════════════════════════════════════════════════════════════

_RAG_QA_TEMPLATE = """
CONTEXT FROM DOCUMENTS:
{context}

---

QUESTION: {question}

INSTRUCTIONS:
- Answer based strictly on the context above.
- At the end of your answer, list the sources you used in this format:
  Sources: [filename, page X], [filename, page Y]
- If the context doesn't answer the question, say so clearly.

ANSWER:"""

RAG_QA_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(_SYSTEM_INSTRUCTION),
    HumanMessagePromptTemplate.from_template(_RAG_QA_TEMPLATE),
])


# ══════════════════════════════════════════════════════════════════════
# Prompt 2 — Conversational RAG (with chat history)
# ══════════════════════════════════════════════════════════════════════

_CONVERSATIONAL_TEMPLATE = """
CONTEXT FROM DOCUMENTS:
{context}

---

CONVERSATION HISTORY:
{chat_history}

---

CURRENT QUESTION: {question}

INSTRUCTIONS:
- Use the conversation history to understand follow-up questions.
- Answer using ONLY the provided context.
- Cite sources at the end of your answer.
- If referring to something mentioned earlier in the conversation,
  make it clear you're doing so.

ANSWER:"""

CONVERSATIONAL_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(_SYSTEM_INSTRUCTION),
    HumanMessagePromptTemplate.from_template(_CONVERSATIONAL_TEMPLATE),
])


# ══════════════════════════════════════════════════════════════════════
# Prompt 3 — Document Summarisation
# ══════════════════════════════════════════════════════════════════════

_SUMMARY_TEMPLATE = """
DOCUMENT CONTENT:
{context}

---

TASK: Provide a structured summary of the document content above.

FORMAT YOUR RESPONSE AS:
## Key Topics
(List the 3-5 main topics covered)

## Summary
(2-3 paragraph summary of the main content)

## Key Facts & Figures
(Bullet list of important numbers, dates, or statistics mentioned)

## Notable Gaps
(What important questions does this document NOT answer?)

SUMMARY:"""

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are PRISM, an expert document analyst. "
        "Summarise the provided content accurately and concisely. "
        "Only include information present in the document."
    ),
    HumanMessagePromptTemplate.from_template(_SUMMARY_TEMPLATE),
])


# ══════════════════════════════════════════════════════════════════════
# Condense Question Prompt (for conversational follow-ups)
# ══════════════════════════════════════════════════════════════════════
# This prompt is used to rephrase follow-up questions into standalone
# questions before retrieval. Without this, "What about Q4?" would
# be searched literally instead of "What was the Q4 revenue?"

_CONDENSE_TEMPLATE = """Given the following conversation history and a follow-up \
question, rephrase the follow-up question to be a standalone question that \
captures the full context needed to answer it.

Conversation History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_CONDENSE_TEMPLATE)


# ══════════════════════════════════════════════════════════════════════
# Factory function
# ══════════════════════════════════════════════════════════════════════

def get_rag_prompt(prompt_type: PromptType = PromptType.QA) -> ChatPromptTemplate:
    """
    Return the appropriate prompt template for the given use case.

    Args:
        prompt_type : Which prompt variant to return

    Returns:
        LangChain ChatPromptTemplate ready for use in a chain
    """
    _PROMPT_MAP = {
        PromptType.QA:             RAG_QA_PROMPT,
        PromptType.CONVERSATIONAL: CONVERSATIONAL_PROMPT,
        PromptType.SUMMARY:        SUMMARY_PROMPT,
    }
    return _PROMPT_MAP[prompt_type]


def format_context(documents) -> str:
    """
    Format retrieved Document objects into a clean context string
    for injection into the prompt.

    Each chunk is labelled with its source and page number so the
    LLM can include accurate citations in its response.

    Args:
        documents : List of LangChain Document objects from retrieval

    Returns:
        Formatted multi-line string ready for {context} injection
    """
    if not documents:
        return "No relevant documents found."

    sections = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Unknown")
        page   = doc.metadata.get("page", "?")
        text   = doc.page_content.strip()

        sections.append(
            f"[{i}] Source: {source} | Page: {page}\n{text}"
        )

    return "\n\n---\n\n".join(sections)
