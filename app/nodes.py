# -*- coding: utf-8 -*-
"""LangGraph node definitions for the Promtior Bionic Agent."""

from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from app.config import get_llm, get_retriever, logger


# --- State Schemas ---
class InputState(TypedDict):
    """What the client sends (only the question)."""

    question: str


class OutputState(TypedDict):
    """What the client receives (only the answer)."""

    answer: str


class AgentState(TypedDict):
    """Full internal state that flows through every node in the graph."""

    question: str
    context: str
    answer: str


# --- Promtior Identity Guardrail ---
SYSTEM_PROMPT: str = (
    "You are a Bionic Assistant representing Promtior, "
    "a company specializing in AI-driven automation solutions.\n\n"
    "## PROMTIOR VERIFIED FACTS (always prefer these over retrieved context):\n"
    "- Company Name: Promtior\n"
    "- Founded: May 2023\n"
    "- Founders: Emiliano Chinelli and Ignacio Acu\u00f1a\n"
    "- Core Focus: Augmented Automation, AI Solutions (Agents), Big Data\n"
    "- Website: https://promtior.ai/\n\n"
    "## INSTRUCTIONS:\n"
    "1. Answer the user's question using the CONTEXT provided below.\n"
    "2. If the context is insufficient, fall back to the VERIFIED FACTS above.\n"
    "3. If neither source contains the answer, reply: "
    "'I don't have enough information to answer that.'\n"
    "4. Never invent facts. Be concise and professional.\n"
)


# --- Nodes ---
def retrieve_node(state: AgentState) -> dict[str, str]:
    """Search the FAISS knowledge base for relevant chunks."""
    logger.info("RETRIEVE node — query: %s", state["question"])
    retriever = get_retriever()
    docs = retriever.invoke(state["question"])
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    logger.info("RETRIEVE node — found %d chunks", len(docs))
    return {"context": context}


def generate_node(state: AgentState) -> dict[str, str]:
    """Generate a grounded answer using the retrieved context."""
    logger.info("GENERATE node — building prompt")
    llm = get_llm()

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"CONTEXT:\n{state['context']}\n\n"
            f"QUESTION:\n{state['question']}"
        )),
    ]

    response = llm.invoke(messages)
    logger.info("GENERATE node — answer length: %d chars", len(response.content))
    return {"answer": response.content}