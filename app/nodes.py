# -*- coding: utf-8 -*-
"""LangGraph node definitions for the Promtior Bionic Agent.

Each node is a pure function: (AgentState) -> partial state dict.
The retrieve node carries source metadata so the generate node can cite.
"""

from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from app.config import get_llm, get_retriever, logger


# ---------------------------------------------------------------------------
# State schemas (Input / Output / Internal)
# ---------------------------------------------------------------------------

class InputState(TypedDict):
    """What the client sends - only the question."""

    question: str


class OutputState(TypedDict):
    """What the client receives - only the answer."""

    answer: str


class AgentState(TypedDict):
    """Full internal state that flows through every node in the graph."""

    question: str
    context: str
    sources: str
    answer: str


# ---------------------------------------------------------------------------
# System prompt - XML-tagged, Chain-of-Thought, non-shy
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = """You are the Promtior Bionic Assistant.

<verified_facts>
- Company Name: Promtior
- Founded: May 2023
- Founders: Emiliano Chinelli and Ignacio Acu\u00f1a
- Core Focus: Augmented Automation, AI Solutions (Agents), Big Data
- Services: GenAI Product Delivery, GenAI Department as a Service, GenAI Adoption Consulting
- Key Clients: CAF, 5M Travel Group, Guyer & Regules, Paigo, Handy, Vangwe, \
Forestal Atl\u00e1ntico Sur, Advice Consulting, Infocorp, L'Or\u00e9al Latam, SIEE, \
ST Consultores, RPA Maker, CIEMSA, S1, Incapital
- Key Stats: 70% productivity boost, 3.5x return per $1 invested, 91% of leading orgs investing in AI
- Website: https://promtior.ai/
</verified_facts>

<instructions>
1. You will receive a <context> block with text retrieved from the Promtior website \
and the AI Engineer presentation PDF, and a <question> from the user.
2. FIRST, silently reason about what information is available in the context AND \
in the verified facts above. Do NOT show your reasoning to the user.
3. If ANY relevant information exists in either source, YOU MUST answer. \
Synthesize and summarize - do NOT say "I don't have enough information" \
when there is relevant data available.
4. When answering, briefly cite the source: (Source: Website) or (Source: Presentation).
5. Only say you cannot answer if the question is completely unrelated to Promtior.
6. Never invent facts. Be concise, professional, and helpful.
</instructions>"""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def retrieve_node(state: AgentState) -> dict[str, str]:
    """Search the FAISS knowledge base and return context with source tags."""
    logger.info("RETRIEVE node - query: %s", state["question"])
    retriever = get_retriever()
    docs = retriever.invoke(state["question"])

    tagged_chunks: list[str] = []
    source_labels: list[str] = []
    for doc in docs:
        source_type = doc.metadata.get("source_type", "unknown")
        source_url = doc.metadata.get("source", "unknown")
        label = f"[{source_type}: {source_url}]"
        tagged_chunks.append(f"{label}\n{doc.page_content}")
        if label not in source_labels:
            source_labels.append(label)

    context = "\n\n---\n\n".join(tagged_chunks)
    sources = ", ".join(source_labels)
    logger.info("RETRIEVE node - %d chunks from sources: %s", len(docs), sources)
    return {"context": context, "sources": sources}


def generate_node(state: AgentState) -> dict[str, str]:
    """Generate a grounded answer using XML-tagged context and CoT reasoning."""
    logger.info("GENERATE node - building prompt")

    if not state["context"].strip():
        logger.warning("GENERATE node - empty context, returning fallback")
        return {
            "answer": (
                "I couldn't find specific information about that in the Promtior "
                "knowledge base. Please try rephrasing your question or ask "
                "about Promtior's services, founders, or case studies."
            )
        }

    llm = get_llm()

    user_message = (
        f"<context>\n{state['context']}\n</context>\n\n"
        f"<question>\n{state['question']}\n</question>"
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    logger.info("GENERATE node - answer length: %d chars", len(response.content))
    return {"answer": response.content}