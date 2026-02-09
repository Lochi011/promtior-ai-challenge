"""LangGraph workflow definition for the Promtior Bionic Agent."""

from langgraph.graph import StateGraph, END
from app.nodes import AgentState, InputState, OutputState, retrieve_node, generate_node
from app.config import logger


def create_agent() -> StateGraph:
    """Build and compile the two-step RAG agent graph.

    Flow:  START → retrieve → generate → END
    """
    logger.info("Building Promtior Bionic Agent graph")
    workflow = StateGraph(AgentState, input=InputState, output=OutputState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    compiled = workflow.compile()
    logger.info("Agent graph compiled successfully")
    return compiled


# Singleton exported to server.py via LangServe
agent_executor = create_agent()