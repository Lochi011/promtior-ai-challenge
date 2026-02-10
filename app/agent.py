"""LangGraph workflow definition for the Promtior Bionic Agent.

Graph:  START -> retrieve -> generate -> END

Uses separate InputState / OutputState so LangServe only requires
{"question": "..."} from the client and only returns {"answer": "..."}.
"""

from langgraph.graph import StateGraph, END
from app.nodes import AgentState, InputState, OutputState, retrieve_node, generate_node
from app.config import logger


def create_agent() -> StateGraph:
    """Build and compile the two-step RAG agent graph."""
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


agent_executor = create_agent()