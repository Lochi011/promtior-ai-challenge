"""FastAPI + LangServe entrypoint for the Promtior Bionic Agent."""

import os

from fastapi import FastAPI
from langserve import add_routes
from app.agent import agent_executor

app = FastAPI(
    title="Promtior Bionic API",
    version="1.0.0",
    description="Professional Agentic RAG for Promtior Challenge",
)

add_routes(app, agent_executor, path="/agent")


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe for Railway / Docker health checks."""
    return {"status": "bionic_online"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.server:app", host="0.0.0.0", port=port, reload=True)