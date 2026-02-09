"""Centralized configuration, validation, and resource factories."""

import os
import logging
from functools import lru_cache
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("PromtiorAgent")


class Config:
    """Single source of truth for all tuneable parameters."""

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    MODEL_NAME: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    INDEX_PATH: str = "faiss_index"
    RETRIEVER_K: int = 5
    PDF_PATH: str = "data/AI Engineer.pdf"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    PROMTIOR_URLS: list[str] = [
        "https://www.promtior.ai/",
        "https://www.promtior.ai/service",
        "https://www.promtior.ai/use-cases",
        "https://www.promtior.ai/contact-us",
        "https://www.promtior.ai/blog",
    ]

    @classmethod
    def validate(cls) -> None:
        """Raise early if required secrets are missing."""
        if not cls.OPENAI_API_KEY:
            raise EnvironmentError(
                "OPENAI_API_KEY is missing. Add it to your .env file."
            )


Config.validate()


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Create and cache a single ChatOpenAI instance."""
    logger.info("Initializing LLM: %s", Config.MODEL_NAME)
    return ChatOpenAI(model=Config.MODEL_NAME, temperature=0)


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    """Create and cache a single OpenAIEmbeddings instance."""
    return OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_retriever() -> VectorStoreRetriever:
    """Load the persisted FAISS index once and cache the retriever.

    Raises:
        FileNotFoundError: If the FAISS index directory does not exist.
    """
    if not os.path.exists(Config.INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at '{Config.INDEX_PATH}'. "
            "Run ingester.py first."
        )
    logger.info("Loading FAISS index from '%s'", Config.INDEX_PATH)
    vector_store = FAISS.load_local(
        Config.INDEX_PATH,
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
    return vector_store.as_retriever(search_kwargs={"k": Config.RETRIEVER_K})