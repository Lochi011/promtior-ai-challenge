"""Dual-source ingestion pipeline — scrapes all Promtior pages AND loads
the local PDF, then combines both into a single FAISS index."""

import os
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from app.config import Config, get_embeddings, logger


def _load_web(urls: list[str]) -> list[Document]:
    """Scrape multiple Promtior pages and tag each with its source URL."""
    all_docs: list[Document] = []
    for url in urls:
        logger.info("Scraping: %s", url)
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = url
            all_docs.extend(docs)
            logger.info("  → loaded %d document(s) from %s", len(docs), url)
        except Exception:
            logger.warning("  ⚠ Failed to scrape %s — skipping", url, exc_info=True)
    return all_docs


def _load_pdf(file_path: str) -> list[Document]:
    """Load the local PDF and return raw documents.

    Raises:
        FileNotFoundError: If the PDF does not exist at the given path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source PDF not found: {file_path}")
    logger.info("Loading PDF from %s", file_path)
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = file_path
    return docs


def run_ingestion(
    file_path: str = Config.PDF_PATH,
    urls: list[str] | None = None,
) -> None:
    """Load all sources, split into chunks, and persist a FAISS index."""
    if urls is None:
        urls = Config.PROMTIOR_URLS

    # --- 1. Gather documents from both sources ---
    web_docs = _load_web(urls)
    pdf_docs = _load_pdf(file_path)
    all_docs = web_docs + pdf_docs
    logger.info(
        "Total: %d web pages + %d PDF pages = %d documents",
        len(web_docs),
        len(pdf_docs),
        len(all_docs),
    )

    if not all_docs:
        logger.error("No documents loaded — aborting ingestion")
        return

    # --- 2. Split into chunks ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(all_docs)
    logger.info(
        "Split into %d chunks (size=%d, overlap=%d)",
        len(chunks),
        Config.CHUNK_SIZE,
        Config.CHUNK_OVERLAP,
    )

    # --- 3. Embed and persist ---
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(Config.INDEX_PATH)
    logger.info("FAISS index saved to '%s'", Config.INDEX_PATH)


if __name__ == "__main__":
    run_ingestion()