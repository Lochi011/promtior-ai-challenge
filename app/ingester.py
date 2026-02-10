"""Deep-crawl ingestion pipeline using Wix Sitemaps + PDF.

Fetches sitemap XML, extracts URLs, then scrapes each page synchronously
with requests + BeautifulSoup.  PyPDFLoader handles the presentation PDF.
A custom _parse_page() strips nav/footer/script noise before chunking.

Safeguards against data pollution:
  - Static asset URLs (.jpg, .png, .svg, etc.) are rejected before fetching.
  - Content-Type must be text/html; binary responses are discarded.
  - Pages with < 200 chars of clean text are dropped (galleries, broken pages).
  - Embeddings are sent in batches to avoid OpenAI 429 rate limits.

Strategy Pattern: each data source has its own loader function returning
list[Document] with standardized metadata (source, source_type, title).
"""

import os
import re
import time
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from app.config import Config, get_embeddings, logger

_HTTP_HEADERS = {
    "User-Agent": "PromtiorBot/1.0 (+https://github.com/lochi011/promtior-ai-challenge)"
}

_STATIC_EXTENSIONS = frozenset([
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
    ".mp4", ".mp3", ".wav", ".avi",
    ".pdf", ".zip", ".gz", ".tar",
    ".css", ".js", ".woff", ".woff2", ".ttf", ".eot",
])

_MIN_TEXT_LENGTH = 200
_EMBED_BATCH_SIZE = 50


# ---------------------------------------------------------------------------
# Parsing & cleaning
# ---------------------------------------------------------------------------

def _parse_page(soup: BeautifulSoup) -> str:
    """Extract clean text from a BeautifulSoup object.

    Removes nav, footer, header, script, style, and noscript tags,
    then extracts text from <main> or <article> if available,
    falling back to the full <body>.
    """
    for tag in soup.find_all(["nav", "footer", "header", "script", "style",
                              "noscript", "iframe", "svg", "picture"]):
        tag.decompose()

    for img in soup.find_all("img"):
        img.decompose()

    main = soup.find("main") or soup.find("article") or soup.find("body")
    if main is None:
        return ""

    text = main.get_text(separator="\n", strip=True)

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

    noise = [r"top of page", r"bottom of page", r"Privacy Policy",
             r"Ancla \d+", r"cookie", r"suscri\w+"]
    for pattern in noise:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()


def _is_static_asset(url: str) -> bool:
    """Return True if the URL points to a static file (image, font, etc.)."""
    path = url.split("?")[0].split("#")[0].lower()
    return any(path.endswith(ext) for ext in _STATIC_EXTENSIONS)


def _should_skip(url: str) -> bool:
    """Return True if the URL matches any skip pattern or is a static asset."""
    if _is_static_asset(url):
        return True
    return any(p in url for p in Config.SKIP_URL_PATTERNS)


# ---------------------------------------------------------------------------
# Source loaders (Strategy Pattern)
# ---------------------------------------------------------------------------

def _extract_urls_from_sitemap(sitemap_url: str) -> list[str]:
    """Fetch sitemap XML and return the list of <loc> URLs."""
    try:
        resp = requests.get(sitemap_url, headers=_HTTP_HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException:
        logger.error("Failed to fetch sitemap XML: %s", sitemap_url, exc_info=True)
        return []

    soup = BeautifulSoup(resp.content, "lxml-xml")
    return [loc.text.strip() for loc in soup.find_all("loc")]


def _fetch_page(url: str) -> str:
    """Fetch a single page and return cleaned text.

    Validates Content-Type is text/html before parsing.
    """
    try:
        resp = requests.get(url, headers=_HTTP_HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException:
        logger.warning("  Could not fetch: %s", url)
        return ""

    content_type = resp.headers.get("Content-Type", "")
    if "text/html" not in content_type:
        logger.info("  Skipping (not HTML, got %s): %s", content_type, url)
        return ""

    soup = BeautifulSoup(resp.text, "lxml")
    return _parse_page(soup)


def _load_sitemap(sitemap_url: str, source_type: str) -> list[Document]:
    """Load all pages from a sitemap using synchronous requests."""
    logger.info("Loading sitemap: %s", sitemap_url)
    urls = _extract_urls_from_sitemap(sitemap_url)
    logger.info("  Found %d raw URLs in sitemap", len(urls))

    cleaned: list[Document] = []
    skipped_filter = 0
    skipped_empty = 0

    for url in urls:
        if _should_skip(url):
            logger.info("  SKIP (filtered): %s", url)
            skipped_filter += 1
            continue

        text = _fetch_page(url)
        if len(text) < _MIN_TEXT_LENGTH:
            logger.info("  SKIP (< %d chars): %s", _MIN_TEXT_LENGTH, url)
            skipped_empty += 1
            continue

        doc = Document(
            page_content=text,
            metadata={
                "source": url,
                "source_type": source_type,
                "title": url.split("/")[-1] or "home",
            },
        )
        cleaned.append(doc)
        logger.info("  OK: %s (%d chars)", url, len(text))

    logger.info(
        "Sitemap %s summary: %d loaded, %d filtered, %d empty/short",
        source_type, len(cleaned), skipped_filter, skipped_empty,
    )
    return cleaned


def _load_pdf(file_path: str) -> list[Document]:
    """Load the local PDF and tag with presentation metadata.

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
        doc.metadata["source_type"] = "presentation"
        doc.metadata["title"] = "AI Engineer Presentation"
    return docs


# ---------------------------------------------------------------------------
# Batched embedding (avoids OpenAI 429 rate limits)
# ---------------------------------------------------------------------------

def _embed_in_batches(chunks: list[Document], batch_size: int = _EMBED_BATCH_SIZE) -> FAISS:
    """Embed documents in batches with a pause between each to avoid 429s."""
    embeddings = get_embeddings()

    logger.info("Embedding batch 1/%d (%d docs)...",
                (len(chunks) + batch_size - 1) // batch_size, len(chunks[:batch_size]))
    vector_store = FAISS.from_documents(chunks[:batch_size], embeddings)

    for i in range(batch_size, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        logger.info("Embedding batch %d/%d (%d docs)...", batch_num, total_batches, len(batch))
        time.sleep(1)
        vector_store.add_documents(batch)

    return vector_store


# ---------------------------------------------------------------------------
# Ingestion orchestrator
# ---------------------------------------------------------------------------

def run_ingestion(file_path: str = Config.PDF_PATH) -> None:
    """Deep-crawl sitemaps + PDF, chunk, embed, and persist FAISS index."""

    # 1. Load from all sources
    pages_docs = _load_sitemap(Config.PAGES_SITEMAP, "website")
    blog_docs = _load_sitemap(Config.BLOG_SITEMAP, "blog")
    pdf_docs = _load_pdf(file_path)

    all_docs = pages_docs + blog_docs + pdf_docs
    logger.info(
        "Total: %d pages + %d blog posts + %d PDF pages = %d documents",
        len(pages_docs), len(blog_docs), len(pdf_docs), len(all_docs),
    )

    if not all_docs:
        logger.error("No documents loaded -- aborting ingestion")
        return

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    logger.info(
        "Split into %d chunks (size=%d, overlap=%d)",
        len(chunks), Config.CHUNK_SIZE, Config.CHUNK_OVERLAP,
    )

    if len(chunks) > 2000:
        logger.warning(
            "Chunk count (%d) seems too high -- check URL filtering!", len(chunks)
        )

    # 3. Embed in batches and persist
    vector_store = _embed_in_batches(chunks)
    vector_store.save_local(Config.INDEX_PATH)
    logger.info("FAISS index saved to '%s' (%d vectors)", Config.INDEX_PATH, len(chunks))


if __name__ == "__main__":
    run_ingestion()