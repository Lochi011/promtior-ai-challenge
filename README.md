# Promtior Bionic Agent: AI Engineer Challenge

An advanced **Agentic RAG** assistant developed for Promtior. This agent leverages a "Bionic" architecture, combining deep website crawling via Sitemaps with high-density corporate data from PDF documents to provide precise, context-aware responses.

> **Live Demo:** [promtior-ai-challenge-production.up.railway.app/agent/playground](https://promtior-ai-challenge-production.up.railway.app/agent/playground/)

---

## 🚀 Key Features

| Feature | Description |
|---|---|
| **🌐 Deep Website Ingestion** | Sitemap-based crawling with `requests` + `BeautifulSoup` to index the entire site (Blog, Use Cases, Services). |
| **🧹 Data Optimization** | Advanced filtering of binary assets (images/videos) and HTML noise to maximize context quality. |
| **🤖 Agentic Orchestration** | Built with **LangGraph** to separate retrieval logic from answer generation. |
| **⚡ Production Stack** | Powered by **FastAPI**, **LangServe**, **FAISS**, and **OpenAI** (`gpt-4o-mini`). |

## 📂 Data Sources

The agent is pre-loaded with high-quality context from:

- **Official Website** — Crawled via sitemaps to cover services, use cases, and blog posts.
- **Corporate Presentation** — The `data/AI Engineer.pdf` file is included in this repository.

---

## 🛠 Quick Start

> **Note:** On macOS/Linux, use `python3` and `pip3` if `python`/`pip` point to Python 2.

### 1. Clone & Enter

```bash
git clone https://github.com/lochi011/promtior-ai-challenge.git
cd promtior-ai-challenge
```

### 2. Create & Activate Virtual Environment

```bash
python -m venv .venv
```

**Activate it:**

| OS | Command |
|---|---|
| **Windows (PowerShell)** | `.venv\Scripts\activate` |
| **Windows (CMD)** | `.venv\Scripts\activate.bat` |
| **macOS / Linux** | `source .venv/bin/activate` |

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Copy the example file and add your `OPENAI_API_KEY`:

| OS | Command |
|---|---|
| **Windows (PowerShell)** | `Copy-Item .env.example .env` |
| **Windows (CMD)** | `copy .env.example .env` |
| **macOS / Linux** | `cp .env.example .env` |

Then edit `.env` and replace `sk-your-key-here` with your actual OpenAI API key.

### 5. Build the Vector Index (Ingestion)

This script crawls the Promtior website and processes the PDF data:

```bash
python -m app.ingester
```

### 6. Run the Server

```bash
python -m app.server
```

Open the interactive chat at: **http://localhost:8000/agent/playground**

---

## 🐳 Docker Support

```bash
docker build -t promtior-bionic .
```

```bash
docker run -p 8000:8000 --env-file .env promtior-bionic
```

---

## 🧪 Testing

Run the automated test suite to verify agent performance:

```bash
python -m app.test_agent
```

---

## 📄 Documentation

For a deep dive into the architecture, component diagrams, and design decisions, see the **[Technical Report](doc/REPORT.md)**.