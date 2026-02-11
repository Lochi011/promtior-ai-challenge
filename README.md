# Promtior Bionic Agent: AI Engineer Challenge
An advanced Agentic RAG assistant developed for Promtior. This agent leverages a "Bionic" architecture, combining deep website crawling via Sitemaps with high-density corporate data from PDF documents to provide precise, context-aware responses.

## 🚀 Key Features
### Deep Website Ingestion: 
Utilizes SitemapLoader to index the entire site (Blog, Use Cases, Services).

### Data Optimization: 
Advanced filtering of binary assets (images/videos) and HTML noise to maximize context quality.

### Agentic Orchestration: 
Built with LangGraph to separate retrieval logic from answer generation.

### Production Stack: 
Powered by FastAPI, LangServe, FAISS, and OpenAI (gpt-4o-mini).

## 🛠 Quick Start
### 1. Clone & Enter
```bash 
1. git clone https://github.com/lochi011/promtior-ai-challenge.git cd promtior-ai-challenge 
```

### 2. Create Virtual Environment
```powershell 
1. python -m venv .venv .venv\Scripts\activate # Windows source .venv/bin/activate # Mac/Linux 
```

### 3. Install Dependencies
```bash
1. pip install -r requirements.txt 
```

### 4. Configure Environment
Copy the example file and add your OPENAI_API_KEY: 
```bash 
cp .env.example .env 
```

### 5. Build the Vector Index (Ingestion)
This script scrapes the Promtior website and processes the PDF data: 
```bash 
python -m app.ingester 
```

### 6. Run the Server
```bash 
python -m app.server 
```

Open the interactive chat at: http://localhost:8000/agent/playground

### 🐳 Docker Support
```bash 
docker build -t promtior-bionic . docker run -p 8000:8000 --env-file .env promtior-bionic
```

### 🧪 Testing
Run the automated test suite to verify agent performance: 
```bash 
python -m app.test_agent 
```

### 📂 Documentation
For a deep dive into the architecture, component diagrams, and design decisions, please refer to the Technical Report (doc/REPORT.md).