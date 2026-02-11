# ---- Stage: Production Image ----
FROM python:3.11-slim

# Prevent .pyc files and ensure logs flush immediately
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install OS-level dependencies for FAISS and web scraping
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run as non-root user for security
RUN useradd -m appuser
USER appuser

# Expose the FastAPI / LangServe port
EXPOSE 8000

# Railway injects PORT; default to 8000 for local Docker
CMD ["sh", "-c", "uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-8000}"]