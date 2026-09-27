FROM python:3.12-slim

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python dependencies ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Application code ----
COPY agent/ ./agent/
COPY memory/ ./memory/
COPY patent_search_mcp/ ./patent_search_mcp/

# ---- Run FastAPI with uvicorn ----
# Use --workers 1 so the in-process MCP client and rate limiter are shared.
# For horizontal scaling, replace the in-process MCP client with an SSE client.
CMD ["uvicorn", "agent.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]