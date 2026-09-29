# Patent Prior-Art Research Agent

An AI backend agent that searches USPTO patents via PatentsView, evaluates prior-art relevance using Groq LLM reasoning, and uses **Hindsight** to remember past research — so future queries build on previous work instead of starting from zero.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Agent (:8000)                  │
│                                                         │
│  POST /research                                         │
│    │                                                     │
│    ├─ 1. Hindsight recall ──────────────► Hindsight     │
│    │      (prior research?)               (:8888)        │
│    │                                                     │
│    ├─ 2. Patent Search MCP tool                         │
│    │      search_patents()  ────────────► Google Patents│
│    │      (in-process FastMCP client)     + local corpus │
│    │                                                     │
│    ├─ 3. Groq LLM evaluate each patent                  │
│    │      (accept / reject + reasoning)  ► Groq API     │
│    │                                                     │
│    └─ 4. Hindsight retain ──────────────► Hindsight     │
│           (store verdict + reasoning)     (:8888)        │
│                                                         │
│  GET /reflect  ────────────────────────► Hindsight      │
│  GET /logs     ────────────────────────► App Postgres   │
│                                           (:5432)        │
└─────────────────────────────────────────────────────────┘
```

### Module Boundaries

| Directory | Owner | Responsibility |
|-----------|-------|----------------|
| `patent_search_mcp/` | Patent data team | FastMCP server combining Google Patents search and the verified local corpus |
| `memory/` | Memory team | Hindsight REST client (recall / retain / reflect) |
| `agent/` | Agent / backend team | Orchestration loop, LLM evaluation, FastAPI app, database |

---

## Prerequisites

- [Docker Desktop](https://docs.docker.com/get-docker/) with Compose v2
- A [Groq API key](https://console.groq.com)

---

## Setup

### 1. Clone / navigate to the project directory

```bash
cd patent-prior-art-agent
```

### 2. Create your `.env` file

```bash
cp .env.example .env
```

Open `.env` and fill in:

| Variable | Description |
|---|---|
| `PATENTSVIEW_API_KEY` | Your PatentsView (USPTO) API key |
| `GROQ_API_KEY` | Your Groq API key (used by the agent for reasoning) |
| `HINDSIGHT_API_LLM_PROVIDER` | LLM provider for Hindsight's internal memory (e.g. `groq`) |
| `HINDSIGHT_API_LLM_API_KEY` | API key for Hindsight's LLM (can reuse Groq key) |

The other variables have sensible defaults for local Docker Compose.

### 3. Start all services

```bash
docker compose up --build
```

Run `docker compose` from the `Smriti` folder, with `Patent Data_Search` and `Smriti` as sibling folders in Documents. Compose mounts the local search project read-only into the agent container. This starts:
- **db** — pgvector/pgvector:pg16 on port 5432
- **hindsight** — ghcr.io/vectorize-io/hindsight:latest on ports 8888 / 9999
- **agent** — FastAPI app on port 8000, searching Google Patents and supplementing matches from the three-record local development corpus

Wait for all health checks to pass (about 30–60 seconds on first start, since Hindsight pulls its image and initialises its embedded Postgres).

### 4. Verify services

```bash
curl http://localhost:8000/health
# → {"status":"ok","service":"patent-prior-art-agent"}

curl http://localhost:8888/health
# → Hindsight health response
```

---

## API Reference

### `POST /research`

Run prior-art research for a given query. Candidate search combines live Google Patents results with matching records from `Patent Data_Search`; duplicate patent IDs are removed. The local corpus contains only three battery thermal-management records and is a development fixture, not a complete patent database.

**Request body:**
```json
{
  "query": "neural network image classification using convolutional layers",
  "cpc_class": "G06N3/04",
  "date_from": "2015-01-01",
  "date_to": "2023-12-31",
  "max_patents": 5
}
```

Only `query` is required. All other fields are optional.

**Example:**
```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "neural network image classification using convolutional layers",
    "cpc_class": "G06N3/04",
    "max_patents": 3
  }'
```

**Response:**
```json
{
  "query": "neural network image classification using convolutional layers",
  "recalled_context": [
    {
      "content": "PATENT PRIOR ART RESEARCH\nResearch Query: ...\nVerdict: ACCEPTED\n...",
      "score": 0.92,
      "metadata": {"patent_id": "10,123,456", "verdict": "ACCEPTED"}
    }
  ],
  "findings": [
    {
      "patent_id": "10,410,105",
      "title": "Deep convolutional neural network for image recognition",
      "abstract": "...",
      "assignee": "Acme Corp",
      "cpc_codes": ["G06N3/04", "G06V10/82"],
      "filing_date": "2017-03-15",
      "grant_date": "2019-09-10",
      "verdict": "ACCEPTED",
      "confidence": "HIGH",
      "reasoning": "This patent directly discloses a CNN architecture for image classification that anticipates the queried invention. Claim 1 recites 'a plurality of convolutional layers' applied to image data, which reads on the query."
    }
  ],
  "summary": "Research complete in 4230ms. Evaluated 3 patent(s): 2 accepted as prior art, 1 rejected. Found 1 relevant prior research memory/memories."
}
```

---

### `GET /reflect`

Synthesise patterns across all accumulated research memories.

```bash
curl "http://localhost:8000/reflect"

# With a focused query:
curl "http://localhost:8000/reflect?query=Which+CPC+classes+are+most+commonly+accepted+as+prior+art"
```

**Response:**
```json
{
  "reflection": "Based on 47 accumulated patent evaluations: CPC class G06N3 (neural networks) patents are accepted as prior art in 78% of queries related to machine learning. The most common rejection reason is 'non-analogous art' — patents in G06F16 (data retrieval) are frequently rejected for ML invention queries because they lack the learning/training aspect..."
}
```

---

### `GET /health`

```bash
curl http://localhost:8000/health
```

---

### `GET /logs`

Return recent request audit logs.

```bash
curl "http://localhost:8000/logs?limit=10"
```

---

### Interactive API docs

FastAPI's built-in Swagger UI is available at: **http://localhost:8000/docs**

---

## Development (without Docker)

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Environment

Copy `.env.example` to `.env` and populate it. Then:

```bash
# On Linux/macOS:
export $(cat .env | grep -v '^#' | xargs)

# On Windows PowerShell:
Get-Content .env | ForEach-Object {
  if ($_ -match '^([^#=]+)=(.*)$') {
    [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2])
  }
}
```

### Start just the databases

```bash
docker compose up db hindsight
```

### Run the agent locally

```bash
APP_DATABASE_URL="postgresql+asyncpg://patentuser:patentpass@localhost:5432/patentdb" \
HINDSIGHT_URL="http://localhost:8888" \
uvicorn agent.app:app --host 0.0.0.0 --port 8000 --reload
```

### Run the MCP server standalone (stdio)

```bash
python -m patent_search_mcp.server
```

### Run the MCP server with SSE transport

```bash
python -m patent_search_mcp.server --sse --port 8001
```

---

## Configuration Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | ✅ | — | Groq API key for agent reasoning |
| `GROQ_MODEL` | ❌ | `openai/gpt-oss-120b` | Groq model for patent evaluation |
| `HINDSIGHT_API_LLM_PROVIDER` | ✅ | `groq` | LLM provider for Hindsight's internal memory |
| `HINDSIGHT_API_LLM_API_KEY` | ✅ | — | API key for Hindsight's LLM |
| `HINDSIGHT_API_LLM_MODEL` | ❌ | `openai/gpt-oss-120b` | Model for Hindsight's memory reasoning |
| `HINDSIGHT_URL` | ❌ | `http://hindsight:8888` | URL of the Hindsight REST API |
| `PATENT_DATA_SEARCH_PATH` | ❌ | sibling `Patent Data_Search/Patent Data_Search` folder | Directory containing `patent_search.py` and `data/sample_patents.json`; set to `/opt/patent_data_search` in Docker Compose |
| `APP_DATABASE_URL` | ✅ | (Docker default) | asyncpg connection string for app Postgres |
| `POSTGRES_USER` | ❌ | `patentuser` | App Postgres username |
| `POSTGRES_PASSWORD` | ❌ | `patentpass` | App Postgres password |
| `POSTGRES_DB` | ❌ | `patentdb` | App Postgres database name |
| `LOG_LEVEL` | ❌ | `INFO` | Python logging level |
| `SQL_ECHO` | ❌ | `false` | Set to `true` to echo SQL queries |

---

## Rate Limiting

The local corpus supplements Google Patents results and requires no API key. Search remains keyword-based for the local records; the corpus has three records, no claims, and no CPC classifications. CPC and grant-date filters therefore exclude local records because those fields are unavailable. Publication dates remain labeled separately from grant dates.

---

## Memory Bank

All research memories are stored in Hindsight under the bank ID **`patent-prior-art`**. The Hindsight control plane UI at **http://localhost:9999** lets you browse, search, and manage stored memories.

---

## Adding a New PatentsView Field

1. Add the field name to `_PATENT_FIELDS` in `patent_search_mcp/client.py`
2. Add normalization logic in `_normalize()` in the same file
3. Add the field to the `Patent` model in `patent_search_mcp/models.py`
4. Optionally add it to `PatentFinding` in `agent/schemas.py`

No other files need to change.

---

## Extending to MCP SSE (Remote MCP Client)

The orchestrator currently uses an in-process FastMCP client for zero-overhead calls. To switch to a remote SSE-based MCP server:

1. Start the MCP server: `python -m patent_search_mcp.server --sse --port 8001`
2. In `agent/orchestrator.py`, replace:
   ```python
   from fastmcp import Client as MCPClient
   from patent_search_mcp.server import mcp as patent_mcp_server
   self._mcp_client = MCPClient(patent_mcp_server)
   ```
   with:
   ```python
   from fastmcp import Client as MCPClient
   self._mcp_client = MCPClient("http://patent-mcp:8001/sse")
   ```
3. Add the MCP server as a separate Docker service.
