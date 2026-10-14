# PolicyPilot | RBI Intelligence AI

<div align="center">

[![Windows](https://img.shields.io/badge/Platform-Windows%2010%20%2F%2011-0078D6?style=flat&logo=windows&logoColor=white)](https://www.microsoft.com/windows)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5.4-646CFF?style=flat&logo=vite&logoColor=white)](https://vitejs.dev/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14+-336791?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![pgvector](https://img.shields.io/badge/pgvector-0.3+-336791?style=flat)](https://github.com/pgvector/pgvector)
[![Docker](https://img.shields.io/badge/Docker-Compose%20Ready-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-blue?style=flat)](https://github.com/langchain-ai/langgraph)
[![LangSmith](https://img.shields.io/badge/LangSmith-Tracing%20%26%20Evals-orange?style=flat)](https://smith.langchain.com/)

**Agentic RAG and Deep Research system for multi-year RBI Annual Reports and financial policy analysis with strict citation grounding.**

[Key Features](#-key-features) • [Architecture](#-architecture) • [Prerequisites](#-prerequisites-windows) • [Docker Quickstart](#-docker--docker-compose-quickstart) • [Step-by-Step Setup](#-step-by-step-setup-windows-powershell) • [Configuration](#-configuration-reference) • [Evaluations](#-langsmith--evaluations) • [Project Structure](#-project-structure)

</div>

---

## 📌 Overview

**PolicyPilot** is an enterprise-grade agentic Question-Answering and Policy Intelligence platform built on top of Reserve Bank of India (RBI) Annual Reports. 

Unlike generic RAG pipelines that hallucinate or mix up financial figures across fiscal years, PolicyPilot guarantees that:
- Every answer is strictly grounded in retrieved text passages.
- Every claim carries **verified report-year and page-level citations**.
- Model-submitted citations undergo deterministic string and fuzzy quote validation directly against raw database chunks.
- If relevant evidence does not exist in the indexed reports, the system refuses to answer rather than fabricating details.

---

## ✨ Key Features

- **🔎 Hybrid Retrieval (Dense + Sparse)**: Combines dense vector semantic search using `BAAI/bge-m3` with PostgreSQL Full-Text Search (tsvector), fused via **Reciprocal Rank Fusion (RRF)**.
- **🤖 Dual-Mode Intelligence**:
  - **Standard RAG Mode**: Fast deterministic workflow powered by **LangGraph** (`plan` → `retrieve` → `generate` → `validate`).
  - **Deep Research Mode**: Multi-turn autonomous investigation driven by the **OpenAI Agents SDK** for multi-step report discovery, cross-referencing, and comparative syntheses.
- **🛡️ Strict Grounding & Citation Validation**:
  - Deterministically verifies `[S#]` source tags against extracted chunk text.
  - Page numbers and report years are read directly from PostgreSQL metadata, never trusted from model hallucinations.
  - Numerical checks flag discrepancies between cited figures and responses.
- **📄 Page-Bounded Document Chunking**: High-fidelity PDF extraction with `pymupdf` or `pdfplumber`, ensuring chunks never span multiple pages so page citations are 100% accurate.
- **🔐 Secure Authentication & Admin Panel**: Role-based access control (Admin & Analyst) using JWT and bcrypt, with an admin portal to upload PDFs and monitor indexing in real time.
- **📊 Observability & Evals**: Native **LangSmith** integration for full trace inspection, feedback collection, and automated benchmark evaluations.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   React + Vite UI (Windows)                 │
│        (Chat Interface, Source Inspector, Admin Dashboard)  │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP / JSON (JWT Auth)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                        │
├──────────────────────────────┬──────────────────────────────┤
│      Standard RAG Mode       │      Deep Research Mode      │
│     (LangGraph Workflow)     │     (OpenAI Agents SDK)      │
│  [Plan ➔ Retrieve ➔ Answer]   │   [Multi-turn Tool Loop]     │
└──────────────┬───────────────┴──────────────┬───────────────┘
               │                              │
               ▼                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Deterministic Validator                 │
│         Verifies [S#] quotes against raw chunk text        │
└──────────────────────────────┬──────────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
┌──────────────────────────────┐ ┌───────────────────────────┐
│     PostgreSQL + pgvector    │ │    HuggingFace Embedder   │
│  - Vectors (BGE-M3 1024d)    │ │        BAAI/bge-m3        │
│  - Full-Text Search (BM25/FTS│ │     (Local CPU / CUDA)    │
│  - Chunks, Reports, Users    │ └───────────────────────────┘
└──────────────────────────────┘
```

---

## 📋 Prerequisites (Windows)

Ensure you have the following installed on Windows 10/11:

1. **Python 3.11+**:
   - Download from [python.org](https://www.python.org/downloads/).
   - ⚠️ **Important:** During installation, check the box: **"Add python.exe to PATH"**.
2. **Node.js 18+ (LTS)** & **npm**:
   - Download from [nodejs.org](https://nodejs.org/).
3. **PostgreSQL 14+ with pgvector**:
   - **Docker Desktop (Recommended)**: Easiest method with no local database installation required. Commands to run `pgvector/pgvector:pg16` are provided in [Step 5](#5-set-up--initialize-database-postgresql--pgvector).
   - **Local Windows Service**: Official PostgreSQL 14+ installer with the `pgvector` Windows extension.
4. **Git for Windows**:
   - Download from [git-scm.com](https://git-scm.com/).
5. **OpenAI API Key**:
   - Required for LLM answer generation and research agent planning.

---

## 🐳 Docker & Docker Compose Quickstart

If you prefer to run the entire PolicyPilot stack (Database, Backend API, and Frontend UI) in containers without manually configuring local Python or Node environments:

### 1. Configure Backend Environment
```powershell
copy backend\.env.example backend\.env
```
Open `backend\.env` and supply your `OPENAI_API_KEY`:
```env
OPENAI_API_KEY=sk-your-openai-api-key-here
```

### 2. Build and Start the Full Stack
```powershell
docker compose up -d
```
Docker Compose will launch:
- `rbi_postgres` (pgvector/pgvector:pg16) on port `5432`
- `rbi_backend` (FastAPI with embeddings & agentic RAG) on port `8000`
- `rbi_frontend` (React + Vite UI) on port `5173`

### 3. Start or Resume Containers Anytime
When resuming development or after restarting your PC:
```powershell
# Start all containers in the stack:
docker compose start

# Or start the PostgreSQL container directly:
docker start rbi_postgres
```

### 4. Check Status & Live Logs
```powershell
# View running containers:
docker compose ps

# Follow backend logs:
docker compose logs -f backend
```

### 5. Stop Containers
```powershell
# Pause / stop services:
docker compose stop

# Or stop and remove containers (database volume persists):
docker compose down
```

Access the application in your browser:
- 💻 **Frontend**: [http://localhost:5173](http://localhost:5173)
- ⚙️ **Backend API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- 🩺 **Health Check**: [http://localhost:8000/api/health](http://localhost:8000/api/health)

---

## 🚀 Step-by-Step Setup (Windows PowerShell)

Open **PowerShell** (Run as Administrator if needed to allow script execution).

### Step 1: Clone the Repository
```powershell
git clone https://github.com/your-username/PolicyPilot.git
cd PolicyPilot
```

---

### Step 2: Backend Setup

#### 1. Navigate to the backend directory
```powershell
cd backend
```

#### 2. Create and activate a Python virtual environment
```powershell
python -m venv .venv
```
If script execution is disabled on your system, enable it for this process:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
Activate the virtual environment:
```powershell
.\.venv\Scripts\Activate.ps1
```
*(Your terminal prompt should now show `(.venv)`)*

#### 3. Install Python dependencies
```powershell
pip install -r requirements.txt
```

#### 4. Configure environment file (`.env`)
Create your `.env` from the template:
```powershell
copy .env.example .env
```
Open `.env` in Notepad or VS Code:
```powershell
notepad .env
```
Update the required variables:
```env
# Database Settings
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rbi_intelligence
DB_USER=postgres
DB_PASSWORD=your_postgres_password

# Authentication
JWT_SECRET=your_custom_32_plus_character_secret_key_here
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin282005

# OpenAI API & Model Selection
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_CHAT_MODEL=gpt-4.1
OPENAI_PLANNER_MODEL=gpt-4.1-mini
OPENAI_VALIDATOR_MODEL=gpt-4.1-mini
OPENAI_AGENT_MODEL=gpt-4.1

# Optional Tracing (LangSmith)
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
```
Save and close Notepad.

#### 5. Set Up & Initialize Database (PostgreSQL + pgvector)

Ensure PostgreSQL is running before initializing the database:

- **Using Docker (Recommended)**:
  ```powershell
  # First time setup (creates and starts container):
  # Note: port 5433 prevents conflicts if another local PostgreSQL service occupies 5432
  docker run -d --name rbi_postgres -p 5433:5432 -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=PolicyPilot pgvector/pgvector:pg16

  # When resuming work or after PC restart (starts existing container):
  docker start rbi_postgres
  ```

- **Using Local Windows PostgreSQL**:
  Ensure the local PostgreSQL service is running on your configured port with credentials matching `.env` and `pgvector` extension installed.

Once PostgreSQL is running, execute the database setup script:
```powershell
python scripts\setup_db.py
```

**Expected output:**
```text
Database 'PolicyPilot' created / already exists
pgvector extension ready (version 0.8.7)
```

> 💡 **Note**: Application tables, HNSW vector indexes, Full-Text Search (GIN) indexes, and the default admin user are automatically bootstrapped when you start the FastAPI server in Step 6.

#### 6. Start the FastAPI Backend Server
```powershell
uvicorn app.main:app --reload --port 8000
```
The backend API is now running at **http://localhost:8000**.
- Interactive API Docs (Swagger): **http://localhost:8000/docs**
- Health check: **http://localhost:8000/api/health**

---

### Step 3: Frontend Setup

Open a **second PowerShell window** (leave the backend server running in the first window):

#### 1. Navigate to the frontend directory
```powershell
cd PolicyPilot\frontend
```

#### 2. Configure environment file (`.env`)
```powershell
copy .env.example .env
```
*(Default settings point to `http://localhost:8000/api` on port `5173`)*

#### 3. Install packages and launch the frontend
```powershell
npm install
npm run dev
```

Open your browser and navigate to:
👉 **http://localhost:5173**

---

## 🔑 Initial Admin Login & Document Ingestion

1. Navigate to **http://localhost:5173/admin/login**.
2. Sign in with the default admin credentials from `.env`:
   - **Username**: `admin`
   - **Password**: `admin282005`
3. **Upload an RBI Annual Report**:
   - Enter the fiscal year for the report (e.g., `2023-24` or `2024`).
   - Select your RBI Annual Report PDF file and click **Upload & Index**.
   - The backend will extract text, chunk it by page, compute embeddings with `BAAI/bge-m3`, and index both vectors and full-text search tokens into PostgreSQL.
4. **Start Querying**:
   - Return to the **Chat** screen (**http://localhost:5173/**).
   - Toggle between **Standard** mode and **Deep Research** mode.
   - Inspect source citations, verification scores, and page-level references.

> **Note on Initial Embedding Model Download**:
> On the very first PDF upload, the system downloads the `BAAI/bge-m3` embedding model (~2.3 GB) from Hugging Face to `%USERPROFILE%\.cache\huggingface`. Subsequent uploads run locally and immediately.

---

## ⚙️ Configuration Reference

All environment variables in `backend\.env`:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DB_HOST` | `localhost` | PostgreSQL host |
| `DB_PORT` | `5432` | PostgreSQL port |
| `DB_NAME` | `rbi_intelligence` | PostgreSQL database name |
| `DB_USER` | `postgres` | PostgreSQL username |
| `DB_PASSWORD` | `postgres` | PostgreSQL password |
| `JWT_SECRET` | *required* | Random string (min 32 chars) for signing auth JWTs |
| `OPENAI_API_KEY` | *required* | Your OpenAI API key |
| `OPENAI_CHAT_MODEL` | `gpt-4.1` | Primary LLM for answering user queries |
| `OPENAI_PLANNER_MODEL`| `gpt-4.1-mini` | LLM for tool-call planning |
| `OPENAI_VALIDATOR_MODEL`| `gpt-4.1-mini` | LLM for citation and factual consistency check |
| `OPENAI_AGENT_MODEL` | `gpt-4.1` | Model for the multi-turn deep research agent |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | Embedding model |
| `EMBEDDING_DEVICE` | `cpu` | Device for embedding inference (`cpu` or `cuda`) |
| `RETRIEVAL_TOP_K` | `8` | Top chunk passages fed to the answering model |
| `MIN_SIMILARITY` | `0.35` | Minimum cosine similarity threshold |
| `HYBRID_SEARCH` | `true` | Combine dense vectors + sparse full-text search |
| `LANGSMITH_TRACING` | `false` | Enable LangSmith observability traces |

---

## 🧪 Testing & Evaluation (Windows PowerShell)

In `backend` with your virtual environment activated:

### Run Unit Tests
```powershell
pip install pytest httpx
pytest
```

### Run Integration Tests
```powershell
# Set throwaway test database in PowerShell
$env:DB_NAME="rbi_test"
python scripts\setup_db.py

$env:RBI_TEST_DB_NAME="rbi_test"
pytest
```

---

## 🔬 LangSmith & Evaluations (Windows PowerShell)

To benchmark answer accuracy, citation fidelity, and latency against evaluation datasets:

1. Set your LangSmith API key in `backend\.env`:
   ```env
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=your_langsmith_key
   LANGSMITH_PROJECT=rbi-intelligence-ai
   ```
2. Run the evaluation script:
   ```powershell
   python -m evals.run_eval
   ```
3. Inspect benchmark curves and traces in your LangSmith web dashboard.

---

## 📂 Project Structure

```text
PolicyPilot\
├── backend\
│   ├── app\              # FastAPI application, AI agent, DB & RAG logic
│   ├── evals\            # LangSmith evaluation datasets and runners
│   ├── prompts\          # System prompts for planner, chat, agent, validator
│   ├── scripts\          # DB setup and admin password utility scripts
│   ├── tests\            # Unit and integration test suite
│   ├── .dockerignore     # Docker build ignore rules
│   ├── .env.example      # Backend environment template
│   ├── Dockerfile        # Backend container specification
│   └── requirements.txt  # Python dependencies
├── frontend\
│   ├── src\              # React components, pages, API clients, styles
│   ├── .dockerignore     # Docker build ignore rules
│   ├── .env.example      # Frontend environment template
│   ├── Dockerfile        # Frontend container specification
│   ├── package.json      # Frontend dependencies
│   └── vite.config.js    # Vite configuration
├── docker-compose.yml    # Full-stack Docker orchestration
├── .gitignore
└── README.md
```

---

## ⚠️ Notes for Windows Users

- **Docker Container Management (`docker start`)**:
  If you restart your PC or stop containers, do **not** re-run `docker run` (which causes a `Conflict: The container name "/rbi_postgres" is already in use` error). Instead, simply start the existing container:
  ```powershell
  # Start the existing database container:
  docker start rbi_postgres

  # Or start the entire Docker Compose stack:
  docker compose start
  ```
- **Docker Engine Not Running**:
  If you encounter `failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine`, ensure **Docker Desktop** is opened and running from the Windows Start menu.
- **PowerShell Execution Policy**: If `Activate.ps1` gives an execution policy error, run:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```
- **Password Rotation**: Generate a bcrypt hash for a new admin password before production:
  ```powershell
  python scripts\hash_password.py
  ```
- **Scanned Documents**: The PDF parser expects selectable text. For scanned PDFs without text layers, run an OCR tool before indexing.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
