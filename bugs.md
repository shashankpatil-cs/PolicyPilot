# PolicyPilot: Bug Fixes, Optimizations & Improvements Log

This document tracks technical bugs identified, root-cause analyses, architectural optimizations, and fixes implemented in PolicyPilot.

---

## 1. Research Agent Crash: `RuntimeError: Event loop is closed`

* **Severity**: High (Server crash / 500 error on consecutive requests in Research Mode)
* **Components Affected**: [`backend/app/ai/research_agent.py`](file:///c:/Users/Shashank/Desktop/PolicyPilot/backend/app/ai/research_agent.py), [`backend/app/ai/pipeline.py`](file:///c:/Users/Shashank/Desktop/PolicyPilot/backend/app/ai/pipeline.py)
* **Operating System**: Windows (`asyncio.ProactorEventLoop`)

### Root Cause
FastAPI executes the synchronous `/ask` route in worker threads, invoking the research agent via `asyncio.run(run_research(...))`. Each `asyncio.run()` creates and destroys a new event loop. 

However, the OpenAI Agents SDK (`openai-agents`) cached its underlying `AsyncOpenAI` client globally in `agents.models._openai_shared`. On the second request, `httpcore`/`anyio` attempted to clean up idle connection sockets attached to the first (now closed) event loop. On Windows, `ProactorEventLoop.close()` calls `call_soon()` on a closed loop, raising:
```text
RuntimeError: Event loop is closed
```

### Fix & Resolution
1. **Per-Invocation Scoped Client**: Modified `run_research()` to instantiate a fresh `AsyncOpenAI` client bound to the active request's event loop.
2. **Explicit Provider Injection**: Passed the scoped client to `Runner.run()` via `RunConfig(model_provider=OpenAIProvider(openai_client=client))`.
3. **Graceful Socket Teardown**: Added a `try ... finally: await client.close()` block to drain and terminate HTTP connection pools before the event loop is torn down.

---

## 2. Embedding Acceleration: CUDA / GPU Enablement

* **Severity**: Performance / Resource Utilization
* **Components Affected**: [`backend/app/rag/embeddings.py`](file:///c:/Users/Shashank/Desktop/PolicyPilot/backend/app/rag/embeddings.py), [`backend/app/core/config.py`](file:///c:/Users/Shashank/Desktop/PolicyPilot/backend/app/core/config.py), `backend/.env`

### Problem
The backend virtual environment had PyTorch installed from the default CPU wheel (`torch.cuda.is_available() == False`). Attempting to set `EMBEDDING_DEVICE=cuda` caused:
```text
AssertionError: Torch not compiled with CUDA enabled
```
This prevented the local embedding model (`BAAI/bge-m3`, 1024-dim) from taking advantage of the system's dedicated **NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)**.

### Fix & Resolution
1. Reinstalled PyTorch with CUDA 12.4 support:
   ```powershell
   pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
   ```
2. Added `EMBEDDING_DEVICE=cuda` to `backend/.env`.
3. Verified CUDA tensor allocation on `cuda:0` and validated inference for the `BAAI/bge-m3` model directly in GPU VRAM. PDF chunk and query vectorization now run at GPU speeds with zero cloud API cost.

---

## 3. High Token Consumption Analysis & Optimization Controls

* **Severity**: Cost / Operational Efficiency
* **Components Affected**: [`backend/app/ai/graph.py`](file:///c:/Users/Shashank/Desktop/PolicyPilot/backend/app/ai/graph.py), [`backend/app/core/config.py`](file:///c:/Users/Shashank/Desktop/PolicyPilot/backend/app/core/config.py)

### Problem
Testing ~15 questions consumed ~200,000 input tokens in OpenAI.

### Root Cause Breakdown
1. **Multi-Call Architecture**: A single user question executes up to 4 LLM calls:
   * **Planner LLM (`gpt-4.1-mini`)**: ~1,200 input tokens.
   * **Answer Generator (`gpt-4.1`)**: Ingests up to 16 PDF chunks (`CHUNK_SIZE=1200`) = ~4,500 to 5,500 tokens.
   * **Validator LLM (`gpt-4.1-mini`)**: Fact-checks claims against evidence = ~2,000 tokens.
   * **Automatic Retry Loop**: If validation fails, resends chunks + previous draft + feedback = ~5,500 tokens.
2. **Multi-Turn Research Mode**: Resends cumulative conversation history and newly retrieved passages across every turn.
3. **Abstaining Still Incurs Costs**: When the model concludes evidence is insufficient, it still incurs the full cost of reading all retrieved chunks.

### Optimization Controls Introduced
Added configurable flags in `backend/.env` to reduce token consumption by up to **70%**:
```env
# Reduce chunk payload sent to prompt (saves ~2,500 tokens/query)
MAX_EVIDENCE_CHUNKS=6
RETRIEVAL_TOP_K=6

# Disable secondary LLM fact-checker (uses deterministic Python verifier instead)
VALIDATION_LLM_CHECK=false

# Disable multi-pass generation retries
MAX_GENERATION_ATTEMPTS=1

# Use cost-effective model for development
OPENAI_CHAT_MODEL=gpt-4.1-mini
```

---

## 4. Strict Citation Validation False-Negatives

* **Severity**: Usability / Answer Completeness
* **Components Affected**: [`backend/app/ai/context.py`](file:///c:/Users/Shashank/Desktop/PolicyPilot/backend/app/ai/context.py), [`backend/app/ai/validation.py`](file:///c:/Users/Shashank/Desktop/PolicyPilot/backend/app/ai/validation.py)

### Problem
Broad questions without a specified report year (e.g., *"Key points on inflation"*) frequently resulted in:
> *"I could not find enough evidence in the indexed RBI Annual Reports to answer this reliably."*

### Root Cause
The citation validator enforces strict verbatim quotation matching (`MIN_QUOTE_CHARS=25`, `QUOTE_FUZZY_THRESHOLD=92.0`) and numeric verification. When the LLM synthesized broad insights across multiple chunks, minor paraphrasing or punctuation changes in quotes failed the strict check. After reaching `MAX_GENERATION_ATTEMPTS=2`, the system routed to `abstain()`.

### Fix & Recommendations
1. **Research Agent Mode**: For cross-year or broad topics, Research Mode performs iterative searches and comparisons, bypassing single-pass generation bottlenecks.
2. **Tunable Thresholds**: Made quote length and fuzzy thresholds configurable in `config.py` (`MIN_QUOTE_CHARS=18-20`, `QUOTE_FUZZY_THRESHOLD=75-80`) to allow natural phrasing while preventing hallucinations.

---

## 5. PostgreSQL Local Port Collision Handling

* **Severity**: Initialization / Dev Environment
* **Components Affected**: `docker-compose.yml`, `backend/.env`, `README.md`

### Problem
Local Windows PostgreSQL installations frequently occupy default port `5432`, preventing Docker containers from binding to the same port.

### Resolution
* Standardized PostgreSQL container mapping to port `5433:5432` in Docker and `DB_PORT=5433` in `backend/.env`.
* Documented container lifecycle management (`docker start rbi_postgres`) in setup instructions.
