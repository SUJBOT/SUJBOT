# SUJBOT - Production RAG System for Legal/Technical Documents

Vision-Language RAG system optimized for legal and technical documentation with autonomous agent framework, knowledge graph, and compliance checking.

## Overview

SUJBOT uses a **VL (Vision-Language) architecture** that processes document pages as images, enabling multimodal understanding via VL-capable LLMs.

### Key Features

- **VL Architecture**: Jina v4 embeddings (2048-dim) + page images + multimodal LLM
- **Knowledge Graph (Graph RAG)**: PostgreSQL-based entity/relationship graph with Leiden communities
- **Compliance Checking**: Community-based regulatory compliance assessment
- **Autonomous Agent**: LLM-driven tool loop with 9 specialized tools
- **State persistence** with PostgreSQL checkpointing
- **Full observability** with LangSmith integration
---

## Quick Start

### Prerequisites

- Python 3.10+
- `uv` package manager ([installation](https://docs.astral.sh/uv/))
- API keys: `ANTHROPIC_API_KEY` and optionally `OPENAI_API_KEY`

### Installation

**macOS/Linux:**
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Configure
cp .env.example .env
# Edit .env with your API keys
```

**Windows:**
```bash
# Install uv
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# IMPORTANT: Install PyTorch FIRST (prevents DLL errors)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
uv sync

# Configure (use cloud embeddings for Windows)
copy .env.example .env
# Edit .env and set EMBEDDING_MODEL=text-embedding-3-large
```

**API Keys (.env):**
```bash
ANTHROPIC_API_KEY=sk-ant-...  # Required
OPENAI_API_KEY=sk-...         # Optional (for OpenAI embeddings)
LLM_MODEL=gpt-4o-mini         # For summaries & agent
EMBEDDING_MODEL=text-embedding-3-large  # Windows
# EMBEDDING_MODEL=bge-m3      # macOS M1/M2/M3 (local, FREE, GPU-accelerated)
```

**For detailed platform-specific instructions, see [INSTALL.md](INSTALL.md).**

---

## 🌐 Web Interface (Recommended)

**Production-ready web UI with real-time agent progress visualization:**

```bash
# Start full stack (PostgreSQL + Backend + Frontend)
docker compose up -d

# OR use convenience script
./start_web.sh

# Access UI
# Frontend: http://localhost:5173
# Backend API: http://localhost:8000/docs
```

**Features:**
- 🔐 **JWT authentication** with Argon2 password hashing
- 💬 **Real-time chat** with agent progress visualization
- 📊 **Cost tracking** per query with agent breakdown
- 🔍 **Tool execution** display (inline)
- 💾 **Persistent conversations** (PostgreSQL)
- 🎨 **Dark/light theme** with smooth transitions

**Default credentials:**
```
Email: admin@sujbot.local
Password: ChangeThisPassword123!
```

**⚠️ IMPORTANT:** Change default password immediately in production!
```bash
# Reset admin password
docker compose exec backend uv run python scripts/reset_admin_password.py
```

**⚠️ SECURITY:** Never commit `config.json` or `.env` files to git!
```bash
# First-time setup (creates config.json from template)
cp config.json.example config.json
# Edit config.json with your settings (API keys, database passwords)

# Verify config.json is in .gitignore
git check-ignore config.json  # Should print: config.json
```

**Full documentation:** [docs/WEB_INTERFACE.md](docs/WEB_INTERFACE.md)

---
## 🔒 Security Features

SUJBOT implements production-grade security following OWASP best practices:

### Authentication & Authorization

**JWT-based Authentication:**
- ✅ Argon2id password hashing (PHC winner, GPU-resistant)
- ✅ httpOnly cookies for token storage (XSS protection)
- ✅ 24-hour token expiry with secure key rotation
- ✅ Admin-only user registration (prevents unauthorized signups)

**Password Requirements (OWASP-compliant):**
- Minimum 8 characters
- At least one uppercase letter
- At least one lowercase letter  
- At least one digit
- At least one special character (@$!%*?&)
- Not in common password blacklist (25 most common passwords)
- No consecutive identical characters (e.g., "aaa", "111")

### Network Security

**HTTP Security Headers:**
- ✅ Content-Security-Policy (XSS protection)
- ✅ X-Frame-Options: DENY (clickjacking protection)
- ✅ X-Content-Type-Options: nosniff (MIME sniffing protection)
- ✅ Strict-Transport-Security (HTTPS enforcement in production)
- ✅ Referrer-Policy (information leakage prevention)
- ✅ Permissions-Policy (disable camera, geolocation, etc.)

**Rate Limiting:**
- ✅ Token bucket algorithm per IP address
- ✅ Login endpoint: 10 requests/minute (brute force protection)
- ✅ Registration endpoint: 5 requests/minute (spam prevention)
- ✅ Default: 60 requests/minute for other endpoints

**CORS Configuration:**
- ✅ Explicit origin allow-list (no wildcards)
- ✅ Restricted HTTP methods and headers
- ✅ Credentials support for cookie-based auth

### Data Protection

**SQL Injection Prevention:**
- ✅ Parameterized queries throughout (asyncpg with $1, $2 placeholders)
- ✅ No string concatenation in SQL statements

**Input Validation:**
- ✅ Pydantic models for all API requests
- ✅ Email format validation
- ✅ Message length limits (50K characters)
- ✅ Conversation title length limits (500 characters)

### Production Deployment Checklist

**Before deploying to production:**

1. **Change Default Credentials**
   ```bash
   # Default admin account
   Email: admin@sujbot.local
   Password: ChangeThisPassword123!
   
   # Reset password immediately
   docker compose exec backend uv run python scripts/reset_admin_password.py
   ```

2. **Generate Secure Keys**
   ```bash
   # AUTH_SECRET_KEY (64 bytes)
   openssl rand -base64 64
   
   # POSTGRES_PASSWORD (32 bytes)
   openssl rand -base64 32
   ```

3. **Set Environment Variables**
   ```bash
   # Edit .env file
   AUTH_SECRET_KEY=<generated-key>
   POSTGRES_PASSWORD=<strong-password>
   BUILD_TARGET=production  # Enables HSTS and other production security
   ```

4. **Enable HTTPS**
   - Configure reverse proxy (Nginx/Caddy) with TLS certificates
   - Let's Encrypt recommended for automatic certificate management
   - Update VITE_API_BASE_URL to use https://

5. **Database Security**
   ```bash
   # ✅ PostgreSQL port NOT exposed by default (secure by design)
   # docker-compose.yml: No port mapping in production
   # docker-compose.override.yml: Port 5432 exposed ONLY in development

   # Restrict PostgreSQL access
   # Edit postgresql.conf:
   listen_addresses = 'localhost'

   # Use strong password for postgres user
   # Generate with: openssl rand -base64 32
   ```

6. **Review Security Logs**
   ```bash
   # Monitor failed login attempts
   docker compose logs backend | grep "Failed login"
   
   # Check rate limit violations
   docker compose logs backend | grep "Rate limit exceeded"
   ```

### Security Considerations

**What's Protected:**
- ✅ User registration (admin-only)
- ✅ Password strength (OWASP requirements)
- ✅ Brute force attacks (rate limiting)
- ✅ XSS attacks (CSP headers + httpOnly cookies)
- ✅ Clickjacking (X-Frame-Options)
- ✅ SQL injection (parameterized queries)
- ✅ CSRF (SameSite=Lax cookies)

**Known Limitations:**
- ⚠️ No multi-factor authentication (planned for future release)
- ⚠️ No token refresh mechanism (tokens expire after 24h)
- ⚠️ In-memory HITL storage (use Redis for multi-instance deployments)
- ⚠️ No audit log for admin actions (planned for future release)

**Reporting Security Issues:**
- Please report security vulnerabilities to the project maintainers privately
- Do not create public GitHub issues for security vulnerabilities



---


## Usage

### 1. Upload Documents

Documents are uploaded via the web UI (`POST /documents/upload`). The upload pipeline:
1. Converts PDF pages to images (stored in `data/vl_pages/`)
2. Embeds pages with Jina v4 (stored in `vectors.vl_pages`)
3. Extracts entities/relationships for the knowledge graph

### 2. Run RAG Agent

```bash
# Launch interactive agent (14 tools)
uv run python -m src.agent.cli

# With specific vector store
uv run python -m src.agent.cli --vector-store output/my_doc/phase4_vector_store

# Debug mode
uv run python -m src.agent.cli --debug
```

**Agent Commands:**
- `/help` - Show available commands and tools
- `/stats` - Show tool usage, conversation stats, session costs
- `/config` - Show current configuration
- `/clear` - Clear conversation history
- `/exit` - Exit agent

**Example Session:**
```
🤖 RAG Agent CLI (14 tools, Claude SDK)
📚 Loaded vector store: output/safety_manual/phase4_vector_store
💰 Session cost: $0.0000 (0 tokens)

You: What are the safety procedures for reactor shutdown?

Agent: [Uses 3 tools: document_search → section_search → extract_text]
Based on the safety manual, reactor shutdown follows these procedures:

1. **Normal Shutdown** (Section 4.2):
   - Reduce power to 50% over 30 minutes
   - Insert control rods gradually...

[Citations: Section 4.2, Page 45-47]

💰 Session cost: $0.0234 (12,450 tokens) | 📦 Cache: 8,500 tokens read (90% saved)
```

### 3. Run Tests

```bash
# All tests
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=src --cov-report=html
```

---

## Architecture

### VL Pipeline Flow

```
Document (PDF)
    ↓
[Upload] VL Indexing Pipeline
    ├─ Convert PDF pages to PNG images
    ├─ Embed pages with Jina v4 (2048-dim)
    ├─ Store in PostgreSQL (vectors.vl_pages)
    └─ Extract entities/relationships (Graph RAG)
    ↓
[Query] Autonomous Agent (SingleAgentRunner)
    ├─ Jina v4 cosine similarity search
    ├─ Top-k page images → multimodal LLM
    ├─ Graph RAG for cross-document reasoning
    └─ Compliance checking via knowledge graph
```

### Agent Tools

**RAG Tools:**
- `search` - VL page search (Jina v4 cosine similarity)
- `expand_context` - Context window expansion
- `get_document_info` - Document metadata and summaries
- `get_document_list` - List indexed documents
- `get_stats` - Retrieval statistics

**Graph RAG Tools:**
- `graph_search` - Semantic entity search
- `graph_context` - N-hop relationship traversal
- `graph_communities` - Semantic community search
- `compliance_check` - Community-based compliance assessment

---

## Research Foundation

Based on **LegalBench-RAG** (Pipitone & Alami, 2024) and **Contextual Retrieval** (Anthropic, 2024).

---

## Configuration

### Key .env Variables

```bash
# API Keys
ANTHROPIC_API_KEY=sk-ant-...
JINA_API_KEY=jnswk_...

# LangSmith
LANGSMITH_API_KEY=lsv2_pt_xxx
LANGSMITH_PROJECT_NAME=sujbot-multi-agent
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com

# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/sujbot
```

Settings are in `config.json` (version-controlled). See `CLAUDE.md` for details.

---

## Project Structure

```
SUJBOT/
├── src/
│   ├── single_agent/                  # Production runner (autonomous tool loop)
│   ├── agent/                         # Agent CLI and tools
│   │   ├── cli.py                     # Interactive CLI
│   │   ├── config.py                  # Agent configuration
│   │   ├── providers/                 # LLM provider implementations
│   │   └── tools/                     # 9 specialized tools
│   ├── graph/                         # Graph RAG (knowledge graph)
│   │   ├── storage.py                 # PostgreSQL graph CRUD
│   │   ├── embedder.py                # multilingual-e5-small embeddings
│   │   ├── entity_extractor.py        # Multimodal entity extraction
│   │   └── community_detector.py      # Leiden community detection
│   ├── vl/                            # Vision-Language RAG module
│   │   ├── embedder.py                # Jina v4 embeddings
│   │   ├── page_store.py              # Page image storage
│   │   └── retriever.py               # VL retrieval
│   └── storage/                       # PostgreSQL vector storage
├── backend/                           # FastAPI web backend
├── frontend/                          # React + Vite web UI
├── tests/                             # Test suite
├── data/                              # Input documents + page images
├── docs/                              # Documentation
├── CLAUDE.md                          # Development guidelines
└── .env.example                       # Environment template
```

---

## 📖 Documentation

### Core Documentation

- **[INSTALL.md](INSTALL.md)** - Platform-specific installation (Windows/macOS/Linux)
- **[CLAUDE.md](CLAUDE.md)** - Development guidelines and project instructions
- **[PIPELINE.md](PIPELINE.md)** - Complete pipeline specification with research

### User Guides

- **[Agent CLI Guide](docs/agent/README.md)** - RAG Agent CLI documentation
- **[macOS Quick Start](docs/how-to-run-macos.md)** - Quick start for macOS users
- **[Vector DB Management](docs/vector-db-management.md)** - Central database tools

### Advanced Topics

- **[Cost Tracking](docs/cost-tracking.md)** - API cost monitoring and optimization
- **[Cost Optimization](docs/development/cost-optimization.md)** - Detailed cost analysis
- **[Batching Optimizations](docs/development/batching-optimizations.md)** - Performance guide

---

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Test agent tools
uv run pytest tests/agent/ -v

# With coverage
uv run pytest tests/ --cov=src --cov-report=html

# Single test
uv run pytest tests/agent/test_validation.py::test_api_key_validation -v
```

---

## Performance Tips

**Prompt Caching (Anthropic only):**
- 90% cost reduction on cached tokens
- Enabled by default for Anthropic models

---

## 🌍 Platform Support

**Tested Platforms:**
- macOS (Apple Silicon M1/M2/M3) - Recommended for local embeddings
- Linux (Ubuntu 20.04+) - Production deployment
- Windows 10/11 - Cloud embeddings recommended

**Embedding Model Selection:**
- **Windows:** `text-embedding-3-large` (cloud) - avoids PyTorch DLL issues
- **macOS M1/M2/M3:** `bge-m3` (local, FREE, GPU-accelerated)
- **Linux GPU:** `bge-m3` (local)
- **Linux CPU:** `text-embedding-3-large` (cloud)

---

## ⚠️ Requirements

- **Python:** >=3.10
- **uv:** Latest version (package manager)
- **Memory:** 8GB+ recommended
- **API Keys:** ANTHROPIC_API_KEY (required), OPENAI_API_KEY (optional)
- **GPU:** Optional (for local embeddings on macOS/Linux)

---

## Acknowledgments

Based on research from:
- Pipitone & Alami (LegalBench-RAG, 2024)
- Anthropic (Contextual Retrieval, 2024)

---

## 📄 License

MIT License

---

**Architecture:** VL-only (Vision-Language)
**Last Updated:** 2026-02-15
