# Deployment and Operations

## Production Stack

The production system runs as Docker containers across two networks:

```
                      Internet
                         |
                      [Nginx]  (SSL termination, port 443)
                      /      \
               [Frontend]   [Backend]  (sujbot_prod_net)
                              |
                         [PostgreSQL]   [Redis]  (sujbot_db_net)
```

### Services

| Service | Container | Image | Port | Purpose |
|---------|-----------|-------|------|---------|
| PostgreSQL | `sujbot_postgres` | Custom (pgvector + pg_trgm) | 5432 | Vector storage, auth, graph |
| Redis | `sujbot_redis` | `redis:7-alpine` | 6379 | Indexing pipeline state |
| Backend | `sujbot_backend` | `sujbot-backend` | 8000 | FastAPI API server |
| Frontend | `sujbot_frontend` | `sujbot-frontend` | 80 | React SPA (nginx) |
| Nginx | `sujbot_nginx` | `nginx:alpine` | 443 | Reverse proxy, SSL |
| Certbot | `sujbot_certbot` | `certbot/certbot` | — | SSL certificate renewal |
| pgAdmin | `sujbot_pgadmin` | `dpage/pgadmin4` | 127.0.0.1:5050 | DB admin (localhost only) |

### Docker Networks

| Network | Subnet | Purpose | Members |
|---------|--------|---------|---------|
| `sujbot_prod_net` | — | Application traffic | nginx, backend, frontend |
| `sujbot_db_net` | 172.20.0.0/16 | Database access | postgres, redis, backend, pgadmin |

The backend must be connected to BOTH networks (prod_net for nginx, db_net for postgres).

## Building Images

### Frontend

The frontend MUST be built with an empty `VITE_API_BASE_URL` for production. This ensures it uses relative URLs (`/api/...`) instead of hardcoded `localhost`.

```bash
docker build -t sujbot-frontend --target production \
  --build-arg VITE_API_BASE_URL="" \
  -f docker/frontend/Dockerfile frontend/
```

### Backend

```bash
docker build -t sujbot-backend -f docker/backend/Dockerfile .
```

### PostgreSQL (custom with pgvector)

```bash
docker compose build postgres
```

## Deploying Containers

### Frontend Deploy

```bash
docker stop sujbot_frontend && docker rm sujbot_frontend
docker run -d --name sujbot_frontend \
  --network sujbot_sujbot_prod_net \
  --network-alias frontend \
  --restart unless-stopped \
  sujbot-frontend
```

- `--network-alias frontend` is REQUIRED — nginx resolves the container by this DNS name

### Backend Deploy (full recreation)

```bash
docker stop sujbot_backend && docker rm sujbot_backend
docker run -d --name sujbot_backend \
  --network sujbot_sujbot_prod_net --network-alias backend \
  --env-file /home/prusemic/SUJBOT/.env \
  -e DATABASE_URL=postgresql://postgres:sujbot_secure_password@sujbot_postgres:5432/sujbot \
  -v $(pwd)/config.json:/app/config.json:ro \
  -v $(pwd)/prompts:/app/prompts:ro \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/data/vl_pages:/app/data/vl_pages \
  -v $(pwd)/vector_db:/app/vector_db:ro \
  --restart unless-stopped sujbot-backend
docker network connect sujbot_sujbot_db_net sujbot_backend
```

Critical notes:
- `--env-file .env` is required (API keys, secrets)
- `-e DATABASE_URL=...@sujbot_postgres:5432/...` overrides the dev port in `.env`
- `/app/data` must be writable (no `:ro`) for document uploads
- `/app/vector_db:ro` is required even if empty (validation check)
- Must connect to BOTH `sujbot_prod_net` AND `sujbot_db_net`

### Database Services

Database services (postgres, redis, pgadmin) are managed via docker-compose:

```bash
# Start shared database services
docker compose up -d

# View logs
docker compose logs -f postgres

# Stop all
docker compose down
```

## SSL/TLS

Certificates are managed by Let's Encrypt via Certbot:

- Certificate path: `/etc/letsencrypt/live/sujbot.fjfi.cvut.cz/`
- Protocols: TLSv1.2, TLSv1.3
- HSTS: 1 year with `includeSubDomains`
- HTTP (port 80) redirects to HTTPS (port 443)
- ACME challenge path: `/.well-known/acme-challenge/`

## Nginx Routing

Nginx (`docker/nginx/reverse-proxy.conf`) routes requests as follows:

| Path Pattern | Destination | Notes |
|-------------|-------------|-------|
| `/api/*` | Backend (strip `/api/` prefix) | SSE-enabled, 300s timeout |
| `/health`, `/chat`, `/auth/*`, `/conversations/*`, etc. | Backend (direct) | Regex match on route prefixes |
| `/admin/login` | GET → Frontend, POST → Backend | Method-based routing |
| `/admin/users/*`, `/admin/documents/*` | `Accept: application/json` → Backend, else → Frontend | Content negotiation |
| `/admin/health`, `/admin/stats` | Backend (always) | Health/metrics endpoints |
| `/*` (fallback) | Frontend | React SPA with WebSocket upgrade |

New backend routes MUST either:
1. Be added to the nginx regex pattern in the "Direct backend endpoints" section, OR
2. Use the `/api/` prefix (auto-proxied, no nginx changes needed)

## Development Setup

### Local Development

```bash
# Install Python dependencies
uv sync

# Start database services
docker compose up -d

# Run backend locally
uv run uvicorn backend.main:app --reload --port 8000

# Run frontend dev server (separate terminal)
cd frontend && npm run dev
```

### Ports

| Port | Service | Environment |
|------|---------|-------------|
| 443 | Nginx (HTTPS) | Production |
| 8000 | Backend (FastAPI) | Both |
| 5173 | Frontend (Vite dev) | Development |
| 5432 | PostgreSQL | Production |
| 5433 | PostgreSQL | Development (often not running) |
| 6379 | Redis | Both |
| 5050 | pgAdmin | Both (localhost only) |

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| 404 on `/admin` | Missing SPA fallback in frontend nginx | Rebuild frontend image |
| 502 Bad Gateway | Backend not on `sujbot_prod_net` or missing `--network-alias` | Re-deploy backend with both networks |
| `localhost:8000` in browser requests | Frontend built with wrong `VITE_API_BASE_URL` | Rebuild with `--build-arg VITE_API_BASE_URL=""` |
| Frontend gets HTML instead of JSON | Backend route not in nginx config | Add route to nginx regex or use `/api/` prefix |
| Database connection refused | Wrong port (5432 prod vs 5433 dev) | Check `DATABASE_URL` in `.env` |
| Backend startup crash | Missing `--env-file .env` or volume mounts | Re-deploy with all required flags |
| LangSmith 403 | Wrong endpoint (EU vs US) | Set `LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com` |

## Database Management

### Backups

```bash
# Dump full database
docker exec sujbot_postgres pg_dump -U postgres sujbot > backup.sql

# Dump specific schema
docker exec sujbot_postgres pg_dump -U postgres -n vectors sujbot > vectors_backup.sql

# Restore
docker exec -i sujbot_postgres psql -U postgres sujbot < backup.sql
```

### Schema Initialization

The PostgreSQL container runs init scripts from `docker/postgres/init/` on first startup. These create the `vectors`, `auth`, and `graph` schemas with all required tables and indexes.

### Resource Limits

| Service | CPU | Memory |
|---------|-----|--------|
| PostgreSQL | 2 cores | 3 GB |
| Redis | 0.5 cores | 256 MB |
| pgAdmin | 0.5 cores | 512 MB |

## Monitoring

### Health Checks

- Backend: `GET /health` — returns service status and database connectivity
- PostgreSQL: `pg_isready` (5s interval)
- Redis: `redis-cli ping` (10s interval)

### Logs

```bash
# Backend logs
docker logs -f sujbot_backend

# All service logs
docker compose logs -f

# Nginx access logs
docker exec sujbot_nginx cat /var/log/nginx/access.log
```

### LangSmith Observability

All agent interactions are traced in LangSmith (see [CONFIGURATION.md](CONFIGURATION.md) for setup). View traces at the LangSmith dashboard to debug agent behavior, tool usage, and costs.

### Security Monitoring

The system includes automated security monitoring configured in `config.json` → `security_monitoring`. It checks container health, analyzes logs for suspicious patterns, monitors disk usage, and validates SSL certificates. Notifications can be sent via Microsoft Teams webhooks.
