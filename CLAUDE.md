# CLAUDE.md - Vývojář: michal

Tento soubor poskytuje instrukce pro Claude Code při práci s tímto repozitářem.

---

## Tvoje prostředí

**Vývojář:** michal  
**Repozitář:** /home/michal/SUJBOT  
**Git větev:** dev/michal  
**Aplikace:** http://localhost:8280

### Tvoje porty

| Služba | Port | URL |
|--------|------|-----|
| Nginx (aplikace) | 8280 | http://localhost:8280 |
| Backend API | 8200 | http://localhost:8200 |
| Frontend dev | 5175 | http://localhost:5175 |
| PgAdmin | 5052 | http://localhost:5052 |

---

## Git Workflow

### Denní rutina

```bash
# 1. Aktualizuj svou větev z main
git fetch origin
git merge origin/main

# 2. Pracuj na změnách
# ... editace souborů ...

# 3. Commit změn
git add .
git commit -m "feat: popis změny"

# 4. Push do remote
git push origin dev/michal

# 5. Vytvoř Pull Request na GitHubu
# https://github.com/SUJBOT/SUJBOT/compare/main...dev/michal
```

### Pravidla pro commity

- `feat:` - nová funkce
- `fix:` - oprava bugu
- `docs:` - dokumentace
- `refactor:` - refaktoring kódu
- `test:` - testy
- `chore:` - údržba

### NIKDY nedělej

- `git push --force` na main
- Přímý push do main (vždy přes PR)
- Commit citlivých dat (.env, API klíče)

---

## Development

### Spuštění aplikace

```bash
cd ~/SUJBOT
docker compose up -d          # Spustit všechny služby
docker compose logs -f backend # Sledovat logy backendu
```

### Zastavení aplikace

```bash
docker compose down           # Zastavit služby
docker compose down -v        # Zastavit + smazat volumes (POZOR!)
```

### Testování

```bash
docker compose exec backend uv run pytest tests/ -v
```

### Hot reload

- **Frontend:** Automatický (Vite) - změny v .tsx/.css se projeví okamžitě
- **Backend:** Automatický (uvicorn --reload)

---

## Sdílená data

| Adresář | Umístění | Popis |
|---------|----------|-------|
| data/ | /opt/sujbot-shared/data | PDF dokumenty (sdílené) |
| output/ | /opt/sujbot-shared/output | Výstupy pipeline (sdílené) |
| .env | ~/SUJBOT/.env | API klíče (kopie ze sdíleného) |

**POZOR:** data/ a output/ jsou symlinky - změny vidí všichni!

---

## Struktura projektu

```
src/
├── agent/          # Agent CLI a nástroje
├── multi_agent/    # LangGraph multi-agent systém
├── retrieval/      # HyDE + Expansion Fusion retrieval
└── graph/          # Knowledge graph (Graphiti)

backend/            # FastAPI web backend
frontend/           # React + Vite UI
docker/             # Docker konfigurace
prompts/            # Systémové prompty (SSOT!)
```

---

## Důležitá pravidla

1. **SSOT (Single Source of Truth)** - jedna implementace na funkci
2. **Autonomní agenti** - LLM rozhoduje, ne hardcoded workflow
3. **Prompty v souborech** - nikdy inline v kódu, vždy v prompts/
4. **Chunking 512 tokenů** - neměnit bez konzultace
5. **API klíče pouze v .env** - nikdy v kódu nebo config.json

---

## Ostatní vývojáři

| Vývojář | Port | Větev |
|---------|------|-------|
| francji1 | 8180 | dev/francji1 |
| michal | 8280 | dev/michal |
| matyas | 8380 | dev/matyas |
| vendula | 8480 | dev/vendula |
| **Produkce** | 80 | main |

---

## Potřebuješ pomoc?

- **Admin:** prusemic
- **Repozitář:** https://github.com/SUJBOT/SUJBOT
- **Dokumentace:** docs/ adresář

