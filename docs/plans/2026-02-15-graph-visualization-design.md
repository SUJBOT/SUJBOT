# Interactive Knowledge Graph Visualization — Design

## Goal

Add an interactive knowledge graph visualization to the admin page. Users can filter by document, community, and entity type, search for specific nodes, and hover to see details.

## Constraints

- ~10,000+ entities in the graph — must filter, never render all at once
- Max 500 nodes per visualization (with warning if filter returns more)
- Static layout (no dragging nodes), just zoom/pan and hover
- WebGL rendering required for performance (Sigma.js)

## Backend API

Two new endpoints in `backend/routes/admin.py`:

### `GET /admin/graph/overview`

Returns stats + filter option lists:

```json
{
  "stats": {"entities": 12000, "relationships": 8500, "communities": 45},
  "documents": [{"document_id": "BZ_VR1", "display_name": "BZ VR1", "entity_count": 340}],
  "communities": [{"community_id": 1, "title": "...", "entity_count": 28}],
  "entity_types": [{"type": "REGULATION", "count": 1200}]
}
```

### `GET /admin/graph/data`

Returns filtered nodes + edges. Query params: `document_ids`, `community_ids`, `entity_types`, `limit` (default 500).

```json
{
  "nodes": [{"id": 1, "name": "...", "type": "SECTION", "description": "...", "document_id": "...", "community_id": 3}],
  "edges": [{"source": 1, "target": 5, "type": "REQUIRES", "description": "...", "weight": 1.0}],
  "total_matching": 1200,
  "limited": true
}
```

## Frontend

### Libraries

- `@sigma/react` — React wrapper for Sigma.js (WebGL graph renderer)
- `graphology` — Graph data structure
- `graphology-layout-forceatlas2` — Force-directed layout (web worker)

### Component: `GraphPage.tsx`

New admin page registered in `AdminApp.tsx` as a `<CustomRoutes>` entry.

**Layout:**
- Left sidebar (280px): Filter panel + search
- Main area: Sigma.js canvas

**Filter panel:**
- Multi-select dropdown: Documents (with entity counts)
- Multi-select dropdown: Communities (with titles + entity counts)
- Multi-select dropdown: Entity types (with counts)
- Text search: Live search across node names (client-side, highlights matches)
- "Load Graph" button (fetches data after filter selection)
- Warning banner if `limited: true`

**Graph canvas:**
- Sigma.js with WebGL rendering
- ForceAtlas2 layout computed client-side after data load (~1s for 500 nodes)
- Color coding: By entity type (fixed palette: REGULATION=blue, ORGANIZATION=green, etc.)
- Node size: Proportional to degree (number of connections)
- Hover tooltip: Entity name, type, description, document
- Edge hover tooltip: Relationship type, description, weight
- Zoom/pan via mouse wheel and drag

**Search highlighting:**
- Typing in search field dims non-matching nodes (opacity 0.1)
- Matching nodes stay fully visible and get a highlight ring
- Camera auto-focuses on matching nodes cluster

## i18n Keys

Under `admin.graph`:
- title, subtitle, filterByDocument, filterByCommunity, filterByType,
  search, searchPlaceholder, loadGraph, loading, noData, limitWarning,
  entityTypes (REGULATION, STANDARD, etc.), stats labels

## No Changes Needed

- `docker/nginx/reverse-proxy.conf` — `/admin/` prefix already covers new endpoints
- Graph schema — existing tables serve all needed data
- `GraphStorageAdapter` — existing methods cover all queries
