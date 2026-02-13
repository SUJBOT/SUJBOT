"""
GraphRAG-style hierarchical topic clustering of VL page embeddings.

Pipeline:
  1. Fetch 2048-dim Jina v4 embeddings from vectors.vl_pages
  2. Build k-NN similarity graph (cosine, weighted edges)
  3. Hierarchical Leiden community detection (GraphRAG approach):
     Level 0: Leiden on page-level k-NN graph (high resolution → small communities)
     Level 1+: Aggregate inter-community edges → community graph → Leiden again
     Soft overlap via cross-community edge weight analysis
  4. Multimodal topic labeling: page images → Claude Sonnet 4.5
  5. UMAP 2D + interactive Plotly visualizations
  6. JSON export with full tree structure

Usage:
    uv run python scripts/topic_clustering.py
"""

import base64
import json
import time
import warnings
from collections import defaultdict
from pathlib import Path

import anthropic
import igraph as ig
import networkx as nx
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import psycopg
import umap

from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)

# ── Config ────────────────────────────────────────────────────────
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "sujbot",
    "user": "postgres",
    "password": "sujbot_secure_password",
}

SHORT_NAMES = {
    "BZ_VR1": "BZ VR1",
    "Sb_2016_263_2024-01-01_IZ": "Sb 263/2016",
    "Sb_2016_151_Castka_OZ": "Sb 151/2016 OZ",
    "zakonyprolidi_cs_2017_021_v20260201": "Zákon 21/2017",
    "Sb_2025_157_PZZ": "Sb 157/2025",
    "Sb_1997_18_2017-01-01_IZ": "Sb 18/1997",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Data fetching ─────────────────────────────────────────────────
def fetch_embeddings():
    """Fetch all page embeddings + metadata from vectors.vl_pages."""
    conn = psycopg.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        SELECT page_id, document_id, page_number, embedding::text
        FROM vectors.vl_pages
        ORDER BY document_id, page_number
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    page_ids, doc_ids, page_nums, embeddings = [], [], [], []
    for page_id, doc_id, page_num, emb_text in rows:
        vec = np.array([float(x) for x in emb_text.strip("[]").split(",")])
        page_ids.append(page_id)
        doc_ids.append(doc_id)
        page_nums.append(page_num)
        embeddings.append(vec)

    return page_ids, doc_ids, page_nums, np.array(embeddings, dtype=np.float32)


# ── GraphRAG Leiden clustering ────────────────────────────────────
def leiden_hierarchy(
    embeddings, k=15, sim_threshold=0.3,
    max_levels=3, base_resolution=3.0, overlap_threshold=0.1,
):
    """GraphRAG-style hierarchical Leiden community detection on k-NN graph.

    Level 0: Leiden on page-level k-NN similarity graph (high resolution → small
             communities). Soft overlap: nodes assigned to additional communities
             where cross-community edge weight > overlap_threshold of total weight.
    Level 1+: Aggregate inter-community edge weights → community graph → Leiden
             at decreasing resolution → broader themes.

    Returns list of level dicts (same structure as before):
      - level, communities, member_pages, centroids, labels, descriptions
    """
    n = len(embeddings)

    # Build page-level k-NN igraph
    print(f"  Building k-NN igraph (k={k}, threshold={sim_threshold})...")
    sim_matrix = embeddings @ embeddings.T
    edges, weights = [], []
    for i in range(n):
        sims = sim_matrix[i].copy()
        sims[i] = -1
        top_k_idx = np.argsort(sims)[::-1][:k]
        for j in top_k_idx:
            if sims[j] > sim_threshold and i < j:
                edges.append((i, j))
                weights.append(float(sims[j]))

    G_pages = ig.Graph(n, edges, directed=False)
    G_pages.es["weight"] = weights
    print(f"  → {G_pages.vcount()} nodes, {G_pages.ecount()} edges")

    tree = []
    current_graph = G_pages
    current_page_sets = [{i} for i in range(n)]

    for level in range(max_levels):
        n_nodes = current_graph.vcount()
        if n_nodes <= 4:
            break

        resolution = base_resolution / (1.5 ** level)
        min_members = 3 if level == 0 else 2

        print(f"\n  Level {level}: Leiden on {n_nodes} nodes "
              f"(resolution={resolution:.2f})...")

        # Run Leiden (hard partition)
        partition = current_graph.community_leiden(
            objective_function="modularity",
            weights="weight",
            resolution=resolution,
            n_iterations=10,
        )

        # Collect hard communities
        hard_comms = {}
        for node, cid in enumerate(partition.membership):
            hard_comms.setdefault(cid, set()).add(node)

        # Soft assignment: add nodes to communities based on
        # cross-community edge weight fraction
        node_to_hard = {}
        for cid, members in hard_comms.items():
            for m in members:
                node_to_hard[m] = cid

        soft_comms = {c: set(members) for c, members in hard_comms.items()}

        for i in range(n_nodes):
            neighbors = current_graph.neighbors(i)
            if not neighbors:
                continue
            comm_w = defaultdict(float)
            total_w = 0.0
            for j in neighbors:
                w = current_graph.es[current_graph.get_eid(i, j)]["weight"]
                total_w += w
                comm_w[node_to_hard.get(j, -1)] += w

            my_comm = node_to_hard.get(i, -1)
            for c, s in comm_w.items():
                if (c != my_comm and c != -1
                        and total_w > 0 and s / total_w > overlap_threshold):
                    soft_comms[c].add(i)

        # Filter small communities and sort by size
        communities = [
            m for m in soft_comms.values() if len(m) >= min_members
        ]
        communities.sort(key=len, reverse=True)

        if len(communities) <= 1:
            print(f"    → only {len(communities)} community, stopping")
            break

        # Resolve original page indices
        member_pages = []
        for members in communities:
            pages = set()
            for m in members:
                pages.update(current_page_sets[m])
            member_pages.append(pages)

        # Centroids (for representative page selection during labeling)
        centroids = []
        for mp in member_pages:
            vecs = embeddings[sorted(mp)]
            c = vecs.mean(axis=0)
            norm = np.linalg.norm(c)
            if norm > 0:
                c /= norm
            centroids.append(c)
        centroids_arr = np.array(centroids, dtype=np.float32)

        tree.append({
            "level": level,
            "n_items": n_nodes,
            "communities": communities,
            "member_pages": member_pages,
            "centroids": centroids_arr,
            "labels": {},
            "descriptions": {},
        })

        sizes = ", ".join(f"{len(mp)}p" for mp in member_pages)
        print(f"    → {len(communities)} communities: [{sizes}]")

        # ── Build community graph for next level ──────────────────
        # Nodes = communities, edge weight = aggregated inter-community
        # edge weights from the current graph (GraphRAG approach).
        n_comms = len(communities)
        node_to_filtered = {}
        for new_cid, members in enumerate(communities):
            for m in members:
                if m not in node_to_filtered:
                    node_to_filtered[m] = new_cid

        comm_edge_weights = defaultdict(float)
        for edge in current_graph.es:
            u, v = edge.source, edge.target
            cu = node_to_filtered.get(u)
            cv = node_to_filtered.get(v)
            if cu is not None and cv is not None and cu != cv:
                key = (min(cu, cv), max(cu, cv))
                comm_edge_weights[key] += edge["weight"]

        if not comm_edge_weights:
            print("    → no inter-community edges, stopping hierarchy")
            break

        comm_edge_list = list(comm_edge_weights.keys())
        comm_weight_list = [comm_edge_weights[e] for e in comm_edge_list]

        current_graph = ig.Graph(n_comms, comm_edge_list, directed=False)
        current_graph.es["weight"] = comm_weight_list
        current_page_sets = member_pages

    return tree


# ── Build node→community membership (for level-0 visualization) ──
def build_membership(communities, n_nodes):
    """Build per-node community membership from community list."""
    node_communities = [[] for _ in range(n_nodes)]
    for cid, members in enumerate(communities):
        for node in members:
            node_communities[node].append(cid)

    primary_label = np.full(n_nodes, -1, dtype=int)
    assignments = []
    for i in range(n_nodes):
        comms = node_communities[i]
        if comms:
            primary_label[i] = comms[0]
            assignments.append([(c, 1.0) for c in comms])
        else:
            assignments.append([])

    return assignments, primary_label


# ── k-NN graph (visualization only) ──────────────────────────────
def build_knn_graph(embeddings, page_ids, k=15, sim_threshold=0.3):
    """Build a weighted k-NN graph from cosine similarity."""
    n = len(page_ids)
    sim_matrix = embeddings @ embeddings.T

    G = nx.Graph()
    for i in range(n):
        G.add_node(i, page_id=page_ids[i])

    for i in range(n):
        sims = sim_matrix[i].copy()
        sims[i] = -1
        top_k = np.argsort(sims)[::-1][:k]
        for j in top_k:
            if sims[j] > sim_threshold and not G.has_edge(i, j):
                G.add_edge(i, j, weight=float(sims[j]))

    return G


# ── UMAP ──────────────────────────────────────────────────────────
def run_umap(embeddings, n_neighbors=15, min_dist=0.1):
    reducer = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
        metric="cosine", random_state=42,
    )
    return reducer.fit_transform(embeddings)


# ── Multimodal LLM labeling ──────────────────────────────────────
def get_image_path(doc_id, page_num):
    """Construct page image path from document_id and page_number."""
    return PROJECT_ROOT / "data" / "vl_pages" / doc_id / f"page_{page_num:03d}.png"


def load_page_image(doc_id, page_num):
    """Load a page image as base64-encoded PNG."""
    path = get_image_path(doc_id, page_num)
    if not path.exists():
        return None
    return base64.standard_b64encode(path.read_bytes()).decode("utf-8")


def label_community_with_llm(cid, page_images, level_hint=""):
    """Call Claude Sonnet 4.5 to label a community from page images."""
    client = anthropic.Anthropic()

    content = [{
        "type": "text",
        "text": (
            "Analyzuj následující stránky z právních/technických dokumentů "
            "(jaderná bezpečnost, radiační ochrana). "
            f"Všechny patří do jedné tématické skupiny.{level_hint}\n\n"
            "Na základě obrázků:\n"
            "1. Urči hlavní téma (max 6 slov česky).\n"
            "2. Napiš stručný popis (1 věta, max 30 slov).\n\n"
            "Odpověz přesně:\nTÉMA: <název>\nPOPIS: <popis>"
        ),
    }]

    for img in page_images:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": img["base64"],
            },
        })
        content.append({"type": "text", "text": f"[{img['label']}]"})

    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=200,
        messages=[{"role": "user", "content": content}],
    )

    text = response.content[0].text.strip()
    topic, desc = "", ""
    for line in text.split("\n"):
        if line.startswith("TÉMA:"):
            topic = line.replace("TÉMA:", "").strip()
        elif line.startswith("POPIS:"):
            desc = line.replace("POPIS:", "").strip()

    return topic, desc, response.usage


def label_tree(tree, embeddings, page_ids, doc_ids, page_nums):
    """Label all communities at all tree levels using multimodal LLM."""
    total_input, total_output = 0, 0

    level_hints = {
        0: " Jde o jemnou tématickou skupinu (detail).",
        1: " Jde o širší tématickou kategorii.",
        2: " Jde o hlavní tématickou oblast (top-level).",
    }

    for level_data in tree:
        level = level_data["level"]
        hint = level_hints.get(level, "")
        n_comms = len(level_data["member_pages"])
        n_images = 5 if level == 0 else 8

        print(f"\n  Labeling level {level} ({n_comms} communities, {n_images} imgs each)...")

        for cid, mp in enumerate(level_data["member_pages"]):
            pages = sorted(mp)
            if not pages:
                level_data["labels"][cid] = f"L{level}·C{cid}"
                level_data["descriptions"][cid] = ""
                continue

            # Representative pages: closest to centroid
            centroid = level_data["centroids"][cid]
            page_embs = embeddings[pages]
            sims = page_embs @ centroid
            top_idx = np.argsort(sims)[::-1][:15]
            top_pages = [pages[i] for i in top_idx]

            # Load page images
            page_images = []
            for pi in top_pages:
                b64 = load_page_image(doc_ids[pi], page_nums[pi])
                if b64:
                    page_images.append({
                        "base64": b64,
                        "label": f"{SHORT_NAMES.get(doc_ids[pi], doc_ids[pi])} p.{page_nums[pi]}",
                    })
                if len(page_images) >= n_images:
                    break

            if not page_images:
                level_data["labels"][cid] = f"L{level}·C{cid}"
                level_data["descriptions"][cid] = ""
                continue

            topic, desc, usage = label_community_with_llm(cid, page_images, hint)
            total_input += usage.input_tokens
            total_output += usage.output_tokens

            level_data["labels"][cid] = topic or f"L{level}·C{cid}"
            level_data["descriptions"][cid] = desc
            print(f"    L{level}·C{cid}: {topic} ({len(mp)} pages)")

            time.sleep(0.3)

    cost = (total_input * 3 / 1_000_000) + (total_output * 15 / 1_000_000)
    print(f"\n  Total: {total_input + total_output:,} tokens (${cost:.4f})")


# ── Page hierarchy mapping ────────────────────────────────────────
def build_page_hierarchy(tree, n_pages):
    """Map each page to its community IDs at each hierarchy level."""
    hierarchy = [{} for _ in range(n_pages)]
    for level_data in tree:
        level = level_data["level"]
        for cid, mp in enumerate(level_data["member_pages"]):
            for p in mp:
                hierarchy[p].setdefault(level, []).append(cid)
    return hierarchy


# ── Visualization ─────────────────────────────────────────────────
def build_community_viz(
    umap_2d, page_ids, doc_ids, page_nums,
    communities, assignments, primary_label,
    tree, title_suffix="",
):
    """Interactive scatter plot colored by level-0 communities, hierarchy in hover."""
    n_comms = len(communities)
    palette = (
        px.colors.qualitative.Set3 + px.colors.qualitative.Pastel1
        + px.colors.qualitative.Bold + px.colors.qualitative.Vivid
    )
    if n_comms > len(palette):
        palette = palette * ((n_comms // len(palette)) + 1)

    page_hier = build_page_hierarchy(tree, len(page_ids))

    fig = go.Figure()

    # Orphan nodes
    orphan_mask = primary_label == -1
    if orphan_mask.any():
        hovers = []
        for i in np.where(orphan_mask)[0]:
            hovers.append(
                f"<b>{page_ids[i]}</b><br>"
                f"Doc: {SHORT_NAMES.get(doc_ids[i], doc_ids[i])}<br>"
                f"Page: {page_nums[i]}<br>No community"
            )
        fig.add_trace(go.Scatter(
            x=umap_2d[orphan_mask, 0], y=umap_2d[orphan_mask, 1],
            mode="markers",
            marker=dict(size=5, color="lightgray", opacity=0.4,
                        line=dict(width=0.5, color="gray")),
            name="Unclustered", hovertext=hovers, hoverinfo="text",
        ))

    # One trace per level-0 community
    for cid in range(n_comms):
        members = sorted(communities[cid])
        if not members:
            continue

        label_0 = tree[0]["labels"].get(cid, f"C{cid}") if tree else f"C{cid}"
        short_label = label_0[:47] + "..." if len(label_0) > 50 else label_0

        xs = [umap_2d[i, 0] for i in members]
        ys = [umap_2d[i, 1] for i in members]
        hovers = []
        for i in members:
            hier_parts = []
            for ld in tree:
                lv = ld["level"]
                comms_at_lv = page_hier[i].get(lv, [])
                if comms_at_lv:
                    lbl = [ld["labels"].get(c, f"C{c}") for c in comms_at_lv]
                    hier_parts.append(f"L{lv}: {', '.join(lbl)}")

            all_l0 = ", ".join(f"C{c[0]}" for c in assignments[i])
            hovers.append(
                f"<b>{page_ids[i]}</b><br>"
                f"Doc: {SHORT_NAMES.get(doc_ids[i], doc_ids[i])}<br>"
                f"Page: {page_nums[i]}<br>"
                f"<b>C{cid}: {label_0}</b><br>"
                f"Communities: {all_l0}<br>"
                + "<br>".join(hier_parts)
            )

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(size=7, color=palette[cid], opacity=0.75,
                        line=dict(width=0.5, color="black")),
            name=f"C{cid}: {short_label} ({len(members)})",
            hovertext=hovers, hoverinfo="text",
        ))

    # Document outlines
    for doc_id in sorted(set(doc_ids)):
        mask = np.array([d == doc_id for d in doc_ids])
        fig.add_trace(go.Scatter(
            x=umap_2d[mask, 0], y=umap_2d[mask, 1], mode="markers",
            marker=dict(size=12, color="rgba(0,0,0,0)",
                        line=dict(width=1.5, color="black"), symbol="circle-open"),
            name=f"Doc: {SHORT_NAMES.get(doc_id, doc_id)}",
            hoverinfo="skip", visible="legendonly",
        ))

    fig.update_layout(
        title=dict(
            text=f"SUJBOT — GraphRAG Hierarchical Topic Clustering{title_suffix}",
            font=dict(size=18)),
        xaxis=dict(title="UMAP 1", showgrid=False, zeroline=False),
        yaxis=dict(title="UMAP 2", showgrid=False, zeroline=False),
        plot_bgcolor="white", width=1200, height=800,
        legend=dict(title="Level-0 Communities",
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="lightgray", borderwidth=1),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    return fig


def build_graph_viz(G, umap_2d, communities, page_ids, doc_ids, tree):
    """Graph structure visualization with k-NN edges."""
    n_comms = len(communities)
    palette = (
        px.colors.qualitative.Set3 + px.colors.qualitative.Pastel1
        + px.colors.qualitative.Bold
    )
    if n_comms > len(palette):
        palette = palette * ((n_comms // len(palette)) + 1)

    node_color = ["lightgray"] * len(page_ids)
    node_comm = [-1] * len(page_ids)
    for cid, members in enumerate(communities):
        for i in members:
            if node_comm[i] == -1:
                node_color[i] = palette[cid]
                node_comm[i] = cid

    fig = go.Figure()

    edges = list(G.edges(data=True))
    strong = [(u, v, d) for u, v, d in edges if d.get("weight", 0) > 0.5]
    edge_x, edge_y = [], []
    for u, v, _ in strong[:3000]:
        edge_x += [umap_2d[u, 0], umap_2d[v, 0], None]
        edge_y += [umap_2d[u, 1], umap_2d[v, 1], None]

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.3, color="rgba(180,180,180,0.3)"),
        hoverinfo="none", name=f"Edges (sim>0.5, {len(strong)})",
    ))

    label_0 = tree[0]["labels"] if tree else {}
    hovers = [
        f"<b>{page_ids[i]}</b><br>"
        f"Doc: {SHORT_NAMES.get(doc_ids[i], doc_ids[i])}<br>"
        f"Degree: {G.degree(i)}<br>"
        f"Community: C{node_comm[i]} — {label_0.get(node_comm[i], '?')}"
        for i in range(len(page_ids))
    ]
    fig.add_trace(go.Scatter(
        x=umap_2d[:, 0], y=umap_2d[:, 1], mode="markers",
        marker=dict(size=6, color=node_color, opacity=0.8,
                    line=dict(width=0.3, color="black")),
        hovertext=hovers, hoverinfo="text", name="Pages",
    ))

    fig.update_layout(
        title="k-NN Similarity Graph — Leiden Community Colors",
        xaxis=dict(title="UMAP 1", showgrid=False, zeroline=False),
        yaxis=dict(title="UMAP 2", showgrid=False, zeroline=False),
        plot_bgcolor="white", width=1200, height=800,
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    return fig


def build_interactive_hierarchy_viz(
    umap_2d, page_ids, doc_ids, page_nums, tree, G_nx,
):
    """Interactive Plotly viz with buttons to switch between hierarchy levels.

    - Buttons at top toggle Level 0 / Level 1 / Level 2 community coloring
    - Graph edges toggleable via legend
    - Hover text shows full hierarchy path for every page
    """
    n = len(page_ids)
    page_hier = build_page_hierarchy(tree, n)

    fig = go.Figure()
    level_trace_ranges = {}  # level -> (start_idx, end_idx)

    # ── Trace 0: Graph edges (legendonly — user toggles on) ───────
    edges = list(G_nx.edges(data=True))
    strong = [(u, v, d) for u, v, d in edges if d.get("weight", 0) > 0.5]
    edge_x, edge_y = [], []
    for u, v, _ in strong[:3000]:
        edge_x += [umap_2d[u, 0], umap_2d[v, 0], None]
        edge_y += [umap_2d[u, 1], umap_2d[v, 1], None]

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.2, color="rgba(150,150,150,0.15)"),
        hoverinfo="none", name=f"k-NN edges ({len(strong)})",
        visible="legendonly",
    ))
    n_traces = 1

    # ── Pre-compute hover hierarchy text for every page ───────────
    hover_hier = []
    for i in range(n):
        parts = []
        for ld in tree:
            lv = ld["level"]
            comms = page_hier[i].get(lv, [])
            if comms:
                labels = [ld["labels"].get(c, f"C{c}") for c in comms]
                prefix = ["Detail", "Kategorie", "Oblast"][lv] if lv < 3 else f"L{lv}"
                parts.append(f"<b>{prefix}:</b> {', '.join(labels)}")
        hover_hier.append("<br>".join(parts))

    # ── Community traces per level ────────────────────────────────
    level_names = {0: "Detail", 1: "Kategorie", 2: "Oblast"}

    for level_data in tree:
        level = level_data["level"]
        member_pages_list = level_data["member_pages"]
        labels = level_data["labels"]
        n_comms = len(member_pages_list)

        palette = (
            px.colors.qualitative.Set3 + px.colors.qualitative.Pastel1
            + px.colors.qualitative.Bold + px.colors.qualitative.Vivid
            + px.colors.qualitative.Dark24
        )
        if n_comms > len(palette):
            palette = palette * ((n_comms // len(palette)) + 1)

        start_idx = n_traces
        marker_size = {0: 6, 1: 8, 2: 10}.get(level, 7)

        for cid, mp in enumerate(member_pages_list):
            pages = sorted(mp)
            if not pages:
                continue

            label = labels.get(cid, f"C{cid}")
            short = label[:35] + "…" if len(label) > 38 else label

            hovers = [
                f"<b>{page_ids[i]}</b><br>"
                f"Doc: {SHORT_NAMES.get(doc_ids[i], doc_ids[i])}<br>"
                f"Page: {page_nums[i]}<br>"
                f"<b>L{level}·C{cid}: {label}</b><br>"
                f"{'─' * 30}<br>"
                f"{hover_hier[i]}"
                for i in pages
            ]

            fig.add_trace(go.Scatter(
                x=[umap_2d[i, 0] for i in pages],
                y=[umap_2d[i, 1] for i in pages],
                mode="markers",
                marker=dict(
                    size=marker_size, color=palette[cid], opacity=0.8,
                    line=dict(width=0.5, color="black"),
                ),
                name=f"C{cid}: {short} ({len(pages)}p)",
                hovertext=hovers, hoverinfo="text",
                visible=(level == 0),
            ))
            n_traces += 1

        # Orphan trace for this level
        all_in_level = set()
        for mp in member_pages_list:
            all_in_level.update(mp)
        orphans = [i for i in range(n) if i not in all_in_level]
        if orphans:
            fig.add_trace(go.Scatter(
                x=[umap_2d[i, 0] for i in orphans],
                y=[umap_2d[i, 1] for i in orphans],
                mode="markers",
                marker=dict(size=4, color="lightgray", opacity=0.4,
                            line=dict(width=0.5, color="gray")),
                name=f"Bez komunity ({len(orphans)})",
                hovertext=[
                    f"<b>{page_ids[i]}</b><br>"
                    f"Doc: {SHORT_NAMES.get(doc_ids[i], doc_ids[i])}<br>"
                    f"Page: {page_nums[i]}<br><i>Orphan at L{level}</i><br>"
                    f"{'─' * 30}<br>{hover_hier[i]}"
                    for i in orphans
                ],
                hoverinfo="text",
                visible=(level == 0),
            ))
            n_traces += 1

        level_trace_ranges[level] = (start_idx, n_traces)

    # ── Document outline traces (always legendonly) ───────────────
    doc_trace_start = n_traces
    for doc_id in sorted(set(doc_ids)):
        mask = np.array([d == doc_id for d in doc_ids])
        fig.add_trace(go.Scatter(
            x=umap_2d[mask, 0], y=umap_2d[mask, 1], mode="markers",
            marker=dict(size=12, color="rgba(0,0,0,0)",
                        line=dict(width=1.5, color="black"),
                        symbol="circle-open"),
            name=f"Doc: {SHORT_NAMES.get(doc_id, doc_id)}",
            hoverinfo="skip", visible="legendonly",
        ))
        n_traces += 1

    # ── Level-switching buttons ───────────────────────────────────
    buttons = []
    for level_data in tree:
        level = level_data["level"]
        n_comms = len(level_data["member_pages"])
        lname = level_names.get(level, f"Level {level}")

        vis = ["legendonly"]  # trace 0: edges stay legendonly

        for ld in tree:
            lv = ld["level"]
            start, end = level_trace_ranges[lv]
            count = end - start
            vis.extend([True] * count if lv == level else [False] * count)

        # Doc outlines: always legendonly
        vis.extend(["legendonly"] * (n_traces - doc_trace_start))

        buttons.append(dict(
            label=f"  {lname} ({n_comms})  ",
            method="update",
            args=[{"visible": vis}],
        ))

    fig.update_layout(
        title=dict(
            text="SUJBOT — GraphRAG Hierarchical Communities (Leiden)",
            font=dict(size=18),
        ),
        xaxis=dict(title="UMAP 1", showgrid=False, zeroline=False),
        yaxis=dict(title="UMAP 2", showgrid=False, zeroline=False),
        plot_bgcolor="white", width=1400, height=900,
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.5, y=1.08, xanchor="center",
            buttons=buttons,
            bgcolor="white",
            bordercolor="gray",
            font=dict(size=13),
            pad=dict(r=10, t=10),
        )],
        legend=dict(
            title="Komunity",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="lightgray", borderwidth=1,
            font=dict(size=11),
        ),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )

    return fig


# ── Summary ───────────────────────────────────────────────────────
def print_hierarchy(tree, page_ids, doc_ids, page_nums):
    """Print the RAPTOR hierarchy tree to console."""
    n = len(page_ids)

    print(f"\n{'='*70}")
    print(f"  GRAPHRAG LEIDEN HIERARCHY SUMMARY")
    print(f"{'='*70}")
    print(f"  Total pages: {n}")
    print(f"  Hierarchy levels: {len(tree)}")

    for level_data in tree:
        level = level_data["level"]
        comms = level_data["member_pages"]
        labels = level_data["labels"]
        descs = level_data["descriptions"]

        print(f"\n  {'─'*60}")
        print(f"  LEVEL {level} — {len(comms)} communities")
        print(f"  {'─'*60}")

        for cid, mp in enumerate(comms):
            pages = sorted(mp)
            docs_in = set(doc_ids[p] for p in pages)
            label = labels.get(cid, f"C{cid}")
            desc = descs.get(cid, "")

            print(f"\n  L{level}·C{cid}: {label}")
            if desc:
                print(f"    {desc}")
            print(f"    {len(pages)} pages from {len(docs_in)} docs")
            for doc in sorted(docs_in):
                doc_pages = sorted(page_nums[p] for p in pages if doc_ids[p] == doc)
                if doc_pages:
                    print(f"      {SHORT_NAMES.get(doc, doc)}: "
                          f"pp.{min(doc_pages)}-{max(doc_pages)} ({len(doc_pages)}p)")

    # Multi-community stats for level 0
    if tree:
        level0 = tree[0]
        page_comm_count = [0] * n
        for mp in level0["member_pages"]:
            for p in mp:
                page_comm_count[p] += 1
        multi = sum(1 for c in page_comm_count if c > 1)
        orphan = sum(1 for c in page_comm_count if c == 0)
        print(f"\n  Level-0 overlap stats:")
        print(f"    Multi-community pages: {multi} ({multi/n*100:.1f}%)")
        print(f"    Orphan pages: {orphan} ({orphan/n*100:.1f}%)")


# ── Main ──────────────────────────────────────────────────────────
def main():
    Path("output").mkdir(exist_ok=True)

    print("Fetching embeddings from vectors.vl_pages...")
    page_ids, doc_ids, page_nums, embeddings = fetch_embeddings()
    n = embeddings.shape[0]
    print(f"  → {n} pages, {embeddings.shape[1]}-dim, {len(set(doc_ids))} documents")

    # ── 1. GraphRAG hierarchical Leiden clustering ──────────────────
    print(f"\n{'='*70}")
    print(f"  GRAPHRAG HIERARCHICAL LEIDEN CLUSTERING")
    print(f"{'='*70}")
    tree = leiden_hierarchy(embeddings, max_levels=3)

    if not tree:
        print("ERROR: No hierarchy levels produced!")
        return

    # ── 2. Multimodal LLM labeling ────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  MULTIMODAL LLM LABELING (Sonnet 4.5)")
    print(f"{'='*70}")
    label_tree(tree, embeddings, page_ids, doc_ids, page_nums)

    # ── 3. UMAP 2D for visualization ──────────────────────────────
    print("\nRunning UMAP (2048-dim → 2D)...")
    umap_2d = run_umap(embeddings)

    # ── 4. k-NN graph (for graph viz) ─────────────────────────────
    print("Building k-NN graph for visualization...")
    G = build_knn_graph(embeddings, page_ids, k=15, sim_threshold=0.3)
    print(f"  → {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── 5. Print hierarchy ────────────────────────────────────────
    print_hierarchy(tree, page_ids, doc_ids, page_nums)

    # ── 6. Interactive visualization ──────────────────────────────
    print("\nBuilding interactive visualization...")

    fig = build_interactive_hierarchy_viz(
        umap_2d, page_ids, doc_ids, page_nums, tree, G,
    )
    fig.write_html("output/topic_communities.html", include_plotlyjs=True)
    print("  → output/topic_communities.html")

    # ── 8. JSON export ────────────────────────────────────────────
    page_hier = build_page_hierarchy(tree, n)
    export = {"pages": [], "tree": {}}

    for i in range(n):
        hier = {}
        for ld in tree:
            lv = ld["level"]
            comms_at_lv = page_hier[i].get(lv, [])
            hier[f"level_{lv}"] = [
                {
                    "id": c,
                    "label": ld["labels"].get(c, f"C{c}"),
                    "description": ld["descriptions"].get(c, ""),
                }
                for c in comms_at_lv
            ]
        export["pages"].append({
            "page_id": page_ids[i],
            "document_id": doc_ids[i],
            "page_number": page_nums[i],
            "umap_x": float(umap_2d[i, 0]),
            "umap_y": float(umap_2d[i, 1]),
            "hierarchy": hier,
            "degree": G.degree(i),
        })

    for ld in tree:
        lv = ld["level"]
        export["tree"][f"level_{lv}"] = {
            str(cid): {
                "label": ld["labels"].get(cid, f"C{cid}"),
                "description": ld["descriptions"].get(cid, ""),
                "page_count": len(mp),
            }
            for cid, mp in enumerate(ld["member_pages"])
        }

    with open("output/topic_clusters_data.json", "w") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)
    print("  → output/topic_clusters_data.json")

    print("\nDone!")


if __name__ == "__main__":
    main()
