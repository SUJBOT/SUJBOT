"""
Community Detector — Leiden hierarchy on entity-relationship graph.

Adapts the GraphRAG-style hierarchical Leiden detection from
scripts/topic_clustering.py for entity-relationship graphs.
"""

import logging
from collections import defaultdict
from typing import Dict, List

import igraph as ig

logger = logging.getLogger(__name__)


class CommunityDetector:
    """
    Hierarchical Leiden community detection on entity-relationship graph.

    Level 0: Leiden on entity graph (high resolution → small communities)
    Level 1+: Aggregate inter-community edges → community graph → Leiden
    """

    def detect(
        self,
        entities: List[Dict],
        relationships: List[Dict],
        max_levels: int = 3,
        base_resolution: float = 1.0,
        min_community_size: int = 3,
    ) -> List[Dict]:
        """
        Build igraph from entities/relationships, run hierarchical Leiden.

        Args:
            entities: List of entity dicts with 'entity_id'
            relationships: List with 'source_entity_id', 'target_entity_id', 'weight'
            max_levels: Maximum hierarchy depth
            base_resolution: Leiden resolution (higher = more communities)
            min_community_size: Minimum entities per community

        Returns:
            List of community dicts: {level, entity_ids, title}
        """
        if not entities:
            return []

        # Build entity_id -> index mapping
        id_to_idx = {e["entity_id"]: i for i, e in enumerate(entities)}
        n = len(entities)

        # Build edges
        edges = []
        weights = []
        for r in relationships:
            src_idx = id_to_idx.get(r["source_entity_id"])
            tgt_idx = id_to_idx.get(r["target_entity_id"])
            if src_idx is not None and tgt_idx is not None and src_idx != tgt_idx:
                edges.append((src_idx, tgt_idx))
                weights.append(r.get("weight", 1.0))

        if not edges:
            logger.warning("No valid edges for community detection")
            return []

        G = ig.Graph(n, edges, directed=False)
        G.es["weight"] = weights
        logger.info(f"Built entity graph: {G.vcount()} nodes, {G.ecount()} edges")

        communities_all = []
        current_graph = G
        # Map from current graph node index → set of original entity indices
        current_entity_sets = [{i} for i in range(n)]

        for level in range(max_levels):
            n_nodes = current_graph.vcount()
            if n_nodes <= min_community_size:
                break

            resolution = base_resolution / (1.5 ** level)
            min_members = min_community_size if level == 0 else 2

            logger.info(
                f"Level {level}: Leiden on {n_nodes} nodes (resolution={resolution:.2f})"
            )

            try:
                partition = current_graph.community_leiden(
                    objective_function="modularity",
                    weights="weight",
                    resolution=resolution,
                    n_iterations=10,
                )
            except Exception as e:
                logger.error(
                    f"Leiden algorithm failed at level {level} "
                    f"({n_nodes} nodes): {e}",
                    exc_info=True,
                )
                break

            # Collect communities
            comm_members: Dict[int, set] = defaultdict(set)
            for node, cid in enumerate(partition.membership):
                comm_members[cid].add(node)

            # Filter small communities
            valid_comms = [
                members for members in comm_members.values()
                if len(members) >= min_members
            ]
            valid_comms.sort(key=len, reverse=True)

            if len(valid_comms) <= 1:
                logger.info(f"Level {level}: only {len(valid_comms)} community, stopping")
                break

            # Resolve to original entity IDs
            for members in valid_comms:
                original_entity_ids = set()
                for m in members:
                    original_entity_ids.update(current_entity_sets[m])

                entity_id_list = [
                    entities[idx]["entity_id"] for idx in sorted(original_entity_ids)
                ]
                communities_all.append({
                    "level": level,
                    "entity_ids": entity_id_list,
                    "title": None,
                    "summary": None,
                })

            sizes = ", ".join(str(len(m)) for m in valid_comms)
            logger.info(f"Level {level}: {len(valid_comms)} communities [{sizes}]")

            # Build community graph for next level
            n_comms = len(valid_comms)
            node_to_comm = {}
            for new_cid, members in enumerate(valid_comms):
                for m in members:
                    if m not in node_to_comm:
                        node_to_comm[m] = new_cid

            comm_edge_weights: Dict[tuple, float] = defaultdict(float)
            for edge in current_graph.es:
                u, v = edge.source, edge.target
                cu = node_to_comm.get(u)
                cv = node_to_comm.get(v)
                if cu is not None and cv is not None and cu != cv:
                    key = (min(cu, cv), max(cu, cv))
                    comm_edge_weights[key] += edge["weight"]

            if not comm_edge_weights:
                logger.info("No inter-community edges, stopping hierarchy")
                break

            comm_edge_list = list(comm_edge_weights.keys())
            comm_weight_list = [comm_edge_weights[e] for e in comm_edge_list]

            current_graph = ig.Graph(n_comms, comm_edge_list, directed=False)
            current_graph.es["weight"] = comm_weight_list

            # Update entity sets for next level
            new_entity_sets = []
            for members in valid_comms:
                merged = set()
                for m in members:
                    merged.update(current_entity_sets[m])
                new_entity_sets.append(merged)
            current_entity_sets = new_entity_sets

        logger.info(f"Community detection complete: {len(communities_all)} total communities")
        return communities_all
