"""
wire_filter.py — Suppression des fils (wires) avant détection de composants.

Idée clé : les fils relient les composants mais NE SONT PAS des composants.
En les supprimant en amont, on élimine d'un coup :
  - Les faux polygones aux croisements de fils
  - Les faux busbars (petits rectangles aux intersections)
  - Les fusions accidentelles entre composants adjacents

Algorithme :
  1. Construire un graphe topologique (nœuds = jonctions, arêtes = segments).
  2. Identifier les "wire chains" : suites de segments quasi-colinéaires
     dont les nœuds intermédiaires ont degré exactement 2.
  3. Identifier les "wire bridges" : segments isolés longs et droits
     reliant deux nœuds de degré ≥ 3.
  4. Retirer tous les segments marqués comme fils.
  5. Retourner les segments restants (= composants).

Pourquoi ça marche :
  - Un fil est un chemin rectiligne entre deux jonctions (T ou +).
  - Un composant est un amas local de segments formant une forme fermée
    ou un symbole complexe. Ses nœuds internes ne sont PAS de degré 2
    (ils forment des coins, des boucles, etc.).
"""

import math
import networkx as nx
from collections import defaultdict
from typing import List, Dict, Tuple, Set

from .vector_utils import VectorSegment


def _round_pt(x: float, y: float, precision: int = 1) -> tuple:
    return (round(x, precision), round(y, precision))


def _segment_angle(seg: VectorSegment) -> float:
    """Angle du segment en radians [0, π)."""
    dx = seg.x2 - seg.x1
    dy = seg.y2 - seg.y1
    angle = math.atan2(dy, dx) % math.pi
    return angle


def _angles_collinear(a1: float, a2: float, tolerance_deg: float = 15.0) -> bool:
    """Vérifie si deux angles sont quasi-parallèles."""
    diff = abs(a1 - a2)
    if diff > math.pi / 2:
        diff = math.pi - diff
    return diff < math.radians(tolerance_deg)


def build_segment_graph(
    segments: List[VectorSegment],
    precision: int = 1,
) -> Tuple[nx.Graph, Dict]:
    """
    Construit un graphe où :
      - Nœuds = points de jonction (coordonnées arrondies)
      - Arêtes = segments vectoriels

    Retourne (Graph, edge_to_segments) pour pouvoir retrouver les
    segments originaux à partir des arêtes du graphe.
    """
    G = nx.Graph()
    # Map: (node_a, node_b) -> list of VectorSegment
    edge_to_segments: Dict[Tuple, List[VectorSegment]] = defaultdict(list)
    # Map: node -> list of segments touching this node
    node_to_segments: Dict[Tuple, List[VectorSegment]] = defaultdict(list)

    for seg in segments:
        p1 = _round_pt(seg.x1, seg.y1, precision)
        p2 = _round_pt(seg.x2, seg.y2, precision)
        if p1 == p2:
            continue

        # Normalize edge key (smaller tuple first)
        edge_key = (min(p1, p2), max(p1, p2))
        G.add_edge(p1, p2)
        edge_to_segments[edge_key].append(seg)
        node_to_segments[p1].append(seg)
        node_to_segments[p2].append(seg)

    return G, edge_to_segments, node_to_segments


def identify_wire_segments(
    segments: List[VectorSegment],
    precision: int = 1,
    min_wire_length: float = 15.0,
    collinear_tolerance_deg: float = 15.0,
    min_chain_length: float = 20.0,
) -> Set[int]:
    """
    Identifie les indices des segments qui sont des fils (wires).

    Stratégie multi-critère :
      A. Wire chains : suites de segments colinéaires passant par des
         nœuds de degré 2. Un fil traverse des nœuds sans se ramifier.
      B. Wire bridges : segments longs et isolés entre deux jonctions
         (degré ≥ 3). Un long trait droit entre deux T = fil.
      C. Straight long segments : segments individuels très longs et
         quasi-horizontaux/verticaux = fils.

    Args:
        segments: Tous les segments de la page.
        precision: Arrondi des coordonnées.
        min_wire_length: Longueur min d'un segment pour être un bridge candidate.
        collinear_tolerance_deg: Tolérance d'angle pour la colinéarité.
        min_chain_length: Longueur totale min d'une chaîne pour être un fil.

    Returns:
        Set d'indices (dans la liste `segments`) des segments identifiés comme fils.
    """
    if not segments:
        return set()

    # Index segments for fast lookup
    seg_index: Dict[int, VectorSegment] = {i: seg for i, seg in enumerate(segments)}

    # Build graph
    G = nx.Graph()
    # Map: edge (normalized) -> list of (seg_index, VectorSegment)
    edge_seg_map: Dict[Tuple, List[Tuple[int, VectorSegment]]] = defaultdict(list)
    node_seg_map: Dict[Tuple, List[int]] = defaultdict(list)  # node -> seg indices

    for i, seg in enumerate(segments):
        p1 = _round_pt(seg.x1, seg.y1, precision)
        p2 = _round_pt(seg.x2, seg.y2, precision)
        if p1 == p2:
            continue
        edge_key = (min(p1, p2), max(p1, p2))
        G.add_edge(p1, p2)
        edge_seg_map[edge_key].append((i, seg))
        node_seg_map[p1].append(i)
        node_seg_map[p2].append(i)

    degrees = dict(G.degree())
    wire_indices: Set[int] = set()

    # ── A. Wire chains ──
    # Find paths through degree-2 nodes (wire routing paths).
    # A wire chain is a sequence of connected segments where
    # all intermediate nodes have degree == 2, and consecutive
    # segments are roughly collinear (same direction).
    visited_nodes = set()

    for start_node in G.nodes():
        if degrees.get(start_node, 0) != 2:
            continue  # Only start from degree-2 nodes (middle of wire)
        if start_node in visited_nodes:
            continue

        # Trace the chain in both directions
        chain_seg_indices = []
        chain_length = 0.0

        for direction in [0, 1]:
            current = start_node
            prev = None

            while True:
                visited_nodes.add(current)
                neighbors = list(G.neighbors(current))

                if prev is not None:
                    neighbors = [n for n in neighbors if n != prev]

                if not neighbors:
                    break

                next_node = neighbors[0]
                edge_key = (min(current, next_node), max(current, next_node))

                if edge_key not in edge_seg_map:
                    break

                # Check collinearity with previous segment
                curr_segs = edge_seg_map[edge_key]
                if chain_seg_indices and curr_segs:
                    prev_seg = seg_index[chain_seg_indices[-1]]
                    curr_seg = curr_segs[0][1]
                    if not _angles_collinear(
                        _segment_angle(prev_seg),
                        _segment_angle(curr_seg),
                        collinear_tolerance_deg,
                    ):
                        break  # Direction changed → end of wire

                for idx, seg in curr_segs:
                    chain_seg_indices.append(idx)
                    chain_length += seg.length

                prev = current
                current = next_node

                # Stop if we hit a junction (degree ≠ 2)
                if degrees.get(current, 0) != 2:
                    break

        # Mark as wire if chain is long enough
        if chain_length >= min_chain_length and len(chain_seg_indices) >= 2:
            wire_indices.update(chain_seg_indices)

    # ── B. Wire bridges ──
    # Individual long segments between two junctions (degree ≥ 3).
    # These are direct wire connections between components.
    for i, seg in enumerate(segments):
        if i in wire_indices:
            continue
        if seg.length < min_wire_length:
            continue

        p1 = _round_pt(seg.x1, seg.y1, precision)
        p2 = _round_pt(seg.x2, seg.y2, precision)
        if p1 == p2:
            continue

        d1 = degrees.get(p1, 0)
        d2 = degrees.get(p2, 0)

        # Both endpoints are junctions → bridge wire
        if d1 >= 3 and d2 >= 3 and seg.length >= min_wire_length:
            # Extra check: is the segment quasi-horizontal or quasi-vertical?
            angle = _segment_angle(seg)
            is_axis_aligned = (
                angle < math.radians(10)
                or angle > math.radians(170)
                or abs(angle - math.pi / 2) < math.radians(10)
            )
            if is_axis_aligned:
                wire_indices.add(i)

    # ── C. Long straight isolated segments ──
    # Very long segments (> 3× min_wire_length) that are axis-aligned
    # are almost always wires, regardless of endpoint degrees.
    long_threshold = min_wire_length * 3
    for i, seg in enumerate(segments):
        if i in wire_indices:
            continue
        if seg.length < long_threshold:
            continue

        angle = _segment_angle(seg)
        is_axis_aligned = (
            angle < math.radians(8)
            or angle > math.radians(172)
            or abs(angle - math.pi / 2) < math.radians(8)
        )
        if is_axis_aligned:
            wire_indices.add(i)

    return wire_indices


def remove_wires(
    segments: List[VectorSegment],
    precision: int = 1,
    min_wire_length: float = 15.0,
    collinear_tolerance_deg: float = 15.0,
    min_chain_length: float = 20.0,
) -> Tuple[List[VectorSegment], List[VectorSegment]]:
    """
    Sépare les segments en composants et fils.

    Returns:
        (component_segments, wire_segments)
    """
    wire_indices = identify_wire_segments(
        segments,
        precision=precision,
        min_wire_length=min_wire_length,
        collinear_tolerance_deg=collinear_tolerance_deg,
        min_chain_length=min_chain_length,
    )

    component_segs = []
    wire_segs = []

    for i, seg in enumerate(segments):
        if i in wire_indices:
            wire_segs.append(seg)
        else:
            component_segs.append(seg)

    return component_segs, wire_segs
