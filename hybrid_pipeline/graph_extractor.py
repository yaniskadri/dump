"""
graph_extractor.py — Extraction de composants fermés par théorie des graphes.

Stratégie :
  1. Construire un graphe NetworkX à partir des segments vectoriels.
  2. Trouver les cycles minimaux (minimum_cycle_basis).
  3. Filtrer les faux cycles (croisements de fils) via le Node Degree Filter :
     - Si > 50% des nœuds d'un cycle ont un degré ≥ 4 → croisement → rejeté.
  4. Convertir les cycles valides en polygones Shapely.
"""

import networkx as nx
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from .config import GraphConfig, ClassifierConfig
from .vector_utils import VectorSegment


def build_graph(segments: list[VectorSegment], precision: int = 1) -> nx.Graph:
    """
    Construit un graphe non-orienté à partir des segments vectoriels.
    Les coordonnées sont arrondies pour fusionner les nœuds proches.
    """
    G = nx.Graph()
    for seg in segments:
        p1, p2 = seg.as_rounded_endpoints(precision)
        if p1 != p2:  # Ignorer les segments dégénérés (point)
            G.add_edge(p1, p2)
    return G


def extract_cycles(G: nx.Graph, config: GraphConfig) -> list[list[tuple]]:
    """
    Extrait les cycles minimaux du graphe et filtre les croisements de fils.
    
    Le Node Degree Filter :
      - Pour chaque cycle, on compte combien de nœuds ont degré ≥ 4.
      - Si ce ratio dépasse max_cross_ratio → c'est un croisement de fils → rejeté.
      - Les vrais composants ont des coins "propres" (degré 2 = angle en L).
    
    Returns:
        Liste de cycles valides (chaque cycle = liste de coordonnées (x, y)).
    """
    try:
        raw_cycles = nx.minimum_cycle_basis(G)
    except Exception:
        return []

    valid_cycles = []

    for cycle_nodes in raw_cycles:
        if len(cycle_nodes) < config.min_cycle_nodes:
            continue

        # ---- NODE DEGREE FILTER ----
        cross_junctions = 0
        for node in cycle_nodes:
            if G.degree[node] >= 4:  # Nœud carrefour (X ou +)
                cross_junctions += 1

        ratio = cross_junctions / len(cycle_nodes) if cycle_nodes else 1.0
        if ratio > config.max_cross_ratio:
            continue  # Croisement de fils → poubelle

        valid_cycles.append(cycle_nodes)

    return valid_cycles


def cycles_to_polygons(
    cycles: list[list[tuple]],
    cls_config: ClassifierConfig,
) -> list[Polygon]:
    """
    Convertit les cycles (listes de points) en polygones Shapely.
    Filtre par aire min/max et validité géométrique.
    """
    polygons = []
    for cycle in cycles:
        if len(cycle) < 3:
            continue
        try:
            poly = Polygon(cycle)
            if not poly.is_valid:
                poly = poly.buffer(0)  # Tenter réparation
            if not poly.is_valid:
                continue
            if poly.area < cls_config.min_area or poly.area > cls_config.max_area:
                continue
            polygons.append(poly)
        except Exception:
            continue
    return polygons


def filter_isolated_empty(
    candidates: list[Polygon],
    text_bboxes: list[tuple],
) -> list[Polygon]:
    """
    Filtre "Vide & Solitaire" : rejette les polygones qui sont à la fois
    vides de texte ET isolés (pas de voisin adjacent).
    
    Un composant légitime a soit du texte à l'intérieur, soit fait partie
    d'une grille de connecteurs (voisins qui se touchent).
    
    Args:
        candidates: Polygones candidats.
        text_bboxes: Tuples (x0, y0, x1, y1, text) des blocs texte.
    
    Returns:
        Polygones filtrés.
    """
    if not candidates:
        return []

    text_polys = [box(t[0], t[1], t[2], t[3]) for t in text_bboxes]

    valid = []
    for i, poly in enumerate(candidates):
        # A. Test de contenu (texte à l'intérieur ou qui touche)
        has_text = any(poly.intersects(tp) for tp in text_polys)

        # B. Test de voisinage (adjacence avec d'autres candidats)
        neighbors = 0
        for j, other in enumerate(candidates):
            if i == j:
                continue
            if poly.touches(other) or poly.intersects(other):
                neighbors += 1

        # C. Décision
        if has_text or neighbors >= 1:
            valid.append(poly)
        # Sinon → vide et solitaire → croisement de fils → rejeté

    return valid


def run_graph_extraction(
    segments: list[VectorSegment],
    text_bboxes: list[tuple],
    graph_config: GraphConfig,
    cls_config: ClassifierConfig,
) -> list[Polygon]:
    """
    Pipeline complète d'extraction par graphe.
    
    Returns:
        Liste de Polygones Shapely représentant les composants fermés détectés.
    """
    # 1. Construction du graphe
    G = build_graph(segments, graph_config.coord_precision)

    if G.number_of_edges() == 0:
        return []

    # 2. Extraction des cycles + Node Degree Filter
    valid_cycles = extract_cycles(G, graph_config)

    # 3. Conversion en polygones
    candidates = cycles_to_polygons(valid_cycles, cls_config)

    # 4. Filtre Vide & Solitaire
    filtered = filter_isolated_empty(candidates, text_bboxes)

    # 5. Fusion des polygones qui se chevauchent
    if not filtered:
        return []

    merged = unary_union(filtered)
    result = []
    if merged.geom_type == "Polygon":
        result.append(merged)
    elif merged.geom_type == "MultiPolygon":
        result.extend(list(merged.geoms))

    return result
