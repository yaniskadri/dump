"""
graph_extractor.py — Extraction de composants fermés par topologie vectorielle.

Stratégie corrigée (v2) :
  1. Shapely `polygonize` sur tous les segments → trouve TOUTES les faces fermées
     (petits rectangles de composants ET artefacts de croisements de fils).
  2. Construire un graphe NetworkX pour calculer le degré de chaque nœud.
  3. Node Degree Filter : pour chaque face, vérifier si ses sommets sont
     majoritairement des croisements (degré ≥ 4) → rejeter.
  4. Filtre "Vide & Solitaire" : rejeter les faces sans texte ni voisins.

Pourquoi pas minimum_cycle_basis ?
  → Il retourne une base algébrique (~10 gros cycles englobants), pas les
    petites faces individuelles. Le degree filter n'a aucun effet dessus.
  → polygonize est conçu exactement pour trouver les faces d'un arrangement
    planaire de lignes — c'est précisément ce dont on a besoin.
"""

import networkx as nx
from shapely.geometry import Polygon, box, LineString
from shapely.ops import unary_union, polygonize

from .config import GraphConfig, ClassifierConfig
from .vector_utils import VectorSegment


def build_graph(segments: list[VectorSegment], precision: int = 1) -> nx.Graph:
    """
    Construit un graphe non-orienté à partir des segments vectoriels.
    Les coordonnées sont arrondies pour fusionner les nœuds proches.
    Sert uniquement à calculer les degrés des nœuds (pas pour les cycles).
    """
    G = nx.Graph()
    for seg in segments:
        p1, p2 = seg.as_rounded_endpoints(precision)
        if p1 != p2:
            G.add_edge(p1, p2)
    return G


def get_node_degrees(G: nx.Graph) -> dict:
    """Retourne un dict {(x,y): degré} pour tous les nœuds du graphe."""
    return dict(G.degree())


def find_all_faces(segments: list[VectorSegment]) -> list[Polygon]:
    """
    Utilise Shapely polygonize pour trouver toutes les faces fermées
    dans l'arrangement planaire des segments vectoriels.
    
    C'est la bonne méthode pour trouver les petits rectangles individuels
    (composants ET croisements de fils — on filtre après).
    """
    lines = []
    for seg in segments:
        ls = seg.as_linestring()
        if ls.length > 0:
            lines.append(ls)

    if not lines:
        return []

    try:
        merged = unary_union(lines)
        faces = list(polygonize(merged))
    except Exception:
        return []

    return faces


def snap_to_graph(coord: tuple, node_degrees: dict, tolerance: float = 1.5) -> int:
    """
    Trouve le degré du nœud du graphe le plus proche d'une coordonnée.
    
    Les coordonnées polygonize ne sont pas toujours exactement alignées
    avec les nœuds du graphe (arrondis), donc on cherche le plus proche.
    
    Returns:
        Degré du nœud le plus proche, ou 0 si aucun nœud trouvé.
    """
    cx, cy = round(coord[0], 1), round(coord[1], 1)
    
    # Essai exact d'abord
    if (cx, cy) in node_degrees:
        return node_degrees[(cx, cy)]
    
    # Recherche par proximité
    best_deg = 0
    best_dist = tolerance
    for (nx_, ny_), deg in node_degrees.items():
        dist = abs(nx_ - coord[0]) + abs(ny_ - coord[1])  # Manhattan rapide
        if dist < best_dist:
            best_dist = dist
            best_deg = deg
    
    return best_deg


def filter_by_node_degree(
    faces: list[Polygon],
    node_degrees: dict,
    config: GraphConfig,
    cls_config: ClassifierConfig,
) -> tuple[list[Polygon], list[Polygon]]:
    """
    Filtre les faces par degré des nœuds + aire.
    
    Logique :
      - Extraire les sommets de chaque face.
      - Compter combien ont degré ≥ 4 dans le graphe (= croisements).
      - Si le ratio de croisements > max_cross_ratio → artefact → rejeté.
      - Un vrai composant a des coins "propres" (degré 2 ou 3).
    
    Returns:
        (faces_gardées, faces_rejetées) pour le debug.
    """
    kept = []
    rejected = []

    for face in faces:
        # Filtre par aire
        if face.area < cls_config.min_area or face.area > cls_config.max_area:
            continue

        if not face.is_valid:
            face = face.buffer(0)
            if not face.is_valid:
                continue

        # Extraire les sommets du polygone (sans le dernier qui = premier)
        coords = list(face.exterior.coords)[:-1]
        if len(coords) < 3:
            continue

        # Compter les nœuds de croisement
        cross_count = 0
        for coord in coords:
            deg = snap_to_graph(coord, node_degrees)
            if deg >= 4:
                cross_count += 1

        ratio = cross_count / len(coords)

        if ratio > config.max_cross_ratio:
            rejected.append(face)
        else:
            kept.append(face)

    return kept, rejected


def filter_isolated_empty(
    candidates: list[Polygon],
    text_bboxes: list[tuple],
) -> list[Polygon]:
    """
    Filtre "Vide & Solitaire" : rejette les polygones qui sont à la fois
    vides de texte ET isolés (pas de voisin adjacent).
    
    Un composant légitime a soit du texte à l'intérieur, soit fait partie
    d'une grille de connecteurs (voisins qui se touchent).
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

    return valid


def smart_merge_faces(
    faces: list[Polygon],
    config: GraphConfig,
) -> list[Polygon]:
    """
    Regroupe les sous-faces qui font partie d'un même composant,
    SANS fusionner deux composants distincts qui se touchent simplement.

    Stratégie :
      - Deux faces sont fusionnables si elles partagent un bord (touchent)
        ET que l'aire combinée ne dépasse pas un seuil raisonnable.
      - On utilise un Union-Find pour regrouper les faces apparentées.
      - Au sein d'un groupe, on fait unary_union pour reconstruire
        le composant complet (ex: L-shape = 2 rectangles fusionnés).

    Cela évite le problème du unary_union aveugle qui fusionnait
    deux symboles de terre adjacents en un seul blob.
    """
    if not faces:
        return []

    n = len(faces)
    if n == 1:
        return faces

    # Union-Find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Calculer les aires individuelles
    areas = [f.area for f in faces]

    # Construire les groupes : fusionner seulement les faces qui
    # partagent un bord ET dont la combinaison reste raisonnable.
    tol = config.merge_neighbor_tolerance
    for i in range(n):
        for j in range(i + 1, n):
            # Test de contact : les faces se touchent ou se chevauchent légèrement
            try:
                if not faces[i].buffer(tol).intersects(faces[j]):
                    continue
            except Exception:
                continue

            # Test d'aire combinée : ne pas fusionner si le résultat est trop gros
            combined_area = areas[i] + areas[j]
            if combined_area > config.merge_max_combined_area:
                continue

            # Test de croissance d'aire : vérifier que le merge ne crée pas
            # un gros trou vide entre les deux faces
            try:
                merged = unary_union([faces[i], faces[j]])
                if merged.area > combined_area * config.merge_max_area_growth:
                    continue
            except Exception:
                continue

            union(i, j)

    # Regrouper par composant
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    # Fusionner au sein de chaque groupe
    result = []
    for indices in groups.values():
        if len(indices) == 1:
            result.append(faces[indices[0]])
        else:
            group_polys = [faces[i] for i in indices]
            merged = unary_union(group_polys)
            if merged.geom_type == "Polygon":
                result.append(merged)
            elif merged.geom_type == "MultiPolygon":
                result.extend(list(merged.geoms))

    return result


def run_graph_extraction(
    segments: list[VectorSegment],
    text_bboxes: list[tuple],
    graph_config: GraphConfig,
    cls_config: ClassifierConfig,
) -> list[Polygon]:
    """
    Pipeline complète d'extraction par topologie vectorielle.
    
    Étapes :
      1. polygonize → toutes les faces fermées
      2. build_graph → degrés des nœuds
      3. Node Degree Filter → rejeter les croisements de fils
      4. Filtre Vide & Solitaire → rejeter les faces sans contenu
      5. Smart Merge → regrouper les sous-faces d'un même composant
    
    Returns:
        Liste de Polygones Shapely représentant les composants fermés détectés.
    """
    if not segments:
        return []

    # 1. Trouver toutes les faces fermées via polygonize
    all_faces = find_all_faces(segments)

    if not all_faces:
        return []

    # 2. Construire le graphe pour les degrés des nœuds
    G = build_graph(segments, graph_config.coord_precision)
    node_degrees = get_node_degrees(G)

    # 3. Node Degree Filter
    candidates, _rejected = filter_by_node_degree(
        all_faces, node_degrees, graph_config, cls_config,
    )

    # 4. Filtre Vide & Solitaire
    filtered = filter_isolated_empty(candidates, text_bboxes)

    # 5. Smart Merge (remplace l'ancien unary_union aveugle)
    if not filtered:
        return []

    result = smart_merge_faces(filtered, graph_config)

    return result
