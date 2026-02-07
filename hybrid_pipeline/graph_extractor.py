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
from typing import Optional, List, Tuple, Dict

from .config import GraphConfig, ClassifierConfig
from .vector_utils import VectorSegment


def build_graph(segments: List[VectorSegment], precision: int = 1) -> nx.Graph:
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


def find_all_faces(segments: List[VectorSegment]) -> List[Polygon]:
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
    faces: List[Polygon],
    node_degrees: dict,
    config: GraphConfig,
    cls_config: ClassifierConfig,
) -> Tuple[List[Polygon], List[Polygon]]:
    """
    Filtre les faces par degré des nœuds + aire.
    
    Logique :
      - Extraire les sommets de chaque face.
      - Compter combien ont degré ≥ 4 dans le graphe (= croisements).
      - Si le ratio de croisements > max_cross_ratio → artefact → rejeté.
      - Un vrai composant a des coins "propres" (degré 2 ou 3).
      - Exception : les faces assez grandes (aire > 4× min_area) avec un
        ratio modéré sont gardées — ce sont des sous-faces de gros
        composants traversés par des fils.
    
    Returns:
        (faces_gardées, faces_rejetées) pour le debug.
    """
    kept = []
    rejected = []

    for face in faces:
        # Filtre par aire (on ne rejette QUE les trop petits ici).
        # Les faces trop grandes sont conservées pour permettre
        # une fusion intelligente en aval (elles peuvent être
        # des groupes de sous-faces). Rejeter uniquement les
        # petites faces bruiteuses.
        if face.area < cls_config.min_area:
            # Exception: keep small faces if they are very compact (circles).
            # Small circles (grounds, connectors) can have area < min_area
            # but are still valid components.
            try:
                import math
                perim = face.length
                if perim > 0:
                    circ = (4 * math.pi * face.area) / (perim ** 2)
                    if circ > 0.65 and face.area > 20:  # Compact + not noise
                        pass  # Keep it — small circle/ellipse
                    else:
                        continue
                else:
                    continue
            except Exception:
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
            # Exception : les faces relativement grandes avec un ratio
            # modéré sont probablement des sous-faces de gros composants
            # (un fil traverse un rectangle → crée des nœuds degré-4).
            # On les garde avec un seuil relaxé.
            is_big = face.area > cls_config.min_area * 4
            has_low_degree_corners = (len(coords) - cross_count) >= 2
            if is_big and has_low_degree_corners and ratio < 0.85:
                kept.append(face)
            else:
                # Also keep compact shapes (circles) even at crossings.
                # A circle at a wire junction has high-degree nodes but
                # its circularity distinguishes it from a wire artifact.
                try:
                    import math
                    perim = face.length
                    if perim > 0:
                        circ = (4 * math.pi * face.area) / (perim ** 2)
                        if circ > 0.55:  # Compact enough to be a real shape
                            kept.append(face)
                            continue
                except Exception:
                    pass
                rejected.append(face)
        else:
            kept.append(face)

    return kept, rejected


def filter_isolated_empty(
    candidates: List[Polygon],
    text_bboxes: List[Tuple],
) -> List[Polygon]:
    """
    Filtre "Vide & Solitaire" : rejette les polygones qui sont à la fois
    vides de texte ET isolés (pas de voisin adjacent).
    
    Exceptions (composant gardé même si isolé sans texte) :
      - Formes compactes (circularité > 0.50 ou g-ratio > 0.80)
      - Texte à proximité (pas uniquement à l'intérieur)
    """
    if not candidates:
        return []

    import math
    text_polys = [box(t[0], t[1], t[2], t[3]) for t in text_bboxes]
    text_proximity = 12.0  # points PDF autour du composant

    valid = []
    for i, poly in enumerate(candidates):
        # A. Test de contenu (texte à l'intérieur ou qui touche)
        has_text = any(poly.intersects(tp) for tp in text_polys)

        # A'. Test de proximité de texte (étiquette à côté du composant)
        has_text_nearby = has_text
        if not has_text:
            poly_buf = poly.buffer(text_proximity)
            has_text_nearby = any(poly_buf.intersects(tp) for tp in text_polys)

        # B. Test de voisinage (adjacence avec d'autres candidats)
        neighbors = 0
        for j, other in enumerate(candidates):
            if i == j:
                continue
            if poly.touches(other) or poly.intersects(other):
                neighbors += 1

        # C. Test de compacité (forme intrinsèquement intéressante)
        is_compact = False
        try:
            perimeter = poly.length
            if perimeter > 0:
                circ = (4 * math.pi * poly.area) / (perimeter ** 2)
                if circ > 0.50:
                    is_compact = True
            mrr = poly.minimum_rotated_rectangle
            if mrr.area > 0 and poly.area / mrr.area > 0.80:
                is_compact = True
        except Exception:
            pass

        # D. Décision : garder si au moins un critère est rempli
        if has_text or has_text_nearby or neighbors >= 1 or is_compact:
            valid.append(poly)

    return valid


def _shared_boundary_length(a: Polygon, b: Polygon, tolerance: float = 1.0) -> float:
    """
    Calcule la longueur du bord partagé entre deux polygones.
    
    Deux sous-faces d'un même rectangle partagent un segment complet
    (longueur significative). Deux symboles distincts reliés par un fil
    ne partagent qu'un point ou un segment minuscule.
    """
    try:
        # Intersection des contours (pas des aires)
        shared = a.boundary.intersection(b.buffer(tolerance).boundary)
        if shared.is_empty:
            return 0.0
        return shared.length
    except Exception:
        return 0.0


def _polygon_aspect_ratio(poly: Polygon) -> float:
    """Calcule l'aspect ratio (long / court côté) du rectangle orienté min."""
    try:
        box_rot = poly.minimum_rotated_rectangle
        if box_rot.is_empty:
            return 1.0
        x, y = box_rot.exterior.coords.xy
        from shapely.geometry import Point as _Pt
        e1 = _Pt(x[0], y[0]).distance(_Pt(x[1], y[1]))
        e2 = _Pt(x[1], y[1]).distance(_Pt(x[2], y[2]))
        short = min(e1, e2)
        if short <= 0:
            return 999.0
        return max(e1, e2) / short
    except Exception:
        return 1.0


def _is_boundary_collinear(a: Polygon, b: Polygon, tolerance: float = 1.0) -> bool:
    """
    Vérifie si le bord partagé entre deux faces est une seule ligne droite.
    
    Si c'est le cas, c'est probablement un fil qui separe deux composants
    distincts (pas des sous-faces d'un même composant).
    Un vrai bord interne de composant est généralement un côté du rectangle,
    mais entre deux composants le bord est un segment de fil.
    """
    try:
        shared = a.boundary.intersection(b.buffer(tolerance).boundary)
        if shared.is_empty:
            return False
        # Si le bord partagé est un seul LineString quasi-droit, c'est un fil
        if shared.geom_type == 'LineString':
            coords = list(shared.coords)
            if len(coords) == 2:
                return True  # Segment simple = fil
            # Vérifier si tous les points sont quasi-colinéaires
            if len(coords) >= 2:
                dx = coords[-1][0] - coords[0][0]
                dy = coords[-1][1] - coords[0][1]
                length = (dx*dx + dy*dy) ** 0.5
                if length > 0:
                    max_dev = 0.0
                    for c in coords[1:-1]:
                        # Distance point-droite
                        dev = abs(dy*(c[0]-coords[0][0]) - dx*(c[1]-coords[0][1])) / length
                        max_dev = max(max_dev, dev)
                    if max_dev < 2.0:  # Quasi-droit
                        return True
        return False
    except Exception:
        return False


def smart_merge_faces(
    faces: List[Polygon],
    config: GraphConfig,
    text_bboxes: Optional[List[Tuple]] = None,
) -> List[Polygon]:
    """
    Regroupe les sous-faces qui font partie d'un même composant,
    SANS fusionner deux composants distincts qui se touchent.

    Critères de fusion :
      1. Les faces se touchent (tolérance configurable).
      2. Elles partagent un vrai bord (pas juste un point de contact).
         → ratio = longueur_bord_commun / périmètre_petite_face.
      3. Le merge ne crée pas de gros trou vide (area growth check).
      4. Le résultat du merge reste compact (aspect ratio check).

    Pourquoi ça sépare les grounds :
      Deux symboles de terre se touchent en un seul point (via le fil)
      → shared boundary ratio ≈ 0 → merge refusé.

    Pourquoi ça reconstruit les gros rectangles :
      Les sous-faces d'un rectangle partagent un côté entier
      → shared boundary ratio élevé → merge autorisé.
    """
    if not faces:
        return []

    n = len(faces)
    if n == 1:
        return faces

    # Pré-calcul : quelles faces contiennent du texte ?
    text_polys = []
    if text_bboxes:
        text_polys = [box(t[0], t[1], t[2], t[3]) for t in text_bboxes]

    face_has_text = []
    for f in faces:
        has_t = any(f.intersects(tp) for tp in text_polys) if text_polys else False
        face_has_text.append(has_t)

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

    # Pré-calcul
    areas = [f.area for f in faces]
    perimeters = [f.length for f in faces]

    tol = config.merge_neighbor_tolerance
    for i in range(n):
        for j in range(i + 1, n):
            # 1. Test de contact
            try:
                if not faces[i].buffer(tol).intersects(faces[j]):
                    continue
            except Exception:
                continue

            # 2. Test de bord partagé
            shared_len = _shared_boundary_length(faces[i], faces[j], tol)
            min_perim = min(perimeters[i], perimeters[j])
            if min_perim > 0:
                shared_ratio = shared_len / min_perim
            else:
                shared_ratio = 0.0

            if shared_ratio < config.merge_min_shared_boundary:
                continue  # Juste un point de contact → pas le même composant

            # 2b. Rejet si les DEUX faces ont du texte
            #     = deux composants distincts (chacun a son label)
            if face_has_text[i] and face_has_text[j]:
                continue

            # 2c. Rejet si le bord partagé est une ligne droite (= fil)
            #     Les sous-faces d'un vrai composant partagent un côté
            #     du rectangle (peut être droit aussi, mais vérifier en combo)
            if _is_boundary_collinear(faces[i], faces[j], tol):
                # Exception : si shared_ratio est très élevé, c'est quand même
                # un vrai bord de composant (ex: rectangle coupé en 2)
                if shared_ratio < 0.25:
                    continue

            # 3. Test de croissance d'aire
            combined_area = areas[i] + areas[j]
            try:
                merged = unary_union([faces[i], faces[j]])
                if merged.area > combined_area * config.merge_max_area_growth:
                    continue
            except Exception:
                continue

            # 4. Test de compacité du résultat
            try:
                if _polygon_aspect_ratio(merged) > config.merge_max_aspect_ratio:
                    continue
            except Exception:
                pass

            union(i, j)

    # Regrouper par composant
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    # Fusionner au sein de chaque groupe
    result = []
    for indices in groups.values():
        if len(indices) == 1:
            result.append(faces[indices[0]])
        else:
            # Si plusieurs sous-faces du groupe ont chacune du texte,
            # ce sont probablement des composants distincts (ex: plusieurs
            # connecteurs côte à côte) — ne pas les fusionner.
            textful = sum(1 for idx in indices if face_has_text[idx])
            if textful > 1:
                for idx in indices:
                    result.append(faces[idx])
                continue

            group_polys = [faces[i] for i in indices]
            merged = unary_union(group_polys)
            if merged.geom_type == "Polygon":
                result.append(merged)
            elif merged.geom_type == "MultiPolygon":
                result.extend(list(merged.geoms))

    return result


def run_graph_extraction(
    segments: List[VectorSegment],
    text_bboxes: List[Tuple],
    graph_config: GraphConfig,
    cls_config: ClassifierConfig,
) -> List[Polygon]:
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

    result = smart_merge_faces(filtered, graph_config, text_bboxes)

    return result
