"""
classifier.py — Classification des polygones détectés en catégories métier.

Reprend la logique de decision tree de vector_engine.py / polygonize.py :
  - G-ratio (rectangularité)
  - D-ratio (densité matière/enveloppe)
  - Circularité
  - Épaisseur du rectangle orienté minimum

Catégories de sortie :
  - Component_Rect     : Composant rectangulaire plein (relais, ECU, etc.)
  - Component_Complex  : Forme complexe dense (L-shape, triangles, etc.)
  - Circle_Component   : Composant circulaire (moteurs, voyants)
  - Hex_Symbol         : Hexagone / symbole spécial
  - Busbar_Power       : Bus de puissance / câble épais
  - Group_Container    : Conteneur / groupe en pointillés
  - Open_Component     : Composant détecté par DBSCAN (forme ouverte)
  - Unknown_Shape      : Forme non classifiée mais gardée
"""

import math
from dataclasses import dataclass
from shapely.geometry import Polygon, Point
from typing import Optional, List

from .config import ClassifierConfig


@dataclass
class DetectedComponent:
    """Un composant détecté et classifié."""
    id: int
    category: str
    bbox: tuple           # (x1, y1, x2, y2) en points PDF
    polygon: Polygon      # Géométrie Shapely
    source: str           # "graph" ou "dbscan"
    thickness: float = 0.0
    circularity: float = 0.0
    g_ratio: float = 0.0  # Rectangularité
    d_ratio: float = 0.0  # Densité
    page_index: int = 0

    def to_dict(self) -> dict:
        """Sérialisation pour export JSON."""
        return {
            "id": self.id,
            "type": self.category,
            "bbox": list(self.bbox),
            "source": self.source,
            "thickness": round(self.thickness, 2),
            "circularity": round(self.circularity, 3),
            "g_ratio": round(self.g_ratio, 3),
            "d_ratio": round(self.d_ratio, 3),
        }


def compute_metrics(poly: Polygon) -> dict:
    """
    Calcule les métriques géométriques d'un polygone.
    
    Returns:
        Dict avec thickness, g_ratio, d_ratio, circularity.
    """
    metrics = {
        "thickness": 0.0,
        "g_ratio": 0.0,
        "d_ratio": 1.0,
        "circularity": 0.0,
    }

    try:
        poly_clean = poly.buffer(0)
        if poly_clean.is_empty or poly_clean.area <= 0:
            return metrics

        # Rectangle orienté minimum → épaisseur
        box_rot = poly_clean.minimum_rotated_rectangle
        if box_rot.is_empty:
            return metrics

        x, y = box_rot.exterior.coords.xy
        edge1 = Point(x[0], y[0]).distance(Point(x[1], y[1]))
        edge2 = Point(x[1], y[1]).distance(Point(x[2], y[2]))
        metrics["thickness"] = min(edge1, edge2)

        # Enveloppe simplifiée pour les ratios
        poly_env = Polygon(poly_clean.exterior).simplify(0.5)
        box_env = poly_env.minimum_rotated_rectangle

        # G-ratio (rectangularité)
        if box_env.area > 0:
            metrics["g_ratio"] = poly_env.area / box_env.area

        # D-ratio (densité matière / enveloppe)
        if poly_env.area > 0:
            metrics["d_ratio"] = poly_clean.area / poly_env.area

        # Circularité : 4π·A / P²
        perimeter = poly_env.length
        if perimeter > 0:
            metrics["circularity"] = (4 * math.pi * poly_env.area) / (perimeter ** 2)

    except Exception:
        pass

    return metrics


def classify_polygon(
    poly: Polygon,
    config: ClassifierConfig,
    source: str = "graph",
) -> Optional[str]:
    """
    Classifie un polygone selon l'arbre de décision.
    
    Args:
        poly: Le polygone à classifier.
        config: Seuils de classification.
        source: "graph" ou "dbscan".
    
    Returns:
        Catégorie (str) ou None si le polygone doit être rejeté.
    """
    m = compute_metrics(poly)

    thickness = m["thickness"]
    g_ratio = m["g_ratio"]
    d_ratio = m["d_ratio"]
    circ = m["circularity"]

    # Les composants DBSCAN sont des bboxes → ratios peu fiables.
    # On les garde tous avec une catégorie spéciale.
    if source == "dbscan":
        return "Open_Component"

    # ── ARBRE DE DÉCISION (formes fermées / graphe) ──

    # 1. Fils fins → rejet
    if thickness < config.thin_wire_threshold:
        return None

    # 1b. Fils allongés → rejet par aspect ratio
    #     Même si l'épaisseur est > thin_wire, une forme très allongée
    #     et mince est un fil (ex: un segment de busbar mal polygonisé).
    if thickness < config.aspect_ratio_max_thickness and thickness > 0:
        box_rot = poly.buffer(0).minimum_rotated_rectangle
        if not box_rot.is_empty:
            bx, by = box_rot.exterior.coords.xy
            from shapely.geometry import Point as _Pt
            e1 = _Pt(bx[0], by[0]).distance(_Pt(bx[1], by[1]))
            e2 = _Pt(bx[1], by[1]).distance(_Pt(bx[2], by[2]))
            long_side = max(e1, e2)
            short_side = min(e1, e2)
            if short_side > 0:
                aspect = long_side / short_side
                if aspect > config.max_aspect_ratio:
                    return None  # Trop allongé → fil

    # 2. Cercles
    if circ > config.circle_threshold:
        return "Circle_Component"

    # 3. Hexagones / symboles
    hex_lo, hex_hi = config.hex_rect_range
    if hex_lo <= g_ratio <= hex_hi and circ > config.hex_circ_min:
        return "Hex_Symbol"

    # 4. Formes rectangulaires
    if g_ratio > config.rect_ratio_threshold:
        if thickness < config.busbar_threshold:
            # ── Busbar candidate ──
            # Guard: wire crossings create tiny perfect rectangles (3-8pt).
            # A real busbar has meaningful thickness (≥ min_busbar_thickness).
            if thickness < config.min_busbar_thickness:
                # Too thin for a busbar — check if it's just a wire crossing.
                # Wire crossings are small, dense, very rectangular squares.
                area = poly.buffer(0).area
                if area < 400:  # Small rectangles at wire crossings
                    return None  # Wire crossing artifact
                # Larger thin rectangles could be real thin components
                if d_ratio > config.density_busbar_min:
                    return "Unknown_Shape"  # Not confident enough for Busbar
                else:
                    return None
            if d_ratio > config.density_busbar_min:
                return "Busbar_Power"
            else:
                return None  # Layout line vide
        else:
            # Gros composant
            if d_ratio > config.density_filled:
                return "Component_Rect"
            elif d_ratio < config.density_empty:
                return None  # Cadre layout vide
            else:
                return "Group_Container"

    # 5. Formes complexes denses
    if d_ratio > 0.75:
        return "Component_Complex"

    # 6. Unknown si assez grand
    poly_env = Polygon(poly.exterior).simplify(0.5) if poly.exterior else poly
    if poly_env.area > config.unknown_min_area:
        return "Unknown_Shape"

    return None  # Trop petit ou indéterminé


def classify_all(
    polygons: List[Polygon],
    config: ClassifierConfig,
    source: str = "graph",
    page_index: int = 0,
    id_offset: int = 0,
) -> List[DetectedComponent]:
    """
    Classifie une liste de polygones.
    
    Args:
        polygons: Les polygones à classifier.
        config: Configuration du classifieur.
        source: Origine ("graph" ou "dbscan").
        page_index: Index de la page PDF.
        id_offset: Offset pour numérotation des IDs.
    
    Returns:
        Liste de DetectedComponent classifiés (les rejets sont exclus).
    """
    components = []

    for i, poly in enumerate(polygons):
        category = classify_polygon(poly, config, source)
        if category is None:
            continue

        m = compute_metrics(poly)
        minx, miny, maxx, maxy = poly.bounds

        comp = DetectedComponent(
            id=id_offset + i,
            category=category,
            bbox=(minx, miny, maxx, maxy),
            polygon=poly,
            source=source,
            thickness=m["thickness"],
            circularity=m["circularity"],
            g_ratio=m["g_ratio"],
            d_ratio=m["d_ratio"],
            page_index=page_index,
        )
        components.append(comp)

    return components
