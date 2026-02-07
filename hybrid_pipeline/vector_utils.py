"""
vector_utils.py — Utilitaires communs d'extraction vectorielle depuis PyMuPDF.

Centralise la lecture des vecteurs PDF pour éviter la duplication
entre les modules graph et DBSCAN.
"""

import fitz
import math
from dataclasses import dataclass
from shapely.geometry import LineString

from .config import GraphConfig


@dataclass
class VectorSegment:
    """Un segment vectoriel extrait d'un PDF."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def length(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    @property
    def center(self) -> tuple:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def bbox(self) -> tuple:
        """Retourne (min_x, min_y, max_x, max_y)."""
        return (
            min(self.x1, self.x2), min(self.y1, self.y2),
            max(self.x1, self.x2), max(self.y1, self.y2),
        )

    def as_linestring(self) -> LineString:
        return LineString([(self.x1, self.y1), (self.x2, self.y2)])

    def as_rounded_endpoints(self, precision: int = 1) -> tuple:
        """Retourne les endpoints arrondis pour la construction du graphe."""
        p1 = (round(self.x1, precision), round(self.y1, precision))
        p2 = (round(self.x2, precision), round(self.y2, precision))
        return p1, p2


def extract_segments_from_page(page: fitz.Page) -> list[VectorSegment]:
    """
    Extrait tous les segments vectoriels d'une page PDF.
    Lignes, courbes (approximées), rectangles (décomposés en 4 côtés).
    
    Returns:
        Liste de VectorSegment.
    """
    paths = page.get_drawings()
    segments = []

    for path in paths:
        for item in path["items"]:
            if item[0] == "l":  # Ligne
                segments.append(VectorSegment(
                    item[1].x, item[1].y, item[2].x, item[2].y
                ))

            elif item[0] == "c":  # Courbe de Bézier → approximation multi-points
                # Une courbe de Bézier cubique a 4 points de contrôle.
                # Approximer avec N segments au lieu d'un seul (start→end)
                # pour capturer les cercles, arcs, et formes courbes.
                p0 = item[1]
                p1 = item[2]
                p2 = item[3]
                p3 = item[4]
                n_steps = 8  # 8 segments par courbe (suffisant pour les cercles)
                prev = p0
                for step in range(1, n_steps + 1):
                    t = step / n_steps
                    t2 = t * t
                    t3 = t2 * t
                    mt = 1 - t
                    mt2 = mt * mt
                    mt3 = mt2 * mt
                    # De Casteljau / formule explicite
                    bx = mt3*p0.x + 3*mt2*t*p1.x + 3*mt*t2*p2.x + t3*p3.x
                    by = mt3*p0.y + 3*mt2*t*p1.y + 3*mt*t2*p2.y + t3*p3.y
                    segments.append(VectorSegment(
                        prev.x, prev.y, bx, by
                    ))
                    # Créer un point temporaire pour le prochain segment
                    class _TmpPt:
                        def __init__(self, x, y):
                            self.x = x
                            self.y = y
                    prev = _TmpPt(bx, by)

            elif item[0] == "re":  # Rectangle → 4 segments
                r = item[1]
                pts = [
                    (r[0], r[1]), (r[2], r[1]),
                    (r[2], r[3]), (r[0], r[3]),
                ]
                for i in range(4):
                    sp1, sp2 = pts[i], pts[(i + 1) % 4]
                    segments.append(VectorSegment(
                        sp1[0], sp1[1], sp2[0], sp2[1]
                    ))

    return segments


def extract_text_blocks(page: fitz.Page) -> list[tuple]:
    """
    Extrait les blocs de texte d'une page PDF.
    
    Returns:
        Liste de tuples (x0, y0, x1, y1, text).
    """
    blocks = page.get_text("blocks")
    result = []
    for b in blocks:
        # b = (x0, y0, x1, y1, "text", block_no, block_type)
        if len(b) >= 5 and isinstance(b[4], str) and b[4].strip():
            result.append((b[0], b[1], b[2], b[3], b[4].strip()))
    return result
