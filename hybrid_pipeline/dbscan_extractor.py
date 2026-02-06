"""
dbscan_extractor.py — Extraction de composants ouverts par clustering spatial.

Utilise DBSCAN sur les centres des segments vectoriels pour regrouper
les traits proches en composants (diodes, terres, symboles complexes).

Ce module est conçu comme FALLBACK : il s'exécute sur les segments
qui n'ont PAS été capturés par le graph_extractor (formes non fermées).
"""

import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import box, Point, Polygon
from shapely.ops import unary_union

from .config import DBSCANConfig, ClassifierConfig
from .vector_utils import VectorSegment


def filter_segments_by_length(
    segments: list[VectorSegment],
    max_length: float,
) -> list[VectorSegment]:
    """Garde uniquement les segments courts (< max_length)."""
    return [s for s in segments if s.length < max_length]


def remove_already_captured(
    segments: list[VectorSegment],
    captured_polygons: list[Polygon],
    buffer: float = 2.0,
) -> list[VectorSegment]:
    """
    Retire les segments dont le centre tombe à l'intérieur d'un polygone
    déjà détecté par le graph_extractor.
    
    Cela évite que DBSCAN re-détecte les mêmes composants.
    
    Args:
        segments: Tous les segments courts.
        captured_polygons: Polygones détectés par le graphe.
        buffer: Marge autour des polygones pour être sûr de capturer les bords.
    
    Returns:
        Segments non capturés (orphelins).
    """
    if not captured_polygons:
        return segments

    # Fusionner tous les polygones capturés en une seule géométrie
    captured_zone = unary_union([p.buffer(buffer) for p in captured_polygons])

    orphans = []
    for seg in segments:
        cx, cy = seg.center
        if not captured_zone.contains(Point(cx, cy)):
            orphans.append(seg)

    return orphans


def cluster_segments(
    segments: list[VectorSegment],
    config: DBSCANConfig,
) -> list[tuple]:
    """
    Applique DBSCAN sur les centres des segments.
    
    Returns:
        Liste de bounding boxes (x1, y1, x2, y2) pour chaque cluster,
        avec le nombre de segments dans le cluster.
    """
    if not segments:
        return []

    centers = np.array([seg.center for seg in segments])

    clustering = DBSCAN(
        eps=config.epsilon,
        min_samples=config.min_samples,
    ).fit(centers)

    labels = clustering.labels_

    clusters = []
    unique_labels = set(labels)

    for label_id in unique_labels:
        if label_id == -1:
            continue  # Bruit

        indices = np.where(labels == label_id)[0]
        cluster_segs = [segments[i] for i in indices]

        # Calculer la BBox globale du cluster
        bboxes = [s.bbox for s in cluster_segs]
        g_x1 = min(b[0] for b in bboxes)
        g_y1 = min(b[1] for b in bboxes)
        g_x2 = max(b[2] for b in bboxes)
        g_y2 = max(b[3] for b in bboxes)

        width = g_x2 - g_x1
        height = g_y2 - g_y1

        # Filtrer les clusters trop gros (harnais de câbles fusionnés)
        if width > config.max_cluster_size or height > config.max_cluster_size:
            continue

        # Filtrer les clusters trop petits (bruit résiduel)
        if width < config.min_cluster_size and height < config.min_cluster_size:
            continue

        clusters.append((g_x1, g_y1, g_x2, g_y2, len(cluster_segs)))

    return clusters


def clusters_to_polygons(
    clusters: list[tuple],
    cls_config: ClassifierConfig,
) -> list[Polygon]:
    """
    Convertit les clusters (bboxes) en polygones Shapely.
    Filtre par aire min/max.
    """
    polygons = []
    for (x1, y1, x2, y2, n_segs) in clusters:
        poly = box(x1, y1, x2, y2)
        if poly.area < cls_config.min_area or poly.area > cls_config.max_area:
            continue
        polygons.append(poly)
    return polygons


def run_dbscan_extraction(
    segments: list[VectorSegment],
    captured_polygons: list[Polygon],
    dbscan_config: DBSCANConfig,
    cls_config: ClassifierConfig,
) -> list[Polygon]:
    """
    Pipeline complète d'extraction par DBSCAN.
    
    Args:
        segments: Tous les segments de la page.
        captured_polygons: Polygones déjà trouvés par le graphe (pour déduplication).
        dbscan_config: Configuration DBSCAN.
        cls_config: Configuration de classification (pour seuils d'aire).
    
    Returns:
        Liste de Polygones (bboxes) des composants ouverts détectés.
    """
    # 1. Ne garder que les segments courts
    short_segs = filter_segments_by_length(segments, dbscan_config.max_segment_length)

    # 2. Retirer les segments déjà capturés par le graphe
    orphan_segs = remove_already_captured(short_segs, captured_polygons)

    if not orphan_segs:
        return []

    # 3. Clustering DBSCAN
    clusters = cluster_segments(orphan_segs, dbscan_config)

    # 4. Conversion en polygones
    polygons = clusters_to_polygons(clusters, cls_config)

    return polygons
