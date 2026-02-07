"""
pipeline.py — Orchestrateur principal de la pipeline hybride.

Enchaîne les étapes :
  1. Extraction vectorielle (segments) depuis le PDF.
  2. Graph Extractor → composants fermés (cycles + Node Degree Filter).
  3. DBSCAN Extractor → composants ouverts (formes non fermées, orphelins).
  4. Déduplication (IoU) entre les deux sources.
  5. Classification (arbre de décision géométrique).
  6. Export (crops PNG + JSON métadonnées + YOLO labels optionnel).

Usage:
    from hybrid_pipeline import HybridPipeline, PipelineConfig
    
    pipeline = HybridPipeline("schema.pdf")
    pipeline.run("output_dataset/")
"""

import fitz
import os
import time
from shapely.geometry import Polygon, box as shp_box
from shapely.ops import unary_union
from typing import Optional, List, Dict


def _log(msg: str) -> None:
    """Print with immediate flush for visibility during long operations."""
    print(msg, flush=True)

from .config import PipelineConfig
from .vector_utils import extract_segments_from_page, extract_text_blocks
from .wire_filter import remove_wires
from .graph_extractor import run_graph_extraction
from .dbscan_extractor import run_dbscan_extraction
from .classifier import classify_all, classify_polygon, compute_metrics, DetectedComponent
from .exporter import (
    render_page_image,
    export_crops,
    export_metadata,
    export_yolo_labels,
)


def compute_iou(poly_a: Polygon, poly_b: Polygon) -> float:
    """Calcule l'Intersection over Union entre deux polygones."""
    try:
        if not poly_a.intersects(poly_b):
            return 0.0
        inter = poly_a.intersection(poly_b).area
        union = poly_a.area + poly_b.area - inter
        if union <= 0:
            return 0.0
        return inter / union
    except Exception:
        return 0.0


def compute_containment(inner: Polygon, outer: Polygon) -> float:
    """
    Calcule le ratio de 'inner' contenu dans 'outer'.
    Returns: fraction de l'aire de inner qui est dans outer (0.0 à 1.0).
    """
    try:
        if inner.area <= 0:
            return 0.0
        if not inner.intersects(outer):
            return 0.0
        inter = inner.intersection(outer).area
        return inter / inner.area
    except Exception:
        return 0.0


def proximity_merge_components(
    components: List[DetectedComponent],
    radius: float = 8.0,
) -> List[DetectedComponent]:
    """
    Fusionne les composants proches en un seul (bounding box englobante).
    
    Cas d'usage principal : un cercle (Graph) connecté à un symbole de terre
    (DBSCAN) forment ensemble un seul composant électrique.
    
    Algorithme :
      1. Pour chaque composant, créer un buffer de `radius` pts.
      2. Si deux composants (Graph+DBSCAN ou DBSCAN+DBSCAN) se touchent
         après buffer ET au moins un est "petit" (Open_Component ou Circle),
         les fusionner en un seul composant avec bbox englobante.
      3. N'absorbe pas les gros composants rectangulaires.
    
    On ne fusionne PAS deux Graph rectangles — le smart_merge s'en occupe.
    """
    if len(components) <= 1:
        return components

    n = len(components)
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

    # Categories that can participate in proximity merge
    mergeable_cats = {
        "Open_Component", "Circle_Component", "Unknown_Shape", "Hex_Symbol",
    }

    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = components[i], components[j]

            # At least one component must be small/open/circle
            i_mergeable = ci.category in mergeable_cats
            j_mergeable = cj.category in mergeable_cats
            if not (i_mergeable or j_mergeable):
                continue

            # Don't merge two large Graph rectangles
            if (ci.source == "graph" and cj.source == "graph"
                    and ci.category == "Component_Rect"
                    and cj.category == "Component_Rect"):
                continue

            # Check proximity: buffer the smaller polygon
            try:
                if ci.polygon.buffer(radius).intersects(cj.polygon):
                    # Don't merge if both are big (area > 2000)
                    if ci.polygon.area > 2000 and cj.polygon.area > 2000:
                        if not i_mergeable and not j_mergeable:
                            continue
                    union(i, j)
            except Exception:
                continue

    # Group by root
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    result = []
    for indices in groups.values():
        if len(indices) == 1:
            result.append(components[indices[0]])
            continue

        # Merge all polygons in the group into a bounding box
        group_comps = [components[i] for i in indices]
        all_x1 = min(c.bbox[0] for c in group_comps)
        all_y1 = min(c.bbox[1] for c in group_comps)
        all_x2 = max(c.bbox[2] for c in group_comps)
        all_y2 = max(c.bbox[3] for c in group_comps)

        merged_poly = shp_box(all_x1, all_y1, all_x2, all_y2)

        # Pick the "best" category from the group
        # Priority: Component_Rect > Circle_Component > Hex_Symbol > Open_Component > Unknown
        cat_priority = {
            "Component_Rect": 6, "Component_Complex": 5,
            "Circle_Component": 4, "Hex_Symbol": 3,
            "Busbar_Power": 2, "Open_Component": 1,
            "Group_Container": 0, "Unknown_Shape": -1,
        }
        best_comp = max(group_comps, key=lambda c: cat_priority.get(c.category, -2))

        # Reclassify if merged shape is significantly different
        merged_m = compute_metrics(merged_poly)

        merged_component = DetectedComponent(
            id=best_comp.id,
            category=best_comp.category,
            bbox=(all_x1, all_y1, all_x2, all_y2),
            polygon=merged_poly,
            source="merged",
            thickness=merged_m["thickness"],
            circularity=merged_m["circularity"],
            g_ratio=merged_m["g_ratio"],
            d_ratio=merged_m["d_ratio"],
            page_index=best_comp.page_index,
        )
        result.append(merged_component)

    return result


def post_classification_cleanup(
    components: List[DetectedComponent],
    containment_threshold: float = 0.8,
) -> List[DetectedComponent]:
    """
    Nettoyage post-classification pour corriger les erreurs résiduelles.
    
    Règles appliquées :
      1. Containment : si un composant est contenu à > seuil dans un autre
         plus grand, supprimer le plus petit (il fait partie du grand).
      2. Wire-in-component : si un composant allongé (aspect ratio élevé)
         est partiellement dans un composant rectangulaire, le supprimer.
      3. Duplicate grounds : si deux composants de même catégorie se
         chevauchent fortement (IoU > 0.5), garder le mieux scoré.
    """
    if len(components) <= 1:
        return components

    to_remove = set()

    # Trier par aire décroissante (les gros composants ont priorité)
    sorted_comps = sorted(components, key=lambda c: c.polygon.area, reverse=True)

    for i in range(len(sorted_comps)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(sorted_comps)):
            if j in to_remove:
                continue

            big = sorted_comps[i]
            small = sorted_comps[j]

            # Règle 1 : Containment — le petit est "avalé" par le grand
            cont = compute_containment(small.polygon, big.polygon)
            if cont > containment_threshold:
                # Le petit est presque entièrement dans le grand → supprimer le petit
                to_remove.add(j)
                continue

            # Règle 2 : Forte IoU entre deux composants de même catégorie
            iou = compute_iou(big.polygon, small.polygon)
            if iou > 0.5 and big.category == small.category:
                # Doublon probable → garder le plus grand
                to_remove.add(j)
                continue

            # Règle 3 : Composant fin qui chevauche un composant normal
            #   (ex: un fil polygonisé qui traverse un composant)
            if small.thickness > 0 and small.g_ratio > 0:
                box_rot = small.polygon.buffer(0).minimum_rotated_rectangle
                if not box_rot.is_empty:
                    bx, by = box_rot.exterior.coords.xy
                    from shapely.geometry import Point as _Pt
                    e1 = _Pt(bx[0], by[0]).distance(_Pt(bx[1], by[1]))
                    e2 = _Pt(bx[1], by[1]).distance(_Pt(bx[2], by[2]))
                    long_s = max(e1, e2)
                    short_s = min(e1, e2)
                    if short_s > 0 and long_s / short_s > 6.0:
                        # Forme très allongée qui chevauche un composant → fil
                        if cont > 0.3:
                            to_remove.add(j)

    # Reconstruire la liste sans les éléments supprimés
    cleaned = [c for idx, c in enumerate(sorted_comps) if idx not in to_remove]
    return cleaned


def deduplicate(
    graph_components: List[DetectedComponent],
    dbscan_components: List[DetectedComponent],
    iou_threshold: float = 0.3,
) -> List[DetectedComponent]:
    """
    Fusionne les résultats Graph + DBSCAN en supprimant les doublons.
    
    Priorité au Graph (meilleure géométrie). Si un composant DBSCAN
    chevauche un composant Graph avec IoU > seuil, il est rejeté.
    """
    # On garde tous les composants Graph
    merged = list(graph_components)

    for db_comp in dbscan_components:
        is_duplicate = False
        for gr_comp in graph_components:
            iou = compute_iou(db_comp.polygon, gr_comp.polygon)
            if iou > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            merged.append(db_comp)

    return merged


class HybridPipeline:
    """
    Pipeline hybride d'extraction de composants électriques depuis PDF.
    
    Combine Graph (NetworkX) + DBSCAN pour couvrir les formes fermées
    et ouvertes, avec classification géométrique et export multi-format.
    """

    def __init__(self, pdf_path: str, config: Optional[PipelineConfig] = None):
        """
        Args:
            pdf_path: Chemin vers le fichier PDF.
            config: Configuration (défaut si None).
        """
        self.pdf_path = pdf_path
        self.config = config or PipelineConfig()
        self.doc = fitz.open(pdf_path)
        self.file_base = os.path.splitext(os.path.basename(pdf_path))[0]

    def process_page(self, page_index: int = 0, verbose: bool = True) -> List[DetectedComponent]:
        """
        Traite une page complète.
        
        Returns:
            Liste de DetectedComponent classifiés et dédupliqués.
        """
        page = self.doc[page_index]
        t0 = time.time()

        # ── 1. EXTRACTION VECTORIELLE ──
        if verbose:
            _log(f"    [1/6] Extraction des vecteurs...")
        segments = extract_segments_from_page(page)
        text_bboxes = extract_text_blocks(page)
        if verbose:
            _log(f"          → {len(segments)} segments, {len(text_bboxes)} textes ({time.time()-t0:.1f}s)")

        if not segments:
            return []

        # ── 1b. WIRE REMOVAL (suppression des fils) ──
        # Retire les segments de fils AVANT polygonize.
        # Élimine les faux polygones aux croisements et simplifie
        # drastiquement la détection de composants.
        wire_cfg = self.config.wire_filter
        if wire_cfg.enabled:
            if verbose:
                _log(f"    [1b/6] Filtrage des fils...")
            t1 = time.time()
            component_segments, wire_segments = remove_wires(
                segments,
                precision=self.config.graph.coord_precision,
                min_wire_length=wire_cfg.min_wire_length,
                collinear_tolerance_deg=wire_cfg.collinear_tolerance_deg,
                min_chain_length=wire_cfg.min_chain_length,
                verbose=verbose,
            )
            if verbose:
                _log(f"          → {len(wire_segments)} fils retirés, {len(component_segments)} segments restants ({time.time()-t1:.1f}s)")
        else:
            component_segments = segments
            wire_segments = []

        # ── 2. GRAPH EXTRACTOR (formes fermées) ──
        if verbose:
            _log(f"    [2/6] Graph extractor (polygonize)...")
            if len(component_segments) > 5000:
                _log(f"          ⚠️  {len(component_segments)} segments — cela peut prendre du temps!")
        t2 = time.time()
        graph_polygons = run_graph_extraction(
            segments=component_segments,
            text_bboxes=text_bboxes,
            graph_config=self.config.graph,
            cls_config=self.config.classifier,
            verbose=verbose,
        )
        if verbose:
            _log(f"          → {len(graph_polygons)} polygones fermés ({time.time()-t2:.1f}s)")

        # ── 3. DBSCAN EXTRACTOR (formes ouvertes / orphelins) ──
        # Feed component_segments (not all segments) — wires are excluded.
        if verbose:
            _log(f"    [3/6] DBSCAN extractor...")
        t3 = time.time()
        dbscan_polygons = run_dbscan_extraction(
            segments=component_segments,
            captured_polygons=graph_polygons,
            dbscan_config=self.config.dbscan,
            cls_config=self.config.classifier,
        )
        if verbose:
            _log(f"          → {len(dbscan_polygons)} clusters ouverts ({time.time()-t3:.1f}s)")

        # ── 4. CLASSIFICATION ──
        if verbose:
            _log(f"    [4/6] Classification...")
        t4 = time.time()
        graph_components = classify_all(
            graph_polygons,
            self.config.classifier,
            source="graph",
            page_index=page_index,
            id_offset=0,
        )

        dbscan_components = classify_all(
            dbscan_polygons,
            self.config.classifier,
            source="dbscan",
            page_index=page_index,
            id_offset=len(graph_components),
        )
        if verbose:
            _log(f"          → {len(graph_components)} graph + {len(dbscan_components)} dbscan ({time.time()-t4:.1f}s)")

        # ── 5. DÉDUPLICATION ──
        if verbose:
            _log(f"    [5/6] Déduplication & merge...")
        t5 = time.time()
        all_components = deduplicate(
            graph_components,
            dbscan_components,
            self.config.dedup_iou_threshold,
        )

        # ── 5b. PROXIMITY MERGE (Graph ↔ DBSCAN grouping) ──
        # Groups nearby components that form a single electrical symbol
        # (e.g., circle + ground, arrow + ground, etc.)
        all_components = proximity_merge_components(
            all_components,
            radius=self.config.proximity_merge_radius,
        )

        # ── 6. POST-CLASSIFICATION CLEANUP ──
        if verbose:
            _log(f"    [6/6] Post-cleanup...")
        all_components = post_classification_cleanup(
            all_components,
            self.config.containment_threshold,
        )
        if verbose:
            _log(f"          → {len(all_components)} composants finaux ({time.time()-t5:.1f}s)")
            _log(f"          Total page: {time.time()-t0:.1f}s")

        return all_components

    def run(
        self,
        output_dir: str,
        pages: Optional[List[int]] = None,
        export_crops_flag: bool = True,
        export_json: bool = True,
        export_yolo: bool = False,
    ) -> Dict[int, List[DetectedComponent]]:
        """
        Exécute la pipeline complète sur le PDF.
        
        Args:
            output_dir: Dossier de sortie racine.
            pages: Liste d'indices de pages à traiter (None = toutes).
            export_crops_flag: Sauvegarder les crops PNG.
            export_json: Sauvegarder les métadonnées JSON.
            export_yolo: Sauvegarder les labels YOLO.
        
        Returns:
            Dict {page_index: [DetectedComponent, ...]}.
        """
        os.makedirs(output_dir, exist_ok=True)

        if pages is None:
            pages = list(range(len(self.doc)))

        all_results: Dict[int, List[DetectedComponent]] = {}
        total_crops = 0

        for page_idx in pages:
            _log(f"\n[Page {page_idx + 1}/{len(self.doc)}] Extraction...")

            # Détection
            components = self.process_page(page_idx, verbose=True)
            all_results[page_idx] = components

            n_graph = sum(1 for c in components if c.source == "graph")
            n_dbscan = sum(1 for c in components if c.source == "dbscan")
            _log(f"  → {len(components)} composants "
                  f"(Graph: {n_graph}, DBSCAN: {n_dbscan})")

            # Stats par catégorie
            cats = {}
            for c in components:
                cats[c.category] = cats.get(c.category, 0) + 1
            for cat, n in sorted(cats.items()):
                _log(f"    • {cat}: {n}")

            # Export crops
            if export_crops_flag and components:
                _log(f"  Rendering crops...")
                page = self.doc[page_idx]
                img = render_page_image(page, self.config.export.dpi)
                crops_dir = os.path.join(output_dir, "crops")
                n = export_crops(
                    components, img, crops_dir,
                    self.config.export, self.file_base,
                )
                total_crops += n
                _log(f"  → {n} crops sauvegardés")

            # Export YOLO labels
            if export_yolo and components:
                page = self.doc[page_idx]
                yolo_dir = os.path.join(output_dir, "labels")
                yolo_path = os.path.join(
                    yolo_dir, f"{self.file_base}_p{page_idx}.txt"
                )
                export_yolo_labels(
                    components,
                    page.rect.width,
                    page.rect.height,
                    yolo_path,
                )

        # Export JSON global
        if export_json:
            json_path = os.path.join(output_dir, f"{self.file_base}_metadata.json")
            export_metadata(all_results, json_path, self.pdf_path)
            _log(f"\n✅ Métadonnées → {json_path}")

        _log(f"\n🏁 Pipeline terminée : {total_crops} crops total "
              f"sur {len(pages)} page(s).")

        return all_results
