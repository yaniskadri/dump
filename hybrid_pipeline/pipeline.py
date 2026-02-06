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
from shapely.geometry import Polygon

from .config import PipelineConfig
from .vector_utils import extract_segments_from_page, extract_text_blocks
from .graph_extractor import run_graph_extraction
from .dbscan_extractor import run_dbscan_extraction
from .classifier import classify_all, DetectedComponent
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


def deduplicate(
    graph_components: list[DetectedComponent],
    dbscan_components: list[DetectedComponent],
    iou_threshold: float = 0.3,
) -> list[DetectedComponent]:
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

    def __init__(self, pdf_path: str, config: PipelineConfig | None = None):
        """
        Args:
            pdf_path: Chemin vers le fichier PDF.
            config: Configuration (défaut si None).
        """
        self.pdf_path = pdf_path
        self.config = config or PipelineConfig()
        self.doc = fitz.open(pdf_path)
        self.file_base = os.path.splitext(os.path.basename(pdf_path))[0]

    def process_page(self, page_index: int = 0) -> list[DetectedComponent]:
        """
        Traite une page complète.
        
        Returns:
            Liste de DetectedComponent classifiés et dédupliqués.
        """
        page = self.doc[page_index]

        # ── 1. EXTRACTION VECTORIELLE ──
        segments = extract_segments_from_page(page)
        text_bboxes = extract_text_blocks(page)

        if not segments:
            return []

        # ── 2. GRAPH EXTRACTOR (formes fermées) ──
        graph_polygons = run_graph_extraction(
            segments=segments,
            text_bboxes=text_bboxes,
            graph_config=self.config.graph,
            cls_config=self.config.classifier,
        )

        # ── 3. DBSCAN EXTRACTOR (formes ouvertes / orphelins) ──
        dbscan_polygons = run_dbscan_extraction(
            segments=segments,
            captured_polygons=graph_polygons,
            dbscan_config=self.config.dbscan,
            cls_config=self.config.classifier,
        )

        # ── 4. CLASSIFICATION ──
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

        # ── 5. DÉDUPLICATION ──
        all_components = deduplicate(
            graph_components,
            dbscan_components,
            self.config.dedup_iou_threshold,
        )

        return all_components

    def run(
        self,
        output_dir: str,
        pages: list[int] | None = None,
        export_crops_flag: bool = True,
        export_json: bool = True,
        export_yolo: bool = False,
    ) -> dict[int, list[DetectedComponent]]:
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

        all_results: dict[int, list[DetectedComponent]] = {}
        total_crops = 0

        for page_idx in pages:
            print(f"[Page {page_idx + 1}/{len(self.doc)}] Extraction...")

            # Détection
            components = self.process_page(page_idx)
            all_results[page_idx] = components

            n_graph = sum(1 for c in components if c.source == "graph")
            n_dbscan = sum(1 for c in components if c.source == "dbscan")
            print(f"  → {len(components)} composants "
                  f"(Graph: {n_graph}, DBSCAN: {n_dbscan})")

            # Stats par catégorie
            cats = {}
            for c in components:
                cats[c.category] = cats.get(c.category, 0) + 1
            for cat, n in sorted(cats.items()):
                print(f"    • {cat}: {n}")

            # Export crops
            if export_crops_flag and components:
                page = self.doc[page_idx]
                img = render_page_image(page, self.config.export.dpi)
                crops_dir = os.path.join(output_dir, "crops")
                n = export_crops(
                    components, img, crops_dir,
                    self.config.export, self.file_base,
                )
                total_crops += n
                print(f"  → {n} crops sauvegardés")

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
            print(f"\n✅ Métadonnées → {json_path}")

        print(f"\n🏁 Pipeline terminée : {total_crops} crops total "
              f"sur {len(pages)} page(s).")

        return all_results
