"""
exporter.py — Export des composants détectés en crops PNG et métadonnées JSON.

Prend les DetectedComponent de classifier.py et produit :
  - Des images PNG découpées (crops) organisées par catégorie.
  - Un fichier JSON de métadonnées (bboxes, catégories, métriques).
"""

import fitz
import cv2
import json
import numpy as np
import os

from .config import ExportConfig
from .classifier import DetectedComponent


def render_page_image(page: fitz.Page, dpi: int = 300) -> np.ndarray:
    """
    Rend une page PDF en image BGR haute résolution (OpenCV format).
    """
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


def export_crops(
    components: list[DetectedComponent],
    page_image: np.ndarray,
    output_root: str,
    config: ExportConfig,
    file_base: str = "doc",
) -> int:
    """
    Découpe et sauvegarde les crops de chaque composant.
    
    Args:
        components: Composants détectés pour cette page.
        page_image: Image BGR haute résolution de la page.
        output_root: Dossier racine de sortie.
        config: Configuration d'export.
        file_base: Nom de base du fichier source (sans extension).
    
    Returns:
        Nombre de crops sauvegardés.
    """
    img_h, img_w = page_image.shape[:2]
    count = 0

    for comp in components:
        x1, y1, x2, y2 = comp.bbox

        # Convertir coordonnées PDF (72 DPI) → pixels image (DPI configuré)
        px1 = int(x1 * config.scale_factor) - config.padding
        py1 = int(y1 * config.scale_factor) - config.padding
        px2 = int(x2 * config.scale_factor) + config.padding
        py2 = int(y2 * config.scale_factor) + config.padding

        # Bornes de sécurité
        px1 = max(0, px1)
        py1 = max(0, py1)
        px2 = min(img_w, px2)
        py2 = min(img_h, py2)

        if px2 <= px1 or py2 <= py1:
            continue

        crop = page_image[py1:py2, px1:px2]
        if crop.size == 0:
            continue

        # Sauvegarder dans un sous-dossier par catégorie
        save_dir = os.path.join(output_root, comp.category)
        os.makedirs(save_dir, exist_ok=True)

        fname = f"{file_base}_p{comp.page_index}_id{comp.id}.png"
        cv2.imwrite(os.path.join(save_dir, fname), crop)
        count += 1

    return count


def export_metadata(
    all_components: dict[int, list[DetectedComponent]],
    output_path: str,
    source_file: str,
) -> None:
    """
    Exporte les métadonnées de tous les composants en JSON.
    
    Args:
        all_components: Dict {page_index: [DetectedComponent, ...]}.
        output_path: Chemin du fichier JSON de sortie.
        source_file: Nom du fichier PDF source.
    """
    data = {
        "source_file": os.path.basename(source_file),
        "pipeline": "hybrid_v1",
        "pages": [],
    }

    for page_idx in sorted(all_components.keys()):
        comps = all_components[page_idx]
        page_data = {
            "page_index": page_idx,
            "total_objects": len(comps),
            "by_source": {
                "graph": sum(1 for c in comps if c.source == "graph"),
                "dbscan": sum(1 for c in comps if c.source == "dbscan"),
            },
            "objects": [c.to_dict() for c in comps],
        }
        data["pages"].append(page_data)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def export_yolo_labels(
    components: list[DetectedComponent],
    page_width: float,
    page_height: float,
    output_path: str,
    category_map: dict[str, int] | None = None,
) -> None:
    """
    Exporte les annotations au format YOLO (txt).
    Chaque ligne : class_id center_x center_y width height (normalisés 0-1).
    
    Args:
        components: Composants de la page.
        page_width: Largeur de la page en points PDF.
        page_height: Hauteur de la page en points PDF.
        output_path: Chemin du fichier .txt YOLO.
        category_map: Mapping catégorie → class_id. Si None, auto-généré.
    """
    if category_map is None:
        # Mapping par défaut
        category_map = {
            "Component_Rect": 0,
            "Component_Complex": 1,
            "Circle_Component": 2,
            "Hex_Symbol": 3,
            "Busbar_Power": 4,
            "Group_Container": 5,
            "Open_Component": 6,
            "Unknown_Shape": 7,
        }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as f:
        for comp in components:
            class_id = category_map.get(comp.category)
            if class_id is None:
                continue

            x1, y1, x2, y2 = comp.bbox
            cx = ((x1 + x2) / 2) / page_width
            cy = ((y1 + y2) / 2) / page_height
            w = (x2 - x1) / page_width
            h = (y2 - y1) / page_height

            # Clamp to [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))

            f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
