"""
visualizer.py — Outil de QA visuel pour la pipeline hybride.

Affiche la page PDF avec les composants détectés en overlay,
colorés par catégorie et source (graphe vs DBSCAN), avec légende.
"""

import fitz
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Polygon
from typing import Optional, List, Tuple

from .classifier import DetectedComponent
from .pipeline import HybridPipeline
from .config import PipelineConfig


# Couleurs par catégorie
CATEGORY_COLORS = {
    "Component_Rect":    "#006400",  # Vert foncé
    "Component_Complex": "#00CED1",  # Cyan
    "Circle_Component":  "#FF00FF",  # Magenta
    "Hex_Symbol":        "#FF8C00",  # Orange
    "Busbar_Power":      "#90EE90",  # Vert clair
    "Group_Container":   "#4169E1",  # Bleu
    "Open_Component":    "#FFD700",  # Or (DBSCAN)
    "Unknown_Shape":     "#FF4444",  # Rouge
}

# Styles par source
SOURCE_STYLES = {
    "graph":  {"linestyle": "-",  "linewidth": 2.0},
    "dbscan": {"linestyle": "--", "linewidth": 1.5},
}


def visualize_page(
    pdf_path: str,
    components: List[DetectedComponent],
    page_index: int = 0,
    figsize: Tuple = (20, 14),
    show_ids: bool = False,
    show_metrics: bool = False,
    title: Optional[str] = None,
):
    """
    Affiche une page PDF avec les composants détectés en overlay.
    
    Args:
        pdf_path: Chemin du PDF.
        components: Composants détectés pour cette page.
        page_index: Index de la page à afficher.
        figsize: Taille de la figure.
        show_ids: Afficher les IDs des composants.
        show_metrics: Afficher G-ratio / D-ratio sur chaque composant.
        title: Titre custom (auto-généré si None).
    """
    doc = fitz.open(pdf_path)
    page = doc[page_index]

    # Rendre le fond à 72 DPI (1:1 avec les coordonnées PDF)
    pix = page.get_pixmap(dpi=72)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)

    # Compteurs par catégorie
    counts = {}

    for comp in components:
        color = CATEGORY_COLORS.get(comp.category, "#888888")
        style = SOURCE_STYLES.get(comp.source, SOURCE_STYLES["graph"])

        # Dessiner le polygone ou la bbox
        try:
            if comp.polygon and not comp.polygon.is_empty:
                x, y = comp.polygon.exterior.xy
                ax.fill(x, y, alpha=0.25, fc=color, zorder=2)
                ax.plot(x, y, color=color, zorder=3, **style)
        except Exception:
            # Fallback : dessiner la bbox
            x1, y1, x2, y2 = comp.bbox
            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=True, facecolor=color, alpha=0.25,
                edgecolor=color, linewidth=style["linewidth"],
                linestyle=style["linestyle"], zorder=2,
            )
            ax.add_patch(rect)

        # Label
        if show_ids or show_metrics:
            cx = (comp.bbox[0] + comp.bbox[2]) / 2
            cy = (comp.bbox[1] + comp.bbox[3]) / 2
            label_parts = []
            if show_ids:
                label_parts.append(f"#{comp.id}")
            if show_metrics:
                label_parts.append(f"G:{comp.g_ratio:.2f}")
                label_parts.append(f"D:{comp.d_ratio:.2f}")
            label_text = " ".join(label_parts)
            ax.text(
                cx, cy, label_text,
                fontsize=6, color="black", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7),
                zorder=4,
            )

        counts[comp.category] = counts.get(comp.category, 0) + 1

    # Légende
    legend_patches = []
    for cat, color in CATEGORY_COLORS.items():
        n = counts.get(cat, 0)
        if n > 0:
            legend_patches.append(
                mpatches.Patch(color=color, label=f"{cat} ({n})")
            )

    # Ajouter indicateur source
    legend_patches.append(
        mpatches.Patch(
            facecolor="white", edgecolor="black",
            linestyle="-", linewidth=2,
            label="── Graphe (fermé)",
        )
    )
    legend_patches.append(
        mpatches.Patch(
            facecolor="white", edgecolor="black",
            linestyle="--", linewidth=1.5,
            label="-- DBSCAN (ouvert)",
        )
    )

    ax.legend(handles=legend_patches, loc="upper right", fontsize=9)

    n_graph = sum(1 for c in components if c.source == "graph")
    n_dbscan = sum(1 for c in components if c.source == "dbscan")

    if title is None:
        title = (
            f"Pipeline Hybride — Page {page_index + 1} — "
            f"{len(components)} objets (Graph: {n_graph}, DBSCAN: {n_dbscan})"
        )
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    doc.close()


def quick_visualize(
    pdf_path: str,
    page_index: int = 0,
    config: Optional[PipelineConfig] = None,
    show_ids: bool = False,
    show_metrics: bool = False,
):
    """
    Raccourci : exécute la pipeline sur une page et affiche directement.
    Pratique pour le debug interactif.
    
    Usage:
        from hybrid_pipeline.visualizer import quick_visualize
        quick_visualize("schema.pdf", page_index=0)
    """
    pipeline = HybridPipeline(pdf_path, config)
    components = pipeline.process_page(page_index)

    print(f"Détecté : {len(components)} composants")
    cats = {}
    for c in components:
        cats[c.category] = cats.get(c.category, 0) + 1
    for cat, n in sorted(cats.items()):
        print(f"  • {cat}: {n}")

    visualize_page(
        pdf_path, components, page_index,
        show_ids=show_ids,
        show_metrics=show_metrics,
    )
