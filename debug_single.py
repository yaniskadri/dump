#!/usr/bin/env python3
"""
debug_single.py — Debug interactif de la pipeline hybride sur UN SEUL PDF.

Produit 6 visualisations détaillées + stats complètes pour ajuster les seuils.

Usage:
    python debug_single.py chemin/vers/schema.pdf
    python debug_single.py chemin/vers/schema.pdf --page 2
    python debug_single.py chemin/vers/schema.pdf --save debug_output/

Visualisations générées :
    1. Vue vectorielle brute (tous les segments extraits)
    2. Graphe NetworkX (nœuds colorés par degré)
    3. Cycles bruts vs filtrés (avant/après Node Degree Filter)
    4. DBSCAN clusters (orphelins regroupés)
    5. Pipeline finale (tous les composants classifiés)
    6. Histogrammes des métriques (épaisseur, G-ratio, D-ratio, circularité)
"""

import sys
import os
import argparse
import time

# Ajouter le parent au path pour l'import du package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fitz
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from shapely.geometry import box
from sklearn.cluster import DBSCAN

from hybrid_pipeline.config import (
    PipelineConfig, GraphConfig, DBSCANConfig, ClassifierConfig,
)
from hybrid_pipeline.vector_utils import extract_segments_from_page, extract_text_blocks
from hybrid_pipeline.graph_extractor import (
    build_graph, find_all_faces, get_node_degrees,
    filter_by_node_degree, filter_isolated_empty,
)
from hybrid_pipeline.dbscan_extractor import (
    filter_segments_by_length, remove_already_captured, cluster_segments,
)
from hybrid_pipeline.classifier import (
    classify_all, compute_metrics, classify_polygon, DetectedComponent,
)
from hybrid_pipeline.visualizer import CATEGORY_COLORS


def save_figure(fig, save_dir, fname, dpi=None):
    """Sauvegarde `fig` dans `save_dir/fname`, ajoute une entrée dans save_manifest.txt

    Retourne le chemin absolu sauvegardé.
    """
    if not save_dir:
        return None
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, fname)
    # Utiliser dpi si fourni
    if dpi:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    else:
        fig.savefig(path, bbox_inches="tight")
    manifest = os.path.join(save_dir, "save_manifest.txt")
    try:
        with open(manifest, "a", encoding="utf-8") as f:
            f.write(path + "\n")
    except Exception:
        pass
    print(f"💾 Saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════
#  UTILITAIRES
# ═══════════════════════════════════════════════════════════════

def get_page_background(page, dpi=72):
    """Rend la page PDF en array numpy (RGB) à la résolution donnée."""
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    # Convertir RGBA → RGB si nécessaire
    if pix.n == 4:
        img = img[:, :, :3]
    return img


def print_separator(title):
    """Affiche un séparateur dans le terminal."""
    w = 60
    print(f"\n{'═' * w}")
    print(f"  {title}")
    print(f"{'═' * w}")


def print_config(config: PipelineConfig):
    """Affiche la config actuelle pour référence."""
    print_separator("CONFIGURATION ACTIVE")
    print(f"  Graph:")
    print(f"    coord_precision    = {config.graph.coord_precision}")
    print(f"    max_cross_ratio    = {config.graph.max_cross_ratio}")
    print(f"    min_cycle_nodes    = {config.graph.min_cycle_nodes}")
    print(f"  DBSCAN:")
    print(f"    epsilon            = {config.dbscan.epsilon}")
    print(f"    min_samples        = {config.dbscan.min_samples}")
    print(f"    max_segment_length = {config.dbscan.max_segment_length}")
    print(f"    max_cluster_size   = {config.dbscan.max_cluster_size}")
    print(f"    min_cluster_size   = {config.dbscan.min_cluster_size}")
    print(f"  Classifier:")
    print(f"    thin_wire_threshold = {config.classifier.thin_wire_threshold}")
    print(f"    busbar_threshold    = {config.classifier.busbar_threshold}")
    print(f"    min_area / max_area = {config.classifier.min_area} / {config.classifier.max_area}")
    print(f"    rect_ratio (G)      = {config.classifier.rect_ratio_threshold}")
    print(f"    circle_threshold    = {config.classifier.circle_threshold}")
    print(f"    density_filled (D)  = {config.classifier.density_filled}")
    print(f"    density_empty       = {config.classifier.density_empty}")


# ═══════════════════════════════════════════════════════════════
#  VIZ 1 — SEGMENTS VECTORIELS BRUTS
# ═══════════════════════════════════════════════════════════════

def viz_raw_segments(ax, page, segments, text_bboxes):
    """Dessine tous les segments vectoriels extraits, colorés par longueur."""
    bg = get_page_background(page)
    ax.imshow(bg, alpha=0.3)

    lengths = [s.length for s in segments]
    if not lengths:
        ax.set_title("VIZ 1 — Segments bruts (AUCUN)")
        return

    max_len = max(lengths)
    cmap = plt.cm.plasma

    for seg in segments:
        ratio = seg.length / max_len if max_len > 0 else 0
        color = cmap(ratio)
        ax.plot(
            [seg.x1, seg.x2], [seg.y1, seg.y2],
            color=color, linewidth=0.5, alpha=0.7,
        )

    # Texte en bleu clair
    for (x0, y0, x1, y1, txt) in text_bboxes:
        rect = mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fill=True, facecolor="#ADD8E6", alpha=0.3,
            edgecolor="#4682B4", linewidth=0.5,
        )
        ax.add_patch(rect)

    ax.set_title(
        f"VIZ 1 — {len(segments)} segments bruts "
        f"(couleur = longueur, bleu = texte)\n"
        f"Min: {min(lengths):.1f}  Max: {max(lengths):.1f}  "
        f"Moy: {np.mean(lengths):.1f} pts",
        fontsize=10,
    )
    ax.set_xlim(0, page.rect.width)
    ax.set_ylim(page.rect.height, 0)
    ax.set_aspect("equal")


# ═══════════════════════════════════════════════════════════════
#  VIZ 2 — GRAPHE NETWORKX (NŒUDS PAR DEGRÉ)
# ═══════════════════════════════════════════════════════════════

def viz_graph_nodes(ax, page, G):
    """Affiche le graphe avec nœuds colorés par degré."""
    bg = get_page_background(page)
    ax.imshow(bg, alpha=0.2)

    if G.number_of_nodes() == 0:
        ax.set_title("VIZ 2 — Graphe (VIDE)")
        return

    # Dessiner les arêtes
    for (u, v) in G.edges():
        ax.plot([u[0], v[0]], [u[1], v[1]], color="#CCCCCC", linewidth=0.3, zorder=1)

    # Nœuds colorés par degré
    degree_colors = {1: "#999999", 2: "#00AA00", 3: "#FFA500", 4: "#FF0000"}
    degree_labels = {1: "Deg 1 (bout)", 2: "Deg 2 (coin ✓)", 3: "Deg 3 (T)", 4: "Deg ≥4 (X ✗)"}

    nodes_by_deg = {}
    for node in G.nodes():
        deg = min(G.degree[node], 4)  # Cap à 4 pour la couleur
        if deg not in nodes_by_deg:
            nodes_by_deg[deg] = []
        nodes_by_deg[deg].append(node)

    for deg in sorted(nodes_by_deg.keys()):
        nodes = nodes_by_deg[deg]
        xs = [n[0] for n in nodes]
        ys = [n[1] for n in nodes]
        color = degree_colors.get(deg, "#FF0000")
        ax.scatter(xs, ys, c=color, s=3 if deg <= 2 else 12,
                   zorder=2 + deg, alpha=0.8, label=f"{degree_labels.get(deg, f'Deg {deg}')} ({len(nodes)})")

    ax.legend(loc="upper right", fontsize=7, markerscale=2)
    ax.set_title(
        f"VIZ 2 — Graphe : {G.number_of_nodes()} nœuds, "
        f"{G.number_of_edges()} arêtes\n"
        f"🟢 Deg 2 = coins composants   🔴 Deg ≥4 = croisements fils",
        fontsize=10,
    )
    ax.set_xlim(0, page.rect.width)
    ax.set_ylim(page.rect.height, 0)
    ax.set_aspect("equal")


# ═══════════════════════════════════════════════════════════════
#  VIZ 3 — CYCLES : BRUTS vs FILTRÉS
# ═══════════════════════════════════════════════════════════════

def viz_faces_comparison(ax, page, all_faces, kept_faces, rejected_faces, config):
    """Montre les faces polygonize avant et après le Node Degree Filter."""
    bg = get_page_background(page)
    ax.imshow(bg, alpha=0.2)

    if not all_faces:
        ax.set_title("VIZ 3 — Faces polygonize (AUCUNE)")
        return

    # Dessiner les rejetées en rouge semi-transparent
    for face in rejected_faces:
        try:
            x, y = face.exterior.xy
            ax.fill(x, y, alpha=0.25, fc="#FF4444", ec="#CC0000", linewidth=0.5, zorder=2)
        except Exception:
            pass

    # Dessiner les gardées en vert
    for face in kept_faces:
        try:
            x, y = face.exterior.xy
            ax.fill(x, y, alpha=0.4, fc="#00CC00", ec="#006400", linewidth=1, zorder=3)
        except Exception:
            pass

    # Faces filtrées par aire (ni dans kept ni dans rejected)
    n_area_filtered = len(all_faces) - len(kept_faces) - len(rejected_faces)

    ax.set_title(
        f"VIZ 3 — Node Degree Filter (cross_ratio={config.graph.max_cross_ratio})\n"
        f"🟢 Gardés: {len(kept_faces)}   "
        f"🔴 Rejetés degré: {len(rejected_faces)}   "
        f"⚪ Filtrés aire: {n_area_filtered}   "
        f"Total polygonize: {len(all_faces)}",
        fontsize=10,
    )
    ax.set_xlim(0, page.rect.width)
    ax.set_ylim(page.rect.height, 0)
    ax.set_aspect("equal")


# ═══════════════════════════════════════════════════════════════
#  VIZ 4 — DBSCAN CLUSTERS (ORPHELINS)
# ═══════════════════════════════════════════════════════════════

def viz_dbscan_clusters(ax, page, segments, graph_polygons, config):
    """Montre les clusters DBSCAN sur les segments orphelins."""
    bg = get_page_background(page)
    ax.imshow(bg, alpha=0.3)

    # Reproduire les étapes DBSCAN
    short_segs = filter_segments_by_length(segments, config.dbscan.max_segment_length)
    orphans = remove_already_captured(short_segs, graph_polygons)

    if not orphans:
        ax.set_title("VIZ 4 — DBSCAN (aucun orphelin)")
        ax.set_xlim(0, page.rect.width)
        ax.set_ylim(page.rect.height, 0)
        ax.set_aspect("equal")
        return 0, 0, 0

    centers = np.array([s.center for s in orphans])
    clustering = DBSCAN(
        eps=config.dbscan.epsilon,
        min_samples=config.dbscan.min_samples,
    ).fit(centers)
    labels = clustering.labels_

    unique = set(labels)
    n_clusters = len(unique) - (1 if -1 in unique else 0)
    n_noise = int(np.sum(labels == -1))

    # Bruit en gris
    noise_mask = labels == -1
    if np.any(noise_mask):
        ax.scatter(
            centers[noise_mask, 0], centers[noise_mask, 1],
            c="#999999", s=1, alpha=0.3, zorder=1, label=f"Bruit ({n_noise})",
        )

    # Clusters colorés
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))
    kept = 0
    rejected_size = 0

    cluster_ids = sorted([l for l in unique if l != -1])
    for idx, label_id in enumerate(cluster_ids):
        mask = labels == label_id
        cluster_segs = [orphans[i] for i in np.where(mask)[0]]
        bboxes = [s.bbox for s in cluster_segs]
        x1 = min(b[0] for b in bboxes)
        y1 = min(b[1] for b in bboxes)
        x2 = max(b[2] for b in bboxes)
        y2 = max(b[3] for b in bboxes)
        w, h = x2 - x1, y2 - y1

        too_big = w > config.dbscan.max_cluster_size or h > config.dbscan.max_cluster_size
        too_small = w < config.dbscan.min_cluster_size and h < config.dbscan.min_cluster_size

        if too_big or too_small:
            ec = "#FF6666"
            alpha = 0.15
            rejected_size += 1
        else:
            ec = colors[idx % len(colors)]
            alpha = 0.3
            kept += 1

        rect = mpatches.Rectangle(
            (x1, y1), w, h,
            fill=True, facecolor=ec, alpha=alpha,
            edgecolor=ec, linewidth=1.5, zorder=2,
        )
        ax.add_patch(rect)
        ax.scatter(centers[mask, 0], centers[mask, 1], c=[ec], s=2, zorder=3)

    ax.legend(loc="upper right", fontsize=7)
    ax.set_title(
        f"VIZ 4 — DBSCAN (ε={config.dbscan.epsilon}, min_samples={config.dbscan.min_samples})\n"
        f"Orphelins: {len(orphans)}/{len(short_segs)} segs   "
        f"Clusters: {kept} gardés, {rejected_size} rejetés taille   "
        f"Bruit: {n_noise}",
        fontsize=10,
    )
    ax.set_xlim(0, page.rect.width)
    ax.set_ylim(page.rect.height, 0)
    ax.set_aspect("equal")

    return len(orphans), kept, n_noise


# ═══════════════════════════════════════════════════════════════
#  VIZ 5 — RÉSULTAT FINAL CLASSIFIÉ
# ═══════════════════════════════════════════════════════════════

def viz_final_result(ax, page, components):
    """Affiche le résultat final avec couleurs par catégorie et source."""
    bg = get_page_background(page)
    ax.imshow(bg, alpha=0.4)

    counts = {}
    for comp in components:
        color = CATEGORY_COLORS.get(comp.category, "#888888")
        ls = "-" if comp.source == "graph" else "--"
        lw = 2 if comp.source == "graph" else 1.5

        try:
            if comp.polygon and not comp.polygon.is_empty:
                x, y = comp.polygon.exterior.xy
                ax.fill(x, y, alpha=0.3, fc=color, zorder=2)
                ax.plot(x, y, color=color, linewidth=lw, linestyle=ls, zorder=3)
        except Exception:
            x1, y1, x2, y2 = comp.bbox
            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=True, facecolor=color, alpha=0.3,
                edgecolor=color, linewidth=lw, linestyle=ls, zorder=2,
            )
            ax.add_patch(rect)

        # ID + catégorie courte
        cx = (comp.bbox[0] + comp.bbox[2]) / 2
        cy = (comp.bbox[1] + comp.bbox[3]) / 2
        short_cat = comp.category.replace("Component_", "").replace("_", " ")[:8]
        ax.text(
            cx, cy, f"#{comp.id}\n{short_cat}",
            fontsize=5, color="black", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor=color, alpha=0.6),
            zorder=4,
        )

        counts[comp.category] = counts.get(comp.category, 0) + 1

    # Légende
    patches = []
    for cat, color in CATEGORY_COLORS.items():
        n = counts.get(cat, 0)
        if n > 0:
            patches.append(mpatches.Patch(color=color, label=f"{cat} ({n})"))
    if patches:
        ax.legend(handles=patches, loc="upper right", fontsize=7)

    n_gr = sum(1 for c in components if c.source == "graph")
    n_db = sum(1 for c in components if c.source == "dbscan")
    ax.set_title(
        f"VIZ 5 — Résultat final : {len(components)} composants\n"
        f"Graph (trait plein): {n_gr}   DBSCAN (pointillé): {n_db}",
        fontsize=10,
    )
    ax.set_xlim(0, page.rect.width)
    ax.set_ylim(page.rect.height, 0)
    ax.set_aspect("equal")


# ═══════════════════════════════════════════════════════════════
#  VIZ 6 — HISTOGRAMMES DES MÉTRIQUES
# ═══════════════════════════════════════════════════════════════

def viz_histograms(fig, gs_row, all_polygons, config):
    """
    4 histogrammes : épaisseur, G-ratio, D-ratio, circularité.
    Avec les seuils de décision en lignes verticales rouges.
    """
    metrics_list = []
    for poly in all_polygons:
        m = compute_metrics(poly)
        if m["thickness"] > 0:
            metrics_list.append(m)

    if not metrics_list:
        ax = fig.add_subplot(gs_row[0])
        ax.text(0.5, 0.5, "Aucune métrique calculable", ha="center", va="center")
        return

    thicknesses = [m["thickness"] for m in metrics_list]
    g_ratios = [m["g_ratio"] for m in metrics_list]
    d_ratios = [m["d_ratio"] for m in metrics_list]
    circularities = [m["circularity"] for m in metrics_list]

    cls = config.classifier

    # Sous-plots pour les histogrammes
    ax1 = fig.add_subplot(gs_row[0])
    ax2 = fig.add_subplot(gs_row[1])
    ax3 = fig.add_subplot(gs_row[2])
    ax4 = fig.add_subplot(gs_row[3])

    # ── Épaisseur ──
    ax1.hist(thicknesses, bins=50, color="#4ECDC4", edgecolor="white", alpha=0.8)
    ax1.axvline(cls.thin_wire_threshold, color="red", linestyle="--", linewidth=1.5,
                label=f"Fil ({cls.thin_wire_threshold})")
    ax1.axvline(cls.busbar_threshold, color="orange", linestyle="--", linewidth=1.5,
                label=f"Busbar ({cls.busbar_threshold})")
    ax1.set_title("Épaisseur (pts PDF)", fontsize=9)
    ax1.set_xlabel("px")
    ax1.legend(fontsize=7)

    # ── G-ratio (rectangularité) ──
    ax2.hist(g_ratios, bins=50, color="#45B7D1", edgecolor="white", alpha=0.8)
    ax2.axvline(cls.rect_ratio_threshold, color="red", linestyle="--", linewidth=1.5,
                label=f"Rect ({cls.rect_ratio_threshold})")
    hex_lo, hex_hi = cls.hex_rect_range
    ax2.axvspan(hex_lo, hex_hi, alpha=0.15, color="orange", label=f"Hex [{hex_lo}-{hex_hi}]")
    ax2.set_title("G-ratio (rectangularité)", fontsize=9)
    ax2.set_xlabel("ratio")
    ax2.legend(fontsize=7)

    # ── D-ratio (densité) ──
    ax3.hist(d_ratios, bins=50, color="#96CEB4", edgecolor="white", alpha=0.8)
    ax3.axvline(cls.density_empty, color="gray", linestyle="--", linewidth=1.5,
                label=f"Vide ({cls.density_empty})")
    ax3.axvline(cls.density_busbar_min, color="orange", linestyle="--", linewidth=1.5,
                label=f"Busbar min ({cls.density_busbar_min})")
    ax3.axvline(cls.density_filled, color="green", linestyle="--", linewidth=1.5,
                label=f"Plein ({cls.density_filled})")
    ax3.set_title("D-ratio (densité)", fontsize=9)
    ax3.set_xlabel("ratio")
    ax3.legend(fontsize=7)

    # ── Circularité ──
    ax4.hist(circularities, bins=50, color="#FFEAA7", edgecolor="white", alpha=0.8)
    ax4.axvline(cls.circle_threshold, color="magenta", linestyle="--", linewidth=1.5,
                label=f"Cercle ({cls.circle_threshold})")
    ax4.axvline(cls.hex_circ_min, color="orange", linestyle="--", linewidth=1.5,
                label=f"Hex min ({cls.hex_circ_min})")
    ax4.set_title("Circularité (4πA/P²)", fontsize=9)
    ax4.set_xlabel("ratio")
    ax4.legend(fontsize=7)


# ═══════════════════════════════════════════════════════════════
#  VIZ BONUS — SCATTER 2D DES MÉTRIQUES
# ═══════════════════════════════════════════════════════════════

def viz_scatter_metrics(fig, gs_slot, components):
    """Scatter plot G-ratio vs D-ratio, coloré par catégorie."""
    ax = fig.add_subplot(gs_slot)

    if not components:
        ax.text(0.5, 0.5, "Aucun composant", ha="center", va="center")
        return

    for comp in components:
        color = CATEGORY_COLORS.get(comp.category, "#888888")
        marker = "o" if comp.source == "graph" else "^"
        ax.scatter(
            comp.g_ratio, comp.d_ratio,
            c=color, marker=marker, s=30, alpha=0.7,
            edgecolors="black", linewidths=0.3,
        )

    # Zones de décision
    ax.axvline(0.70, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(0.80, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(0.25, color="gray", linestyle=":", alpha=0.3)
    ax.axhline(0.50, color="gray", linestyle=":", alpha=0.3)

    # Annotations des zones
    ax.text(0.85, 0.9, "Comp Rect", fontsize=7, ha="center", color="#006400", alpha=0.7)
    ax.text(0.85, 0.5, "Group", fontsize=7, ha="center", color="#4169E1", alpha=0.7)
    ax.text(0.85, 0.1, "Layout ✗", fontsize=7, ha="center", color="gray", alpha=0.7)
    ax.text(0.35, 0.85, "Complexe", fontsize=7, ha="center", color="#00CED1", alpha=0.7)

    ax.set_xlabel("G-ratio (rectangularité)")
    ax.set_ylabel("D-ratio (densité)")
    ax.set_title("Scatter G vs D — ● Graph  ▲ DBSCAN", fontsize=9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.2)


# ═══════════════════════════════════════════════════════════════
#  STATS TEXTUELLES
# ═══════════════════════════════════════════════════════════════

def print_detailed_stats(components, segments, G, raw_cycle_count, valid_cycle_count,
                         orphan_count, dbscan_kept, elapsed):
    """Affiche un rapport de stats complet dans le terminal."""

    print_separator("STATISTIQUES DÉTAILLÉES")

    print(f"\n⏱  Temps d'exécution : {elapsed:.2f}s")

    print(f"\n📐 Segments vectoriels : {len(segments)}")
    lengths = [s.length for s in segments]
    if lengths:
        print(f"   Longueur min/max/moy : {min(lengths):.1f} / {max(lengths):.1f} / {np.mean(lengths):.1f}")

    print(f"\n🔗 Graphe NetworkX :")
    print(f"   Nœuds : {G.number_of_nodes()}")
    print(f"   Arêtes : {G.number_of_edges()}")
    deg_counts = {}
    for node in G.nodes():
        d = G.degree[node]
        d_key = min(d, 5)
        deg_counts[d_key] = deg_counts.get(d_key, 0) + 1
    for d in sorted(deg_counts):
        label = {1: "bout", 2: "coin ✓", 3: "T-junction", 4: "croisement ✗"}.get(d, f"≥{d}")
        print(f"   Degré {d} ({label}) : {deg_counts[d]}")

    print(f"\n🔄 Faces (polygonize) :")
    print(f"   Brutes (polygonize) : {raw_cycle_count}")
    print(f"   Après Node Degree Filter : {valid_cycle_count}")
    print(f"   Rejetés (croisements) : {raw_cycle_count - valid_cycle_count}")
    if raw_cycle_count > 0:
        print(f"   Taux de rejet : {(raw_cycle_count - valid_cycle_count) / raw_cycle_count * 100:.1f}%")

    print(f"\n🔬 DBSCAN :")
    print(f"   Segments orphelins : {orphan_count}")
    print(f"   Clusters gardés : {dbscan_kept}")

    print(f"\n🏷  Classification finale : {len(components)} composants")
    cats = {}
    sources = {"graph": 0, "dbscan": 0}
    for c in components:
        cats[c.category] = cats.get(c.category, 0) + 1
        sources[c.source] = sources.get(c.source, 0) + 1

    for cat in sorted(cats):
        color_indicator = "🟢" if "Component" in cat else "🟡" if "Busbar" in cat or "Open" in cat else "🔵" if "Group" in cat else "🔴"
        print(f"   {color_indicator} {cat:25s} : {cats[cat]}")

    print(f"\n   Source Graph  : {sources['graph']}")
    print(f"   Source DBSCAN : {sources['dbscan']}")

    # Métriques des composants graph
    graph_comps = [c for c in components if c.source == "graph"]
    if graph_comps:
        print(f"\n📊 Métriques (composants Graph) :")
        thick = [c.thickness for c in graph_comps]
        g_vals = [c.g_ratio for c in graph_comps]
        d_vals = [c.d_ratio for c in graph_comps]
        print(f"   Épaisseur  : min={min(thick):.1f}  max={max(thick):.1f}  moy={np.mean(thick):.1f}")
        print(f"   G-ratio    : min={min(g_vals):.3f}  max={max(g_vals):.3f}  moy={np.mean(g_vals):.3f}")
        print(f"   D-ratio    : min={min(d_vals):.3f}  max={max(d_vals):.3f}  moy={np.mean(d_vals):.3f}")

    # Table détaillée des 20 premiers composants
    print(f"\n{'─' * 90}")
    print(f"  {'ID':>4}  {'Catégorie':25s} {'Source':8s} {'Thick':>7} {'G-ratio':>8} {'D-ratio':>8} {'Circ':>6}  {'Aire':>8}")
    print(f"{'─' * 90}")
    for c in components[:30]:
        area = c.polygon.area if c.polygon else 0
        print(f"  {c.id:>4}  {c.category:25s} {c.source:8s} {c.thickness:>7.1f} {c.g_ratio:>8.3f} {c.d_ratio:>8.3f} {c.circularity:>6.3f}  {area:>8.1f}")
    if len(components) > 30:
        print(f"  ... et {len(components) - 30} de plus")
    print(f"{'─' * 90}")


# ═══════════════════════════════════════════════════════════════
#  MAIN — ORCHESTRATEUR DEBUG
# ═══════════════════════════════════════════════════════════════

def run_debug(pdf_path, page_index=0, config=None, save_dir=None):
    """Exécute le debug complet sur une page."""

    config = config or PipelineConfig()

    print_separator(f"DEBUG — {os.path.basename(pdf_path)} — Page {page_index}")
    print_config(config)

    doc = fitz.open(pdf_path)
    if page_index >= len(doc):
        print(f"❌ Page {page_index} inexistante (max: {len(doc) - 1})")
        return
    page = doc[page_index]

    t0 = time.time()

    # ── Étape 1 : Extraction ──
    print("\n⏳ Extraction des segments vectoriels...")
    segments = extract_segments_from_page(page)
    text_bboxes = extract_text_blocks(page)
    print(f"   {len(segments)} segments, {len(text_bboxes)} blocs texte")

    # ── Étape 2 : Graphe ──
    print("⏳ Construction du graphe...")
    G = build_graph(segments, config.graph.coord_precision)
    print(f"   {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")

    # ── Étape 3 : Faces fermées (polygonize) ──
    print("⏳ Polygonize → détection de toutes les faces fermées...")
    all_faces = find_all_faces(segments)
    raw_count = len(all_faces)
    print(f"   {raw_count} faces fermées trouvées par polygonize")

    print("⏳ Node Degree Filter...")
    node_degrees = get_node_degrees(G)
    candidates, rejected_faces = filter_by_node_degree(
        all_faces, node_degrees, config.graph, config.classifier,
    )
    valid_count = len(candidates)
    print(f"   {raw_count} brutes → {valid_count} gardées, {len(rejected_faces)} rejetées (croisements)")

    # ── Étape 4 : Filtre vide/solitaire ──
    print("⏳ Filtre vide & solitaire...")
    graph_polygons = filter_isolated_empty(candidates, text_bboxes)
    print(f"   {len(candidates)} candidats → {len(graph_polygons)} après filtre contenu")

    # ── Étape 5 : DBSCAN ──
    print("⏳ DBSCAN sur orphelins...")
    short_segs = filter_segments_by_length(segments, config.dbscan.max_segment_length)
    orphans = remove_already_captured(short_segs, graph_polygons)
    dbscan_clusters = cluster_segments(orphans, config.dbscan)
    dbscan_polygons = []
    for (x1, y1, x2, y2, n) in dbscan_clusters:
        poly = box(x1, y1, x2, y2)
        if config.classifier.min_area <= poly.area <= config.classifier.max_area:
            dbscan_polygons.append(poly)
    print(f"   {len(orphans)} orphelins → {len(dbscan_clusters)} clusters → {len(dbscan_polygons)} gardés")

    # ── Étape 6 : Classification ──
    print("⏳ Classification...")
    graph_components = classify_all(graph_polygons, config.classifier, "graph", page_index, 0)
    dbscan_components = classify_all(dbscan_polygons, config.classifier, "dbscan", page_index, len(graph_components))

    # Dédup
    from hybrid_pipeline.pipeline import deduplicate
    all_components = deduplicate(graph_components, dbscan_components, config.dedup_iou_threshold)

    elapsed = time.time() - t0

    # ── Stats ──
    print_detailed_stats(
        all_components, segments, G,
        raw_count, valid_count,
        len(orphans), len(dbscan_polygons),
        elapsed,
    )

    # ── VISUALISATIONS ──
    print("\n⏳ Génération des visualisations...")

    # Par défaut on affiche la figure combinée haute résolution si split False
    if not getattr(config, "_viz_split", False):
        # Figure principale : 3 lignes × 2 colonnes + 1 ligne histogrammes
        fig = plt.figure(figsize=(24, 28))
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.25)

        # Ligne 1 : Segments bruts + Graphe nœuds
        ax1 = fig.add_subplot(gs[0, :2])
        viz_raw_segments(ax1, page, segments, text_bboxes)

        ax2 = fig.add_subplot(gs[0, 2:])
        viz_graph_nodes(ax2, page, G)

        # Ligne 2 : Cycles comparaison + DBSCAN
        ax3 = fig.add_subplot(gs[1, :2])
        viz_faces_comparison(ax3, page, all_faces, candidates, rejected_faces, config)

        ax4 = fig.add_subplot(gs[1, 2:])
        viz_dbscan_clusters(ax4, page, segments, graph_polygons, config)

        # Ligne 3 : Résultat final + Scatter métriques
        ax5 = fig.add_subplot(gs[2, :2])
        viz_final_result(ax5, page, all_components)

        viz_scatter_metrics(fig, gs[2, 2:], all_components)

        # Ligne 4 : Histogrammes (4 colonnes)
        all_polys = graph_polygons + dbscan_polygons
        viz_histograms(fig, [gs[3, i] for i in range(4)], all_polys, config)

        fig.suptitle(
            f"🔍 DEBUG — {os.path.basename(pdf_path)} — Page {page_index + 1}\n"
            f"{len(all_components)} composants (Graph: {sum(1 for c in all_components if c.source == 'graph')}, "
            f"DBSCAN: {sum(1 for c in all_components if c.source == 'dbscan')}) — {elapsed:.2f}s",
            fontsize=16, fontweight="bold", y=0.995,
        )

        if save_dir:
            fname = f"debug_{os.path.splitext(os.path.basename(pdf_path))[0]}_p{page_index}.png"
            save_figure(fig, save_dir, fname, dpi=getattr(config, "_viz_dpi", 150))

        plt.show()
        doc.close()

    else:
        # Split mode: afficher chaque panneau séparément en haute résolution
        viz_dpi = getattr(config, "_viz_dpi", 200)
        # VIZ 1
        fig = plt.figure(figsize=(12, 9), dpi=viz_dpi)
        ax = fig.add_subplot(1, 1, 1)
        viz_raw_segments(ax, page, segments, text_bboxes)
        if save_dir:
            save_figure(fig, save_dir, f"viz1_raw_segments_p{page_index}.png", dpi=viz_dpi)
        plt.show()
        plt.close(fig)

        # VIZ 2
        fig = plt.figure(figsize=(12, 9), dpi=viz_dpi)
        ax = fig.add_subplot(1, 1, 1)
        viz_graph_nodes(ax, page, G)
        if save_dir:
            save_figure(fig, save_dir, f"viz2_graph_nodes_p{page_index}.png", dpi=viz_dpi)
        plt.show()
        plt.close(fig)

        # VIZ 3
        fig = plt.figure(figsize=(12, 9), dpi=viz_dpi)
        ax = fig.add_subplot(1, 1, 1)
        viz_faces_comparison(ax, page, all_faces, candidates, rejected_faces, config)
        if save_dir:
            save_figure(fig, save_dir, f"viz3_cycles_p{page_index}.png", dpi=viz_dpi)
        plt.show()
        plt.close(fig)

        # VIZ 4
        fig = plt.figure(figsize=(12, 9), dpi=viz_dpi)
        ax = fig.add_subplot(1, 1, 1)
        viz_dbscan_clusters(ax, page, segments, graph_polygons, config)
        if save_dir:
            save_figure(fig, save_dir, f"viz4_dbscan_p{page_index}.png", dpi=viz_dpi)
        plt.show()
        plt.close(fig)

        # VIZ 5
        fig = plt.figure(figsize=(12, 9), dpi=viz_dpi)
        ax = fig.add_subplot(1, 1, 1)
        viz_final_result(ax, page, all_components)
        if save_dir:
            save_figure(fig, save_dir, f"viz5_final_p{page_index}.png", dpi=viz_dpi)
        plt.show()
        plt.close(fig)

        # VIZ 6 (histograms)
        fig = plt.figure(figsize=(16, 6), dpi=viz_dpi)
        gs_row = [GridSpec(1, 4, figure=fig)[0, i] for i in range(4)]
        all_polys = graph_polygons + dbscan_polygons
        viz_histograms(fig, gs_row, all_polys, config)
        if save_dir:
            save_figure(fig, save_dir, f"viz6_histograms_p{page_index}.png", dpi=viz_dpi)
        plt.show()
        plt.close(fig)

        doc.close()

    print(f"\n✅ Debug terminé. Ajustez les seuils dans config et relancez !")
    return all_components


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Debug visuel de la pipeline hybride sur un PDF unique",
    )
    parser.add_argument("pdf", help="Chemin du fichier PDF")
    parser.add_argument("--page", type=int, default=0, help="Page à analyser (0-indexed)")
    parser.add_argument("--save", type=str, default=None, help="Dossier pour sauvegarder la figure")

    # Tuning rapide
    parser.add_argument("--epsilon", type=float, default=None, help="DBSCAN epsilon")
    parser.add_argument("--cross-ratio", type=float, default=None, help="Max cross ratio (graph)")
    parser.add_argument("--min-area", type=float, default=None, help="Aire minimum")
    parser.add_argument("--thin-wire", type=float, default=None, help="Seuil fil fin")
    parser.add_argument("--busbar", type=float, default=None, help="Seuil busbar")
    parser.add_argument("--split", action="store_true", help="Afficher les visualisations UNE PAR UNE (haute résolution)")
    parser.add_argument("--viz-dpi", type=int, default=None, help="DPI pour l'affichage/sauvegarde des visualisations (ex: 200)")

    args = parser.parse_args()

    if not os.path.isfile(args.pdf):
        print(f"❌ Fichier introuvable : {args.pdf}")
        sys.exit(1)

    config = PipelineConfig()

    # Overrides CLI
    if args.epsilon is not None:
        config.dbscan.epsilon = args.epsilon
    if args.cross_ratio is not None:
        config.graph.max_cross_ratio = args.cross_ratio
    if args.min_area is not None:
        config.classifier.min_area = args.min_area
    if args.thin_wire is not None:
        config.classifier.thin_wire_threshold = args.thin_wire
    if args.busbar is not None:
        config.classifier.busbar_threshold = args.busbar
    # Visualization mode overrides
    if args.split:
        config._viz_split = True
    if args.viz_dpi is not None:
        config._viz_dpi = args.viz_dpi

    run_debug(args.pdf, args.page, config, args.save)
