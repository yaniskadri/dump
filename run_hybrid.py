#!/usr/bin/env python3
"""
run_hybrid.py — Script rapide pour lancer la pipeline hybride.

Modifiez PDF_PATH et OUTPUT_DIR puis lancez :
    python run_hybrid.py

Pour plus d'options, utilisez le CLI :
    python -m hybrid_pipeline --help
"""

from hybrid_pipeline.pipeline import HybridPipeline
from hybrid_pipeline.config import PipelineConfig
from hybrid_pipeline.visualizer import visualize_page

# ╔══════════════════════════════════════════════════════╗
# ║           CONFIGURATION — MODIFIEZ ICI              ║
# ╚══════════════════════════════════════════════════════╝

PDF_PATH = "VotreFichier.pdf"       # ← Remplacez par votre PDF
OUTPUT_DIR = "output_hybrid"         # ← Dossier de sortie
PAGES = None                         # None = toutes, ou [0, 1, 2] pour spécifique

# Paramètres de tuning (ajustez si besoin)
config = PipelineConfig()
# config.dbscan.epsilon = 20.0       # Augmenter si DBSCAN coupe les composants
# config.graph.max_cross_ratio = 0.4 # Baisser pour être plus strict sur les croisements
# config.classifier.min_area = 100   # Augmenter pour ignorer les petits artefacts

# ╔══════════════════════════════════════════════════════╗
# ║                  EXÉCUTION                          ║
# ╚══════════════════════════════════════════════════════╝

if __name__ == "__main__":
    pipeline = HybridPipeline(PDF_PATH, config)
    
    # Lancer l'extraction complète
    results = pipeline.run(
        output_dir=OUTPUT_DIR,
        pages=PAGES,
        export_crops_flag=True,   # Sauver les crops PNG
        export_json=True,          # Sauver les métadonnées JSON
        export_yolo=False,         # Mettre True pour labels YOLO
    )
    
    # Visualisation de la première page
    if results:
        first_page = min(results.keys())
        visualize_page(
            PDF_PATH,
            results[first_page],
            page_index=first_page,
            show_ids=True,
            show_metrics=False,
        )
