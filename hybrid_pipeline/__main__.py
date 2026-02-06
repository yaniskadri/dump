"""
__main__.py — Point d'entrée CLI de la pipeline hybride.

Usage:
    python -m hybrid_pipeline input.pdf -o output_dir/
    python -m hybrid_pipeline input.pdf -o output_dir/ --yolo --viz
    python -m hybrid_pipeline input.pdf --pages 0 1 2 --viz --ids
"""

import argparse
import sys
import os

from .config import PipelineConfig, GraphConfig, DBSCANConfig, ClassifierConfig, ExportConfig
from .pipeline import HybridPipeline
from .visualizer import visualize_page


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Hybride — Extraction de composants électriques depuis PDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Extraction simple (crops + JSON)
  python -m hybrid_pipeline schema.pdf -o dataset/

  # Avec labels YOLO
  python -m hybrid_pipeline schema.pdf -o dataset/ --yolo

  # Visualisation de la page 0
  python -m hybrid_pipeline schema.pdf --viz --page 0

  # Tuning des paramètres
  python -m hybrid_pipeline schema.pdf -o dataset/ --epsilon 20 --cross-ratio 0.4
        """,
    )

    # Positional
    parser.add_argument("pdf", help="Chemin du fichier PDF à traiter")

    # Output
    parser.add_argument("-o", "--output", default="output_hybrid",
                        help="Dossier de sortie (défaut: output_hybrid)")

    # Pages
    parser.add_argument("--pages", nargs="+", type=int, default=None,
                        help="Pages à traiter (0-indexed). Défaut: toutes")

    # Export options
    parser.add_argument("--no-crops", action="store_true",
                        help="Ne pas exporter les crops PNG")
    parser.add_argument("--no-json", action="store_true",
                        help="Ne pas exporter le JSON de métadonnées")
    parser.add_argument("--yolo", action="store_true",
                        help="Exporter aussi les labels YOLO")

    # Visualization
    parser.add_argument("--viz", action="store_true",
                        help="Afficher la visualisation matplotlib")
    parser.add_argument("--page", type=int, default=0,
                        help="Page à visualiser (avec --viz). Défaut: 0")
    parser.add_argument("--ids", action="store_true",
                        help="Afficher les IDs sur la visualisation")
    parser.add_argument("--metrics", action="store_true",
                        help="Afficher G/D-ratios sur la visualisation")

    # Tuning parameters
    parser.add_argument("--epsilon", type=float, default=15.0,
                        help="DBSCAN epsilon (défaut: 15.0)")
    parser.add_argument("--cross-ratio", type=float, default=0.5,
                        help="Max ratio croisements pour le filtre graph (défaut: 0.5)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI pour les crops (défaut: 300)")
    parser.add_argument("--min-area", type=float, default=80.0,
                        help="Aire minimum des composants (défaut: 80)")
    parser.add_argument("--max-area", type=float, default=50000.0,
                        help="Aire maximum des composants (défaut: 50000)")

    args = parser.parse_args()

    # Vérifier que le PDF existe
    if not os.path.isfile(args.pdf):
        print(f"❌ Fichier introuvable : {args.pdf}")
        sys.exit(1)

    # Construire la config
    config = PipelineConfig(
        graph=GraphConfig(max_cross_ratio=args.cross_ratio),
        dbscan=DBSCANConfig(epsilon=args.epsilon),
        classifier=ClassifierConfig(min_area=args.min_area, max_area=args.max_area),
        export=ExportConfig(dpi=args.dpi),
    )

    # Lancer la pipeline
    pipeline = HybridPipeline(args.pdf, config)

    print(f"📄 Fichier  : {args.pdf}")
    print(f"📁 Sortie   : {args.output}")
    print(f"⚙️  Config   : ε={config.dbscan.epsilon}, "
          f"cross_ratio={config.graph.max_cross_ratio}, "
          f"DPI={config.export.dpi}")
    print("─" * 60)

    results = pipeline.run(
        output_dir=args.output,
        pages=args.pages,
        export_crops_flag=not args.no_crops,
        export_json=not args.no_json,
        export_yolo=args.yolo,
    )

    # Visualisation optionnelle
    if args.viz:
        page_idx = args.page
        if page_idx in results:
            visualize_page(
                args.pdf,
                results[page_idx],
                page_index=page_idx,
                show_ids=args.ids,
                show_metrics=args.metrics,
            )
        else:
            print(f"⚠️  Page {page_idx} n'a pas été traitée. "
                  f"Pages disponibles : {list(results.keys())}")


if __name__ == "__main__":
    main()
