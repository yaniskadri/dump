"""
hybrid_pipeline — Extraction de composants électriques depuis PDF vectoriels.

Combine :
  • Graph/NetworkX  → formes fermées (cycles + Node Degree Filter)
  • DBSCAN          → formes ouvertes (diodes, terres, symboles)
  • Classification  → arbre de décision géométrique (G-ratio, D-ratio, etc.)

Usage rapide :
    from hybrid_pipeline import HybridPipeline
    pipeline = HybridPipeline("schema.pdf")
    pipeline.run("output_dataset/")

CLI :
    python -m hybrid_pipeline schema.pdf -o output_dataset/ --viz
"""

__version__ = "1.0.0"

from .config import PipelineConfig
from .pipeline import HybridPipeline
from .visualizer import quick_visualize
