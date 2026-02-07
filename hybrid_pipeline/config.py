"""
config.py — Configuration centralisée de la pipeline hybride.

Tous les seuils et paramètres ajustables sont regroupés ici
pour faciliter le tuning sans toucher au code métier.
"""

from dataclasses import dataclass, field


@dataclass
class GraphConfig:
    """Paramètres pour l'extraction par théorie des graphes (NetworkX)."""
    # Précision d'arrondi des coordonnées (en points PDF) pour fusionner
    # les nœuds proches. Plus le chiffre est petit, plus on fusionne.
    coord_precision: int = 1

    # Ratio max de nœuds degré ≥ 4 dans un cycle pour le garder.
    # Au-dessus → croisement de fils → rejeté.
    max_cross_ratio: float = 0.5

    # Nombre minimum de nœuds pour qu'un cycle soit considéré.
    min_cycle_nodes: int = 3

    # ── Smart Merge (remplacement de unary_union aveugle) ──
    # Distance max entre deux faces pour les considérer comme voisines
    # d'un même composant (en points PDF).
    merge_neighbor_tolerance: float = 1.0

    # Ratio min de bord partagé entre deux faces pour autoriser le merge.
    # = longueur du bord commun / périmètre de la plus petite face.
    # Sous-faces d'un même rectangle partagent un long bord → ratio élevé.
    # Deux symboles distincts ne se touchent qu'en un point → ratio ~0.
    merge_min_shared_boundary: float = 0.08

    # Ratio max entre l'aire combinée et la somme des aires individuelles.
    # Un ratio élevé signifie que le merge ajoute beaucoup de "vide" → rejet.
    merge_max_area_growth: float = 1.8

    # Aspect ratio max du résultat après merge.
    # Empêche de fusionner deux symboles côte à côte en une forme allongée.
    merge_max_aspect_ratio: float = 6.0


@dataclass
class DBSCANConfig:
    """Paramètres pour le clustering DBSCAN (formes ouvertes)."""
    # Distance max (en points PDF) entre deux traits pour les regrouper.
    epsilon: float = 15.0

    # Nombre minimum de segments pour former un cluster.
    min_samples: int = 2

    # Longueur max d'un segment individuel (au-delà = fil long → ignoré).
    max_segment_length: float = 200.0

    # Taille max du cluster final (largeur ou hauteur en points PDF).
    max_cluster_size: float = 400.0

    # Taille min du cluster (évite le bruit).
    min_cluster_size: float = 8.0


@dataclass
class ClassifierConfig:
    """Seuils de classification des composants détectés."""
    # Épaisseur minimale du rectangle orienté (en points PDF).
    # En dessous → fil de commande (ignoré).
    thin_wire_threshold: float = 5.0

    # Épaisseur max pour un busbar. Au-dessus → vrai composant.
    busbar_threshold: float = 40.0

    # Aire min/max d'un polygone pour être considéré (en points PDF²).
    min_area: float = 80.0
    max_area: float = 150000.0

    # Ratio rectangularité (G-ratio) : aire_poly / aire_bbox_orienté.
    # > 0.70 → rectangulaire.
    rect_ratio_threshold: float = 0.70

    # Circularité : 4π·A / P². > 0.85 → cercle.
    circle_threshold: float = 0.85

    # Densité (D-ratio) : aire_matière / aire_enveloppe.
    density_filled: float = 0.80     # > 0.80 → composant plein
    density_empty: float = 0.25      # < 0.25 → cadre vide (layout)
    density_busbar_min: float = 0.50  # seuil densité pour busbar

    # Hexagone / Symbole
    hex_rect_range: tuple = (0.70, 0.82)
    hex_circ_min: float = 0.60

    # Aire min pour garder un "Unknown"
    unknown_min_area: float = 200.0

    # ── Filtre aspect ratio (rejet des fils allongés) ──
    # Ratio d'aspect max (longueur / épaisseur) pour qu'une face
    # soit considérée comme composant. Au-delà → fil ou wire → rejeté.
    max_aspect_ratio: float = 8.0

    # Épaisseur max pour appliquer le filtre aspect ratio.
    # Au-dessus de cette épaisseur, on ne rejette pas (gros busbar).
    aspect_ratio_max_thickness: float = 15.0


@dataclass
class ExportConfig:
    """Configuration de l'export (crops et JSON)."""
    dpi: int = 300
    padding: int = 20  # Marge autour des crops (en pixels)
    scale_factor: float = field(init=False)

    def __post_init__(self):
        self.scale_factor = self.dpi / 72.0


@dataclass
class PipelineConfig:
    """Configuration globale de la pipeline."""
    graph: GraphConfig = field(default_factory=GraphConfig)
    dbscan: DBSCANConfig = field(default_factory=DBSCANConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    # Seuil IoU pour dédupliquer les détections Graph vs DBSCAN.
    dedup_iou_threshold: float = 0.3

    # ── Post-classification cleanup ──
    # Seuil de containment : si un composant est contenu à ce ratio
    # dans un autre, le plus petit est supprimé.
    containment_threshold: float = 0.8

    # Ratio de chevauchement min pour fusionner deux composants DBSCAN.
    dbscan_overlap_merge_ratio: float = 0.5
