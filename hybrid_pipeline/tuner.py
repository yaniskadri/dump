"""
tuner.py — Optimisation automatique des hyperparamètres de la pipeline.

Utilise Optuna (optimisation bayésienne) pour trouver les meilleurs
seuils de détection, classification et merge.

Deux modes de scoring :
  A. Avec ground truth (annotations YOLO .txt) → mAP / F1 par IoU matching.
  B. Sans ground truth (heuristique) → score basé sur la qualité des détections
     (nombre de composants, pas de chevauchements, pas de fils, aspect ratio, etc.)

Usage :
    # Mode heuristique (pas besoin d'annotations)
    python -m hybrid_pipeline.tuner schema.pdf --trials 100

    # Mode supervisé (avec labels YOLO)
    python -m hybrid_pipeline.tuner schema.pdf --gt labels/schema_p0.txt --trials 200

    # Mode multi-PDF (répertoire)
    python -m hybrid_pipeline.tuner pdf_dir/ --gt-dir labels/ --trials 300

API :
    from hybrid_pipeline.tuner import PipelineTuner
    tuner = PipelineTuner(["schema.pdf"])
    best_config = tuner.run(n_trials=100)
"""

import os
import json
import math
from pathlib import Path
from dataclasses import asdict

import optuna
from typing import Optional, List, Dict, Tuple
from shapely.geometry import Polygon, box

from .config import (
    PipelineConfig,
    GraphConfig,
    DBSCANConfig,
    ClassifierConfig,
    WireFilterConfig,
)
from .pipeline import HybridPipeline, compute_iou
from .classifier import DetectedComponent


# ══════════════════════════════════════════════════════════════
#                   SCORING FUNCTIONS
# ══════════════════════════════════════════════════════════════

def load_yolo_gt(label_path: str, page_width: float, page_height: float) -> List[Dict]:
    """
    Charge un fichier YOLO .txt et retourne les bboxes en coordonnées PDF.
    
    Format YOLO : class_id cx cy w h (normalisés 0-1).
    """
    gt_boxes = []
    if not os.path.exists(label_path):
        return gt_boxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            # Dénormaliser en coordonnées PDF
            x1 = (cx - w / 2) * page_width
            y1 = (cy - h / 2) * page_height
            x2 = (cx + w / 2) * page_width
            y2 = (cy + h / 2) * page_height

            gt_boxes.append({
                "class_id": cls_id,
                "bbox": (x1, y1, x2, y2),
                "polygon": box(x1, y1, x2, y2),
            })

    return gt_boxes


def score_supervised(
    components: List[DetectedComponent],
    gt_boxes: List[Dict],
    iou_threshold: float = 0.5,
) -> float:
    """
    Score supervisé : F1 basé sur le matching IoU avec ground truth.
    
    - True Positive : composant détecté qui matche un GT (IoU > seuil).
    - False Positive : composant détecté sans match GT.
    - False Negative : GT sans composant détecté.
    
    Returns:
        F1-score (0.0 à 1.0). Plus c'est haut, mieux c'est.
    """
    if not gt_boxes and not components:
        return 1.0
    if not gt_boxes or not components:
        return 0.0

    matched_gt = set()
    matched_pred = set()

    # Greedy matching par IoU décroissante
    pairs = []
    for i, comp in enumerate(components):
        for j, gt in enumerate(gt_boxes):
            iou = compute_iou(comp.polygon, gt["polygon"])
            if iou >= iou_threshold:
                pairs.append((iou, i, j))

    pairs.sort(reverse=True)  # Meilleur IoU d'abord

    for iou_val, pred_idx, gt_idx in pairs:
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)

    tp = len(matched_pred)
    fp = len(components) - tp
    fn = len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def score_heuristic(components: List[DetectedComponent]) -> float:
    """
    Score heuristique (sans ground truth).
    
    Pénalise les symptômes de mauvaise détection :
      - Trop peu ou trop de composants
      - Chevauchements élevés entre composants
      - Formes très allongées (fils classés comme composants)
      - Composants énormes (faux merges)
      - Trop de "Unknown_Shape" (classifier indécis)
    
    Returns:
        Score (0.0 à 1.0). Plus c'est haut, mieux c'est.
    """
    if not components:
        return 0.0

    n = len(components)
    score = 0.0

    # ── A. Pénalité nombre (trop peu = sous-détection, trop = sur-détection)
    # On s'attend à ~10-100 composants par page de schéma électrique.
    if n < 3:
        score -= 0.3
    elif n > 200:
        score -= 0.2 * min(1.0, (n - 200) / 200)

    # ── B. Pénalité chevauchements
    overlap_penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            iou = compute_iou(components[i].polygon, components[j].polygon)
            if iou > 0.3:
                overlap_penalty += iou
    # Normaliser par le nombre max de paires
    max_pairs = n * (n - 1) / 2 if n > 1 else 1
    score -= 0.3 * min(1.0, overlap_penalty / max(1, min(max_pairs, 50)))

    # ── C. Pénalité formes allongées (fils dans les composants)
    wire_count = 0
    for comp in components:
        if comp.source == "graph" and comp.thickness > 0:
            try:
                box_rot = comp.polygon.buffer(0).minimum_rotated_rectangle
                if not box_rot.is_empty:
                    x, y = box_rot.exterior.coords.xy
                    from shapely.geometry import Point as _Pt
                    e1 = _Pt(x[0], y[0]).distance(_Pt(x[1], y[1]))
                    e2 = _Pt(x[1], y[1]).distance(_Pt(x[2], y[2]))
                    short = min(e1, e2)
                    if short > 0 and max(e1, e2) / short > 10.0:
                        wire_count += 1
            except Exception:
                pass
    score -= 0.2 * min(1.0, wire_count / max(1, n))

    # ── D. Pénalité composants énormes (faux merges)
    huge_count = sum(1 for c in components if c.polygon.area > 10000)
    score -= 0.1 * min(1.0, huge_count / max(1, n))

    # ── E. Pénalité Unknown_Shape excessive
    unknown_ratio = sum(1 for c in components if c.category == "Unknown_Shape") / n
    score -= 0.1 * unknown_ratio

    # ── F. Bonus diversité de catégories (bon classifieur)
    categories = set(c.category for c in components)
    if len(categories) >= 3:
        score += 0.15
    elif len(categories) >= 2:
        score += 0.05

    # ── G. Bonus ratio graph vs dbscan raisonnable
    n_graph = sum(1 for c in components if c.source == "graph")
    if n > 0 and 0.3 < n_graph / n < 0.95:
        score += 0.1

    # Normaliser entre 0 et 1
    return max(0.0, min(1.0, score + 0.5))  # Offset pour centrer autour de 0.5


# ══════════════════════════════════════════════════════════════
#                   OPTUNA OBJECTIVE
# ══════════════════════════════════════════════════════════════

def build_config_from_trial(trial: optuna.Trial) -> PipelineConfig:
    """
    Construit une PipelineConfig à partir des suggestions Optuna.
    
    Chaque paramètre est échantillonné dans un range raisonnable
    centré autour de la valeur par défaut actuelle.
    """
    config = PipelineConfig()

    # ── Graph Config ──
    config.graph.max_cross_ratio = trial.suggest_float(
        "graph.max_cross_ratio", 0.3, 0.8, step=0.05,
    )
    config.graph.merge_neighbor_tolerance = trial.suggest_float(
        "graph.merge_neighbor_tol", 0.5, 3.0, step=0.25,
    )
    config.graph.merge_min_shared_boundary = trial.suggest_float(
        "graph.merge_min_shared_boundary", 0.02, 0.20, step=0.02,
    )
    config.graph.merge_max_area_growth = trial.suggest_float(
        "graph.merge_max_area_growth", 1.2, 2.5, step=0.1,
    )
    config.graph.merge_max_aspect_ratio = trial.suggest_float(
        "graph.merge_max_aspect_ratio", 3.0, 10.0, step=0.5,
    )

    # ── DBSCAN Config ──
    config.dbscan.epsilon = trial.suggest_float(
        "dbscan.epsilon", 8.0, 30.0, step=1.0,
    )
    config.dbscan.min_samples = trial.suggest_int(
        "dbscan.min_samples", 2, 6,
    )
    config.dbscan.max_segment_length = trial.suggest_float(
        "dbscan.max_segment_length", 80.0, 400.0, step=20.0,
    )
    config.dbscan.max_cluster_size = trial.suggest_float(
        "dbscan.max_cluster_size", 150.0, 600.0, step=25.0,
    )
    config.dbscan.min_cluster_size = trial.suggest_float(
        "dbscan.min_cluster_size", 4.0, 20.0, step=2.0,
    )

    # ── Classifier Config ──
    config.classifier.thin_wire_threshold = trial.suggest_float(
        "cls.thin_wire_threshold", 2.0, 10.0, step=0.5,
    )
    config.classifier.busbar_threshold = trial.suggest_float(
        "cls.busbar_threshold", 20.0, 80.0, step=5.0,
    )
    config.classifier.min_busbar_thickness = trial.suggest_float(
        "cls.min_busbar_thickness", 5.0, 20.0, step=1.0,
    )
    config.classifier.min_area = trial.suggest_float(
        "cls.min_area", 30.0, 200.0, step=10.0,
    )
    config.classifier.rect_ratio_threshold = trial.suggest_float(
        "cls.rect_ratio_threshold", 0.55, 0.85, step=0.05,
    )
    config.classifier.circle_threshold = trial.suggest_float(
        "cls.circle_threshold", 0.75, 0.95, step=0.05,
    )
    config.classifier.density_filled = trial.suggest_float(
        "cls.density_filled", 0.60, 0.95, step=0.05,
    )
    config.classifier.density_empty = trial.suggest_float(
        "cls.density_empty", 0.10, 0.40, step=0.05,
    )
    config.classifier.max_aspect_ratio = trial.suggest_float(
        "cls.max_aspect_ratio", 4.0, 15.0, step=1.0,
    )
    config.classifier.aspect_ratio_max_thickness = trial.suggest_float(
        "cls.aspect_ratio_max_thickness", 8.0, 25.0, step=1.0,
    )

    # ── Pipeline-level ──
    config.dedup_iou_threshold = trial.suggest_float(
        "dedup_iou_threshold", 0.15, 0.60, step=0.05,
    )
    config.containment_threshold = trial.suggest_float(
        "containment_threshold", 0.6, 0.95, step=0.05,
    )
    config.proximity_merge_radius = trial.suggest_float(
        "proximity_merge_radius", 3.0, 20.0, step=1.0,
    )

    # ── Wire Filter Config ──
    config.wire_filter.enabled = trial.suggest_categorical(
        "wire.enabled", [True, False],
    )
    config.wire_filter.min_wire_length = trial.suggest_float(
        "wire.min_wire_length", 8.0, 40.0, step=2.0,
    )
    config.wire_filter.collinear_tolerance_deg = trial.suggest_float(
        "wire.collinear_tol_deg", 5.0, 30.0, step=5.0,
    )
    config.wire_filter.min_chain_length = trial.suggest_float(
        "wire.min_chain_length", 10.0, 50.0, step=5.0,
    )

    return config


# ══════════════════════════════════════════════════════════════
#                       TUNER CLASS
# ══════════════════════════════════════════════════════════════

class PipelineTuner:
    """
    Optimiseur automatique de la pipeline hybride.
    
    Évalue des configurations candidates sur un ensemble de PDF
    et retourne la meilleure configuration trouvée.
    
    Args:
        pdf_paths: Liste de chemins de PDF à utiliser pour l'évaluation.
        gt_labels: Dict {pdf_basename: label_path} pour le mode supervisé.
                   Si None, utilise le scoring heuristique.
        pages: Pages à traiter par PDF (None = page 0 seulement pour la vitesse).
    """

    def __init__(
        self,
        pdf_paths: List[str],
        gt_labels: Optional[Dict[str, str]] = None,
        pages: Optional[List[int]] = None,
    ):
        self.pdf_paths = pdf_paths
        self.gt_labels = gt_labels or {}
        self.pages = pages or [0]
        self.best_score = -1.0
        self.best_config = None

    def _evaluate(self, config: PipelineConfig) -> float:
        """Évalue une configuration sur tous les PDF."""
        scores = []

        for pdf_path in self.pdf_paths:
            try:
                pipeline = HybridPipeline(pdf_path, config)

                for page_idx in self.pages:
                    if page_idx >= len(pipeline.doc):
                        continue

                    components = pipeline.process_page(page_idx)
                    basename = os.path.splitext(os.path.basename(pdf_path))[0]

                    # Mode supervisé ou heuristique
                    label_key = f"{basename}_p{page_idx}"
                    label_path = self.gt_labels.get(label_key)

                    if label_path and os.path.exists(label_path):
                        page = pipeline.doc[page_idx]
                        gt = load_yolo_gt(label_path, page.rect.width, page.rect.height)
                        s = score_supervised(components, gt)
                    else:
                        s = score_heuristic(components)

                    scores.append(s)

                pipeline.doc.close()

            except Exception as e:
                print(f"  ⚠ Erreur sur {pdf_path}: {e}")
                scores.append(0.0)

        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    def _objective(self, trial: optuna.Trial) -> float:
        """Fonction objectif pour Optuna."""
        config = build_config_from_trial(trial)
        score = self._evaluate(config)

        # Track best
        if score > self.best_score:
            self.best_score = score
            self.best_config = config

        return score

    def run(
        self,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        study_name: str = "pipeline_tuning",
        storage: Optional[str] = None,
        show_progress: bool = True,
    ) -> PipelineConfig:
        """
        Lance l'optimisation.
        
        Args:
            n_trials: Nombre d'essais Optuna.
            timeout: Timeout en secondes (None = pas de limite).
            study_name: Nom de l'étude (pour persistence).
            storage: URL SQLite pour sauvegarder les résultats
                     (ex: "sqlite:///tuning.db"). None = en mémoire.
            show_progress: Afficher la barre de progression.
        
        Returns:
            La meilleure PipelineConfig trouvée.
        """
        # Évaluer d'abord la config par défaut comme baseline
        default_config = PipelineConfig()
        default_score = self._evaluate(default_config)
        print(f"📊 Score baseline (config par défaut): {default_score:.4f}")
        self.best_score = default_score
        self.best_config = default_config

        # Configurer Optuna
        verbosity = optuna.logging.WARNING if not show_progress else optuna.logging.INFO
        optuna.logging.set_verbosity(verbosity)

        sampler = optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=max(10, n_trials // 5),  # Explorer d'abord
        )

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            sampler=sampler,
            load_if_exists=True,
        )

        # Enqueuler la config par défaut comme premier trial
        study.enqueue_trial(
            {
                "graph.max_cross_ratio": 0.5,
                "graph.merge_neighbor_tol": 1.0,
                "graph.merge_min_shared_boundary": 0.08,
                "graph.merge_max_area_growth": 1.8,
                "graph.merge_max_aspect_ratio": 6.0,
                "dbscan.epsilon": 15.0,
                "dbscan.min_samples": 2,
                "dbscan.max_segment_length": 200.0,
                "dbscan.max_cluster_size": 400.0,
                "dbscan.min_cluster_size": 8.0,
                "cls.thin_wire_threshold": 5.0,
                "cls.busbar_threshold": 40.0,
                "cls.min_busbar_thickness": 10.0,
                "cls.min_area": 80.0,
                "cls.rect_ratio_threshold": 0.70,
                "cls.circle_threshold": 0.85,
                "cls.density_filled": 0.80,
                "cls.density_empty": 0.25,
                "cls.max_aspect_ratio": 8.0,
                "cls.aspect_ratio_max_thickness": 15.0,
                "dedup_iou_threshold": 0.3,
                "containment_threshold": 0.8,
                "proximity_merge_radius": 8.0,
                "wire.enabled": True,
                "wire.min_wire_length": 15.0,
                "wire.collinear_tol_deg": 15.0,
                "wire.min_chain_length": 20.0,
            }
        )

        # Lancer l'optimisation
        study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress,
        )

        # Reconstruire le meilleur config
        best_trial = study.best_trial
        print(f"\n{'='*60}")
        print(f"🏆 Meilleur score: {best_trial.value:.4f} (trial #{best_trial.number})")
        print(f"   Amélioration vs baseline: {best_trial.value - default_score:+.4f}")
        print(f"\n📋 Meilleurs paramètres:")
        for key, val in sorted(best_trial.params.items()):
            print(f"   {key}: {val}")
        print(f"{'='*60}")

        # Reconstruire la config depuis le best trial
        best_config = self._trial_params_to_config(best_trial.params)

        return best_config

    @staticmethod
    def _trial_params_to_config(params: dict) -> PipelineConfig:
        """Reconstruit une PipelineConfig depuis les params d'un trial."""
        config = PipelineConfig()

        config.graph.max_cross_ratio = params["graph.max_cross_ratio"]
        config.graph.merge_neighbor_tolerance = params["graph.merge_neighbor_tol"]
        config.graph.merge_min_shared_boundary = params["graph.merge_min_shared_boundary"]
        config.graph.merge_max_area_growth = params["graph.merge_max_area_growth"]
        config.graph.merge_max_aspect_ratio = params["graph.merge_max_aspect_ratio"]

        config.dbscan.epsilon = params["dbscan.epsilon"]
        config.dbscan.min_samples = params["dbscan.min_samples"]
        config.dbscan.max_segment_length = params["dbscan.max_segment_length"]
        config.dbscan.max_cluster_size = params["dbscan.max_cluster_size"]
        config.dbscan.min_cluster_size = params["dbscan.min_cluster_size"]

        config.classifier.thin_wire_threshold = params["cls.thin_wire_threshold"]
        config.classifier.busbar_threshold = params["cls.busbar_threshold"]
        config.classifier.min_busbar_thickness = params["cls.min_busbar_thickness"]
        config.classifier.min_area = params["cls.min_area"]
        config.classifier.rect_ratio_threshold = params["cls.rect_ratio_threshold"]
        config.classifier.circle_threshold = params["cls.circle_threshold"]
        config.classifier.density_filled = params["cls.density_filled"]
        config.classifier.density_empty = params["cls.density_empty"]
        config.classifier.max_aspect_ratio = params["cls.max_aspect_ratio"]
        config.classifier.aspect_ratio_max_thickness = params["cls.aspect_ratio_max_thickness"]

        config.dedup_iou_threshold = params["dedup_iou_threshold"]
        config.containment_threshold = params["containment_threshold"]
        if "proximity_merge_radius" in params:
            config.proximity_merge_radius = params["proximity_merge_radius"]

        # Wire filter
        if "wire.enabled" in params:
            config.wire_filter.enabled = params["wire.enabled"]
        if "wire.min_wire_length" in params:
            config.wire_filter.min_wire_length = params["wire.min_wire_length"]
        if "wire.collinear_tol_deg" in params:
            config.wire_filter.collinear_tolerance_deg = params["wire.collinear_tol_deg"]
        if "wire.min_chain_length" in params:
            config.wire_filter.min_chain_length = params["wire.min_chain_length"]

        return config

    @staticmethod
    def export_best_config(config: PipelineConfig, output_path: str) -> None:
        """
        Sauvegarde la meilleure config en JSON pour réutilisation.
        
        Usage pour recharger :
            import json
            from hybrid_pipeline.tuner import PipelineTuner
            config = PipelineTuner.load_config("best_config.json")
        """
        data = {
            "graph": {
                "coord_precision": config.graph.coord_precision,
                "max_cross_ratio": config.graph.max_cross_ratio,
                "min_cycle_nodes": config.graph.min_cycle_nodes,
                "merge_neighbor_tolerance": config.graph.merge_neighbor_tolerance,
                "merge_min_shared_boundary": config.graph.merge_min_shared_boundary,
                "merge_max_area_growth": config.graph.merge_max_area_growth,
                "merge_max_aspect_ratio": config.graph.merge_max_aspect_ratio,
            },
            "dbscan": {
                "epsilon": config.dbscan.epsilon,
                "min_samples": config.dbscan.min_samples,
                "max_segment_length": config.dbscan.max_segment_length,
                "max_cluster_size": config.dbscan.max_cluster_size,
                "min_cluster_size": config.dbscan.min_cluster_size,
            },
            "classifier": {
                "thin_wire_threshold": config.classifier.thin_wire_threshold,
                "busbar_threshold": config.classifier.busbar_threshold,
                "min_busbar_thickness": config.classifier.min_busbar_thickness,
                "min_area": config.classifier.min_area,
                "max_area": config.classifier.max_area,
                "rect_ratio_threshold": config.classifier.rect_ratio_threshold,
                "circle_threshold": config.classifier.circle_threshold,
                "density_filled": config.classifier.density_filled,
                "density_empty": config.classifier.density_empty,
                "density_busbar_min": config.classifier.density_busbar_min,
                "max_aspect_ratio": config.classifier.max_aspect_ratio,
                "aspect_ratio_max_thickness": config.classifier.aspect_ratio_max_thickness,
            },
            "dedup_iou_threshold": config.dedup_iou_threshold,
            "containment_threshold": config.containment_threshold,
            "proximity_merge_radius": config.proximity_merge_radius,
            "wire_filter": {
                "enabled": config.wire_filter.enabled,
                "min_wire_length": config.wire_filter.min_wire_length,
                "collinear_tolerance_deg": config.wire_filter.collinear_tolerance_deg,
                "min_chain_length": config.wire_filter.min_chain_length,
            },
        }

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"💾 Config sauvegardée → {output_path}")

    @staticmethod
    def load_config(json_path: str) -> PipelineConfig:
        """Charge une PipelineConfig depuis un JSON exporté."""
        with open(json_path, "r") as f:
            data = json.load(f)

        config = PipelineConfig()

        for key, val in data.get("graph", {}).items():
            if hasattr(config.graph, key):
                setattr(config.graph, key, val)

        for key, val in data.get("dbscan", {}).items():
            if hasattr(config.dbscan, key):
                setattr(config.dbscan, key, val)

        for key, val in data.get("classifier", {}).items():
            if hasattr(config.classifier, key):
                setattr(config.classifier, key, val)

        if "dedup_iou_threshold" in data:
            config.dedup_iou_threshold = data["dedup_iou_threshold"]
        if "containment_threshold" in data:
            config.containment_threshold = data["containment_threshold"]
        if "proximity_merge_radius" in data:
            config.proximity_merge_radius = data["proximity_merge_radius"]

        for key, val in data.get("wire_filter", {}).items():
            if hasattr(config.wire_filter, key):
                setattr(config.wire_filter, key, val)

        return config


# ══════════════════════════════════════════════════════════════
#               AUTO-CALIBRATION (SELF-OPTIMIZING)
# ══════════════════════════════════════════════════════════════

def diagnose_detections(components: List[DetectedComponent]) -> Dict:
    """
    Analyse les détections et identifie les symptômes de mauvaise config.
    
    Retourne un dict de symptômes avec leur sévérité (0-1).
    Chaque symptôme pointe vers les paramètres à ajuster.
    """
    if not components:
        return {"no_detections": 1.0}

    n = len(components)
    symptoms = {}

    # 1. Fils dans les composants (aspect ratio > 8)
    wire_count = 0
    for comp in components:
        try:
            box_rot = comp.polygon.buffer(0).minimum_rotated_rectangle
            if not box_rot.is_empty:
                x, y = box_rot.exterior.coords.xy
                from shapely.geometry import Point as _Pt
                e1 = _Pt(x[0], y[0]).distance(_Pt(x[1], y[1]))
                e2 = _Pt(x[1], y[1]).distance(_Pt(x[2], y[2]))
                short = min(e1, e2)
                if short > 0 and max(e1, e2) / short > 8.0:
                    wire_count += 1
        except Exception:
            pass
    if wire_count > 0:
        symptoms["wires_as_components"] = min(1.0, wire_count / max(n, 1))

    # 2. Chevauchements excessifs (composants qui se superposent)
    overlap_pairs = 0
    for i in range(min(n, 50)):  # Cap pour performance
        for j in range(i + 1, min(n, 50)):
            iou = compute_iou(components[i].polygon, components[j].polygon)
            if iou > 0.3:
                overlap_pairs += 1
    if overlap_pairs > 0:
        symptoms["excessive_overlaps"] = min(1.0, overlap_pairs / max(n, 1))

    # 3. Composants énormes (faux merge)
    areas = [c.polygon.area for c in components]
    median_area = sorted(areas)[n // 2] if n > 0 else 0
    huge = [a for a in areas if a > median_area * 20 and a > 5000]
    if huge:
        symptoms["huge_false_merges"] = min(1.0, len(huge) / max(n, 1))

    # 4. Trop d'Unknown (classifier ne reconnaît rien)
    unknown_count = sum(1 for c in components if c.category == "Unknown_Shape")
    if unknown_count > n * 0.3:
        symptoms["too_many_unknowns"] = unknown_count / n

    # 5. Sous-détection (très peu de composants)
    if n < 5:
        symptoms["under_detection"] = 1.0 - (n / 5)

    # 6. Sur-détection (beaucoup trop de composants)
    if n > 150:
        symptoms["over_detection"] = min(1.0, (n - 150) / 150)

    # 7. Ratio graph/dbscan déséquilibré
    n_graph = sum(1 for c in components if c.source == "graph")
    n_dbscan = n - n_graph
    if n > 5:
        if n_graph == 0:
            symptoms["graph_finds_nothing"] = 1.0
        elif n_dbscan == 0:
            symptoms["dbscan_finds_nothing"] = 0.3  # Moins grave
        elif n_graph / n > 0.98:
            symptoms["dbscan_too_weak"] = 0.5

    return symptoms


def compute_corrections(symptoms: dict) -> dict:
    """
    Traduit les symptômes en corrections de paramètres.
    
    Retourne un dict {param_path: (direction, magnitude)} où direction
    est "increase" ou "decrease" et magnitude est le facteur d'ajustement.
    """
    corrections = {}

    if "wires_as_components" in symptoms:
        sev = symptoms["wires_as_components"]
        corrections["cls.max_aspect_ratio"] = ("decrease", 0.7 + 0.3 * (1 - sev))
        corrections["cls.thin_wire_threshold"] = ("increase", 1.0 + 0.5 * sev)

    if "excessive_overlaps" in symptoms:
        sev = symptoms["excessive_overlaps"]
        corrections["containment_threshold"] = ("decrease", 0.9 - 0.1 * sev)
        corrections["dedup_iou_threshold"] = ("decrease", 0.9 - 0.1 * sev)

    if "huge_false_merges" in symptoms:
        sev = symptoms["huge_false_merges"]
        corrections["graph.merge_max_area_growth"] = ("decrease", 0.85)
        corrections["graph.merge_min_shared_boundary"] = ("increase", 1.3)
        corrections["graph.merge_max_aspect_ratio"] = ("decrease", 0.8)

    if "too_many_unknowns" in symptoms:
        sev = symptoms["too_many_unknowns"]
        corrections["cls.rect_ratio_threshold"] = ("decrease", 0.9)
        corrections["cls.density_filled"] = ("decrease", 0.9)
        corrections["cls.circle_threshold"] = ("decrease", 0.95)

    if "under_detection" in symptoms:
        sev = symptoms["under_detection"]
        corrections["cls.min_area"] = ("decrease", 0.6)
        corrections["cls.thin_wire_threshold"] = ("decrease", 0.7)
        corrections["graph.max_cross_ratio"] = ("increase", 1.3)
        corrections["dbscan.epsilon"] = ("increase", 1.3)

    if "over_detection" in symptoms:
        sev = symptoms["over_detection"]
        corrections["cls.min_area"] = ("increase", 1.5)
        corrections["cls.thin_wire_threshold"] = ("increase", 1.3)
        corrections["dbscan.min_samples"] = ("increase", 1.5)

    if "graph_finds_nothing" in symptoms:
        corrections["graph.max_cross_ratio"] = ("increase", 1.5)
        corrections["cls.min_area"] = ("decrease", 0.5)

    return corrections


def apply_corrections(config: PipelineConfig, corrections: dict) -> PipelineConfig:
    """
    Applique les corrections à une PipelineConfig.
    """
    import copy
    new_config = copy.deepcopy(config)

    param_map = {
        "cls.max_aspect_ratio": (new_config.classifier, "max_aspect_ratio"),
        "cls.thin_wire_threshold": (new_config.classifier, "thin_wire_threshold"),
        "cls.min_area": (new_config.classifier, "min_area"),
        "cls.rect_ratio_threshold": (new_config.classifier, "rect_ratio_threshold"),
        "cls.density_filled": (new_config.classifier, "density_filled"),
        "cls.circle_threshold": (new_config.classifier, "circle_threshold"),
        "cls.busbar_threshold": (new_config.classifier, "busbar_threshold"),
        "graph.max_cross_ratio": (new_config.graph, "max_cross_ratio"),
        "graph.merge_max_area_growth": (new_config.graph, "merge_max_area_growth"),
        "graph.merge_min_shared_boundary": (new_config.graph, "merge_min_shared_boundary"),
        "graph.merge_max_aspect_ratio": (new_config.graph, "merge_max_aspect_ratio"),
        "graph.merge_neighbor_tol": (new_config.graph, "merge_neighbor_tolerance"),
        "dbscan.epsilon": (new_config.dbscan, "epsilon"),
        "dbscan.min_samples": (new_config.dbscan, "min_samples"),
        "dedup_iou_threshold": (new_config, "dedup_iou_threshold"),
        "containment_threshold": (new_config, "containment_threshold"),
    }

    for param_path, (direction, magnitude) in corrections.items():
        if param_path not in param_map:
            continue
        obj, attr = param_map[param_path]
        old_val = getattr(obj, attr)

        if direction == "increase":
            new_val = old_val * magnitude
        elif direction == "decrease":
            new_val = old_val * magnitude
        else:
            continue

        # Clamp to reasonable bounds
        if isinstance(old_val, int):
            new_val = max(1, int(round(new_val)))
        else:
            new_val = max(0.01, round(new_val, 4))

        setattr(obj, attr, new_val)

    return new_config


class AutoCalibrator:
    """
    Boucle d'auto-calibration qui détecte les problèmes et corrige
    automatiquement la configuration.
    
    Idéal pour les nouveaux constructeurs : on lance une fois et ça
    converge vers laff bonne config sans intervention manuelle.
    
    Stratégie en 2 phases :
      Phase 1 — Diagnostic rapide (3-5 itérations) :
        Analyse les symptômes → applique des corrections ciblées.
        Converge vite vers une config "pfas loin".
      Phase 2 — Fine-tuning Optuna (optionnel) :
        Affine les paramètres autour de la zone trouvée en phase 1.
    
    Usage :
        from hybrid_pipeline.tuner import AutoCalibrator
        
        calibrator = AutoCalibrator(["schema_constructeur_X.pdf"])
        best_config = calibrator.run()
        
        # Utiliser directement
        pipeline = HybridPipeline("autre_schema.pdf", best_config)
    """

    def __init__(
        self,
        pdf_paths: List[str],
        gt_labels: Optional[Dict[str, str]] = None,
        pages: Optional[List[int]] = None,
    ):
        self.pdf_paths = pdf_paths
        self.gt_labels = gt_labels or {}
        self.pages = pages or [0]

    def _run_and_diagnose(
        self, config: PipelineConfig,
    ) -> Tuple[List[DetectedComponent], Dict, float]:
        """Run pipeline with config and return components, symptoms, score."""
        all_components = []

        for pdf_path in self.pdf_paths:
            try:
                pipeline = HybridPipeline(pdf_path, config)
                for page_idx in self.pages:
                    if page_idx >= len(pipeline.doc):
                        continue
                    components = pipeline.process_page(page_idx)
                    all_components.extend(components)
                pipeline.doc.close()
            except Exception as e:
                print(f"  ⚠ {pdf_path}: {e}")

        symptoms = diagnose_detections(all_components)

        # Score
        if self.gt_labels:
            scores = []
            for pdf_path in self.pdf_paths:
                try:
                    pipeline = HybridPipeline(pdf_path, config)
                    basename = os.path.splitext(os.path.basename(pdf_path))[0]
                    for page_idx in self.pages:
                        key = f"{basename}_p{page_idx}"
                        label_path = self.gt_labels.get(key)
                        if label_path and os.path.exists(label_path):
                            comps = pipeline.process_page(page_idx)
                            page = pipeline.doc[page_idx]
                            gt = load_yolo_gt(label_path, page.rect.width, page.rect.height)
                            scores.append(score_supervised(comps, gt))
                    pipeline.doc.close()
                except Exception:
                    pass
            score = sum(scores) / len(scores) if scores else score_heuristic(all_components)
        else:
            score = score_heuristic(all_components)

        return all_components, symptoms, score

    def run(
        self,
        max_iterations: int = 8,
        optuna_trials: int = 50,
        do_optuna_phase: bool = True,
        output_path: Optional[str] = None,
    ) -> PipelineConfig:
        """
        Lance l'auto-calibration complète.
        
        Phase 1 : Diagnostic itératif (rapide, ~5 itérations).
        Phase 2 : Fine-tuning Optuna autour de la zone trouvée (optionnel).
        
        Args:
            max_iterations: Max itérations pour la phase diagnostic.
            optuna_trials: Nombre de trials Optuna pour la phase 2.
            do_optuna_phase: Lancer la phase 2 (True par défaut).
            output_path: Si fourni, sauvegarde la config en JSON.
        
        Returns:
            La meilleure PipelineConfig trouvée.
        """
        config = PipelineConfig()

        print("═" * 60)
        print("🔧 AUTO-CALIBRATION — Phase 1 : Diagnostic itératif")
        print("═" * 60)

        best_score = -1.0
        best_config = config
        no_improve_count = 0

        for iteration in range(max_iterations):
            components, symptoms, score = self._run_and_diagnose(config)

            n_comp = len(components)
            cats = {}
            for c in components:
                cats[c.category] = cats.get(c.category, 0) + 1

            print(f"\n── Itération {iteration + 1}/{max_iterations} ──")
            print(f"   Score: {score:.4f} | Composants: {n_comp}")
            print(f"   Catégories: {dict(sorted(cats.items()))}")

            if symptoms:
                print(f"   Symptômes: {', '.join(f'{k}({v:.2f})' for k, v in symptoms.items())}")
            else:
                print(f"   ✅ Aucun symptôme détecté!")

            if score > best_score:
                best_score = score
                best_config = config
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Convergence : pas d'amélioration depuis 2 itérations
            if no_improve_count >= 2:
                print(f"\n   ⏹ Convergence atteinte (pas d'amélioration depuis 2 itérations)")
                break

            # Pas de symptômes → on est bon
            if not symptoms:
                print(f"\n   ⏹ Configuration optimale trouvée!")
                break

            # Appliquer les corrections
            corrections = compute_corrections(symptoms)
            if corrections:
                print(f"   Corrections: {list(corrections.keys())}")
                config = apply_corrections(config, corrections)
            else:
                break

        print(f"\n📊 Phase 1 terminée — Score: {best_score:.4f}")

        # ── Phase 2 : Fine-tuning Optuna ──
        if do_optuna_phase and optuna_trials > 0:
            print(f"\n{'═' * 60}")
            print(f"🎯 AUTO-CALIBRATION — Phase 2 : Fine-tuning Optuna ({optuna_trials} trials)")
            print(f"{'═' * 60}")

            tuner = PipelineTuner(
                pdf_paths=self.pdf_paths,
                gt_labels=self.gt_labels if self.gt_labels else None,
                pages=self.pages,
            )

            # Seed Optuna avec la config trouvée en phase 1
            tuner.best_score = best_score
            tuner.best_config = best_config

            optuna_config = tuner.run(
                n_trials=optuna_trials,
                show_progress=True,
            )

            if tuner.best_score > best_score:
                best_config = optuna_config
                best_score = tuner.best_score
                print(f"   Optuna a amélioré le score: {best_score:.4f}")
            else:
                print(f"   Optuna n'a pas amélioré (phase 1 était déjà optimale)")

        # Sauvegarder si demandé
        if output_path:
            PipelineTuner.export_best_config(best_config, output_path)

        print(f"\n{'═' * 60}")
        print(f"🏆 Score final: {best_score:.4f}")
        print(f"{'═' * 60}")

        return best_config


# ══════════════════════════════════════════════════════════════
#                          CLI
# ══════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimisation automatique des hyperparamètres de la pipeline hybride.",
    )
    parser.add_argument(
        "input",
        help="Chemin PDF ou dossier de PDFs.",
    )
    parser.add_argument(
        "--gt", "--ground-truth",
        default=None,
        help="Fichier label YOLO (.txt) pour un seul PDF.",
    )
    parser.add_argument(
        "--gt-dir",
        default=None,
        help="Dossier de labels YOLO pour multi-PDF (nommés <basename>_p<page>.txt).",
    )
    parser.add_argument(
        "--trials", "-n",
        type=int, default=100,
        help="Nombre d'essais Optuna (défaut: 100).",
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int, default=None,
        help="Timeout en secondes.",
    )
    parser.add_argument(
        "--pages", "-p",
        type=int, nargs="+", default=[0],
        help="Pages à traiter (défaut: 0).",
    )
    parser.add_argument(
        "--output", "-o",
        default="best_config.json",
        help="Chemin de sortie pour la meilleure config JSON.",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="SQLite storage pour persistence Optuna (ex: sqlite:///tuning.db).",
    )
    parser.add_argument(
        "--auto", "-a",
        action="store_true",
        help="Mode auto-calibration (diagnostic + Optuna). Recommandé pour nouveaux constructeurs.",
    )
    parser.add_argument(
        "--auto-only",
        action="store_true",
        help="Mode auto-calibration SANS phase Optuna (diagnostic seul, très rapide).",
    )

    args = parser.parse_args()

    # Collecter les PDFs
    input_path = Path(args.input)
    if input_path.is_dir():
        pdf_paths = sorted(str(p) for p in input_path.glob("*.pdf"))
        if not pdf_paths:
            print(f"❌ Aucun PDF trouvé dans {input_path}")
            return
        print(f"📁 {len(pdf_paths)} PDFs trouvés dans {input_path}")
    else:
        pdf_paths = [str(input_path)]

    # Collecter les labels GT
    gt_labels = {}
    if args.gt:
        # Un seul label pour un seul PDF
        basename = os.path.splitext(os.path.basename(pdf_paths[0]))[0]
        for p in args.pages:
            gt_labels[f"{basename}_p{p}"] = args.gt
    elif args.gt_dir:
        # Dossier de labels
        gt_dir = Path(args.gt_dir)
        for label_file in gt_dir.glob("*.txt"):
            key = label_file.stem  # ex: "schema_p0"
            gt_labels[key] = str(label_file)
        print(f"📋 {len(gt_labels)} labels GT trouvés")

    # Lancer le tuning
    if args.auto or args.auto_only:
        # Mode auto-calibration
        calibrator = AutoCalibrator(
            pdf_paths=pdf_paths,
            gt_labels=gt_labels if gt_labels else None,
            pages=args.pages,
        )
        best_config = calibrator.run(
            optuna_trials=args.trials if not args.auto_only else 0,
            do_optuna_phase=not args.auto_only,
            output_path=args.output,
        )
    else:
        # Mode Optuna pur
        tuner = PipelineTuner(
            pdf_paths=pdf_paths,
            gt_labels=gt_labels if gt_labels else None,
            pages=args.pages,
        )

        best_config = tuner.run(
            n_trials=args.trials,
            timeout=args.timeout,
            storage=args.db,
        )

        # Sauvegarder
        PipelineTuner.export_best_config(best_config, args.output)

    print(f"\n✅ Pour utiliser cette config :")
    print(f'   from hybrid_pipeline.tuner import PipelineTuner')
    print(f'   config = PipelineTuner.load_config("{args.output}")')
    print(f'   pipeline = HybridPipeline("schema.pdf", config)')


if __name__ == "__main__":
    main()
