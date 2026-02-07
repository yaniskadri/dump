# Hybrid Pipeline — Extraction de composants électriques depuis PDF

Pipeline d'extraction automatique de composants électriques (relais, ECU, connecteurs, symboles) depuis des schémas PDF vectoriels.

## Architecture

```
dump/
├── hybrid_pipeline/          # Package principal
│   ├── __init__.py
│   ├── __main__.py           # CLI: python -m hybrid_pipeline
│   ├── config.py             # Configurations (seuils, paramètres)
│   ├── pipeline.py           # Orchestrateur principal
│   ├── vector_utils.py       # Extraction vectorielle PyMuPDF
│   ├── wire_filter.py        # Suppression des fils avant détection
│   ├── graph_extractor.py    # Détection formes fermées (NetworkX + Shapely)
│   ├── dbscan_extractor.py   # Détection formes ouvertes (DBSCAN clustering)
│   ├── classifier.py         # Arbre de décision géométrique
│   ├── exporter.py           # Export crops PNG, JSON, YOLO
│   ├── visualizer.py         # Visualisation debug
│   └── tuner.py              # Auto-calibration Optuna
├── train_classifier.py       # Entraînement CNN (optionnel)
├── run_hybrid.py             # Script rapide d'exécution
├── debug_single.py           # Debug détaillé sur un PDF
├── requirements.txt
├── Doc/                      # Documentation technique
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

**Dépendances principales :**
- PyMuPDF (fitz) — extraction vectorielle PDF
- NetworkX — topologie de graphe
- Shapely — géométrie computationnelle
- scikit-learn — DBSCAN clustering
- Optuna — optimisation bayésienne
- torch/torchvision — entraînement CNN (optionnel)

## Utilisation rapide

### 1. Exécuter la pipeline

```bash
# Via CLI
python -m hybrid_pipeline schema.pdf -o output/

# Via script
python run_hybrid.py  # (modifier PDF_PATH dans le fichier)
```

### 2. Visualisation debug

```python
from hybrid_pipeline import quick_visualize
quick_visualize("schema.pdf", page_index=0)
```

### 3. Debug complet (6 visualisations)

```bash
python debug_single.py schema.pdf --save debug_output/
```

---

## Méthodologie

### Phase 1 : Extraction vectorielle

Le PDF est lu via PyMuPDF (`get_drawings()`). Tous les éléments vectoriels sont convertis en segments :
- **Lignes** → segments directs
- **Courbes de Bézier** → approximation en 8 segments (capture les cercles)
- **Rectangles** → 4 segments

### Phase 2 : Suppression des fils (Wire Filter)

**Problème :** Les fils créent des "faux polygones" aux croisements, qui sont ensuite mal classifiés comme composants.

**Solution :** Avant la polygonisation, on identifie et retire les fils grâce à 3 critères :

1. **Wire Chains** : Suites de segments quasi-colinéaires passant par des nœuds de degré 2 (jonctions simples).
2. **Wire Bridges** : Longs segments droits entre deux jonctions de degré ≥3 (raccords T ou +).
3. **Long Straight Segments** : Segments très longs et axis-aligned (>45pt).

### Phase 3 : Graph Extractor (formes fermées)

1. **Polygonize** (Shapely) : Trouve toutes les faces fermées dans le réseau de segments.
2. **Node Degree Filter** : Rejette les faces dont la majorité des sommets ont degré ≥4 (= croisements de fils).
3. **Filtre Vide & Solitaire** : Rejette les faces sans texte ni voisins (cadres layout).
4. **Smart Merge** : Regroupe les sous-faces adjacentes qui partagent un vrai bord (+8% shared boundary).

### Phase 4 : DBSCAN Extractor (formes ouvertes)

Pour les symboles non fermés (terre, diodes, flèches) :

1. Ne garder que les segments courts (<200pt).
2. Retirer ceux déjà capturés par le Graph Extractor.
3. Clustering DBSCAN sur les centres des segments.
4. Créer une bounding box par cluster.

### Phase 5 : Classification géométrique

Arbre de décision basé sur 4 métriques :

| Métrique | Définition | Utilité |
|----------|------------|---------|
| **Thickness** | Épaisseur min du rectangle englobant orienté | Distinguer fils (fins) / composants (épais) |
| **G-ratio** | `Area_poly / Area_bounding_box` | 1.0 = rectangle, ~0.78 = cercle |
| **D-ratio** | `Area_remplie / Area_enveloppe` | 1.0 = solide, <0.5 = creux |
| **Circularité** | `4π·A / P²` | >0.85 = cercle |

**Catégories de sortie :**
- `Component_Rect` — Composants rectangulaires pleins
- `Component_Complex` — Formes complexes (L-shape, triangles)
- `Circle_Component` — Cercles (moteurs, voyants)
- `Hex_Symbol` — Hexagones
- `Busbar_Power` — Bus de puissance (épais)
- `Group_Container` — Conteneurs (cadres pointillés)
- `Open_Component` — Détecté par DBSCAN
- `Unknown_Shape` — Non classifié mais gardé

### Phase 6 : Post-traitement

- **Déduplication** : Supprime les doublons Graph/DBSCAN par IoU.
- **Proximity Merge** : Fusionne les composants proches (ex: cercle + terre).
- **Containment Filter** : Supprime les petits composants contenus dans des grands.

---

## Entraîner le classificateur CNN

Le classificateur géométrique (arbre de décision) fonctionne bien pour la majorité des cas, mais un CNN peut améliorer la précision sur les cas ambigus.

### Étape 1 : Générer un dataset

```bash
# Génère des crops depuis vos PDFs
python train_classifier.py prepare schema1.pdf schema2.pdf -o dataset/crops
```

Cela crée un dossier par catégorie :
```
dataset/crops/
├── Component_Rect/
│   ├── crop_001.png
│   └── ...
├── Circle_Component/
├── Open_Component/
├── Busbar_Power/
└── Unknown_Shape/
```

### Étape 2 : Corriger manuellement les labels

**C'est l'étape la plus importante !**

1. Ouvrir le dossier `dataset/crops/`
2. Créer un dossier `False_Positive/` pour les erreurs
3. Déplacer les crops mal classifiés dans le bon dossier
4. Optionnel : créer de nouvelles catégories (`Ground_Symbol/`, `Arrow/`, etc.)

### Étape 3 : Entraîner le modèle

```bash
python train_classifier.py train \
    --data dataset/crops \
    --epochs 30 \
    --model resnet18 \
    --output component_classifier.pth
```

**Options de modèles :**
- `resnet18` — 11M params, bon équilibre (recommandé)
- `mobilenet` — 3.4M params, plus léger
- `simple` — ~200K params, rapide si <500 images

**Sortie typique :**
```
📊 Dataset: 1234 images, 8 classes
   Component_Rect: 456 images
   Circle_Component: 123 images
   ...
🖥️  Device: mps
🧠 Model: resnet18 (11M params)

Epoch | Train Loss | Train Acc | Val Loss | Val Acc
    1 |     0.8234 |    72.3%  |   0.5123 |   78.5%
    2 |     0.4521 |    84.1%  |   0.3892 |   85.2%
   ...
   30 |     0.0512 |    98.7%  |   0.1234 |   94.3%

✅ Meilleur modèle sauvegardé → component_classifier.pth
   Val accuracy: 94.3%
```

### Étape 4 : Utiliser le modèle

```python
from train_classifier import ComponentClassifier

# Charger le modèle
clf = ComponentClassifier("component_classifier.pth")

# Prédire une image
category, confidence = clf.predict("crop.png")
print(f"{category} ({confidence:.1%})")

# Prédire plusieurs images
results = clf.predict_batch(["img1.png", "img2.png"])
```

### Intégration dans la pipeline

Pour utiliser le CNN comme post-filtre dans la pipeline, modifiez `classifier.py` :

```python
# Dans classify_polygon(), après l'arbre de décision :
if category == "Unknown_Shape" and confidence < 0.7:
    # Utiliser le CNN pour trancher
    cnn_category, cnn_conf = cnn_classifier.predict(crop_path)
    if cnn_conf > 0.8:
        category = cnn_category
```

---

## Auto-calibration (Optuna)

Pour optimiser automatiquement les seuils sur vos PDFs :

```bash
# Mode heuristique (sans labels)
python -m hybrid_pipeline.tuner schema.pdf --trials 100

# Mode supervisé (avec labels YOLO)
python -m hybrid_pipeline.tuner schema.pdf --gt labels/schema_p0.txt --trials 200

# Auto-calibration complète
python -m hybrid_pipeline.tuner schema.pdf --auto --output best_config.json
```

**Utiliser la config optimisée :**
```python
from hybrid_pipeline.tuner import PipelineTuner
from hybrid_pipeline import HybridPipeline

config = PipelineTuner.load_config("best_config.json")
pipeline = HybridPipeline("nouveau_schema.pdf", config)
```

---

## Configuration

Les seuils sont dans `config.py`. Principaux paramètres à ajuster :

```python
# Classifier
thin_wire_threshold = 5.0    # Épaisseur max pour être un fil
busbar_threshold = 40.0      # Épaisseur min pour un busbar
min_area = 80.0              # Aire min pour garder un composant

# DBSCAN
epsilon = 15.0               # Distance max entre segments d'un même cluster
max_segment_length = 200.0   # Longueur max des segments DBSCAN

# Wire Filter
min_wire_length = 15.0       # Longueur min pour être un fil
min_chain_length = 20.0      # Longueur totale min d'une chaîne de fils
```

---

## Licence

Projet académique — H26/P4
