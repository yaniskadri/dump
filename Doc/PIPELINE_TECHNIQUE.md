# Pipeline Hybride d'Extraction de Composants Électriques

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Architecture globale](#2-architecture-globale)
3. [Étapes de la pipeline](#3-étapes-de-la-pipeline)
4. [Modules détaillés](#4-modules-détaillés)
5. [Hyperparamètres et configuration](#5-hyperparamètres-et-configuration)
6. [Auto-calibration](#6-auto-calibration)
7. [Cas d'usage et workflows](#7-cas-dusage-et-workflows)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Vue d'ensemble

### 1.1 Objectif

Cette pipeline extrait automatiquement les **composants électriques** depuis des schémas PDF vectoriels et les classe en catégories métier (relais, moteurs, busbars, connecteurs, etc.).

**Pourquoi "hybride" ?**

Les schémas électriques contiennent deux types de formes géométriques :
- **Formes fermées** (rectangles, cercles) → composants "classiques"
- **Formes ouvertes** (flèches, symboles de terre, diodes) → cas particuliers

Une approche unique ne suffit pas. La pipeline combine donc **deux extracteurs complémentaires** :
- **Graph Extractor** (NetworkX + polygonization) → formes fermées
- **DBSCAN Extractor** (clustering spatial) → formes ouvertes

### 1.2 Workflow simplifié

```
PDF vectoriel
    ↓
[1] Extraction segments (PyMuPDF)
    ↓
[2] Graph Extractor ──→ Polygones fermés (rectangles, cercles)
[3] DBSCAN Extractor ──→ Clusters ouverts (flèches, terres)
    ↓
[4] Déduplication (IoU matching)
    ↓
[5] Classification géométrique (arbre de décision)
    ↓
[6] Post-cleanup (containment, chevauchements)
    ↓
[7] Export (PNG crops + JSON métadonnées)
```

### 1.3 Technologies clés

| Librairie | Rôle |
|-----------|------|
| **PyMuPDF (fitz)** | Extraction des vecteurs PDF (lignes, courbes, rectangles) |
| **Shapely** | Géométrie computationnelle (polygonisation, unions, IoU) |
| **NetworkX** | Analyse de graphe (degrés de nœuds, topologie) |
| **scikit-learn** | DBSCAN clustering pour formes ouvertes |
| **Optuna** | Auto-tuning bayésien des hyperparamètres |

---

## 2. Architecture globale

### 2.1 Structure des modules

```
hybrid_pipeline/
├── config.py              # Configuration centralisée (tous les hyperparamètres)
├── pipeline.py            # Orchestrateur principal
├── vector_utils.py        # Extraction vectorielle depuis PDF
├── graph_extractor.py     # Détection par topologie de graphe
├── dbscan_extractor.py    # Détection par clustering DBSCAN
├── classifier.py          # Arbre de décision géométrique
├── exporter.py            # Export des crops et métadonnées
├── visualizer.py          # Outils de QA visuel
└── tuner.py               # Auto-calibration Optuna
```

### 2.2 Flux de données

```
          ┌──────────────┐
          │  PDF Input   │
          └──────┬───────┘
                 │
                 ▼
     ┌───────────────────────┐
     │  Vector Extraction    │  ← vector_utils.py
     │  (segments + text)    │
     └─────────┬─────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌─────────────┐  ┌─────────────┐
│   Graph     │  │   DBSCAN    │
│  Extractor  │  │  Extractor  │
│ (fermé)     │  │  (ouvert)   │
└──────┬──────┘  └──────┬──────┘
       │                │
       └────────┬───────┘
                ▼
       ┌────────────────┐
       │ Deduplicate    │
       │   (IoU)        │
       └────────┬───────┘
                ▼
       ┌────────────────┐
       │  Classifier    │
       │ (G/D-ratio)    │
       └────────┬───────┘
                ▼
       ┌────────────────┐
       │ Post-cleanup   │
       │ (containment)  │
       └────────┬───────┘
                ▼
       ┌────────────────┐
       │     Export     │
       │  (PNG + JSON)  │
       └────────────────┘
```

---

## 3. Étapes de la pipeline

### Étape 1 : Extraction vectorielle

**Fichier** : `vector_utils.py`

**Que fait cette étape ?**

Lit le PDF page par page et extrait :
- **Segments vectoriels** : lignes (l), courbes de Bézier (c), rectangles (re)
- **Blocs de texte** : bounding boxes + contenu texte

**Pourquoi c'est important ?**

Un PDF n'est **pas** une image — c'est une liste d'instructions vectorielles. Un rectangle n'est pas stocké comme "rectangle", mais comme 4 segments de ligne déconnectés. La pipeline doit reconstruire les formes à partir de ces segments.

**Technique clé : approximation des courbes de Bézier**

Les cercles sont encodés avec 4 courbes de Bézier cubiques. L'ancienne méthode (1 segment droit par courbe) créait un losange au lieu d'un cercle.

**Solution actuelle** : chaque courbe est échantillonnée en **8 segments** via la formule de De Casteljau :

```python
# Courbe de Bézier cubique : p0, p1, p2, p3
for t in [1/8, 2/8, ..., 8/8]:
    x = (1-t)³·p0.x + 3(1-t)²t·p1.x + 3(1-t)t²·p2.x + t³·p3.x
    y = (idem pour y)
```

Résultat : un cercle = 32 petits segments → polygone quasi-circulaire.

**Sortie** :
- Liste de `VectorSegment(x1, y1, x2, y2)`
- Liste de blocs texte `(x0, y0, x1, y1, "text")`

---

### Étape 2 : Graph Extractor (formes fermées)

**Fichier** : `graph_extractor.py`

**Objectif** : Détecter les rectangles, cercles, et autres formes **fermées** (qui ont un contour complet).

#### 2.1 Sous-étape : Polygonisation

Utilise `shapely.ops.polygonize` pour trouver **toutes les faces fermées** dans l'arrangement planaire des segments.

**Analogie** : imagine un dessin de fils qui se croisent sur une feuille. Chaque zone délimitée par ces fils est une "face". `polygonize` trouve toutes ces zones automatiquement.

**Problème** : ça trouve AUSSI les artefacts (petits rectangles aux croisements de fils).

#### 2.2 Sous-étape : Node Degree Filter

Pour distinguer les vrais composants des croisements de fils :
1. Construire un graphe NetworkX où chaque jonction est un nœud
2. Calculer le **degré** de chaque nœud (= nombre de fils qui se rejoignent)
3. Pour chaque face, compter le ratio de sommets de **degré ≥ 4** (= croisements)
4. Si ratio > `max_cross_ratio` → c'est un artefact de croisement → **rejet**

**Exemple** :
- Rectangle de composant : 4 coins avec degré 2 ou 3 → ratio faible → **gardé**
- Petit carré au croisement de 2 fils : 4 coins avec degré 4 → ratio élevé → **rejeté**

**Exception** : les grandes faces (> 4× min_area) avec au moins 2 coins "propres" sont gardées même si le ratio est modéré (cas des gros rectangles traversés par des fils).

#### 2.3 Sous-étape : Filtre "Vide & Solitaire"

Rejette les faces qui sont à la fois :
- **Vides** : pas de texte à l'intérieur ni à proximité (12pt autour)
- **Isolées** : aucun voisin adjacent

**Exceptions** (gardées même si vides et isolées) :
- **Formes compactes** : circularité > 0.50 ou g-ratio > 0.80 → un cercle/carré est un composant par nature
- **Texte à proximité** : un label est souvent À CÔTÉ du composant, pas dedans

#### 2.4 Sous-étape : Smart Merge

Fusionne les sous-faces qui appartiennent au même composant **SANS** fusionner deux composants distincts qui se touchent.

**Algorithme Union-Find** : regroupe les faces en "familles" selon des critères de fusion.

**Critères de fusion** (tous doivent être vrais) :

| Critère | Seuil | Signification |
|---------|-------|---------------|
| **Contact** | buffer(tolerance) | Les faces se touchent physiquement |
| **Bord partagé** | `merge_min_shared_boundary` (0.08) | Longueur bord commun / périmètre petit polygone > seuil |
| **Croissance d'aire** | `merge_max_area_growth` (1.8) | Aire après fusion < somme aires × ratio |
| **Aspect ratio** | `merge_max_aspect_ratio` (6.0) | Le résultat ne doit pas être trop allongé |

**Gardes anti-merge** (rejettent la fusion même si les critères passent) :
- **Deux textes** : si les DEUX faces contiennent du texte → deux composants distincts
- **Bord = fil** : si le bord partagé est une ligne droite simple (colinéarité) avec ratio faible → séparés par un fil

**Résultat** : deux symboles de terre adjacents restent séparés, mais un rectangle coupé en sous-faces est reconstruit.

**Cas spécial : groupes multi-texte**

Si un groupe fusionné contient **plusieurs sous-faces avec chacune un texte**, elles sont gardées séparées (pas fusionnées). Exemple : une rangée de connecteurs côte à côte, chacun avec son label.

---

### Étape 3 : DBSCAN Extractor (formes ouvertes)

**Fichier** : `dbscan_extractor.py`

**Objectif** : Détecter les formes **non fermées** que `polygonize` rate (flèches, symboles de terre, diodes, etc.).

**Principe** : clustering spatial des segments courts orphelins.

#### 3.1 Filtrage des segments

1. **Longueur** : ne garder que les segments < `max_segment_length` (200pt par défaut)
2. **Déjà capturés** : retirer les segments dont le centre tombe dans un polygone Graph

#### 3.2 DBSCAN clustering

Utilise `sklearn.cluster.DBSCAN` sur les **centres** des segments :
- `epsilon` (15pt) : distance max pour qu'un segment rejoigne un cluster
- `min_samples` (2) : nombre min de segments pour former un cluster

**Filtres post-cluster** :
- Trop grand (> `max_cluster_size` 400pt) → rejeté (harnais de câbles fusionnés)
- Trop petit (< `min_cluster_size` 8pt) → rejeté (bruit)

**Sortie** : liste de **bounding boxes** englobant chaque cluster.

**Note** : DBSCAN retourne des rectangles englobants, pas des formes précises. Suffisant pour les symboles simples.

---

### Étape 4 : Déduplication

**Fichier** : `pipeline.py` → fonction `deduplicate()`

**Problème** : Graph et DBSCAN peuvent détecter le même composant (ex: un rectangle détecté par Graph ET clustérisé par DBSCAN).

**Solution** : calcul d'**IoU** (Intersection over Union) entre tous les polygones Graph et toutes les bboxes DBSCAN.

```
IoU = Aire(A ∩ B) / Aire(A ∪ B)
```

**Règle** : si IoU > `dedup_iou_threshold` (0.3), le composant DBSCAN est **rejeté** (priorité au Graph, qui a une meilleure géométrie).

---

### Étape 5 : Classification

**Fichier** : `classifier.py`

**Objectif** : Attribuer une **catégorie métier** à chaque polygone détecté.

#### 5.1 Métriques géométriques

Pour chaque polygone, on calcule :

| Métrique | Formule | Signification |
|----------|---------|---------------|
| **Thickness** | min(largeur, hauteur) du rectangle orienté minimum | Épaisseur du composant |
| **G-ratio** | Aire_poly / Aire_bbox_orienté | Rectangularité (1.0 = rectangle parfait, 0.78 ≈ cercle) |
| **D-ratio** | Aire_matière / Aire_enveloppe | Densité (1.0 = plein, <0.3 = cadre vide) |
| **Circularity** | 4π·Aire / Périmètre² | Circularité (1.0 = cercle parfait) |

#### 5.2 Arbre de décision

**Pour les composants DBSCAN** : tous classés en `Open_Component` (géométrie bbox peu fiable).

**Pour les composants Graph** (formes fermées) :

```
┌─ thickness < thin_wire_threshold (5pt) ?
│   └─ YES → REJET (fil de commande)
│
├─ thickness < 15pt ET aspect_ratio > 8 ?
│   └─ YES → REJET (fil allongé)
│
├─ circularity > 0.85 ?
│   └─ YES → Circle_Component
│
├─ 0.70 < g_ratio < 0.82 ET circ > 0.60 ?
│   └─ YES → Hex_Symbol
│
├─ g_ratio > 0.70 (rectangulaire) ?
│   ├─ thickness < 40pt ?
│   │   ├─ d_ratio > 0.50 → Busbar_Power
│   │   └─ d_ratio ≤ 0.50 → REJET (layout line vide)
│   └─ thickness ≥ 40pt ?
│       ├─ d_ratio > 0.80 → Component_Rect
│       ├─ d_ratio < 0.25 → REJET (cadre layout)
│       └─ 0.25 ≤ d_ratio ≤ 0.80 → Group_Container
│
├─ d_ratio > 0.75 ?
│   └─ YES → Component_Complex (L-shape, triangle)
│
├─ area > 200 ?
│   └─ YES → Unknown_Shape
│
└─ SINON → REJET (trop petit ou indéterminé)
```

**Catégories finales** :

| Catégorie | Description | Exemples |
|-----------|-------------|----------|
| `Component_Rect` | Rectangle plein dense | Relais, ECU, disjoncteurs |
| `Component_Complex` | Forme complexe dense | L-shapes, triangles, connecteurs spéciaux |
| `Circle_Component` | Forme circulaire | Moteurs, voyants, bornes rondes |
| `Hex_Symbol` | Hexagone | Connecteurs off-page |
| `Busbar_Power` | Bus de puissance fin | Rails d'alimentation |
| `Group_Container` | Conteneur logique | Groupes en pointillés |
| `Open_Component` | Forme ouverte (DBSCAN) | Flèches, terres, diodes |
| `Unknown_Shape` | Non classifié | Formes ambiguës |

---

### Étape 6 : Post-Classification Cleanup

**Fichier** : `pipeline.py` → fonction `post_classification_cleanup()`

**Objectif** : Corriger les erreurs résiduelles après classification.

#### 6.1 Règles de nettoyage

| Règle | Condition | Action |
|-------|-----------|--------|
| **Containment** | Composant A contenu à >80% dans B | Supprimer A (sous-partie de B) |
| **Duplicate IoU** | IoU > 0.5 entre A et B, même catégorie | Supprimer le plus petit |
| **Fil traversant** | A très allongé (aspect > 6) et overlap > 30% avec B | Supprimer A (fil polygonisé) |

**Tri** : les composants sont triés par **aire décroissante** pour donner priorité aux gros (évite de supprimer un gros rectangle au profit d'un petit artefact).

---

### Étape 7 : Export

**Fichier** : `exporter.py`

#### 7.1 Crops PNG

Pour chaque composant détecté :
1. Rendre la page PDF en image haute résolution (300 DPI par défaut)
2. Découper un crop autour de la bbox du composant (avec padding de 20px)
3. Sauvegarder dans `output/crops/{category}/{filename}_p{page}_id{id}.png`

**Conversion de coordonnées** :
```python
# PDF (72 DPI) → Image (300 DPI)
px = x_pdf * (300 / 72) = x_pdf * 4.167
```

#### 7.2 Métadonnées JSON

Structure :
```json
{
  "source_file": "schema.pdf",
  "pipeline": "hybrid_v1",
  "pages": [
    {
      "page_index": 0,
      "total_objects": 42,
      "by_source": {"graph": 38, "dbscan": 4},
      "objects": [
        {
          "id": 0,
          "type": "Component_Rect",
          "bbox": [100.5, 200.3, 150.8, 250.6],
          "source": "graph",
          "thickness": 12.5,
          "circularity": 0.85,
          "g_ratio": 0.92,
          "d_ratio": 0.88
        }
      ]
    }
  ]
}
```

#### 7.3 Labels YOLO (optionnel)

Format : `class_id cx cy w h` (normalisés 0-1)

Mapping par défaut :
```python
{
  "Component_Rect": 0,
  "Component_Complex": 1,
  "Circle_Component": 2,
  "Hex_Symbol": 3,
  "Busbar_Power": 4,
  "Group_Container": 5,
  "Open_Component": 6,
  "Unknown_Shape": 7
}
```

---

## 4. Modules détaillés

### 4.1 vector_utils.py

**Responsabilité** : Interface avec PyMuPDF pour l'extraction vectorielle.

**Classe principale** : `VectorSegment`

```python
@dataclass
class VectorSegment:
    x1, y1, x2, y2: float
    
    @property
    def length(self) -> float
        # Distance euclidienne
    
    @property
    def center(self) -> tuple
        # Point milieu
    
    def as_linestring(self) -> LineString
        # Conversion Shapely pour géométrie
```

**Fonctions clés** :

- `extract_segments_from_page(page)` : parcourt tous les paths du PDF, gère 3 types :
  - `"l"` (line) → 1 segment
  - `"c"` (curve) → 8 segments (approximation multi-points)
  - `"re"` (rectangle) → 4 segments (un par côté)

- `extract_text_blocks(page)` : retourne les bboxes + texte de tous les blocs textuels

---

### 4.2 graph_extractor.py

**Responsabilité** : Détection de formes fermées par analyse topologique.

**Fonctions principales** :

#### `find_all_faces(segments)`
```python
# Fusionne tous les segments
merged = unary_union(lines)
# Trouve toutes les faces fermées
faces = list(polygonize(merged))
```

#### `build_graph(segments, precision)`
```python
G = nx.Graph()
for seg in segments:
    p1, p2 = seg.as_rounded_endpoints(precision)
    G.add_edge(p1, p2)
return G
```

#### `filter_by_node_degree(faces, node_degrees, config)`
```python
for face in faces:
    coords = face.exterior.coords[:-1]
    cross_count = sum(1 for c in coords if snap_to_graph(c, node_degrees) >= 4)
    ratio = cross_count / len(coords)
    if ratio > config.max_cross_ratio:
        # Exception pour grandes faces
        if not (is_big and has_low_degree_corners and ratio < 0.85):
            reject(face)
```

#### `smart_merge_faces(faces, config, text_bboxes)`

Union-Find avec 4 tests de fusion + 2 gardes anti-merge :

**Tests** :
1. Contact physique (buffer tolerance)
2. Bord partagé significatif (> `merge_min_shared_boundary`)
3. Croissance d'aire raisonnable (< `merge_max_area_growth`)
4. Aspect ratio compact (< `merge_max_aspect_ratio`)

**Gardes** :
- Si les deux faces ont du texte → pas de merge
- Si le bord est colinéaire (fil) avec ratio faible → pas de merge

**Cas spécial** : lors de la fusion finale, si un groupe contient plusieurs sous-faces avec chacune du texte, elles sont gardées individuellement (pas fusionnées en un gros blob).

---

### 4.3 dbscan_extractor.py

**Responsabilité** : Clustering des segments orphelins.

**Pipeline interne** :

```python
def run_dbscan_extraction(segments, captured_polygons, config):
    # 1. Filtrer par longueur
    short_segs = [s for s in segments if s.length < max_segment_length]
    
    # 2. Retirer les segments capturés par Graph
    orphans = remove_already_captured(short_segs, captured_polygons)
    
    # 3. DBSCAN sur les centres
    centers = np.array([seg.center for seg in orphans])
    clustering = DBSCAN(eps=epsilon, min_samples=min_samples).fit(centers)
    
    # 4. Construire les bboxes des clusters
    for label_id in set(clustering.labels_):
        cluster_segs = [orphans[i] for i in np.where(labels == label_id)]
        bbox = (min_x, min_y, max_x, max_y)
        
        # Filtres taille
        if width > max_cluster_size or height > max_cluster_size:
            continue  # Trop gros
        if width < min_cluster_size and height < min_cluster_size:
            continue  # Trop petit
        
        clusters.append(bbox)
    
    return [box(*bbox) for bbox in clusters]
```

**Fonction clé** : `remove_already_captured()`

```python
captured_zone = unary_union([p.buffer(2.0) for p in captured_polygons])
orphans = [seg for seg in segments 
           if not captured_zone.contains(Point(seg.center))]
```

---

### 4.4 classifier.py

**Responsabilité** : Calcul des métriques et arbre de décision.

**Fonction `compute_metrics(poly)`** :

```python
def compute_metrics(poly):
    poly_clean = poly.buffer(0)  # Fix topologique
    
    # Rectangle orienté minimum
    box_rot = poly_clean.minimum_rotated_rectangle
    coords = box_rot.exterior.coords.xy
    edge1 = distance(coords[0], coords[1])
    edge2 = distance(coords[1], coords[2])
    thickness = min(edge1, edge2)
    
    # Enveloppe simplifiée
    poly_env = Polygon(poly_clean.exterior).simplify(0.5)
    box_env = poly_env.minimum_rotated_rectangle
    
    # G-ratio
    g_ratio = poly_env.area / box_env.area if box_env.area > 0 else 0
    
    # D-ratio
    d_ratio = poly_clean.area / poly_env.area if poly_env.area > 0 else 0
    
    # Circularité
    perimeter = poly_env.length
    circularity = (4 * π * poly_env.area) / (perimeter²) if perimeter > 0 else 0
    
    return {thickness, g_ratio, d_ratio, circularity}
```

**Fonction `classify_polygon(poly, config, source)`** :

Implémente l'arbre de décision décrit dans la section 3.5.

---

### 4.5 config.py

**Responsabilité** : Centraliser TOUS les hyperparamètres.

**Structure** :
```python
@dataclass
class GraphConfig:
    # ... paramètres Graph
    
@dataclass
class DBSCANConfig:
    # ... paramètres DBSCAN
    
@dataclass
class ClassifierConfig:
    # ... paramètres Classification
    
@dataclass
class PipelineConfig:
    graph: GraphConfig
    dbscan: DBSCANConfig
    classifier: ClassifierConfig
    export: ExportConfig
    dedup_iou_threshold: float
    containment_threshold: float
```

Voir section 5 pour la liste complète.

---

## 5. Hyperparamètres et configuration

### 5.1 GraphConfig (graph_extractor)

| Paramètre | Défaut | Unité | Effet si trop bas | Effet si trop haut |
|-----------|--------|-------|-------------------|-------------------|
| `coord_precision` | 1 | décimales | Nœuds non fusionnés → graphe fragmenté | Nœuds sur-fusionnés → perte de géométrie |
| `max_cross_ratio` | 0.5 | ratio | Rejette des vrais composants traversés | Garde des croisements de fils |
| `merge_neighbor_tolerance` | 1.0 | pts PDF | Faces adjacentes non détectées | Fusionne des composants distants |
| `merge_min_shared_boundary` | 0.08 | ratio | Fusionne des composants distincts | Empêche reconstruction de L-shapes |
| `merge_max_area_growth` | 1.8 | facteur | Fusionne des formes avec gros trous vides | Empêche fusion de sous-faces légitimes |
| `merge_max_aspect_ratio` | 6.0 | ratio | Fusionne deux symboles côte à côte | Empêche fusion de rectangles allongés |

**Interactions clés** :

- `merge_min_shared_boundary` vs `merge_max_area_growth` : équilibre entre "partager un vrai bord" et "ne pas créer de trou vide"
- `max_cross_ratio` haut + `merge_max_aspect_ratio` bas : aggressif sur la détection mais conservateur sur la fusion

---

### 5.2 DBSCANConfig (dbscan_extractor)

| Paramètre | Défaut | Unité | Effet si trop bas | Effet si trop haut |
|-----------|--------|-------|-------------------|-------------------|
| `epsilon` | 15.0 | pts PDF | Segments isolés → sous-détection | Clusters fusionnés → sur-détection |
| `min_samples` | 2 | segments | Bruit classé comme composant | Symboles simples ratés |
| `max_segment_length` | 200.0 | pts PDF | Rate les grands symboles ouverts | Inclut des fils longs → faux clusters |
| `max_cluster_size` | 400.0 | pts PDF | Rate les gros groupes ouverts | Fusionne des harnais entiers |
| `min_cluster_size` | 8.0 | pts PDF | Garde du bruit résiduel | Rejette de petits symboles |

**Cas d'usage** :

- **Symboles denses** (connecteurs) : `epsilon` bas (10-12), `min_samples` élevé (3-4)
- **Flèches longues** : `max_segment_length` haut (300+), `epsilon` moyen (15-20)
- **Schémas complexes** : `max_cluster_size` haut (600+)

---

### 5.3 ClassifierConfig (classifier)

#### 5.3.1 Seuils de rejet (fils)

| Paramètre | Défaut | Unité | Description | Tuning |
|-----------|--------|-------|-------------|--------|
| `thin_wire_threshold` | 5.0 | pts PDF | Épaisseur min pour un composant | ↓ si fins fils classés comme composants |
| `max_aspect_ratio` | 8.0 | ratio | Aspect max longueur/largeur | ↓ si des fils allongés passent |
| `aspect_ratio_max_thickness` | 15.0 | pts PDF | Épaisseur max pour appliquer aspect_ratio | ↑ pour appliquer le filtre aux busbars |

**Workflow de tuning fils** :
1. Si des fils verticaux/horizontaux sont détectés comme composants :
   - Baisser `max_aspect_ratio` (6.0 → 5.0)
   - Monter `thin_wire_threshold` (5.0 → 7.0)
2. Si des busbars fins sont rejetés à tort :
   - Monter `aspect_ratio_max_thickness` (15 → 25)

#### 5.3.2 Seuils de classification

| Paramètre | Défaut | Unité | Catégorie affectée | Effet si modifié |
|-----------|--------|-------|-------------------|------------------|
| `rect_ratio_threshold` | 0.70 | ratio | Tous rectangulaires | ↓ = plus permissif (rectangles "sales" acceptés) |
| `circle_threshold` | 0.85 | ratio | Circle_Component | ↓ = accepte des formes moins rondes |
| `density_filled` | 0.80 | ratio | Component_Rect | ↓ = accepte des composants moins denses |
| `density_empty` | 0.25 | ratio | Rejet (layout) | ↑ = rejette plus de cadres vides |
| `density_busbar_min` | 0.50 | ratio | Busbar_Power | ↑ = busbars doivent être plus pleins |
| `busbar_threshold` | 40.0 | pts PDF | Busbar vs Component | ↑ = plus de composants classés comme busbars |

**Matrice G-ratio / D-ratio** :

```
D-ratio ↑
   1.0 │ Component_Rect │ Component_Complex
   0.8 │────────────────┼─────────────────
       │                │
   0.5 │   Busbar       │
   0.25│────────────────┼─────────────────
       │   Layout       │
   0.0 └────────────────┴─────────────────→ G-ratio
       0.0            0.70              1.0
```

#### 5.3.3 Seuils d'aire

| Paramètre | Défaut | Unité | Effet |
|-----------|--------|-------|-------|
| `min_area` | 80.0 | pts²PDF | Rejette les petits artefacts |
| `max_area` | 150000.0 | pts²PDF | **⚠ Ne pas trop contraindre** (voir note) |
| `unknown_min_area` | 200.0 | pts²PDF | Aire min pour garder un Unknown |

**⚠ Note importante sur `max_area`** :

Ce paramètre est appliqué **après classification**, pas pendant l'extraction Graph. Si tu le baisses trop :
- Les gros composants **passent** quand même par Graph/DBSCAN
- Mais sont **rejetés** lors du filtre d'aire en classification
- ❌ **Conséquence** : les sous-faces internes (qui étaient dans le gros composant) sont perdues car elles ont été fusionnées

**Recommandation** : garder `max_area` très élevé (150000) et filtrer en post-traitement si besoin.

---

### 5.4 PipelineConfig (pipeline-level)

| Paramètre | Défaut | Unité | Rôle |
|-----------|--------|-------|------|
| `dedup_iou_threshold` | 0.3 | ratio | Seuil IoU pour dédupliquer Graph vs DBSCAN |
| `containment_threshold` | 0.8 | ratio | Seuil de containment pour supprimer les sous-composants |

**Effet `dedup_iou_threshold`** :
- Trop bas (0.1) : garde des doublons (même composant détecté 2 fois)
- Trop haut (0.6) : perd des composants légèrement différents Graph/DBSCAN

**Effet `containment_threshold`** :
- Trop bas (0.5) : supprime des composants valides partiellement dans un autre
- Trop haut (0.95) : garde des doublons quasi-identiques

---

### 5.5 ExportConfig (export)

| Paramètre | Défaut | Unité | Description |
|-----------|--------|-------|-------------|
| `dpi` | 300 | dpi | Résolution des crops PNG |
| `padding` | 20 | pixels | Marge autour des crops |

**Choix DPI** :
- 150 DPI : rapide, suffisant pour preview
- 300 DPI : standard industrie, bon équilibre
- 600 DPI : haute qualité, fichiers lourds

---

## 6. Auto-calibration

### 6.1 Pourquoi auto-calibrer ?

**Problème** : les schémas électriques varient énormément selon le constructeur (Schneider, ABB, Siemens, etc.). Une config optimale pour un constructeur peut être catastrophique pour un autre.

**Solution** : deux systèmes complémentaires.

---

### 6.2 PipelineTuner (optimisation Optuna)

**Principe** : optimisation bayésienne de Tree-structured Parzen Estimator (TPE).

**Fichier** : `tuner.py` → classe `PipelineTuner`

#### Mode 1 : Heuristique (sans annotations)

Score basé sur la **qualité intrinsèque** des détections :

| Critère | Poids | Symptôme |
|---------|-------|----------|
| Nombre de composants | -0.3 | Trop peu (<3) ou trop (>200) |
| Chevauchements | -0.3 | IoU > 0.3 entre composants |
| Fils classés comme composants | -0.2 | Aspect ratio > 10 |
| Composants énormes (faux merges) | -0.1 | Aire > 10000 |
| Trop d'Unknown | -0.1 | >30% de la catégorie Unknown |
| Diversité de catégories | +0.15 | ≥3 catégories trouvées |
| Ratio Graph/DBSCAN équilibré | +0.1 | 30% < ratio < 95% |

**Formule** : `score = 0.5 + bonus - pénalités` (clamped [0, 1])

#### Mode 2 : Supervisé (avec annotations YOLO)

Score = **F1-score** par matching IoU avec ground truth :

```python
# Pour chaque paire (détection, GT) avec IoU > 0.5
TP = nombre de matches
FP = détections sans match
FN = GT sans détection

Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 · Precision · Recall / (Precision + Recall)
```

#### Usage

```python
from hybrid_pipeline.tuner import PipelineTuner

tuner = PipelineTuner(
    pdf_paths=["schema1.pdf", "schema2.pdf"],
    gt_labels={"schema1_p0": "labels/schema1_p0.txt"},  # Optionnel
    pages=[0]
)

best_config = tuner.run(n_trials=100, storage="sqlite:///tuning.db")
PipelineTuner.export_best_config(best_config, "best_config.json")
```

**CLI** :
```bash
# Mode heuristique
python -m hybrid_pipeline.tuner schema.pdf --trials 100 -o config.json

# Mode supervisé
python -m hybrid_pipeline.tuner schema.pdf --gt labels/schema_p0.txt --trials 200

# Multi-PDF
python -m hybrid_pipeline.tuner pdf_folder/ --gt-dir labels/ --trials 300
```

---

### 6.3 AutoCalibrator (diagnostic itératif)

**Principe** : analyse les symptômes → applique des corrections ciblées → converge en 3-5 itérations.

**Fichier** : `tuner.py` → classe `AutoCalibrator`

#### Phase 1 : Diagnostic rapide

**Algorithme** :

```python
for iteration in range(max_iterations):
    # Run pipeline avec config actuelle
    components = run_pipeline(config)
    
    # Analyser les symptômes
    symptoms = diagnose_detections(components)
    # Ex: {"wires_as_components": 0.3, "huge_false_merges": 0.15}
    
    # Calculer le score
    score = score_heuristic(components)
    
    # Convergence ?
    if score > best_score:
        best_score = score
        best_config = config
    elif no_improvement_count >= 2:
        break  # Converged
    
    # Appliquer les corrections
    corrections = compute_corrections(symptoms)
    # Ex: {"cls.max_aspect_ratio": ("decrease", 0.8)}
    config = apply_corrections(config, corrections)
```

**Symptômes détectés** :

| Symptôme | Condition | Correction appliquée |
|----------|-----------|---------------------|
| `wires_as_components` | Aspect ratio > 8 | ↓ `max_aspect_ratio` (×0.7-0.9)<br>↑ `thin_wire_threshold` |
| `excessive_overlaps` | IoU > 0.3 entre paires | ↓ `containment_threshold`<br>↓ `dedup_iou_threshold` |
| `huge_false_merges` | Énormes composants | ↓ `merge_max_area_growth` (×0.85)<br>↑ `merge_min_shared_boundary` (×1.3) |
| `too_many_unknowns` | >30% Unknown | ↓ `rect_ratio_threshold`, `density_filled`, `circle_threshold` |
| `under_detection` | <5 composants | ↓ `min_area`, `thin_wire_threshold`<br>↑ `max_cross_ratio`, `epsilon` |
| `over_detection` | >150 composants | ↑ `min_area`, `thin_wire_threshold`, `min_samples` |
| `graph_finds_nothing` | 0 Graph détections | ↑ `max_cross_ratio` (×1.5)<br>↓ `min_area` (×0.5) |

#### Phase 2 : Fine-tuning Optuna (optionnel)

Lance un tuner Optuna classique avec la config de Phase 1 comme seed. Affine les paramètres autour de cette zone.

#### Usage

```python
from hybrid_pipeline.tuner import AutoCalibrator

calibrator = AutoCalibrator(["schema_new_vendor.pdf"])
best_config = calibrator.run(
    max_iterations=8,        # Phase 1
    optuna_trials=50,        # Phase 2
    do_optuna_phase=True,
    output_path="config_vendor.json"
)
```

**CLI** :
```bash
# Auto complet (diagnostic + Optuna)
python -m hybrid_pipeline.tuner schema.pdf --auto -o config.json

# Diagnostic seul (rapide, ~30 sec)
python -m hybrid_pipeline.tuner schema.pdf --auto-only -o config.json

# Multi-PDF même constructeur
python -m hybrid_pipeline.tuner pdf_vendor/ --auto -o config_vendor.json
```

**Workflow recommandé pour un nouveau constructeur** :
1. Prendre 2-3 PDF représentatifs
2. Lancer `--auto` sur ces PDFs
3. Sauvegarder la config obtenue
4. Réutiliser cette config pour tous les PDFs du même constructeur
5. Re-calibrer si les schémas changent significativement

---

## 7. Cas d'usage et workflows

### 7.1 Workflow basique

```python
from hybrid_pipeline import HybridPipeline, PipelineConfig

# Config par défaut
pipeline = HybridPipeline("schema.pdf")
results = pipeline.run(
    output_dir="output/",
    pages=[0],  # Première page
    export_crops_flag=True,
    export_json=True,
    export_yolo=False
)

# Accéder aux composants détectés
page_0_components = results[0]
for comp in page_0_components:
    print(f"{comp.category}: bbox={comp.bbox}, source={comp.source}")
```

### 7.2 Config custom

```python
from hybrid_pipeline import PipelineConfig
from hybrid_pipeline.config import GraphConfig, ClassifierConfig

config = PipelineConfig()

# Ajuster pour des fils plus épais
config.classifier.thin_wire_threshold = 8.0
config.classifier.busbar_threshold = 50.0

# Être plus strict sur les fusions
config.graph.merge_min_shared_boundary = 0.12
config.graph.merge_max_area_growth = 1.5

pipeline = HybridPipeline("schema.pdf", config)
```

### 7.3 Charger une config sauvegardée

```python
from hybrid_pipeline.tuner import PipelineTuner

config = PipelineTuner.load_config("best_config.json")
pipeline = HybridPipeline("new_schema.pdf", config)
```

### 7.4 Batch processing

```python
import os
from pathlib import Path

config = PipelineTuner.load_config("config_schneider.json")
pdf_dir = Path("pdfs_schneider/")

for pdf_file in pdf_dir.glob("*.pdf"):
    print(f"Processing {pdf_file.name}...")
    pipeline = HybridPipeline(str(pdf_file), config)
    output_dir = f"output/{pdf_file.stem}"
    pipeline.run(output_dir, export_crops_flag=True)
```

### 7.5 QA visuel

```python
from hybrid_pipeline.visualizer import visualize_page

# Extraire et visualiser
pipeline = HybridPipeline("schema.pdf")
components = pipeline.process_page(0)

visualize_page(
    "schema.pdf",
    components,
    page_index=0,
    figsize=(20, 14),
    show_ids=True,         # Afficher les IDs
    show_metrics=True,     # Afficher G/D-ratio sur chaque composant
    title="QA Page 0"
)
```

### 7.6 Tuning itératif

```bash
# 1. Baseline avec config par défaut
python run_hybrid.py  # Ajuster PDF_PATH dans le script

# 2. Vérifier visuellement les erreurs
python -c "
from hybrid_pipeline import HybridPipeline
from hybrid_pipeline.visualizer import visualize_page
p = HybridPipeline('schema.pdf')
c = p.process_page(0)
visualize_page('schema.pdf', c, 0, show_ids=True)
"

# 3. Auto-calibration
python -m hybrid_pipeline.tuner schema.pdf --auto -o tuned.json

# 4. Tester la config tuned
python -c "
from hybrid_pipeline import HybridPipeline
from hybrid_pipeline.tuner import PipelineTuner
config = PipelineTuner.load_config('tuned.json')
p = HybridPipeline('schema.pdf', config)
p.run('output_tuned/')
"
```

---

## 8. Troubleshooting

### 8.1 Problèmes fréquents

#### Cercles non détectés

**Symptôme** : Les composants circulaires ne sont pas trouvés.

**Causes** :
1. ❌ Ancienne version avec mauvaise approximation Bézier → **corrigé** (8 segments par courbe)
2. Rejetés par `filter_isolated_empty` (solo sans texte) → **corrigé** (exception pour circularité > 0.50)
3. `circle_threshold` trop haut

**Diagnostic** :
```python
# Vérifier la circularité
from hybrid_pipeline.classifier import compute_metrics
metrics = compute_metrics(polygon_suspect)
print(f"Circularité: {metrics['circularity']}")  # Doit être > 0.85
```

**Solutions** :
- Baisser `circle_threshold` (0.85 → 0.75)
- Vérifier que le critère "forme compacte" de `filter_isolated_empty` est actif

---

#### Deux composants fusionnés (ex : deux flèches)

**Symptôme** : Deux symboles adjacents deviennent un seul composant.

**Causes** :
1. `merge_min_shared_boundary` trop bas
2. Pas de texte dans les faces → le garde "double texte" ne s'applique pas
3. Le fil entre eux est trop court → le garde "bord colinéaire" ne s'applique pas

**Solutions** :
- Monter `merge_min_shared_boundary` (0.08 → 0.12)
- Baisser `merge_max_aspect_ratio` (6.0 → 4.0)
- Si chaque symbole a un label, vérifier que le garde "deux textes" fonctionne

**Diagnostic** :
```python
# Vérifier le shared boundary ratio entre deux faces
from hybrid_pipeline.graph_extractor import _shared_boundary_length
shared_len = _shared_boundary_length(face_a, face_b)
ratio = shared_len / min(face_a.length, face_b.length)
print(f"Shared boundary ratio: {ratio}")  # Doit être > 0.08
```

---

#### Fils classés comme composants

**Symptôme** : Des segments de fil horizontaux/verticaux sont détectés comme busbars ou composants.

**Causes** :
1. `thin_wire_threshold` trop bas
2. `max_aspect_ratio` trop élevé
3. `aspect_ratio_max_thickness` trop bas (le filtre ne s'applique pas)

**Solutions** :
```python
config.classifier.thin_wire_threshold = 7.0      # Was 5.0
config.classifier.max_aspect_ratio = 6.0         # Was 8.0
config.classifier.aspect_ratio_max_thickness = 20.0  # Was 15.0
```

**Diagnostic** :
```python
# Pour un composant suspect
metrics = compute_metrics(polygon)
if metrics['thickness'] < 15:
    aspect = calculate_aspect_ratio(polygon)
    print(f"Aspect ratio: {aspect}")  # Si > 8, c'est un fil
```

---

#### Composants perdus quand on change `max_area`

**Symptôme** : En augmentant `max_area`, un gros groupe apparaît mais les sous-composants disparaissent.

**Cause** : `max_area` est vérifié au niveau du **node degree filter**, rejetant les grandes faces avant la fusion. Si elles sont fusionnées en amont, les sous-faces internes sont perdues.

**Solution** : ✅ **Corrigé dans la dernière version** — `max_area` n'est plus appliqué dans `filter_by_node_degree`. Seul `min_area` rejette les petites faces bruiteuses. Les grandes faces passent et la fusion intelligente décide si elle les fusionne ou non.

**Fallback** : si le problème persiste, activer le garde "multi-texte" dans `smart_merge_faces` :
```python
# Dans smart_merge_faces, après fusion d'un groupe :
textful = sum(1 for idx in indices if face_has_text[idx])
if textful > 1:
    for idx in indices:
        result.append(faces[idx])  # Garder séparé
    continue
```

---

#### Trop de "Unknown_Shape"

**Symptôme** : Beaucoup de composants finissent en Unknown.

**Causes** :
1. Seuils de classification trop stricts
2. Formes particulières au constructeur

**Solutions** :
```python
# Relaxer les seuils
config.classifier.rect_ratio_threshold = 0.65   # Was 0.70
config.classifier.density_filled = 0.75         # Was 0.80
config.classifier.circle_threshold = 0.80       # Was 0.85
```

**Ou** : lancer l'auto-calibration qui détecte ce symptôme et ajuste automatiquement.

---

#### DBSCAN ne trouve rien

**Symptôme** : `n_dbscan = 0` dans les stats.

**Causes** :
1. `epsilon` trop bas → segments trop espacés
2. `min_samples` trop haut
3. `max_segment_length` trop bas → segments filtrés avant clustering

**Solutions** :
```python
config.dbscan.epsilon = 20.0              # Was 15.0
config.dbscan.min_samples = 2             # Keep low
config.dbscan.max_segment_length = 300.0  # Was 200.0
```

---

### 8.2 Workflow de debug

1. **Visualiser la page** avec `show_ids=True` et `show_metrics=True`
2. **Identifier le composant problématique** par son ID
3. **Extraire ses métriques** :
   ```python
   comp = components[id_probleme]
   print(f"Category: {comp.category}")
   print(f"Thickness: {comp.thickness}")
   print(f"G-ratio: {comp.g_ratio}")
   print(f"D-ratio: {comp.d_ratio}")
   print(f"Circularity: {comp.circularity}")
   ```
4. **Comparer aux seuils** dans `ClassifierConfig`
5. **Ajuster le paramètre pertinent**
6. **Re-test**

---

### 8.3 Vérifier l'impact d'un paramètre

```python
from hybrid_pipeline import HybridPipeline, PipelineConfig

# Baseline
config = PipelineConfig()
p1 = HybridPipeline("schema.pdf", config)
r1 = p1.process_page(0)

# Variant
config.classifier.thin_wire_threshold = 8.0
p2 = HybridPipeline("schema.pdf", config)
r2 = p2.process_page(0)

# Comparaison
print(f"Baseline: {len(r1)} composants")
print(f"Variant:  {len(r2)} composants")

# Différences par catégorie
from collections import Counter
c1 = Counter(c.category for c in r1)
c2 = Counter(c.category for c in r2)
for cat in set(c1.keys()) | set(c2.keys()):
    print(f"{cat}: {c1.get(cat, 0)} → {c2.get(cat, 0)}")
```

---

## 9. Référence rapide

### 9.1 Commandes CLI essentielles

```bash
# Extraction simple
python run_hybrid.py  # Ajuster PDF_PATH dans le script

# Auto-calibration rapide
python -m hybrid_pipeline.tuner schema.pdf --auto-only -o config.json

# Auto-calibration complète (Optuna)
python -m hybrid_pipeline.tuner schema.pdf --auto --trials 100 -o config.json

# Tuning supervisé (avec labels YOLO)
python -m hybrid_pipeline.tuner schema.pdf --gt labels/schema_p0.txt --trials 200

# Visualisation QA
python -c "
from hybrid_pipeline import HybridPipeline
from hybrid_pipeline.visualizer import visualize_page
p = HybridPipeline('schema.pdf')
c = p.process_page(0)
visualize_page('schema.pdf', c, 0, show_ids=True)
"
```

### 9.2 Paramètres les plus impactants

| Rang | Paramètre | Impact sur |
|------|-----------|-----------|
| 🥇 1 | `thin_wire_threshold` | Rejet des fils |
| 🥈 2 | `merge_min_shared_boundary` | Fusion de composants distincts |
| 🥉 3 | `max_cross_ratio` | Faux positifs aux croisements |
| 4 | `max_aspect_ratio` | Fils allongés classés comme composants |
| 5 | `dbscan.epsilon` | Détection des formes ouvertes |
| 6 | `rect_ratio_threshold` | Classification rectangles vs complexes |
| 7 | `density_filled` | Distinction composant plein / cadre |
| 8 | `merge_max_area_growth` | Faux merges |

### 9.3 Cheat sheet tuning

| Symptôme | Paramètre à ajuster | Direction |
|----------|-------------------|-----------|
| Fils détectés comme composants | `thin_wire_threshold`<br>`max_aspect_ratio` | ↑<br>↓ |
| Deux symboles fusionnés | `merge_min_shared_boundary`<br>`merge_max_aspect_ratio` | ↑<br>↓ |
| Cercles ratés | `circle_threshold`<br>`filter_isolated_empty` | ↓<br>Check |
| Beaucoup d'Unknown | `rect_ratio_threshold`<br>`density_filled` | ↓<br>↓ |
| Sous-détection | `min_area`<br>`thin_wire_threshold` | ↓<br>↓ |
| Sur-détection | `min_area`<br>`min_samples` | ↑<br>↑ |
| DBSCAN vide | `epsilon`<br>`max_segment_length` | ↑<br>↑ |
| Croisements de fils détectés | `max_cross_ratio` | ↓ |

---

## 10. Annexes

### 10.1 Format JSON métadonnées

```json
{
  "source_file": "schema_electrical.pdf",
  "pipeline": "hybrid_v1",
  "pages": [
    {
      "page_index": 0,
      "total_objects": 42,
      "by_source": {
        "graph": 38,
        "dbscan": 4
      },
      "objects": [
        {
          "id": 0,
          "type": "Component_Rect",
          "bbox": [100.5, 200.3, 150.8, 250.6],
          "source": "graph",
          "thickness": 12.5,
          "circularity": 0.12,
          "g_ratio": 0.92,
          "d_ratio": 0.88
        },
        {
          "id": 1,
          "type": "Circle_Component",
          "bbox": [300.0, 150.0, 350.0, 200.0],
          "source": "graph",
          "thickness": 25.0,
          "circularity": 0.95,
          "g_ratio": 0.78,
          "d_ratio": 1.0
        }
      ]
    }
  ]
}
```

### 10.2 Format YOLO labels

Fichier : `{basename}_p{page}.txt`

Format : `class_id cx cy w h` (normalisés 0-1)

```
0 0.250000 0.350000 0.120000 0.080000
2 0.600000 0.450000 0.090000 0.090000
1 0.800000 0.200000 0.150000 0.100000
```

Mapping par défaut :
```python
{
    "Component_Rect": 0,
    "Component_Complex": 1,
    "Circle_Component": 2,
    "Hex_Symbol": 3,
    "Busbar_Power": 4,
    "Group_Container": 5,
    "Open_Component": 6,
    "Unknown_Shape": 7
}
```

---

## Glossaire

| Terme | Définition |
|-------|------------|
| **Polygonize** | Algorithme Shapely qui trouve toutes les faces fermées dans un arrangement de lignes |
| **Node Degree** | Nombre d'arêtes connectées à un nœud dans un graphe |
| **G-ratio** | Geometry ratio = Aire polygone / Aire bbox orienté (mesure la rectangularité) |
| **D-ratio** | Density ratio = Aire matière / Aire enveloppe (mesure le remplissage) |
| **IoU** | Intersection over Union = Aire(A∩B) / Aire(A∪B) |
| **Smart Merge** | Fusion de sous-faces avec critères géométriques + gardes anti-merge |
| **Union-Find** | Structure de données pour regrouper des éléments en ensembles disjoints |
| **DBSCAN** | Density-Based Spatial Clustering of Applications with Noise |
| **TPE** | Tree-structured Parzen Estimator (algorithme d'optimisation Optuna) |
| **Bézier curve** | Courbe paramétrique utilisée dans les PDFs pour les formes courbes |

---

**Version** : 1.0  
**Date** : Février 2026  
**Auteur** : Pipeline Hybride d'Extraction de Composants  
**Contact** : Voir README.md pour support
