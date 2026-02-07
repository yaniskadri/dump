import numpy as np
import cv2
import torch
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from segment_anything import sam_model_registry, SamPredictor

# --- 1. FONCTION DE GROUPEMENT (LES ISLANDS) ---
def get_islands(polygons_coords, gap=20):
    """
    Groupe les polygones granulaires en 'Islands'.
    Retourne une liste d'islands, chaque island contenant ses polygones membres.
    """
    shapely_polys = [Polygon(p) for p in polygons_coords if len(p) >= 3]
    
    # On crée des zones tampons pour qu'ils se touchent
    buffered = [p.buffer(gap) for p in shapely_polys]
    merged = unary_union(buffered)
    
    # Séparer en entités distinctes (Islands)
    if isinstance(merged, Polygon):
        island_shapes = [merged]
    else:
        island_shapes = list(merged.geoms)
        
    islands_data = []
    for island_shape in island_shapes:
        # Trouver quels polygones originaux appartiennent à cette île
        members = [p for p in shapely_polys if island_shape.intersects(p)]
        # Trier par aire pour identifier le 'Parent' (le plus grand rectangle de l'île)
        members.sort(key=lambda x: x.area, reverse=True)
        
        islands_data.append({
            'bbox': list(island_shape.bounds), # [xmin, ymin, xmax, ymax]
            'parent': members[0],              # Le plus grand polygone de l'île
            'children': members[1:]            # Tout le reste à l'intérieur
        })
    return islands_data

# --- 2. TRAITEMENT SAM (ISLAND PAR ISLAND) ---
def run_sam_hierarchical(image, polygons_coords):
    # Setup SAM (9GB VRAM safe)
    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
    sam.to(device="cuda").half()
    predictor = SamPredictor(sam)
    predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Création des Islands
    islands = get_islands(polygons_coords)
    print(f"Nombre d'islands détectées : {len(islands)}")

    final_masks = []

    for idx, island in enumerate(islands):
        # PROMPT : Box de l'île entière
        box = np.array(island['bbox'])
        
        # PROMPT : Point positif sur le parent (le grand rectangle)
        pos_point = [island['parent'].centroid.x, island['parent'].centroid.y]
        coords = [pos_point]
        labels = [1]
        
        # PROMPT : Points négatifs sur les enfants (les composants internes)
        # On en prend max 10 pour la stabilité
        for child in island['children'][:10]:
            coords.append([child.centroid.x, child.centroid.y])
            labels.append(0) # LABEL 0 = EXCLUSION

        # Appel SAM pour cette île spécifique
        masks, _, _ = predictor.predict(
            point_coords=np.array(coords),
            point_labels=np.array(labels),
            box=box,
            multimask_output=False
        )
        
        final_masks.append({
            'mask': masks[0],
            'box': box,
            'id': idx
        })
        
    return final_masks, image

# --- 3. VISUALISATION DES RÉSULTATS ---
def visualize(image, results):
    overlay = image.copy()
    for res in results:
        color = np.random.randint(0, 255, (3,)).tolist()
        mask = res['mask']
        # Appliquer la couleur là où le masque est vrai
        overlay[mask] = overlay[mask] * 0.5 + np.array(color) * 0.5
        # Dessiner la box de l'île
        bx = res['box']
        cv2.rectangle(overlay, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), color, 2)
    
    cv2.imshow("Detection par Island", overlay)
    cv2.waitKey(0)