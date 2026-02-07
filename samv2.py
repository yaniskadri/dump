import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

# 1. Configuration et Chargement (Optimisé pour 9GB VRAM)
checkpoint = "sam_vit_b_01ec64.pth" # Utilise le modèle 'base' pour la vitesse
model_type = "vit_b"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

def get_mask_overlay(mask, color):
    """Crée une superposition colorée pour le masque."""
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    color_mask[mask] = color
    return color_mask

def run_sam_on_islands(image_path, list_of_island_data):
    # Charger l'image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    
    output_image = image.copy()
    
    with torch.inference_mode():
        for item in list_of_island_data:
            # 'box' : [xmin, ymin, xmax, ymax]
            # 'internal_points' : [[x,y], [x,y]...] des composants à exclure
            box = np.array(item['box'])
            
            # Stratégie élégante : 1 point positif au centre, points négatifs sur les switches
            coords = [item['center']] # Point positif
            labels = [1]
            
            if 'internal_points' in item:
                coords.extend(item['internal_points'])
                labels.extend([0] * len(item['internal_points']))
            
            masks, scores, _ = predictor.predict(
                point_coords=np.array(coords),
                point_labels=np.array(labels),
                box=box,
                multimask_output=False
            )
            
            # Visualisation
            mask = masks[0]
            random_color = np.random.randint(0, 255, (3,)).tolist()
            overlay = get_mask_overlay(mask, random_color)
            
            # Fusion avec l'image originale (alpha blending)
            cv2.addWeighted(overlay, 0.4, output_image, 1.0, 0, output_image)
            cv2.rectangle(output_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), random_color, 2)

    return output_image

# --- EXEMPLE D'UTILISATION ---
# Supposons que tu as une grande île (rectangle de switches)
test_island = {
    'box': [100, 100, 800, 500],
    'center': [150, 150], # Un coin du grand rectangle souvent vide
    'internal_points': [[200, 250], [300, 250], [400, 250]] # Centres des switches à ne PAS fusionner
}

final_view = run_sam_on_islands("diagramme.png", [test_island])

plt.figure(figsize=(12, 12))
plt.imshow(final_view)
plt.axis('off')
plt.show()