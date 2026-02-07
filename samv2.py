import fitz  # PyMuPDF
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from shapely.geometry import Polygon

# --- CONFIGURATION ---
PDF_PATH = "ton_schema.pdf"
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"
DEVICE = "cuda"

# 1. CONVERSION PDF EN IMAGE (300 DPI pour la précision)
def pdf_to_image(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)  # Première page
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# 2. CHARGEMENT SAM
sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=DEVICE).half() # Pour tes 9GB de VRAM
predictor = SamPredictor(sam)

# 3. LOGIQUE DE DÉTECTION ET VISUALISATION
def process_and_visualize(image, polygons_coords):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    
    # Trouver qui est dans quoi
    hierarchy = find_hierarchy(polygons_coords)
    
    canvas = image_rgb.copy()
    
    for item in hierarchy:
        # Préparation des prompts
        box = np.array(item['box'])
        coords = [item['center']]
        labels = [1] # Positif (le contenant)
        
        # Ajout des points négatifs pour les composants internes
        if item['internal_points']:
            # On en prend max 10 pour ne pas saturer SAM
            internals = item['internal_points'][:10] 
            coords.extend(internals)
            labels.extend([0] * len(internals)) # 0 = Négatif (exclure)

        # Prediction
        masks, scores, _ = predictor.predict(
            point_coords=np.array(coords),
            point_labels=np.array(labels),
            box=box,
            multimask_output=False
        )
        
        # Dessiner le masque
        mask = masks[0]
        color = np.random.randint(0, 255, (3,)).tolist()
        
        # Création d'un calque coloré
        mask_overlay = np.zeros_like(canvas)
        mask_overlay[mask] = color
        cv2.addWeighted(mask_overlay, 0.5, canvas, 1.0, 0, canvas)
        
        # Dessiner la boite englobante
        cv2.rectangle(canvas, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)

    return canvas

# --- EXÉCUTION ---
img = pdf_to_image(PDF_PATH)

# Remplace ceci par tes vraies coordonnées de polygones extraites
# Exemple fictif : un grand rectangle et deux petits à l'intérieur
fake_polys = [
    [[100, 100], [900, 100], [900, 600], [100, 600]], # Le contenant
    [[200, 200], [300, 200], [300, 300], [200, 300]], # Switch 1
    [[400, 200], [500, 200], [500, 300], [400, 300]], # Switch 2
]

result_img = process_and_visualize(img, fake_polys)

plt.figure(figsize=(20, 10))
plt.imshow(result_img)
plt.title("Détection SAM : Contenants vs Composants Internes")
plt.axis('off')
plt.show()