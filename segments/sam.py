from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch

# Une seule fois : charger le modèle
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
sam.to(device="cuda")
mask_generator = SamAutomaticMaskGenerator(sam)

# Pour chaque page de diagram
def extract_all_components(pdf_path, page_idx):
    # 1. Rasterize PDF en haute résolution
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    pix = page.get_pixmap(dpi=300)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    
    # 2. SAM trouve TOUS les objets automatiquement
    masks = mask_generator.generate(image)
    
    # 3. Filtrer par taille (optionnel)
    components = []
    for mask in masks:
        area = mask['area']
        if 100 < area < 50000:  # Filtre basique
            components.append({
                'mask': mask['segmentation'],
                'bbox': mask['bbox'],
                'area': area,
                'stability_score': mask['stability_score']
            })
    
    return components

# C'est tout. Zero hardcoding.