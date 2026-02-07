import fitz
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch

def visualize_sam_results(pdf_path, page_idx=0):
    """
    Visualise les masks détectés par SAM sur un wiring diagram.
    """
    # 1. Charger SAM
    print("Loading SAM model...")
    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,           # Grille de points pour la détection
        pred_iou_thresh=0.86,         # Seuil de qualité
        stability_score_thresh=0.92,  # Seuil de stabilité
        min_mask_region_area=100,     # Aire minimale (en pixels)
    )
    
    # 2. Rasteriser le PDF
    print(f"Rendering PDF page {page_idx}...")
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    pix = page.get_pixmap(dpi=300)  # Haute résolution pour SAM
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    
    # 3. Lancer SAM
    print("Running SAM segmentation...")
    masks = mask_generator.generate(image)
    print(f"SAM found {len(masks)} objects")
    
    # 4. Visualiser
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # (a) Image originale
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # (b) Tous les masks colorés
    axes[1].imshow(image)
    show_masks_on_image(image, masks, axes[1])
    axes[1].set_title(f"All Masks ({len(masks)} objects)")
    axes[1].axis('off')
    
    # (c) Juste les bounding boxes
    axes[2].imshow(image)
    show_boxes_on_image(masks, axes[2])
    axes[2].set_title("Bounding Boxes")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 5. Stats
    print("\n=== SAM Statistics ===")
    areas = [m['area'] for m in masks]
    print(f"Total objects: {len(masks)}")
    print(f"Area range: {min(areas):.0f} - {max(areas):.0f} pixels²")
    print(f"Mean area: {np.mean(areas):.0f} pixels²")
    
    return masks, image


def show_masks_on_image(image, masks, ax):
    """Affiche tous les masks avec des couleurs aléatoires"""
    if len(masks) == 0:
        return
    
    # Trier par aire (les plus grands en premier, pour pas qu'ils cachent les petits)
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    # Créer une image de couleurs aléatoires pour chaque mask
    overlay = np.zeros((*image.shape[:2], 4))
    
    for i, mask_data in enumerate(sorted_masks):
        mask = mask_data['segmentation']
        
        # Couleur aléatoire
        color = np.random.random(3)
        
        # Appliquer le mask avec transparence
        overlay[mask] = [*color, 0.35]  # Alpha = 0.35
    
    ax.imshow(overlay)


def show_boxes_on_image(masks, ax):
    """Affiche juste les bounding boxes"""
    import matplotlib.patches as patches
    
    for mask_data in masks:
        bbox = mask_data['bbox']  # Format: [x, y, width, height]
        
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2], bbox[3],
            linewidth=1,
            edgecolor='red',
            facecolor='none',
            alpha=0.8
        )
        ax.add_patch(rect)


# Usage
masks, image = visualize_sam_results("diagram.pdf", page_idx=0)