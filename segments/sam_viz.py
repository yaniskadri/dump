import fitz
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import numpy as np

def visualize_segments(pdf_path, page_idx=0, show_text=True):
    """
    Visualise les segments extraits par-dessus le PDF rasterisé.
    """
    # Ouvrir PDF
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    
    # Extraire segments (ton code existant)
    segments = extract_segments_from_page(page)
    text_blocks = extract_text_blocks(page) if show_text else []
    
    # Rasteriser le PDF en fond
    pix = page.get_pixmap(dpi=150)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    
    # Créer figure
    fig, ax = plt.subplots(figsize=(20, 14))
    ax.imshow(img, alpha=0.3)  # PDF en semi-transparent
    
    # Conversion PDF coords → Image coords (72 DPI → 150 DPI)
    scale = 150 / 72
    
    # Dessiner les segments
    lines = []
    colors = []
    
    for seg in segments:
        # Ligne de (x1,y1) à (x2,y2)
        lines.append([
            (seg.x1 * scale, seg.y1 * scale),
            (seg.x2 * scale, seg.y2 * scale)
        ])
        
        # Couleur selon type (si tu as marqué les courbes vs lignes)
        if hasattr(seg, 'is_curve') and seg.is_curve:
            colors.append('red')  # Courbes en rouge
        else:
            colors.append('blue')  # Lignes en bleu
    
    # Afficher toutes les lignes d'un coup (plus rapide)
    lc = LineCollection(lines, colors=colors, linewidths=0.5, alpha=0.7)
    ax.add_collection(lc)
    
    # Afficher les blocs de texte (optionnel)
    if show_text:
        for block in text_blocks:
            x0, y0, x1, y1, text = block
            rect = patches.Rectangle(
                (x0 * scale, y0 * scale),
                (x1 - x0) * scale,
                (y1 - y0) * scale,
                linewidth=1,
                edgecolor='green',
                facecolor='none',
                alpha=0.5
            )
            ax.add_patch(rect)
    
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)  # Inverser Y
    ax.set_title(f"Segments extracted: {len(segments)}")
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Usage
visualize_segments("diagram.pdf", page_idx=0, show_text=True)