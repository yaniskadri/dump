import fitz  # PyMuPDF
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
import numpy as np
import os

# --- COLOR MAPPING (Matches your Analyzer categories) ---
COLOR_MAP = {
    "Circle_Component": "#ff00ff",   # Magenta (Moteurs, Voyants)
    "Hex_Symbol":       "#ffa500",   # Orange (Symboles spécifiques)
    "Busbar_Power":     "#32cd32",   # Lime Green (Barres de puissance)
    "Component_Rect":   "#006400",   # Dark Green (Les vrais composants)
    "Group_Container":  "#0000ff",   # Blue (Conteneurs pointillés)
    "Component_Complex":"#00ffff",   # Cyan (Formes bizarres denses)
    "Unknown_Shape":    "#ff0000"    # Red (À surveiller)
}

def visualize_page(pdf_path, json_path, page_index=0):
    # 1. Load Data
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Find objects for this page
    page_data = next((p for p in data["pages"] if p["page_index"] == page_index), None)
    if not page_data:
        print(f"No data found for page {page_index}")
        return

    # 2. Render PDF to Image (Background)
    # We render at 72 DPI to match the PDF coordinate system 1:1
    pix = page.get_pixmap(dpi=72)
    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    
    # Setup Plot
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(img_array)
    
    # 3. Draw Bounding Boxes
    counts = {k: 0 for k in COLOR_MAP.keys()}
    
    for obj in page_data["objects"]:
        cat = obj["type"]
        x1, y1, x2, y2 = obj["bbox"]
        
        # Default color if category not in map
        color = COLOR_MAP.get(cat, "gray")
        
        # Style adjustments for Groups vs Components
        style = '--' if "Group" in cat else '-'
        fill = False # Don't fill, we want to see the drawing
        linewidth = 2 if "Component" in cat else 1
        
        # Create Rectangle Patch
        width = x2 - x1
        height = y2 - y1
        rect = mpatches.Rectangle(
            (x1, y1), width, height,
            linewidth=linewidth, edgecolor=color, facecolor='none', linestyle=style
        )
        ax.add_patch(rect)
        
        # Count for stats
        if cat in counts: counts[cat] += 1

    # 4. Create Legend
    patches = [mpatches.Patch(color=v, label=f"{k} ({counts.get(k, 0)})") 
               for k, v in COLOR_MAP.items() if counts.get(k, 0) > 0]
    ax.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.title(f"Classification Analysis - Page {page_index+1}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# --- USAGE ---
# 1. Assurez-vous d'avoir généré le JSON avec le script précédent
# visualize_page("wiring_diagram.pdf", "analysis_data.json", page_index=0)