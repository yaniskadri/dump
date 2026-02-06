import fitz  # PyMuPDF
import cv2
import numpy as np
import os
import glob


#Better morphological cleaning with ink density check and does not reject rectangles with text inside

s
# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FOLDER = "input_pdfs"
OUTPUT_FOLDER = "extracted_v6_smartcheck"
DEBUG_FOLDER = "debug_views_v6"

DPI = 300
PADDING = 20

# 1. Morphologie (La Glue) -> RÉDUIT pour arrêter d'inventer des lignes
# (3, 3) est très doux. Si vos pointillés se cassent encore, essayez (5, 5).
MORPH_KERNEL = (3, 3) 

# 2. Filtres de taille
MIN_AREA = 300
MAX_AREA = 1000000

# 3. Densité d'Encre (Le juge de paix)
# On baisse un peu le seuil pour accepter les textes légers
MIN_INK_DENSITY = 0.005 # 0.5% de remplissage suffit

# 4. Filtre Ratio (Forme)
MIN_RATIO = 0.15 
MAX_RATIO = 6.0
# ==========================================

def get_page_data(page):
    """
    Retourne 3 versions :
    1. img_vis : L'image normale (pour l'humain et la vérification de densité TEXTE)
    2. img_mask : L'image sans texte (pour la DÉTECTION de formes)
    """
    # 1. Image Visuelle
    pix = page.get_pixmap(dpi=DPI)
    img_vis = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4: img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGBA2RGB)
    else: img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)

    # 2. Image Masque (Sans texte)
    shape = page.new_shape()
    for block in page.get_text("dict")["blocks"]:
        if block['type'] == 0: 
            shape.draw_rect(block['bbox'])
            shape.finish(color=(1, 1, 1), fill=(1, 1, 1))
    shape.commit()
    
    pix_mask = page.get_pixmap(dpi=DPI)
    img_mask = np.frombuffer(pix_mask.samples, dtype=np.uint8).reshape(pix_mask.h, pix_mask.w, pix_mask.n)
    if pix_mask.n == 4: img_mask = cv2.cvtColor(img_mask, cv2.COLOR_RGBA2RGB)
    else: img_mask = cv2.cvtColor(img_mask, cv2.COLOR_RGB2BGR)
    
    return img_vis, img_mask

def process_v6():
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    if not os.path.exists(DEBUG_FOLDER): os.makedirs(DEBUG_FOLDER)

    files = glob.glob(os.path.join(INPUT_FOLDER, "*.pdf"))

    for pdf_file in files:
        filename = os.path.basename(pdf_file)
        print(f"--- Traitement V6 : {filename} ---")
        doc = fitz.open(pdf_file)

        for i, page in enumerate(doc):
            img_vis, img_mask = get_page_data(page)
            debug_img = img_vis.copy()

            # --- DÉTECTION (Sur l'image NETTOYÉE sans texte) ---
            gray_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
            _, binary_mask = cv2.threshold(gray_mask, 200, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

            # Glue légère
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL)
            closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            dilated = cv2.dilate(closed, np.ones((3,3), np.uint8), iterations=1) # Dilatation légère

            contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # --- VÉRIFICATION (Sur l'image VISUELLE avec texte) ---
            # On prépare une version binaire de l'image AVEC texte pour calculer la densité réelle
            gray_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2GRAY)
            _, binary_vis = cv2.threshold(gray_vis, 200, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

            kept_count = 0
            if hierarchy is not None:
                for idx, cnt in enumerate(contours):
                    x, y, w, h = cv2.boundingRect(cnt)
                    area = w * h
                    
                    rejection_reason = ""
                    
                    # 1. Filtres Géométriques de base
                    if area < MIN_AREA: rejection_reason = "Too Small"
                    elif area > MAX_AREA: rejection_reason = "Too Big"
                    
                    ratio = float(w)/h
                    if not rejection_reason and (ratio < MIN_RATIO or ratio > MAX_RATIO):
                        rejection_reason = "Bad Ratio"

                    # 2. Filtre Densité INTELLIGENT
                    if not rejection_reason:
                        # On regarde la densité sur l'image AVEC TEXTE (binary_vis)
                        roi_vis = binary_vis[y:y+h, x:x+w]
                        non_zero_vis = cv2.countNonZero(roi_vis)
                        density_vis = non_zero_vis / (w*h)
                        
                        # Si même avec le texte, c'est vide -> C'est un faux rectangle (fils croisés)
                        if density_vis < MIN_INK_DENSITY:
                            rejection_reason = f"Empty Box ({density_vis:.3f})"

                    # --- DECISION ---
                    if rejection_reason:
                        # Rouge = Rejeté
                        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    else:
                        # Vert = Gardé
                        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        x_pad = max(0, x - PADDING)
                        y_pad = max(0, y - PADDING)
                        w_pad = min(img_vis.shape[1] - x_pad, w + 2*PADDING)
                        h_pad = min(img_vis.shape[0] - y_pad, h + 2*PADDING)
                        
                        roi = img_vis[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
                        out_name = f"{os.path.splitext(filename)[0]}_p{i}_{x}_{y}.png"
                        cv2.imwrite(os.path.join(OUTPUT_FOLDER, out_name), roi)
                        kept_count += 1

            cv2.imwrite(os.path.join(DEBUG_FOLDER, f"DEBUG_V6_{filename}_p{i}.jpg"), debug_img)
            print(f"  Page {i}: {kept_count} symboles extraits.")

if __name__ == "__main__":
    process_v6()