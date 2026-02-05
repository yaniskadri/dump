import fitz  # PyMuPDF
import cv2
import numpy as np
import os
import glob

# ==========================================
# CONFIGURATION (Ajustez ici !)
# ==========================================
INPUT_FOLDER = "input_pdfs"
OUTPUT_FOLDER = "extracted_v5_cleaned"
DEBUG_FOLDER = "debug_views" # Pour voir ce qu'il fait

DPI = 300
PADDING = 25  # Marge de sécurité augmentée

# --- FILTRES DE NETTOYAGE ---

# 1. Taille
MIN_AREA = 400          # On ignore les tout petits points
MAX_AREA = 800000       # On ignore les cadres géants

# 2. Densité (L'Anti-Fils Intercroisés)
# Un composant doit avoir au moins X% de "dessin" à l'intérieur
# 0.01 = 1% (Très permissif). 0.05 = 5% (Strict).
# Essayez 0.015 (1.5%) pour commencer.
MIN_INK_DENSITY = 0.015 

# 3. Ratio (Forme)
# On jette les trucs trop allongés (bouts de fils restants)
MIN_RATIO = 0.2  # Pas trop fin verticalement
MAX_RATIO = 5.0  # Pas trop fin horizontalement

# 4. Morphologie (Fusion)
MORPH_KERNEL = (9, 9) # Fusionne les pointillés et traits proches
# ==========================================

def get_page_data(page):
    """Récupère l'image VISUELLE (pour l'humain) et l'image MASQUE (pour le robot)"""
    # 1. Image Visuelle (Avec tout)
    pix = page.get_pixmap(dpi=DPI)
    img_vis = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4: img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGBA2RGB)
    else: img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)

    # 2. Image Masque (On cache le texte via PyMuPDF)
    shape = page.new_shape()
    for block in page.get_text("dict")["blocks"]:
        if block['type'] == 0: # Texte
            shape.draw_rect(block['bbox'])
            shape.finish(color=(1, 1, 1), fill=(1, 1, 1))
    shape.commit()
    
    pix_mask = page.get_pixmap(dpi=DPI)
    img_mask = np.frombuffer(pix_mask.samples, dtype=np.uint8).reshape(pix_mask.h, pix_mask.w, pix_mask.n)
    if pix_mask.n == 4: img_mask = cv2.cvtColor(img_mask, cv2.COLOR_RGBA2RGB)
    else: img_mask = cv2.cvtColor(img_mask, cv2.COLOR_RGB2BGR)
    
    return img_vis, img_mask

def process_cleaning():
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    if not os.path.exists(DEBUG_FOLDER): os.makedirs(DEBUG_FOLDER)

    files = glob.glob(os.path.join(INPUT_FOLDER, "*.pdf"))

    for pdf_file in files:
        filename = os.path.basename(pdf_file)
        print(f"--- Traitement : {filename} ---")
        doc = fitz.open(pdf_file)

        for i, page in enumerate(doc):
            img_vis, img_mask = get_page_data(page)
            debug_img = img_vis.copy() # On dessinera dessus

            # Prétraitement
            gray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

            # Fusion Morphologique (Pour recoller les morceaux)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL)
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Dilatation (Pour fermer les contours ouverts)
            dilated = cv2.dilate(closed, np.ones((3,3), np.uint8), iterations=2)

            # Détection
            contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            kept_count = 0
            
            if hierarchy is not None:
                for idx, cnt in enumerate(contours):
                    x, y, w, h = cv2.boundingRect(cnt)
                    area = w * h
                    
                    # --- FILTRES ---
                    rejection_reason = ""
                    
                    # 1. Filtre Taille
                    if area < MIN_AREA: rejection_reason = "Too Small"
                    elif area > MAX_AREA: rejection_reason = "Too Big"
                    
                    # 2. Filtre Ratio
                    ratio = float(w)/h
                    if not rejection_reason:
                        if ratio < MIN_RATIO or ratio > MAX_RATIO: rejection_reason = "Bad Ratio (Wire)"

                    # 3. Filtre Densité (CRITIQUE POUR LES FILS INTERCROISÉS)
                    if not rejection_reason:
                        # On regarde à l'intérieur de la boîte, dans l'image BINAIRE originale (non dilatée)
                        # Pour voir s'il y a de l'encre réelle.
                        roi_binary = binary[y:y+h, x:x+w]
                        non_zero = cv2.countNonZero(roi_binary)
                        density = non_zero / (w*h)
                        
                        if density < MIN_INK_DENSITY:
                            rejection_reason = f"Empty Box ({density:.3f})"

                    # --- DECISION ---
                    if rejection_reason:
                        # Dessiner en ROUGE sur le debug
                        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    else:
                        # C'EST BON ! Dessiner en VERT
                        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Découpage avec marge
                        x_pad = max(0, x - PADDING)
                        y_pad = max(0, y - PADDING)
                        w_pad = min(img_vis.shape[1] - x_pad, w + 2*PADDING)
                        h_pad = min(img_vis.shape[0] - y_pad, h + 2*PADDING)
                        
                        roi = img_vis[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
                        
                        out_name = f"{os.path.splitext(filename)[0]}_p{i}_{x}_{y}.png"
                        cv2.imwrite(os.path.join(OUTPUT_FOLDER, out_name), roi)
                        kept_count += 1

            # Sauvegarde de l'image de Debug
            cv2.imwrite(os.path.join(DEBUG_FOLDER, f"DEBUG_{filename}_p{i}.jpg"), debug_img)
            print(f"  Page {i}: {kept_count} symboles gardés.")

if __name__ == "__main__":
    process_cleaning()