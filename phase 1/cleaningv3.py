import fitz  # PyMuPDF
import cv2
import numpy as np
import os
import glob

# ==========================================
# CONFIGURATION V7
# ==========================================
INPUT_FOLDER = "input_pdfs"
OUTPUT_FOLDER = "extracted_v7_architect"
DEBUG_FOLDER = "debug_views_v7"

DPI = 300
PADDING = 30  # Marge confortable

# 1. PARAMÈTRE CRITIQUE : La "Force de Connexion"
# (9, 9) devrait suffire à fermer les coins ouverts.
# Si vous avez encore des "bouts de lignes rouges", PASSEZ À (15, 15).
MORPH_KERNEL_SIZE = (9, 9) 

# 2. Seuils de taille
MIN_AREA = 300
MAX_AREA = 2000000 # Augmenté pour accepter les très grands cadres

# 3. NOUVEAU : Seuil d'Encre ABSOLU (et non plus relatif)
# Si on trouve au moins 50 pixels noirs (un peu de texte ou un symbole), on garde.
MIN_PIXELS_COUNT = 50 

# 4. Filtre Ratio (Élargi pour accepter les boîtes plates)
MIN_RATIO = 0.1 
MAX_RATIO = 10.0
# ==========================================

def get_page_data(page):
    pix = page.get_pixmap(dpi=DPI)
    img_vis = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4: img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGBA2RGB)
    else: img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)

    # Masque sans texte (pour la forme)
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

def process_v7():
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    if not os.path.exists(DEBUG_FOLDER): os.makedirs(DEBUG_FOLDER)

    files = glob.glob(os.path.join(INPUT_FOLDER, "*.pdf"))

    for pdf_file in files:
        filename = os.path.basename(pdf_file)
        print(f"--- Traitement V7 : {filename} ---")
        doc = fitz.open(pdf_file)

        for i, page in enumerate(doc):
            img_vis, img_mask = get_page_data(page)
            debug_img = img_vis.copy()

            # --- 1. RECONSTRUCTION (Sur l'image masque) ---
            gray_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
            _, binary_mask = cv2.threshold(gray_mask, 200, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

            # C'est ici que tout se joue : on applique une fermeture FORTE
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)
            
            # Close = Dilate puis Erode -> Bouche les trous
            closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            
            # Petite dilatation supplémentaire pour être sûr de tout attraper
            dilated_mask = cv2.dilate(closed_mask, np.ones((3,3), np.uint8), iterations=1)

            # Sauvegarde de la "Vision Robot" pour debug
            cv2.imwrite(os.path.join(DEBUG_FOLDER, f"MORPH_{filename}_p{i}.jpg"), dilated_mask)

            # --- 2. DÉTECTION ---
            contours, hierarchy = cv2.findContours(dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Préparation Image Visuelle Binaire (Pour compter les pixels noirs réels)
            gray_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2GRAY)
            _, binary_vis = cv2.threshold(gray_vis, 200, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

            kept_count = 0
            if hierarchy is not None:
                for idx, cnt in enumerate(contours):
                    x, y, w, h = cv2.boundingRect(cnt)
                    area = w * h
                    
                    rejection_reason = ""
                    
                    # Filtres Géométriques
                    if area < MIN_AREA: rejection_reason = "Too Small"
                    elif area > MAX_AREA: rejection_reason = "Too Big"
                    
                    ratio = float(w)/h
                    if not rejection_reason and (ratio < MIN_RATIO or ratio > MAX_RATIO):
                        rejection_reason = "Bad Ratio"

                    # --- 3. VÉRIFICATION DE CONTENU (NOUVEAU) ---
                    if not rejection_reason:
                        # On regarde la zone dans l'image ORIGINALE (avec texte)
                        # Mais attention : on regarde l'image binaire brute, pas celle dilatée/bavée
                        roi_vis = binary_vis[y:y+h, x:x+w]
                        
                        # On compte juste les pixels noirs. Point barre.
                        pixels_count = cv2.countNonZero(roi_vis)
                        
                        if pixels_count < MIN_PIXELS_COUNT:
                            rejection_reason = f"Empty ({pixels_count} px)"

                    # --- DÉCISION ---
                    if rejection_reason:
                        # Rouge = Rejeté
                        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        # Optionnel : écrire la raison sur l'image pour comprendre
                        # cv2.putText(debug_img, rejection_reason, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
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

            cv2.imwrite(os.path.join(DEBUG_FOLDER, f"RESULT_{filename}_p{i}.jpg"), debug_img)
            print(f"  Page {i}: {kept_count} symboles extraits.")

if __name__ == "__main__":
    process_v7()