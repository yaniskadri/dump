import fitz  # PyMuPDF
import cv2
import numpy as np
import os
import glob

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FOLDER = "input_pdfs"
OUTPUT_FOLDER = "extracted_smart_v3"
DPI = 300

# Seuils de taille
MIN_AREA = 100         # Petits symboles (Diodes)
MAX_AREA_SMALL = 5000  # Limite entre "Petit symbole" et "Grand Conteneur"
MAX_AREA_LARGE = 500000 # Très grands cadres

PADDING = 15
# ==========================================

def get_page_images(pdf_path):
    """
    Retourne DEUX versions de chaque page :
    1. img_visual : L'image normale (pour la sauvegarde/l'humain)
    2. img_detect : L'image sans texte (pour l'algorithme)
    """
    doc = fitz.open(pdf_path)
    pairs = []

    for page in doc:
        # 1. Version Normale (avec texte)
        pix_normal = page.get_pixmap(dpi=DPI)
        img_normal = np.frombuffer(pix_normal.samples, dtype=np.uint8).reshape(pix_normal.h, pix_normal.w, pix_normal.n)
        if pix_normal.n == 4: img_normal = cv2.cvtColor(img_normal, cv2.COLOR_RGBA2RGB)
        else: img_normal = cv2.cvtColor(img_normal, cv2.COLOR_RGB2BGR)

        # 2. Version "Detection" (On cache le texte)
        shape = page.new_shape()
        text_instances = page.get_text("dict")["blocks"]
        for block in text_instances:
            if block['type'] == 0:
                shape.draw_rect(block['bbox'])
                shape.finish(color=(1, 1, 1), fill=(1, 1, 1))
        shape.commit()
        
        pix_clean = page.get_pixmap(dpi=DPI)
        img_clean = np.frombuffer(pix_clean.samples, dtype=np.uint8).reshape(pix_clean.h, pix_clean.w, pix_clean.n)
        if pix_clean.n == 4: img_clean = cv2.cvtColor(img_clean, cv2.COLOR_RGBA2RGB)
        else: img_clean = cv2.cvtColor(img_clean, cv2.COLOR_RGB2BGR)

        pairs.append((img_normal, img_clean))
        
    return pairs

def process_smart_extraction():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    files = glob.glob(os.path.join(INPUT_FOLDER, "*.pdf"))

    for pdf_file in files:
        filename = os.path.basename(pdf_file)
        print(f"Traitement : {filename}")
        
        try:
            image_pairs = get_page_images(pdf_file)
        except Exception as e:
            print(f"Erreur: {e}")
            continue

        for i, (img_vis, img_det) in enumerate(image_pairs):
            gray = cv2.cvtColor(img_det, cv2.COLOR_BGR2GRAY)
            # Binarisation simple
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            
            # ICI est la magie : RETR_TREE récupère la hiérarchie (qui est dans quoi)
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if hierarchy is None: continue
            
            # Hierarchy format: [Next, Previous, First_Child, Parent]
            # On va ignorer les contours qui sont "parents" du cadre de la page entière
            
            count = 0
            for idx, cnt in enumerate(contours):
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                
                # Filtre de bruit de base
                if area < MIN_AREA: continue
                
                # --- LOGIQUE DE TRI ---
                
                is_container = False
                parent_idx = hierarchy[0][idx][3]
                child_idx = hierarchy[0][idx][2]
                
                # 1. C'est un conteneur (Grand Rectangle) si :
                # - Il a une grande aire ET
                # - Il a des enfants (child_idx != -1) OU ressemble à une boite carrée
                if area > MAX_AREA_SMALL and area < MAX_AREA_LARGE:
                    # Vérification géométrique : Est-ce que le contour est FERMÉ ?
                    # Périmètre vs Aire peut nous aider, mais ici on assume que findContours 
                    # a trouvé un rectangle fermé.
                    folder_sub = "Containers"
                    is_container = True
                
                # 2. C'est un symbole standard si :
                # - Aire moyenne
                elif area <= MAX_AREA_SMALL:
                    folder_sub = "Symbols"
                else:
                    continue # Trop gros (cadre de page)

                # Sauvegarde
                # On sauvegarde depuis img_vis (AVEC TEXTE) pour pouvoir lire ce que c'est
                x_pad = max(0, x - PADDING)
                y_pad = max(0, y - PADDING)
                w_pad = min(img_vis.shape[1] - x_pad, w + 2*PADDING)
                h_pad = min(img_vis.shape[0] - y_pad, h + 2*PADDING)
                
                roi = img_vis[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
                
                # On sépare dans des dossiers pour faciliter votre tri
                save_path = os.path.join(OUTPUT_FOLDER, folder_sub)
                if not os.path.exists(save_path): os.makedirs(save_path)
                
                out_name = f"{os.path.splitext(filename)[0]}_p{i}_{x}_{y}_{'CONT' if is_container else 'SYMB'}.png"
                cv2.imwrite(os.path.join(save_path, out_name), roi)
                count += 1
                
            print(f"  Page {i}: {count} éléments extraits.")

if __name__ == "__main__":
    process_smart_extraction()