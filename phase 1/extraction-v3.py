import fitz  # PyMuPDF
import cv2
import numpy as np
import os
import glob

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FOLDER = "input_pdfs"
OUTPUT_FOLDER = "extracted_v4_morphology"
DPI = 300

# Paramètres de Morphologie (Ajustez si nécessaire)
# Plus le Kernel est grand, plus il fusionne des éléments distants
# (5,5) est bon pour des pointillés serrés. (9,9) pour des pointillés espacés.
MORPH_KERNEL_SIZE = (7, 7) 

# Seuils de taille (Ajustés pour ignorer les petits bouts de fils seuls)
MIN_AREA = 300         
MAX_AREA_LARGE = 1000000 

PADDING = 20 # Marge de sécurité augmentée
# ==========================================

def get_page_images(pdf_path):
    # (Même fonction que V3 - lecture PDF sans texte)
    doc = fitz.open(pdf_path)
    pairs = []
    for page in doc:
        # 1. Visuel (Avec Texte)
        pix_normal = page.get_pixmap(dpi=DPI)
        img_normal = np.frombuffer(pix_normal.samples, dtype=np.uint8).reshape(pix_normal.h, pix_normal.w, pix_normal.n)
        if pix_normal.n == 4: img_normal = cv2.cvtColor(img_normal, cv2.COLOR_RGBA2RGB)
        else: img_normal = cv2.cvtColor(img_normal, cv2.COLOR_RGB2BGR)

        # 2. Détection (Sans Texte)
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

def process_morphology_extraction():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    files = glob.glob(os.path.join(INPUT_FOLDER, "*.pdf"))

    for pdf_file in files:
        filename = os.path.basename(pdf_file)
        print(f"Traitement V4 : {filename}")
        
        try:
            image_pairs = get_page_images(pdf_file)
        except Exception as e:
            print(f"Erreur: {e}")
            continue

        for i, (img_vis, img_det) in enumerate(image_pairs):
            # 1. Conversion gris et seuillage
            gray = cv2.cvtColor(img_det, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            
            # --- CORRECTION MAJEURE ICI ---
            
            # A. Morphological Closing : Fusionne les pointillés
            kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)
            morph_img = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_morph)
            
            # B. Dilatation : Épaissit tout pour fermer les coins ouverts
            # Cela assure que les boîtes sont bien "fermées"
            dilate_kernel = np.ones((3,3), np.uint8)
            processed_img = cv2.dilate(morph_img, dilate_kernel, iterations=2)
            
            # ------------------------------

            # Utilisation de RETR_EXTERNAL pour n'avoir que les enveloppes extérieures
            # (Évite de détecter l'intérieur d'un trait épais)
            # Si vous avez des composants DANS des composants, utilisez RETR_TREE comme avant
            contours, hierarchy = cv2.findContours(processed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if hierarchy is None: continue
            
            count = 0
            for idx, cnt in enumerate(contours):
                # Récupérer la bounding box
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                
                # Filtres
                if area < MIN_AREA or area > MAX_AREA_LARGE: continue
                
                # Vérification de ratio pour éviter les lignes infinies (fils) qui auraient "bavé"
                aspect_ratio = float(w)/h
                if aspect_ratio > 10 or aspect_ratio < 0.1: continue 

                # --- Logique Hiérarchie ---
                # On ne veut pas les enfants (ex: l'intérieur d'un 'O' ou le trou d'un trait épais)
                # hierarchy[0][idx][3] est l'index du PARENT.
                # Si Parent != -1, c'est que ce contour est DANS un autre.
                # Pour les conteneurs complexes, on garde tout pour l'instant et on triera visuellement.
                
                # Découpage (Sur l'image VISUELLE avec texte)
                x_pad = max(0, x - PADDING)
                y_pad = max(0, y - PADDING)
                w_pad = min(img_vis.shape[1] - x_pad, w + 2*PADDING)
                h_pad = min(img_vis.shape[0] - y_pad, h + 2*PADDING)
                
                roi = img_vis[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
                
                # Nommage
                out_name = f"{os.path.splitext(filename)[0]}_p{i}_{x}_{y}.png"
                cv2.imwrite(os.path.join(OUTPUT_FOLDER, out_name), roi)
                count += 1
                
            print(f"  Page {i}: {count} éléments extraits (Fusionnés).")

if __name__ == "__main__":
    process_morphology_extraction()