import fitz
import numpy as np
from sklearn.cluster import DBSCAN
import cv2
import os
import math

class DensityExtractor:
    def __init__(self, pdf_path):
        self.doc = fitz.open(pdf_path)
        
        # --- CONFIGURATION DU CLUSTERING ---
        # 1. On ignore tout ce qui est plus grand que ça (Fils longs / Cadres)
        self.MAX_ELEMENT_SIZE = 300  
        
        # 2. Distance max entre deux traits pour qu'ils fassent partie du même objet
        # Si une diode est faite de 2 morceaux séparés de 5 pixels, EPS=15 les réunira.
        self.EPSILON = 15 
        
        # 3. Minimum de traits pour faire un objet (évite le bruit)
        self.MIN_SAMPLES = 2 

    def get_lines_from_page(self, page):
        """
        Extrait tous les petits traits vectoriels individuels (lignes, courbes).
        Ne construit PAS de polygones. On travaille sur la "soupe de traits".
        """
        paths = page.get_drawings()
        elements = [] # Liste de dictionnaires {'bbox': [x1,y1,x2,y2], 'center': [cx, cy]}

        for path in paths:
            for item in path["items"]:
                p1, p2 = None, None
                
                # Récupère les extrémités des segments
                if item[0] == "l": # Ligne
                    p1, p2 = item[1], item[2]
                elif item[0] == "c": # Courbe (Bezier)
                    p1, p2 = item[1], item[-1] # On approxime par une ligne start-end
                elif item[0] == "re": # Rectangle (décomposé en 4 lignes)
                    r = item[1]
                    pts = [(r[0], r[1]), (r[2], r[1]), (r[2], r[3]), (r[0], r[3])]
                    # On ajoute les 4 segments individuellement
                    for i in range(4):
                        sp1 = pts[i]
                        sp2 = pts[(i+1)%4]
                        # Calcul longueur
                        dist = math.hypot(sp2[0]-sp1[0], sp2[1]-sp1[1])
                        if dist < self.MAX_ELEMENT_SIZE:
                            cx = (sp1[0] + sp2[0]) / 2
                            cy = (sp1[1] + sp2[1]) / 2
                            elements.append({'bbox': [min(sp1[0], sp2[0]), min(sp1[1], sp2[1]), 
                                                      max(sp1[0], sp2[0]), max(sp1[1], sp2[1])],
                                             'center': [cx, cy]})
                    continue # On a traité le rect, on passe au suivant

                if p1 and p2:
                    # Filtre : On jette les très longs fils de connexion
                    length = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
                    if length < self.MAX_ELEMENT_SIZE:
                        cx = (p1[0] + p2[0]) / 2
                        cy = (p1[1] + p2[1]) / 2
                        elements.append({'bbox': [min(p1.x, p2.x), min(p1.y, p2.y), 
                                                  max(p1.x, p2.x), max(p1.y, p2.y)],
                                         'center': [cx, cy]})
        return elements

    def process_extraction(self, output_folder, dpi=300):
        if not os.path.exists(output_folder): os.makedirs(output_folder)
        scale = dpi / 72.0
        padding = 15

        for i, page in enumerate(self.doc):
            print(f"Processing page {i+1}...")
            
            # 1. Récupérer la "Soupe de petits traits"
            elements = self.get_lines_from_page(page)
            if not elements: continue

            # 2. Préparer les données pour le Clustering (Centres des traits)
            centers = np.array([e['center'] for e in elements])

            # 3. MAGIE : DBSCAN CLUSTERING
            # Regroupe les traits proches spatialement
            clustering = DBSCAN(eps=self.EPSILON, min_samples=self.MIN_SAMPLES).fit(centers)
            labels = clustering.labels_

            # 4. Rendu Image
            pix = page.get_pixmap(dpi=dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else: img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # 5. Fusionner les clusters en Bounding Boxes
            unique_labels = set(labels)
            
            count = 0
            for label_id in unique_labels:
                if label_id == -1: continue # -1 = Bruit (traits isolés)

                # Récupérer tous les éléments de ce cluster
                indices = np.where(labels == label_id)[0]
                cluster_bboxes = [elements[idx]['bbox'] for idx in indices]

                # Calculer la BBox globale du cluster (min_x, min_y, max_x, max_y)
                # C'est l'enveloppe qui contient tous les traits du composant
                g_x1 = min([b[0] for b in cluster_bboxes])
                g_y1 = min([b[1] for b in cluster_bboxes])
                g_x2 = max([b[2] for b in cluster_bboxes])
                g_y2 = max([b[3] for b in cluster_bboxes])

                # Vérifier si l'objet final n'est pas trop gros (ex: tout un harnais de câbles)
                if (g_x2 - g_x1) > 500 or (g_y2 - g_y1) > 500:
                    continue 

                # Crop
                px1 = int(g_x1 * scale) - padding
                py1 = int(g_y1 * scale) - padding
                px2 = int(g_x2 * scale) + padding
                py2 = int(g_y2 * scale) + padding

                px1, py1 = max(0, px1), max(0, py1)
                px2, py2 = min(img.shape[1], px2), min(img.shape[0], py2)

                crop = img[py1:py2, px1:px2]
                
                if crop.size == 0: continue

                # Sauvegarde brute (On ne sait pas ce que c'est, mais c'est un objet)
                # L'utilisateur triera plus tard
                fname = f"p{i}_cluster{label_id}.png"
                cv2.imwrite(os.path.join(output_folder, fname), crop)
                count += 1
            
            print(f"  -> {count} objets détectés (Diodes, Grounds, Rectangles inclus).")

# Utilisation
# extractor = DensityExtractor("schema.pdf")
# extractor.process_extraction("output_clusters")