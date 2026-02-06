import fitz
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union
import collections

def diagnostic_epaisseurs(pdf_path):
    # 1. Extraction (Classique)
    doc = fitz.open(pdf_path)
    page = doc[0]
    paths = page.get_drawings()
    
    lignes_brutes = []
    for path in paths:
        for item in path["items"]:
            if item[0] == "l":
                lignes_brutes.append(LineString([item[1], item[2]]))
            elif item[0] == "c":
                lignes_brutes.append(LineString([item[1], item[-1]]))
            elif item[0] == "re":
                r = item[1]
                p1, p2, p3, p4 = (r[0], r[1]), (r[2], r[1]), (r[2], r[3]), (r[0], r[3])
                lignes_brutes.extend([LineString([p1,p2]), LineString([p2,p3]), LineString([p3,p4]), LineString([p4,p1])])

    # 2. Reconstruction
    reseau = unary_union(lignes_brutes)
    polygones = list(polygonize(reseau))
    
    # 3. MESURE PRÉCISE DES ÉPAISSEURS
    epaisseurs = []
    
    print(f"Analyse de {len(polygones)} objets fermés...")
    
    for poly in polygones:
        # On nettoie la géométrie
        p = poly.buffer(0)
        
        # On calcule le rectangle orienté le plus serré (Rotated Rectangle)
        box = p.minimum_rotated_rectangle
        
        if box.is_empty: continue
        
        # On extrait les 4 coins du rectangle orienté
        x, y = box.exterior.coords.xy
        p0, p1, p2 = (x[0], y[0]), (x[1], y[1]), (x[2], y[2])
        
        # Calcul des deux côtés adjacents (Pythagore)
        cote_1 = ((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)**0.5
        cote_2 = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
        
        # L'épaisseur est TOUJOURS le plus petit côté
        thickness = min(cote_1, cote_2)
        
        # On arrondit à 2 décimales pour grouper les valeurs proches
        epaisseurs.append(round(thickness, 3))

    # 4. STATISTIQUES
    compteur = collections.Counter(epaisseurs)
    
    print("\n=== DISTRIBUTION DES ÉPAISSEURS (Histogramme) ===")
    print("Épaisseur (px)  |  Nombre d'objets  |  Interprétation probable")
    print("-" * 60)
    
    # On trie par épaisseur croissante
    sorted_stats = sorted(compteur.items())
    
    for ep, count in sorted_stats:
        # On ignore les tout petits bruits (< 0.01)
        if ep < 0.01: continue
        
        barre = "*" * (count // 2) # Visualisation ascii
        note = ""
        
        if count > len(polygones) * 0.1: # Si c'est fréquent
            note = "<-- PIC MAJEUR"
            
        print(f"{ep:.3f}           |  {count:3d} {barre} {note}")

diagnostic_epaisseurs("mon_fichier.pdf")