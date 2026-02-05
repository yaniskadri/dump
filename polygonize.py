import fitz
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union
import numpy as np

def visualiser_reconstruction(pdf_path):
    # 1. Extraction des données (comme expliqué précédemment)
    doc = fitz.open(pdf_path)
    page = doc[0]
    paths = page.get_drawings()
    
    lignes_brutes = []
    print("Extraction des lignes...")
    
    for path in paths:
        for item in path["items"]:
            # On traite lignes (l) et courbes (c) comme des segments droits pour la structure
            if item[0] == "l":
                lignes_brutes.append(LineString([item[1], item[2]]))
            elif item[0] == "c":
                lignes_brutes.append(LineString([item[1], item[-1]]))
            elif item[0] == "re":
                # Si par miracle il y a un rect, on le convertit en 4 lignes
                rect = item[1] # (x0, y0, x1, y1)
                p1, p2, p3, p4 = (rect[0], rect[1]), (rect[2], rect[1]), (rect[2], rect[3]), (rect[0], rect[3])
                lignes_brutes.extend([LineString([p1, p2]), LineString([p2, p3]), LineString([p3, p4]), LineString([p4, p1])])

    # 2. Algorithme de reconstruction
    print(f"Traitement de {len(lignes_brutes)} segments...")
    try:
        reseau_lignes = unary_union(lignes_brutes)
        polygones = list(polygonize(reseau_lignes))
    except Exception as e:
        print(f"Erreur lors de la reconstruction : {e}")
        return

    print(f"--> {len(polygones)} formes fermées détectées.")

    # 3. Visualisation avec Matplotlib
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(f"Reconstruction Structurelle : {len(polygones)} objets détectés")

    # A. D'abord, on dessine les lignes brutes en gris clair (le fond)
    for line in lignes_brutes:
        x, y = line.xy
        ax.plot(x, y, color='lightgray', linewidth=0.5, zorder=1)

    # B. Ensuite, on dessine les formes reconstruites par dessus
    count_rect = 0
    
    for poly in polygones:
        # ... dans la boucle for poly in polygones ...
    
    # 1. Nettoyage de base (simplification des micro-détails)
        poly_clean = poly.simplify(0.1)
        
        # 2. Calcul du "Rectangle Orienté Minimum" (La boîte idéale)
        box = poly_clean.minimum_rotated_rectangle
        
        # 3. Calcul du score de ressemblance (Ratio d'aire)
        # Si poly_clean est un rectangle parfait, aire_poly == aire_box, donc ratio = 1.0
        aire_poly = poly_clean.area
        aire_box = box.area
        
        is_rectangle = False
        
        # Sécurité division par zéro
        if aire_box > 0:
            ratio = aire_poly / aire_box
            
            # SI la forme remplit plus de 99% de sa boîte idéale
            # ALORS on considère que c'est un rectangle
            if ratio > 0.99:
                is_rectangle = True
        
        # --- Classification et Affichage ---
        
        x, y = poly_clean.exterior.xy
        
        if is_rectangle:
            # C'est un rectangle (même avec des points en trop sur les bords)
            couleur = 'green'
            label = "Rectangle (Corrigé)"
            # Optionnel : Si vous voulez "nettoyer" la donnée pour l'export, 
            # vous pouvez remplacer 'poly' par 'box' ici.
        elif len(poly_clean.exterior.coords) - 1 == 3:
            couleur = 'blue'
            label = "Triangle"
        else:
            couleur = 'red'
            label = "Autre"

        # Dessin
        ax.fill(x, y, alpha=0.4, fc=couleur, ec='black', linewidth=1, zorder=2)
        
        # DEBUG : Afficher les points SEULEMENT si c'est encore rouge
        if couleur == 'red':
            ax.plot(x, y, 'o', color='yellow', markersize=4)

    # Inverser l'axe Y (car en PDF (0,0) est souvent en haut, en plot c'est en bas)
    ax.invert_yaxis()
    ax.set_aspect('equal') # Important pour ne pas déformer les rectangles
    
    print(f"Affichage... (Rectangles identifiés : {count_rect})")
    print("Fermez la fenêtre pour terminer le script.")
    plt.show()

# Lancez le script
visualiser_reconstruction("mon_fichier.pdf")