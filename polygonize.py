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
        from shapely.geometry import Polygon

# ... (votre code précédent) ...

for poly in polygones:
    
    # ASTUCE CLÉ : On recrée un polygone NEUF en utilisant seulement l'extérieur.
    # Cela "bouche" tous les trous créés par les rectangles internes.
    poly_plein = Polygon(poly.exterior)

    # On simplifie un peu l'extérieur pour les petits défauts d'alignement
    poly_clean = poly_plein.simplify(1.0) # Tolérance de 1 unité
    
    # --- Calcul du Ratio (Sur la version "pleine") ---
    box = poly_clean.minimum_rotated_rectangle
    
    is_rectangle = False
    
    if box.area > 0:
        # On compare l'aire du polygone PLEIN avec sa boîte englobante
        ratio = poly_clean.area / box.area
        
        # Tolérance de 0.90 (90%)
        if ratio > 0.90:
            is_rectangle = True

    # --- Affichage ---
    x, y = poly_clean.exterior.xy
    
    if is_rectangle:
        couleur = 'green'
        # On affiche "Container" pour différencier des rectangles simples
        if poly.area != poly_plein.area:
            label = "Rect (Conteneur)"
            alpha_val = 0.2 # Plus transparent pour voir ce qu'il y a dedans
        else:
            label = "Rectangle"
            alpha_val = 0.4
    else:
        couleur = 'red'
        label = f"Ratio: {ratio:.2f}"

    ax.fill(x, y, alpha=alpha_val, fc=couleur, ec='black', linewidth=1, zorder=2)
    # Inverser l'axe Y (car en PDF (0,0) est souvent en haut, en plot c'est en bas)
    ax.invert_yaxis()
    ax.set_aspect('equal') # Important pour ne pas déformer les rectangles
    
    print(f"Affichage... (Rectangles identifiés : {count_rect})")
    print("Fermez la fenêtre pour terminer le script.")
    plt.show()

# Lancez le script
visualiser_reconstruction("mon_fichier.pdf")