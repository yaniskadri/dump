import fitz
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union
import warnings

# On ignore les warnings de géométrie pour ne pas polluer la console
warnings.filterwarnings("ignore")

def analyse_structurelle_finale(pdf_path):
    # --- 1. EXTRACTION ---
    doc = fitz.open(pdf_path)
    page = doc[0]
    paths = page.get_drawings()
    
    lignes_brutes = []
    print("Extraction des vecteurs...")
    
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

    # --- 2. RECONSTRUCTION ---
    print("Calcul des intersections et formes...")
    try:
        reseau = unary_union(lignes_brutes)
        polygones = list(polygonize(reseau))
    except Exception as e:
        print(f"Erreur critique de reconstruction : {e}")
        return

    print(f"--> {len(polygones)} formes potentielles trouvées.")

    # --- 3. ANALYSE & DESSIN ---
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(f"Analyse Structurelle : {len(polygones)} objets")
    
    # Fond : Lignes brutes en gris très clair
    for line in lignes_brutes:
        x, y = line.xy
        ax.plot(x, y, color='#e0e0e0', linewidth=0.5, zorder=1)

    compteur_rect = 0

    for i, poly in enumerate(polygones):
        try:
            # A. NETTOYAGE : On "bouche" les trous (Donut -> Plein)
            # On prend l'extérieur uniquement
            poly_plein = Polygon(poly.exterior)
            
            # B. SIMPLIFICATION : On lisse les coins (Tolérance 1.0)
            poly_clean = poly_plein.simplify(1.0)
            
            # C. CALCUL DU RATIO (Aire forme / Aire boite)
            box = poly_clean.minimum_rotated_rectangle
            
            ratio = 0
            if box.area > 0:
                ratio = poly_clean.area / box.area
            
            # D. DECISION
            is_rectangle = False
            if ratio > 0.88: # Tolérance 88% (assez large pour capturer les rectangles sales)
                is_rectangle = True
            
            # E. PREPARATION DESSIN (Couleurs)
            if is_rectangle:
                couleur = 'green'
                alpha_val = 0.4
                label = "Rect"
                compteur_rect += 1
                
                # Petit détail : Si l'aire a changé radicalement, c'était un conteneur
                if abs(poly.area - poly_plein.area) > 1:
                    couleur = '#006400' # Vert foncé pour les conteneurs
                    label = "Container"
            else:
                couleur = 'red'
                alpha_val = 0.2
                label = f"{ratio:.2f}"

            # F. DESSIN (Sécurisé)
            # On dessine poly_clean (la forme pleine et lissée)
            if not poly_clean.is_empty:
                x, y = poly_clean.exterior.xy
                ax.fill(x, y, alpha=alpha_val, fc=couleur, ec='black', linewidth=0.5, zorder=2)
                
                # Si c'est rouge, on écrit le ratio pour comprendre pourquoi
                if not is_rectangle and poly_clean.area > 50: # On n'écrit pas sur les minuscules trucs
                    cx, cy = poly_clean.centroid.x, poly_clean.centroid.y
                    ax.text(cx, cy, label, fontsize=6, color='darkred', ha='center')

        except Exception as e:
            # Si un polygone spécifique plante, on l'ignore et on continue les autres
            print(f"Erreur sur le polygone {i}: {e}")
            continue

    ax.invert_yaxis()
    ax.set_aspect('equal')
    print(f"Terminé ! {compteur_rect} rectangles validés (Vert).")
    plt.show()

# Lancer
analyse_structurelle_finale("mon_fichier.pdf")