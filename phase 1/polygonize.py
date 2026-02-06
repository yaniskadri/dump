import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import warnings
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import polygonize, unary_union

# On ignore les warnings géométriques mineurs
warnings.filterwarnings("ignore")

def analyser_diagramme_final(pdf_path):
    print(f"--- Analyse de : {pdf_path} ---")

    # ==========================================
    # 1. EXTRACTION
    # ==========================================
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
                lignes_brutes.extend([
                    LineString([p1, p2]), LineString([p2, p3]), 
                    LineString([p3, p4]), LineString([p4, p1])
                ])

    # ==========================================
    # 2. RECONSTRUCTION
    # ==========================================
    print("Reconstruction de la topologie...")
    try:
        reseau = unary_union(lignes_brutes)
        polygones = list(polygonize(reseau))
    except Exception as e:
        print(f"Erreur critique : {e}")
        return

    print(f"--> {len(polygones)} formes détectées.")

    # ==========================================
    # 3. CLASSIFICATION ET DESSIN
    # ==========================================
    fig, ax = plt.subplots(figsize=(16, 16)) # Grande image pour bien voir
    ax.set_title(f"Diagramme Électrique : {len(polygones)} objets")
    
    # Fond gris pâle
    for line in lignes_brutes:
        x, y = line.xy
        ax.plot(x, y, color='#f0f0f0', linewidth=0.5, zorder=0)

    # --- PARAMÈTRES (Ajustables) ---
    SEUIL_FIL_FIN = 5.0      # En dessous de 5px = Fil commande (Jaune)
    SEUIL_BUSBAR = 40.0      # Entre 5px et 40px = Busbar/Puissance (Vert Clair)
                             # Au dessus de 40px = Vrai Composant (Vert Foncé)

    compteurs = {"Fil": 0, "Busbar": 0, "Composant": 0, "Groupe": 0}

    for i, poly in enumerate(polygones):
        try:
            # A. NETTOYAGE
            poly_clean = poly.buffer(0) # Répare les géométries
            
            # B. MESURE ÉPAISSEUR (Crucial pour fils vs composants)
            box_rot = poly_clean.minimum_rotated_rectangle
            if box_rot.is_empty: continue
            
            x, y = box_rot.exterior.coords.xy
            edge1 = Point(x[0], y[0]).distance(Point(x[1], y[1]))
            edge2 = Point(x[1], y[1]).distance(Point(x[2], y[2]))
            
            thickness = min(edge1, edge2)
            length = max(edge1, edge2)
            
            # C. CALCUL RATIOS
            poly_enveloppe = Polygon(poly_clean.exterior).simplify(0.5)
            poly_matiere = poly_clean.simplify(0.5)

            # G: Géométrie (Ressemblance Rectangle)
            box_env = poly_enveloppe.minimum_rotated_rectangle
            ratio_rect = 0
            if box_env.area > 0:
                ratio_rect = poly_enveloppe.area / box_env.area

            # D: Densité (Plein vs Vide)
            ratio_densite = 1.0
            if poly_enveloppe.area > 0:
                ratio_densite = poly_matiere.area / poly_enveloppe.area
            
            # C: Circularité
            perimetre = poly_enveloppe.length
            circularity = 0
            if perimetre > 0:
                circularity = (4 * math.pi * poly_enveloppe.area) / (perimetre ** 2)

            # --- ARBRE DE DÉCISION (LOGIQUE METIER) ---
            
            label = ""
            label_debug = f"G:{ratio_rect:.2f}\nD:{ratio_densite:.2f}"
            couleur = "red"
            alpha_val = 0.5
            z_order = 2
            style = "solid"

            # 1. FIL DE COMMANDE (Très fin)
            if thickness < SEUIL_FIL_FIN:
                couleur = '#FFD700' # Gold/Jaune
                label = "Fil (Cmd)"
                alpha_val = 0.4
                z_order = 1
                compteurs["Fil"] += 1
            
            # 2. BUSBAR / CÂBLE PUISSANCE (Rectangle moyen et allongé)
            # C'est ici que vos rectangles verts vont atterrir !
            elif thickness < SEUIL_BUSBAR and ratio_rect > 0.75:
                # On vérifie juste qu'il n'est pas "vide" (ce serait un cadre)
                if ratio_densite > 0.5: 
                    couleur = '#90EE90' # LightGreen (Vert Clair)
                    label = "Busbar/Power"
                    alpha_val = 0.6
                    z_order = 1
                    compteurs["Busbar"] += 1
                else:
                    couleur = 'gray' # Cadre vide fin
                    label = "Layout"
                    alpha_val = 0.05
                    z_order = 0

            # 3. CERCLE (Composant type moteur/voyant)
            elif circularity > 0.85:
                couleur = 'magenta'
                label = "Cercle"
                z_order = 3
                compteurs["Composant"] += 1

            # 4. HEXAGONE
            elif 0.70 <= ratio_rect <= 0.82 and circularity > 0.6:
                couleur = 'orange'
                label = "Symbole"
                z_order = 3
                compteurs["Composant"] += 1

            # 5. GROS COMPOSANT RECTANGULAIRE (> 40px)
            elif ratio_rect > 0.70:
                if ratio_densite > 0.80:
                    couleur = '#006400' # DarkGreen (Vert Foncé)
                    label = "Composant"
                    z_order = 3
                    compteurs["Composant"] += 1
                elif ratio_densite < 0.25:
                    couleur = 'gray'
                    label = "Layout"
                    alpha_val = 0.05
                    z_order = 0
                else:
                    couleur = 'blue' # Groupe/Conteneur
                    label = "Groupe"
                    style = "dashed"
                    alpha_val = 0.1
                    z_order = 1
                    compteurs["Groupe"] += 1

            # 6. FORME COMPLEXE DENSE (L-Shape / Triangles)
            elif ratio_densite > 0.75:
                couleur = 'cyan'
                label = "Complexe"
                z_order = 3
                compteurs["Composant"] += 1

            # 7. ERREUR / INCONNU
            else:
                couleur = 'red'
                label = "Inconnu"
                alpha_val = 0.2

            # --- DESSIN ---
            
            if label == "Groupe":
                x, y = poly_enveloppe.exterior.xy
                ax.plot(x, y, color='blue', linewidth=1, linestyle='--', zorder=z_order)
                if poly_enveloppe.area > 300:
                    cx, cy = poly_enveloppe.centroid.x, poly_enveloppe.centroid.y
                    ax.text(cx, cy, "Groupe", fontsize=6, color='blue', ha='center', alpha=0.7)

            elif label == "Layout":
                pass 

            elif label != "Inconnu":
                # Dessin standard (Fils, Busbars, Composants)
                x, y = poly_matiere.exterior.xy
                ax.fill(x, y, alpha=alpha_val, fc=couleur, ec='black', linewidth=0.5, zorder=z_order)

            else:
                # Erreurs (Rouge) avec debug
                if poly_enveloppe.area > 50:
                    x, y = poly_matiere.exterior.xy
                    ax.fill(x, y, alpha=alpha_val, fc='red', ec='red', linewidth=1, zorder=z_order)
                    cx, cy = poly_enveloppe.centroid.x, poly_enveloppe.centroid.y
                    ax.text(cx, cy, label_debug, fontsize=7, color='darkred', weight='bold', ha='center')

        except Exception as e:
            continue

    # Légende
    patches = [
        mpatches.Patch(color='#FFD700', label=f'Fil Cmd (<{SEUIL_FIL_FIN}px)'),
        mpatches.Patch(color='#90EE90', label=f'Busbar/Power (<{SEUIL_BUSBAR}px)'),
        mpatches.Patch(color='#006400', label='Composant Rect (>40px)'),
        mpatches.Patch(color='cyan', label='Composant Complexe'),
        mpatches.Patch(color='magenta', label='Cercle'),
        mpatches.Patch(color='orange', label='Hexagone'),
        mpatches.Patch(color='blue', alpha=0.3, linestyle='--', label='Groupe'),
        mpatches.Patch(color='red', label='Inconnu')
    ]
    ax.legend(handles=patches, loc='upper right')
    
    ax.invert_yaxis()
    ax.set_aspect('equal')
    print(f"Terminé. Stats : {compteurs}")
    plt.show()

# Remplacer par votre fichier
analyser_diagramme_final("mon_fichier.pdf")