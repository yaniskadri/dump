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

    # ... (Dans la boucle for poly in polygones) ...

            # 1. PRÉPARATION DES VERSIONS
            # Version "Pleine" (On bouche les trous pour voir la forme globale)
            poly_enveloppe = Polygon(poly.exterior).simplify(1.0)
            # Version "Matière" (La forme réelle avec ses trous éventuels)
            poly_matiere = poly.simplify(1.0)

            # 2. CALCUL DU RATIO GÉOMÉTRIQUE (Basé sur l'enveloppe extérieure)
            box = poly_enveloppe.minimum_rotated_rectangle
            
            ratio_forme = 0
            if box.area > 0:
                ratio_forme = poly_enveloppe.area / box.area
            
            # 3. CALCUL DU RATIO DE DENSITÉ (Plein ou Vide ?)
            # Si densité ~ 1.0, c'est un bloc plein. Si densité ~ 0.2, c'est un cadre.
            ratio_densite = 0
            if poly_enveloppe.area > 0:
                ratio_densite = poly_matiere.area / poly_enveloppe.area

            # --- ARBRE DE DÉCISION ---
            label = ""
            couleur = "red"
            alpha_val = 0.4
            z_order = 2
            is_valid_shape = False

            # A. EST-CE UNE FORME RECTANGULAIRE EXTÉRIEUREMENT ?
            if ratio_forme > 0.88:
                is_valid_shape = True
                
                # C'est un rectangle. Mais quel type ?
                if ratio_densite > 0.90:
                    # C'est un OBJET PLEIN (ex: un mur, une machine)
                    couleur = 'green'
                    label = "Objet"
                    alpha_val = 0.5
                    z_order = 3 # Au dessus des conteneurs
                else:
                    # C'est un CONTENEUR (ex: le cadre à 0.19)
                    # Il est rectangulaire dehors, mais vide dedans.
                    couleur = 'blue' 
                    label = "Groupe/Conteneur"
                    alpha_val = 0.1 # Très transparent pour voir ce qu'il y a dedans
                    z_order = 1 # En arrière plan
            
            # B. EST-CE UN HEXAGONE ?
            elif 0.70 < ratio_forme < 0.80:
                couleur = 'orange'
                label = "Hexagone"
                is_valid_shape = True

            # C. LE RESTE (Formes complexes non identifiées)
            else:
                couleur = 'red'
                label = f"R:{ratio_forme:.2f}"
                alpha_val = 0.2

            # --- DESSIN ---
            if is_valid_shape:
                # On dessine l'enveloppe pour les conteneurs (pour avoir un fond uni)
                # ou la matière pour les objets (pour garder la précision)
                
                if label == "Groupe/Conteneur":
                    # Pour le conteneur, on dessine juste le cadre extérieur en pointillé (simulé)
                    # On utilise l'enveloppe
                    x, y = poly_enveloppe.exterior.xy
                    ax.plot(x, y, color='blue', linewidth=1.5, linestyle='--') # Bordure bleue
                    ax.fill(x, y, alpha=0.05, fc='blue', zorder=z_order) # Fond très léger
                    
                    # Optionnel : Afficher le label en haut à gauche du conteneur
                    minx, miny, maxx, maxy = poly_enveloppe.bounds
                    ax.text(minx, maxy, "GROUPE", color='blue', fontsize=6, verticalalignment='bottom')
                    
                else:
                    # Pour les objets verts/oranges, on remplit normalement
                    x, y = poly_matiere.exterior.xy
                    ax.fill(x, y, alpha=alpha_val, fc=couleur, ec='black', linewidth=0.5, zorder=z_order)

            else:
                # Erreurs (Rouge)
                x, y = poly_matiere.exterior.xy
                ax.fill(x, y, alpha=alpha_val, fc=couleur, ec=couleur, zorder=z_order)
                
    ax.invert_yaxis()
    ax.set_aspect('equal')
    print(f"Terminé ! {compteur_rect} rectangles validés (Vert).")
    plt.show()

# Lancer
analyse_structurelle_finale("mon_fichier.pdf")