import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import warnings
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import polygonize, unary_union

# Ignorer les avertissements géométriques mineurs
warnings.filterwarnings("ignore")

def analyser_diagramme_final(pdf_path):
    print(f"--- Analyse de : {pdf_path} ---")

    # ==========================================
    # 1. EXTRACTION DES DONNÉES (Lignes brutes)
    # ==========================================
    doc = fitz.open(pdf_path)
    page = doc[0] # Première page
    paths = page.get_drawings()
    
    lignes_brutes = []
    print("Extraction des vecteurs...")
    
    for path in paths:
        for item in path["items"]:
            # Ligne ('l')
            if item[0] == "l":
                lignes_brutes.append(LineString([item[1], item[2]]))
            # Courbe ('c') - On la linéarise simplement par sa corde
            elif item[0] == "c":
                lignes_brutes.append(LineString([item[1], item[-1]]))
            # Rectangle natif ('re')
            elif item[0] == "re":
                r = item[1]
                p1, p2, p3, p4 = (r[0], r[1]), (r[2], r[1]), (r[2], r[3]), (r[0], r[3])
                lignes_brutes.extend([
                    LineString([p1, p2]), LineString([p2, p3]), 
                    LineString([p3, p4]), LineString([p4, p1])
                ])

    # ==========================================
    # 2. RECONSTRUCTION GÉOMÉTRIQUE
    # ==========================================
    print("Reconstruction des formes fermées...")
    try:
        # unary_union : Coupe les lignes aux intersections et soude les bouts
        reseau = unary_union(lignes_brutes)
        # polygonize : Trouve les cycles fermés
        polygones = list(polygonize(reseau))
    except Exception as e:
        print(f"Erreur critique lors de la reconstruction : {e}")
        return

    print(f"--> {len(polygones)} formes détectées.")

    # ==========================================
    # 3. ANALYSE ET CLASSIFICATION
    # ==========================================
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_title(f"Diagramme Reconstruit : {len(polygones)} objets")
    
    # Fond : Lignes brutes en gris très pâle
    for line in lignes_brutes:
        x, y = line.xy
        ax.plot(x, y, color='#f0f0f0', linewidth=0.5, zorder=0)

    # Paramètres calibrés selon votre pic à 7.2px
    SEUIL_EPAISSEUR_FIL = 8.0  # Tout ce qui est < 8px est suspecté d'être un fil
    SEUIL_ELANCEMENT = 4.0     # Il faut être 4x plus long que large pour être un fil

    compteurs = {"Objet": 0, "Fil": 0, "Groupe": 0, "Inconnu": 0}

    for i, poly in enumerate(polygones):
        try:
            # A. RÉPARATION TOPOLOGIQUE
            # Répare les géométries invalides (auto-intersection, points de contact)
            poly_clean = poly.buffer(0)
            
            # B. MESURE DE L'ÉPAISSEUR (Pour détecter les fils)
            box_rot = poly_clean.minimum_rotated_rectangle
            if box_rot.is_empty: continue
            
            x, y = box_rot.exterior.coords.xy
            # Calcul des longueurs des deux arêtes du rectangle orienté
            edge1 = Point(x[0], y[0]).distance(Point(x[1], y[1]))
            edge2 = Point(x[1], y[1]).distance(Point(x[2], y[2]))
            
            thickness = min(edge1, edge2)
            length = max(edge1, edge2)
            
            # C. FILTRE "FIL DE FER"
            is_wire = False
            # Si c'est fin (<8px) ET élancé (Ratio > 4), c'est un fil
            if thickness < SEUIL_EPAISSEUR_FIL and (length / thickness) > SEUIL_ELANCEMENT:
                is_wire = True
            # Sécurité pour les traits très fins (< 2px)
            elif thickness < 2.0:
                is_wire = True

            # D. PRÉPARATION GEOMETRIE (Enveloppe vs Matière)
            poly_enveloppe = Polygon(poly_clean.exterior).simplify(0.5)
            poly_matiere = poly_clean.simplify(0.5)

            # E. CALCUL DES RATIOS CLÉS
            # G: Ratio Géométrique (Ressemblance Rectangle)
            box_env = poly_enveloppe.minimum_rotated_rectangle
            ratio_rect = 0
            if box_env.area > 0:
                ratio_rect = poly_enveloppe.area / box_env.area

            # D: Ratio Densité (Plein vs Vide)
            ratio_densite = 1.0
            if poly_enveloppe.area > 0:
                ratio_densite = poly_matiere.area / poly_enveloppe.area
            
            # C: Circularité
            perimetre = poly_enveloppe.length
            circularity = 0
            if perimetre > 0:
                circularity = (4 * math.pi * poly_enveloppe.area) / (perimetre ** 2)

            # --- ARBRE DE DÉCISION ---
            label = ""
            label_debug = f"G:{ratio_rect:.2f}\nD:{ratio_densite:.2f}"
            couleur = "red"
            alpha_val = 0.5
            z_order = 2
            style = "solid" # solid, dashed
            
            # 1. C'EST UN FIL (Priorité absolue)
            if is_wire:
                couleur = 'yellow'
                label = "Fil"
                alpha_val = 0.3
                z_order = 1
                compteurs["Fil"] += 1

            # 2. C'EST UN CERCLE
            elif circularity > 0.88:
                couleur = 'magenta'
                label = "Cercle"
                z_order = 3
                compteurs["Objet"] += 1

            # 3. C'EST UN RECTANGLE (ou quasi-rectangle)
            # Seuil à 0.70 pour accepter les rectangles imparfaits
            elif ratio_rect > 0.70:
                
                # Trie par densité
                if ratio_densite > 0.80:
                    couleur = 'green'     # OBJET PLEIN (Vert)
                    label = "Objet"
                    z_order = 3
                    compteurs["Objet"] += 1
                    
                elif ratio_densite < 0.25:
                    couleur = 'gray'      # ZONE VIDE / LAYOUT (Gris invisible)
                    label = "Layout"
                    alpha_val = 0.05
                    z_order = 0
                    
                else:
                    couleur = 'blue'      # GROUPE / CONTENEUR (Bleu pointillé)
                    label = "Groupe"
                    alpha_val = 0.1
                    style = "dashed"
                    z_order = 1
                    compteurs["Groupe"] += 1

            # 4. C'EST UN HEXAGONE (Symbole)
            elif 0.70 <= ratio_rect <= 0.82 and circularity > 0.6:
                couleur = 'orange'
                label = "Hexagone"
                z_order = 3
                compteurs["Objet"] += 1

            # 5. C'EST UNE FORME COMPLEXE DENSE (Renegade L/Tri)
            # G faible mais D forte -> C'est de la matière bizarre mais valide
            elif 0.30 < ratio_rect <= 0.70 and ratio_densite > 0.75:
                couleur = 'cyan'
                label = "Forme L/Tri"
                z_order = 3
                compteurs["Objet"] += 1

            # 6. INCONNU / ERREUR
            else:
                couleur = 'red'
                label = "Inconnu"
                alpha_val = 0.2
                compteurs["Inconnu"] += 1

            # --- DESSIN ---
            
            if label == "Fil":
                # On dessine les fils en jaune discret
                x, y = poly_matiere.exterior.xy
                ax.fill(x, y, alpha=alpha_val, fc='yellow', ec='none', zorder=0)

            elif label == "Groupe":
                # Conteneur bleu pointillé
                x, y = poly_enveloppe.exterior.xy
                ax.plot(x, y, color='blue', linewidth=1, linestyle='--', zorder=z_order)
                # Debug discret
                cx, cy = poly_enveloppe.centroid.x, poly_enveloppe.centroid.y
                if poly_enveloppe.area > 200:
                    ax.text(cx, cy, label_debug, fontsize=6, color='blue', ha='center', alpha=0.7)

            elif label == "Layout":
                pass # On ne dessine pas les zones vides

            elif label != "Inconnu":
                # Objets valides (Vert, Cyan, Orange, Magenta)
                x, y = poly_matiere.exterior.xy
                ax.fill(x, y, alpha=alpha_val, fc=couleur, ec='black', linewidth=0.5, zorder=z_order)

            elif label == "Inconnu" and poly_enveloppe.area > 50:
                # Erreurs rouges avec debug texte
                x, y = poly_matiere.exterior.xy
                ax.fill(x, y, alpha=alpha_val, fc='red', ec='red', linewidth=1, zorder=z_order)
                cx, cy = poly_enveloppe.centroid.x, poly_enveloppe.centroid.y
                ax.text(cx, cy, label_debug, fontsize=7, color='darkred', weight='bold', ha='center')

        except Exception as e:
            print(f"Erreur poly {i}: {e}")
            continue

    # Légende et affichage
    patches = [
        mpatches.Patch(color='green', alpha=0.5, label='Objet Rect (G>0.7, D>0.8)'),
        mpatches.Patch(color='cyan', alpha=0.5, label='Objet L/Tri (G<0.7, D>0.75)'),
        mpatches.Patch(color='orange', alpha=0.5, label='Hexagone'),
        mpatches.Patch(color='blue', alpha=0.2, linestyle='--', label='Groupe (Conteneur)'),
        mpatches.Patch(color='yellow', alpha=0.5, label=f'Fil (<{SEUIL_EPAISSEUR_FIL}px)'),
        mpatches.Patch(color='red', alpha=0.5, label='Inconnu (Erreur)')
    ]
    ax.legend(handles=patches, loc='upper right')
    
    ax.invert_yaxis()
    ax.set_aspect('equal')
    print(f"Terminé. Stats : {compteurs}")
    plt.show()

# Lancer le script
analyser_diagramme_final("mon_fichier.pdf")