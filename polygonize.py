import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import math
import warnings
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

# On ignore les warnings géométriques mineurs pour garder la console propre
warnings.filterwarnings("ignore")

def analyser_diagramme_complet(pdf_path):
    print(f"--- Analyse de : {pdf_path} ---")

    # ==========================================
    # 1. EXTRACTION DES LIGNES (RAW DATA)
    # ==========================================
    doc = fitz.open(pdf_path)
    page = doc[0]
    paths = page.get_drawings()
    
    lignes_brutes = []
    print("Extraction des vecteurs...")
    
    for path in paths:
        for item in path["items"]:
            # Ligne simple
            if item[0] == "l":
                lignes_brutes.append(LineString([item[1], item[2]]))
            # Courbe (on prend la corde pour simplifier la structure)
            elif item[0] == "c":
                lignes_brutes.append(LineString([item[1], item[-1]]))
            # Rectangle natif (rare, mais on le convertit en 4 lignes)
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
    print("Soudure des lignes et détection des cycles...")
    try:
        # unary_union gère les intersections et "snappe" les points proches
        reseau = unary_union(lignes_brutes)
        # polygonize trouve toutes les boucles fermées
        polygones = list(polygonize(reseau))
    except Exception as e:
        print(f"Erreur critique lors de la reconstruction : {e}")
        return

    print(f"--> {len(polygones)} formes fermées détectées.")

    # ==========================================
    # 3. ANALYSE ET CLASSIFICATION
    # ==========================================
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_title(f"Analyse Diagramme : {len(polygones)} objets")
    
    # Fond : Lignes brutes en gris très clair pour le contexte
    for line in lignes_brutes:
        x, y = line.xy
        ax.plot(x, y, color='#e0e0e0', linewidth=0.5, zorder=0)

    for i, poly in enumerate(polygones):
        try:
            # --- A. NETTOYAGE CHIRURGICAL (Anti-Fils) ---
            # On érode de 2 unités pour supprimer les fils connectés (les "tentacules")
            # Puis on dilate pour revenir à la taille approximative.
            poly_core = poly.buffer(-2.0).buffer(2.0)
            
            # Si l'objet est trop fin (ex: un câble pur), il disparaît. On garde l'original.
            if poly_core.is_empty:
                poly_core = poly

            # --- B. PRÉPARATION DES GEOMETRIES ---
            # 1. L'Enveloppe : Forme extérieure bouchée (pour vérifier si c'est un rectangle)
            poly_enveloppe = Polygon(poly_core.exterior).simplify(1.0)
            
            # 2. La Matière : La forme réelle (pour vérifier si c'est plein ou vide)
            poly_matiere = poly_core.simplify(1.0)

            # --- C. CALCUL DES INDICATEURS (RATIOS) ---
            
            # G = Ratio Géométrique (Ressemblance Rectangle)
            # 1.0 = Rectangle parfait. 0.5 = Triangle ou L.
            box = poly_enveloppe.minimum_rotated_rectangle
            ratio_rect = 0
            if box.area > 0:
                ratio_rect = poly_enveloppe.area / box.area

            # D = Ratio Densité (Plein vs Vide)
            # 1.0 = Bloc plein. <0.2 = Cadre vide.
            ratio_densite = 1.0
            if poly_enveloppe.area > 0:
                ratio_densite = poly_matiere.area / poly_enveloppe.area

            # C = Circularité (Pour Cercles/Hexagones)
            perimetre = poly_enveloppe.length
            circularity = 0
            if perimetre > 0:
                circularity = (4 * math.pi * poly_enveloppe.area) / (perimetre ** 2)

            # --- D. ARBRE DE DÉCISION (CLASSIFICATION) ---
            
            label = ""
            label_debug = f"G:{ratio_rect:.2f}\nD:{ratio_densite:.2f}" # Tableau de bord
            is_valid = False
            z_order = 2
            
            # 1. CERCLE (Circularité très haute)
            if circularity > 0.88:
                couleur = 'magenta'
                label = "Cercle"
                is_valid = True
                alpha_val = 0.5

            # 2. RECTANGLE (G > 0.85)
            # Note : On accepte 0.85 pour tolérer les coins arrondis ou fils résiduels
            elif ratio_rect > 0.85:
                is_valid = True
                
                # Sous-classification par densité
                if ratio_densite > 0.80:
                    couleur = 'green'     # OBJET PLEIN (La cible principale)
                    label = "Objet"
                    alpha_val = 0.5
                    z_order = 3
                elif ratio_densite < 0.25:
                    couleur = 'gray'      # ZONE / LAYOUT (Invisible)
                    label = "Layout"
                    alpha_val = 0.05
                    z_order = 1
                else:
                    couleur = 'blue'      # CONTENEUR (Groupe d'objets)
                    label = "Groupe"
                    alpha_val = 0.1
                    z_order = 1

            # 3. HEXAGONE (G entre 0.7 et 0.82)
            elif 0.70 < ratio_rect < 0.82:
                couleur = 'orange'
                label = "Hexagone"
                is_valid = True
                alpha_val = 0.5

            # 4. FORME COMPLEXE VALIDE (Triangle ou L-Shape)
            # C'est ici que tombe le "Rénégat" s'il est mal nettoyé mais dense
            elif 0.40 < ratio_rect < 0.65 and ratio_densite > 0.80:
                couleur = 'cyan'
                label = "Forme L/Tri"
                is_valid = True
                alpha_val = 0.5

            # 5. INCONNU / ERREUR
            else:
                couleur = 'red'
                label = "Inconnu"
                alpha_val = 0.2

            # --- E. DESSIN ---
            
            # Cas spécial : Groupe (Bordure pointillée)
            if label == "Groupe":
                x, y = poly_enveloppe.exterior.xy
                ax.plot(x, y, color='blue', linewidth=1, linestyle='--', zorder=z_order)
                ax.fill(x, y, alpha=alpha_val, fc='blue', zorder=z_order)
                # Affichage discret du debug
                cx, cy = poly_enveloppe.centroid.x, poly_enveloppe.centroid.y
                if poly_enveloppe.area > 100: # Pas sur les minuscules
                    ax.text(cx, cy, label_debug, fontsize=6, color='blue', ha='center')

            # Cas objets valides (Vert, Orange, Cyan)
            elif is_valid and label != "Layout":
                x, y = poly_matiere.exterior.xy
                ax.fill(x, y, alpha=alpha_val, fc=couleur, ec='black', linewidth=0.5, zorder=z_order)

            # Cas Layout (Gris) -> On ne dessine presque rien
            elif label == "Layout":
                pass 

            # Cas Erreur (Rouge) -> On affiche le DEBUG pour comprendre
            elif couleur == 'red' and poly_enveloppe.area > 50:
                x, y = poly_matiere.exterior.xy
                ax.fill(x, y, alpha=alpha_val, fc='red', ec='red', linewidth=1, zorder=z_order)
                cx, cy = poly_enveloppe.centroid.x, poly_enveloppe.centroid.y
                ax.text(cx, cy, label_debug, fontsize=7, color='darkred', weight='bold', ha='center')

        except Exception as e:
            print(f"Erreur sur forme {i}: {e}")
            continue

    # Finalisation plot
    ax.invert_yaxis()
    ax.set_aspect('equal')
    plt.tight_layout()
    print("Affichage du résultat...")
    plt.show()

# --- Lancer le script ---
# Remplacez par votre fichier
analyser_diagramme_complet("votre_fichier.pdf")