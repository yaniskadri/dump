import fitz  # PyMuPDF

def visualiser_vecteurs(pdf_path, output_image="visualisation_debug.png"):
    doc = fitz.open(pdf_path)
    page = doc[0]  # On travaille sur la première page
    
    # On prépare un objet "Shape" pour dessiner par-dessus la page existante
    overlay = page.new_shape()
    
    chemins = page.get_drawings()
    print(f"Analyse et dessin de {len(chemins)} objets...")
    
    for chemin in chemins:
        items = chemin["items"]
        rect_bbox = fitz.Rect(chemin["rect"]) # La boîte englobante de l'objet
        nb_segments = len(items)
        est_ferme = chemin["closePath"]
        
        # --- Logique de détection (Simplifiée pour l'exemple) ---
        is_rect = False
        is_tri = False
        
        # Détection Rectangle basique
        if (nb_segments == 4 and est_ferme) or (nb_segments == 1 and items[0][0] == "re"):
             is_rect = True
        # Détection Triangle basique
        elif nb_segments == 3 and est_ferme:
             is_tri = True

        # --- Choix des couleurs pour le débogage ---
        if is_rect:
            # Vert semi-transparent pour les rectangles
            color_fill = (0, 1, 0)
            opacity = 0.3 
            debug_msg = "Rectangle"
        elif is_tri:
            # Bleu semi-transparent pour les triangles
            color_fill = (0, 0, 1)
            opacity = 0.3
            debug_msg = "Triangle"
        else:
            # Rouge vif plus transparent pour les complexes
            color_fill = (1, 0, 0)
            opacity = 0.2
            debug_msg = f"Complexe ({nb_segments} seg)"

        # 1. Dessiner la boîte englobante colorée
        overlay.draw_rect(rect_bbox)
        overlay.finish(color=color_fill, fill=color_fill, fill_opacity=opacity, width=0.5)
        
        # 2. VISUALISER LES SEGMENTS (Pour les objets complexes)
        # Si ce n'est pas un rectangle ou triangle simple, on marque les sommets
        if not is_rect and not is_tri:
            for item in items:
                # Chaque 'item' finit par un point (x, y). 
                # item[-1] est le point d'arrivée du segment.
                end_point = item[-1] 
                
                # On dessine un petit cercle jaune à chaque sommet
                # Le rayon est de 2 pixels
                overlay.draw_circle(end_point, 2)
                overlay.finish(color=(1, 1, 0), fill=(1, 1, 0), width=0.5)

    # Appliquer les dessins sur la page
    overlay.commit()
    
    # Rendre la page en image haute résolution
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # Zoom x2 pour la clarté
    pix.save(output_image)
    print(f"Terminé. Image de débogage enregistrée sous : {output_image}")

# Remplacez par votre fichier
visualiser_vecteurs("mon_fichier_vectoriel.pdf")