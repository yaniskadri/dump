import fitz  # PyMuPDF

def visualiser_vecteurs(pdf_path, output_image="visualisation_debug.png"):
    doc = fitz.open(pdf_path)
    page = doc[0]
    
    overlay = page.new_shape()
    chemins = page.get_drawings()
    
    print(f"Analyse de {len(chemins)} objets...")
    
    for chemin in chemins:
        items = chemin["items"]

        # ... début de la boucle ...
        items = chemin["items"]
        
        # Récupérer l'épaisseur du trait (width)
        # Parfois c'est stocké dans chemin['width'], parfois il faut regarder le style
        stroke_width = chemin.get("width", 0) # 0 par défaut si non trouvé
        
        # Si c'est un seul segment mais qu'il est épais, c'est un "Faux Rectangle"
        if len(items) == 1 and stroke_width > 1:
            print(f"Objet {chemin['seqno']} : Ligne unique mais ÉPAISSE (Width: {stroke_width}) -> Ressemble à un rectangle !")
        
        # ... suite du script ...
        
        # Sécurité : Si l'objet n'a pas de bounding box valide, on passe
        if not chemin["rect"]:
            continue
            
        rect_bbox = fitz.Rect(chemin["rect"])
        nb_segments = len(items)
        est_ferme = chemin["closePath"]
        
        # --- Logique de détection ---
        is_rect = False
        is_tri = False
        
        # 1. Cas du Rectangle natif ('re')
        if nb_segments == 1 and items[0][0] == "re":
            is_rect = True
        # 2. Cas du Rectangle dessiné (4 lignes)
        elif nb_segments == 4 and est_ferme:
            is_rect = True
        # 3. Cas du Triangle
        elif nb_segments == 3 and est_ferme:
            is_tri = True

        # --- Couleurs ---
        if is_rect:
            color_fill = (0, 1, 0) # Vert
            opacity = 0.3 
        elif is_tri:
            color_fill = (0, 0, 1) # Bleu
            opacity = 0.3
        else:
            color_fill = (1, 0, 0) # Rouge (Complexe)
            opacity = 0.1 # Très transparent pour voir les points

        # Dessiner la boite globale
        overlay.draw_rect(rect_bbox)
        overlay.finish(color=color_fill, fill=color_fill, fill_opacity=opacity, width=0.5)
        
        # --- VISUALISER LES POINTS (CORRECTION DU BUG) ---
        # On ne dessine les points jaunes que pour les formes complexes (ni rect, ni tri)
        if not is_rect and not is_tri:
            for item in items:
                type_seg = item[0] # "l" (ligne), "c" (courbe), "re" (rect)
                point_a_dessiner = None

                # On extrait le point final selon le type de segment
                if type_seg == "l":  # Ligne : ('l', p1, p2)
                    point_a_dessiner = item[-1]
                elif type_seg == "c": # Courbe : ('c', p1, p2, p3, p4)
                    point_a_dessiner = item[-1]
                # Si c'est 're', c'est un rectangle entier, pas un point, on ignore.
                
                if point_a_dessiner:
                    # On dessine le petit point jaune
                    overlay.draw_circle(point_a_dessiner, 2)
                    overlay.finish(color=(1, 1, 0), fill=(1, 1, 0), width=0)

    overlay.commit()
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    pix.save(output_image)
    print(f"Terminé : {output_image}")

# Lancez le script
visualiser_vecteurs("mon_fichier.pdf")