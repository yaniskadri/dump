for i, poly in enumerate(polygones):
        try:
            # 1. RÉPARATION TOPOLOGIQUE (Fix "touching at a point")
            # buffer(0) répare les auto-intersections et les points de contact invalides
            poly_clean = poly.buffer(0)
            
            # 2. CALCUL DE LA BOÎTE (Pour mesurer l'épaisseur)
            box = poly_clean.minimum_rotated_rectangle
            
            if box.is_empty: continue
            
            # On récupère les coordonnées pour calculer largeur/hauteur
            x, y = box.exterior.coords.xy
            # Astuce pour avoir longueur et largeur du rectangle orienté
            edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])), 
                           Point(x[1], y[1]).distance(Point(x[2], y[2])))
            width = min(edge_length)
            length = max(edge_length)
            
            # 3. FILTRE "FIL DE FER" (WIRE DETECTOR)
            # Si l'objet est très fin (ex: épaisseur < 3 unités) OU très élancé (Ratio > 1:20)
            # Alors c'est un FIL -> On l'ignore ou on le grise.
            is_wire = False
            if width < 3.0 or (length > 0 and width/length < 0.05):
                is_wire = True

            # 4. PRÉPARATION ANALYSE (Sur la forme réparée)
            poly_enveloppe = Polygon(poly_clean.exterior).simplify(0.5)
            poly_matiere = poly_clean.simplify(0.5)

            # 5. CALCULS DES RATIOS
            # G (Géométrie)
            box_env = poly_enveloppe.minimum_rotated_rectangle
            ratio_rect = 0
            if box_env.area > 0:
                ratio_rect = poly_enveloppe.area / box_env.area

            # D (Densité)
            ratio_densite = 1.0
            if poly_enveloppe.area > 0:
                ratio_densite = poly_matiere.area / poly_enveloppe.area
            
            # Label de debug
            label_debug = f"G:{ratio_rect:.2f} D:{ratio_densite:.2f}"

            # --- ARBRE DE DÉCISION ---
            label = ""
            is_valid = False
            z_order = 2
            alpha_val = 0.5
            
            # A. C'EST UN FIL (Nouveau !)
            if is_wire:
                couleur = 'yellow' # Ou 'none' si vous voulez les cacher
                label = "Fil"
                alpha_val = 0.3
                z_order = 0 # Au fond
                # On le marque comme traité pour ne pas qu'il finisse en rouge
                
            # B. OBJET VALIDE (Vert / Cyan / Orange)
            elif ratio_rect > 0.70: # Seuil tolérant
                is_valid = True
                
                # Vérif densité
                if ratio_densite > 0.85:
                    couleur = 'green'    # Objet Plein
                    label = "Objet"
                elif ratio_densite < 0.25:
                    couleur = 'gray'     # Layout vide
                    label = "Layout"
                    alpha_val = 0.05
                    z_order = 0
                else:
                    couleur = 'blue'     # Groupe
                    label = "Groupe"
                    alpha_val = 0.1
                    z_order = 1

            # C. TRIANGLE / FORME EN L (Votre Rénégat dense)
            elif 0.40 < ratio_rect < 0.70 and ratio_densite > 0.80:
                couleur = 'cyan'
                label = "Forme L/Tri"
                is_valid = True

            # D. ERREUR
            else:
                couleur = 'red'
                label = "Inconnu"
                alpha_val = 0.2

            # --- DESSIN ---
            if label == "Fil":
                # On dessine les fils en jaune discret (ou on met 'pass' pour les cacher)
                x, y = poly_matiere.exterior.xy
                ax.fill(x, y, alpha=0.3, fc='yellow', ec='none', zorder=0)
            
            elif label == "Groupe":
                x, y = poly_enveloppe.exterior.xy
                ax.plot(x, y, color='blue', linewidth=1, linestyle='--', zorder=z_order)
                # Debug texte
                cx, cy = poly_enveloppe.centroid.x, poly_enveloppe.centroid.y
                ax.text(cx, cy, label_debug, fontsize=6, color='blue', ha='center')

            elif is_valid and label != "Layout":
                x, y = poly_matiere.exterior.xy
                ax.fill(x, y, alpha=alpha_val, fc=couleur, ec='black', linewidth=0.5, zorder=z_order)

            elif couleur == 'red' and poly_enveloppe.area > 50:
                x, y = poly_matiere.exterior.xy
                ax.fill(x, y, alpha=alpha_val, fc='red', ec='red', linewidth=1, zorder=z_order)
                cx, cy = poly_enveloppe.centroid.x, poly_enveloppe.centroid.y
                ax.text(cx, cy, label_debug, fontsize=7, color='darkred', weight='bold', ha='center')

        except Exception as e:
            continue