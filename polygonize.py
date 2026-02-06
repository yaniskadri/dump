# ... (après les calculs de perimetre, aire, box) ...

            # 1. Ratio Rectangle (Géométrie extérieure)
            box = poly_enveloppe.minimum_rotated_rectangle
            ratio_rect = 0
            if box.area > 0:
                ratio_rect = poly_enveloppe.area / box.area

            # 2. Ratio Densité (Remplissage matière)
            ratio_densite = 1.0
            if poly_enveloppe.area > 0:
                ratio_densite = poly_matiere.area / poly_enveloppe.area
            
            # --- CLASSIFICATION ---
            is_valid = False
            label = ""
            label_debug = f"G:{ratio_rect:.2f}\nD:{ratio_densite:.2f}" # Pour le debug
            z_order = 2
            
            # A. C'est GEOMETRIQUEMENT un Rectangle (ou presque)
            # On baisse à 0.80 pour accepter vos "crayons" (rectangles avec pointes)
            if ratio_rect > 0.80:
                is_valid = True
                
                # Maintenant, on trie selon ce qu'il y a DEDANS (Densité)
                
                if ratio_densite > 0.85:
                    # CAS 1 : C'est PLEIN -> OBJET (Vert)
                    couleur = 'green'
                    label = "Objet"
                    alpha_val = 0.5
                    z_order = 3
                    
                elif ratio_densite < 0.20:
                    # CAS 2 : C'est VIDE -> ZONE/LAYOUT (Gris invisible)
                    couleur = 'gray'
                    label = "Layout"
                    alpha_val = 0.05
                    z_order = 0
                    
                else:
                    # CAS 3 (Le Rénégat) : C'est MOITIÉ-MOITIÉ -> GROUPE (Bleu pointillé)
                    # C'est un conteneur qui englobe d'autres objets
                    couleur = 'blue'
                    label = "Groupe"
                    alpha_val = 0.1 # Fond très léger
                    z_order = 1 # Derrière les objets verts
            
            # B. C'est un CERCLE (Circularité > 0.85)
            elif circularity > 0.85:
                couleur = 'magenta'
                label = "Cercle"
                is_valid = True
                alpha_val = 0.5

            # C. C'est un HEXAGONE (Ratio rect ~0.75)
            elif 0.70 < ratio_rect < 0.80:
                couleur = 'orange'
                label = "Hexagone"
                is_valid = True
                alpha_val = 0.5
            
            # D. POUBELLE (Vraies erreurs)
            else:
                couleur = 'red'
                label = "Inconnu"
                alpha_val = 0.2

            # --- DESSIN ---
            if label == "Groupe":
                # Dessin spécifique pour le rénégat : Cadre bleu pointillé
                x, y = poly_enveloppe.exterior.xy
                ax.plot(x, y, color='blue', linewidth=1, linestyle='--', zorder=z_order)
                ax.fill(x, y, alpha=alpha_val, fc='blue', zorder=z_order)
                # On affiche le debug pour comprendre
                cx, cy = poly_enveloppe.centroid.x, poly_enveloppe.centroid.y
                ax.text(cx, cy, label_debug, fontsize=7, color='blue', ha='center')
                
            elif is_valid and label != "Layout":
                # Objets valides (Verts, Oranges...)
                x, y = poly_matiere.exterior.xy
                ax.fill(x, y, alpha=alpha_val, fc=couleur, ec='black', linewidth=0.5, zorder=z_order)
            
            elif couleur == 'red':
                # Erreurs : On affiche en rouge AVEC LE DEBUG
                x, y = poly_matiere.exterior.xy
                ax.fill(x, y, alpha=alpha_val, fc='red', ec='red', linewidth=1, zorder=z_order)
                cx, cy = poly_enveloppe.centroid.x, poly_enveloppe.centroid.y
                ax.text(cx, cy, label_debug, fontsize=7, color='darkred', weight='bold', ha='center')