import fitz  # PyMuPDF

def analyser_vecteurs(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]  # On analyse la première page
    
    # get_drawings() extrait tous les vecteurs
    chemins = page.get_drawings()
    
    print(f"Analyse de {len(chemins)} objets vectoriels :\n")
    
    for i, chemin in enumerate(chemins):
        items = chemin["items"] # Les segments (lignes, courbes)
        type_objet = "Inconnu"
        details = ""

        # Cas 1 : C'est un rectangle natif (instruction 're')
        # PyMuPDF détecte souvent les rectangles explicitement
        if chemin["rect"] and len(items) == 1 and items[0][0] == "re":
            type_objet = "Rectangle (Objet natif)"
        
        # Cas 2 : C'est un chemin composé de lignes
        else:
            nb_segments = len(items)
            # Vérifier si c'est fermé (le dernier point touche le premier)
            # Note: simplifiée pour l'exemple
            est_ferme = chemin["closePath"]
            
            if nb_segments == 3 and est_ferme:
                type_objet = "Triangle (Chemin fermé)"
            elif nb_segments == 4 and est_ferme:
                type_objet = "Quadrilatère/Rectangle (Chemin fermé)"
            elif nb_segments == 1:
                type_objet = "Ligne simple"
            else:
                type_objet = f"Polygone ou courbe complexe ({nb_segments} segments)"

        # Détection de "Groupes" (Approximation)
        # Dans les PDF, les groupes visuels n'existent pas toujours techniquement.
        # On peut parfois les deviner si des objets partagent la même couleur/épaisseur
        fill = chemin.get("fill")
        stroke = chemin.get("color")
        
        print(f"Objet {i+1}: {type_objet}")
        print(f"   - Segments: {len(items)}")
        print(f"   - Remplissage: {fill}, Contour: {stroke}")
        print("-" * 30)

analyser_vecteurs("mon_fichier.pdf")