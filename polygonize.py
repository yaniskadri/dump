import math

# ... (dans la boucle) ...

# --- CALCULS AVANCÉS ---
perimetre = poly_enveloppe.length
aire = poly_enveloppe.area

# 1. Ratio Rectangle (Box Coverage)
box = poly_enveloppe.minimum_rotated_rectangle
ratio_rect = 0
if box.area > 0:
    ratio_rect = poly_enveloppe.area / box.area

# 2. Ratio Circularité (Isoperimetric Quotient)
# Cercle parfait = 1.0, Carré = 0.78, Triangle = 0.6
circularity = 0
if perimetre > 0:
    circularity = (4 * math.pi * aire) / (perimetre ** 2)

# 3. Densité (Plein vs Vide)
ratio_densite = 1.0
if poly_enveloppe.area > 0:
    ratio_densite = poly_matiere.area / poly_enveloppe.area

# --- ARBRE DE DÉCISION ---
is_valid = False
label = ""
z_order = 2

# A. C'est un ROND (Cercle parfait ou presque)
if circularity > 0.88:
    couleur = 'magenta' # Couleur distincte pour les cercles
    label = "Cercle"
    is_valid = True
    alpha_val = 0.5

# B. C'est un RECTANGLE (ou rect arrondi)
elif ratio_rect > 0.85:
    is_valid = True
    
    if ratio_densite > 0.85:
        couleur = 'green' # Objet Plein
        label = "Rect"
        alpha_val = 0.4
        z_order = 3
    elif ratio_densite < 0.30:
        # C'est votre RÉNÉGAT ! (Cadre vide ou Zone)
        couleur = 'gray' 
        label = "Zone/Layout"
        alpha_val = 0.05 # Quasi invisible
        z_order = 0 # Tout au fond
    else:
        # C'est un conteneur (ex: cadre avec du texte)
        couleur = 'blue'
        label = "Conteneur"
        alpha_val = 0.1
        z_order = 1

# C. C'est un HEXAGONE (ou Octogone)
elif 0.70 < ratio_rect < 0.82:
    couleur = 'orange'
    label = "Hexagone"
    is_valid = True
    alpha_val = 0.5

# D. C'est un TRIANGLE
elif 0.40 < ratio_rect < 0.60:
    couleur = 'cyan'
    label = "Triangle"
    is_valid = True
    alpha_val = 0.5

# E. POUBELLE (Formes bizarres)
else:
    couleur = 'red'
    label = f"Rect:{ratio_rect:.2f} Circ:{circularity:.2f}"
    alpha_val = 0.2
    
# ... (suite code affichage) ...