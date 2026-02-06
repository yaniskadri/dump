import fitz
import networkx as nx
from shapely.geometry import Polygon, LineString, box
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class SmartCircuitExtractor:
    def __init__(self, pdf_path):
        self.doc = fitz.open(pdf_path)
        self.MIN_AREA = 50
        self.MAX_AREA = 50000

    def get_cycles_with_context(self, page_index=0):
        page = self.doc[page_index]
        paths = page.get_drawings()
        text_blocks = page.get_text("blocks") # (x0, y0, x1, y1, "text", ...)
        
        # 1. Construction du Graphe (Identique V4)
        G = nx.Graph()
        all_lines = [] # Pour détecter le contenu orphelin plus tard
        
        for path in paths:
            for item in path["items"]:
                if item[0] == "l":
                    p1, p2 = (round(item[1].x, 1), round(item[1].y, 1)), (round(item[2].x, 1), round(item[2].y, 1))
                    G.add_edge(p1, p2)
                    all_lines.append(LineString([item[1], item[2]]))
                elif item[0] == "re":
                    r = item[1]
                    pts = [(round(r[0], 1), round(r[1], 1)), (round(r[2], 1), round(r[1], 1)), 
                           (round(r[2], 1), round(r[3], 1)), (round(r[0], 1), round(r[3], 1))]
                    for i in range(4):
                        G.add_edge(pts[i], pts[(i+1)%4])
                        all_lines.append(LineString([pts[i], pts[(i+1)%4]]))

        # 2. Extraction des Cycles (Candidats)
        try:
            cycles = nx.minimum_cycle_basis(G)
        except Exception:
            return []

        # 2b. Filtrage topologique par degré des nœuds (Node Degree Filter)
        # Un croisement de fils (#) crée un faux cycle dont TOUS les coins sont degré ≥ 4.
        # Un vrai composant a des coins "propres" (degré 2).
        filtered_cycles = []
        for cycle_nodes in cycles:
            if len(cycle_nodes) < 3:
                continue
            
            cross_junctions = 0
            for node in cycle_nodes:
                degree = G.degree[node]
                if degree >= 4:  # Nœud = croisement (X ou +)
                    cross_junctions += 1
            
            # Si plus de 50% des coins sont des croisements -> artefact de grillage
            if cross_junctions > (len(cycle_nodes) * 0.5):
                continue
            
            filtered_cycles.append(cycle_nodes)

        # 2c. Conversion en Polygones et filtrage par aire
        candidates = []
        for cycle in filtered_cycles:
            poly = Polygon(cycle)
            if not poly.is_valid or poly.area < self.MIN_AREA or poly.area > self.MAX_AREA:
                continue
            candidates.append(poly)

        # ====================================================
        # 3. LE FILTRE "VIDE & SOLITAIRE" (Le Cœur du Fix)
        # ====================================================
        valid_components = []
        
        # Convertir tout le texte en Polygons pour check rapide
        text_polys = [box(b[0], b[1], b[2], b[3]) for b in text_blocks]
        
        for i, poly in enumerate(candidates):
            # A. Test de Contenu (Texte)
            has_text = False
            for t_poly in text_polys:
                if poly.intersects(t_poly): # Si du texte touche ou est dedans
                    has_text = True
                    break
            
            # B. Test de Voisinage (Adjacence)
            # Un connecteur est une grille, donc les cases se touchent.
            # Un croisement de fils est souvent isolé géométriquement des autres boucles.
            neighbors = 0
            for j, other_poly in enumerate(candidates):
                if i == j: continue
                # Si l'intersection est une ligne (touche par le bord), c'est un voisin
                if poly.touches(other_poly): 
                    neighbors += 1

            # C. DÉCISION
            is_connector_part = (neighbors >= 1) # Fait partie d'une grille
            is_filled = has_text # A du texte (On pourrait ajouter check de symboles internes)

            if is_filled or is_connector_part:
                # C'est un BON candidat (Relais rempli OU Connecteur grille)
                valid_components.append(poly)
            else:
                # C'est VIDE et SOLITAIRE -> C'est un croisement de fils (#)
                # On le jette !
                pass 

        # 4. Fusion finale
        final_geom = unary_union(valid_components)
        final_list = []
        if final_geom.geom_type == 'Polygon': final_list.append(final_geom)
        elif final_geom.geom_type == 'MultiPolygon': final_list.extend(list(final_geom.geoms))
        
        return final_list

    def visualize(self, page_index=0):
        # ... (Code de visualisation identique à V4) ...
        # Utilisez le code de viz précédent mais appelez self.get_cycles_with_context(0)
        components = self.get_cycles_with_context(page_index)
        
        # Setup Plot simple
        page = self.doc[page_index]
        fig, ax = plt.subplots(figsize=(20, 14))
        ax.set_xlim(0, page.rect.width)
        ax.set_ylim(page.rect.height, 0)
        ax.set_aspect('equal')
        
        # Dessin
        for poly in components:
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.5, fc='#00FF00', ec='black', linewidth=1)
            
        plt.title(f"Smart Extraction - {len(components)} Objets (Fils croisés filtrés)")
        plt.show()

# --- EXECUTION ---
extractor = SmartCircuitExtractor("VotreFichier.pdf")
extractor.visualize(0)