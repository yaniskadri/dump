import fitz  # PyMuPDF
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
import math

class GraphComponentExtractor:
    def __init__(self, pdf_path):
        self.doc = fitz.open(pdf_path)
        
        # --- CONFIGURATION ---
        self.MIN_AREA = 50           # Ignore le bruit / texte minuscule
        self.MAX_AREA = 100000       # Ignore le cadre de la page
        
        # Le seuil de tolérance pour considérer un cycle comme "Grillage de fils"
        # Si plus de 50% des coins du cycle sont des intersections en X (degré 4), on jette.
        self.WIRE_CROSSING_RATIO = 0.5 

    def _build_graph(self, page):
        """
        Convertit les vecteurs de la page en un graphe non-dirigé NetworkX.
        Les noeuds sont les coordonnées (x, y) arrondies pour fusionner les points proches.
        """
        G = nx.Graph()
        paths = page.get_drawings()
        
        for path in paths:
            for item in path["items"]:
                # On arrondit à 1 décimale pour que les segments qui se touchent 
                # partagent mathématiquement le même noeud.
                if item[0] == "l": # Ligne
                    p1 = (round(item[1].x, 1), round(item[1].y, 1))
                    p2 = (round(item[2].x, 1), round(item[2].y, 1))
                    G.add_edge(p1, p2)
                
                elif item[0] == "re": # Rectangle (converti en 4 lignes)
                    r = item[1]
                    pts = [
                        (round(r[0], 1), round(r[1], 1)), 
                        (round(r[2], 1), round(r[1], 1)), 
                        (round(r[2], 1), round(r[3], 1)), 
                        (round(r[0], 1), round(r[3], 1))
                    ]
                    for i in range(4):
                        G.add_edge(pts[i], pts[(i+1)%4])
                        
        # Nettoyage : On supprime les petits traits isolés (bruit) qui ne forment pas de boucles
        G = nx.k_core(G, k=2)
        return G

    def analyze_page(self, page_index=0):
        page = self.doc[page_index]
        G = self._build_graph(page)
        
        print(f"--> Graphe construit : {G.number_of_nodes()} noeuds, {G.number_of_edges()} arêtes.")
        
        try:
            # Trouve toutes les boucles fermées minimales
            cycle_basis = nx.minimum_cycle_basis(G)
        except Exception as e:
            print(f"Erreur extraction cycles: {e}")
            return [], []

        kept_components = []
        rejected_wires = []

        print(f"--> Analyse de {len(cycle_basis)} cycles candidats...")

        for cycle_nodes in cycle_basis:
            if len(cycle_nodes) < 3: continue
            
            # 1. Création Géométrie (pour l'aire)
            poly = Polygon(cycle_nodes)
            if not poly.is_valid: continue
            
            # Filtre de taille
            if poly.area < self.MIN_AREA or poly.area > self.MAX_AREA: 
                continue # Trop petit ou trop gros pour être un composant

            # 2. FILTRE TOPOLOGIQUE (L vs X)
            # On compte combien de coins du cycle sont des "Intersections Complexes"
            high_degree_nodes = 0
            for node in cycle_nodes:
                # Le degré dans le graphe G complet (pas juste le cycle)
                degree = G.degree[node]
                
                # Degré 2 = Coin propre (L-shape) -> Typique d'un composant
                # Degré 3 = T-junction
                # Degré 4 = Croisement (X-shape) -> Typique d'un maillage de fils
                if degree >= 4:
                    high_degree_nodes += 1
            
            # Ratio de "complexité" des coins
            crossing_ratio = high_degree_nodes / len(cycle_nodes)

            if crossing_ratio > self.WIRE_CROSSING_RATIO:
                # C'est un artefact de fils croisés
                rejected_wires.append(poly)
            else:
                # C'est un vrai composant (majorité de coins propres)
                kept_components.append(poly)

        # 3. Fusion des Composants Adjacents (Grilles de connecteurs)
        # Si on a gardé plusieurs cases d'un connecteur, on veut 1 seul objet.
        final_components = []
        if kept_components:
            merged = unary_union(kept_components)
            if merged.geom_type == 'Polygon':
                final_components.append(merged)
            elif merged.geom_type == 'MultiPolygon':
                final_components.extend(list(merged.geoms))

        return final_components, rejected_wires

    def visualize(self, page_index=0):
        """
        Génère une visualisation Debug :
        - Vert : Composants détectés
        - Rouge (Hachuré) : Faux positifs rejetés (Fils croisés)
        """
        print(f"Visualisation Page {page_index+1}...")
        components, rejected = self.analyze_page(page_index)
        
        page = self.doc[page_index]
        
        # Rendu Haute Définition pour le fond
        pix = page.get_pixmap(dpi=200)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        
        fig, ax = plt.subplots(figsize=(20, 15))
        
        # Mapping Image (Pixels) -> Coordonnées (Points PDF)
        ax.imshow(img_array, extent=[0, page.rect.width, page.rect.height, 0])
        
        # 1. Dessiner les REJETÉS (Rouge)
        for poly in rejected:
            x, y = poly.exterior.xy
            # Hachures rouges pour montrer que c'est "interdit"
            ax.fill(x, y, facecolor='none', hatch='//', edgecolor='red', linewidth=1, alpha=0.5)
            
        # 2. Dessiner les GARDÉS (Vert)
        for poly in components:
            x, y = poly.exterior.xy
            # Fond vert semi-transparent + Bordure solide
            ax.fill(x, y, color='#00FF00', alpha=0.4) 
            ax.plot(x, y, color='darkgreen', linewidth=2)
            
            # Bounding Box (pour vérifier le crop futur)
            minx, miny, maxx, maxy = poly.bounds
            rect = mpatches.Rectangle((minx, miny), maxx-minx, maxy-miny, 
                                      fill=False, edgecolor='blue', linewidth=1, linestyle='--')
            ax.add_patch(rect)

        # Légende
        patch_kept = mpatches.Patch(color='#00FF00', label=f'Composants ({len(components)})')
        patch_rejected = mpatches.Patch(facecolor='none', hatch='//', edgecolor='red', label=f'Fils Croisés Rejetés ({len(rejected)})')
        ax.legend(handles=[patch_kept, patch_rejected], loc='upper right', fontsize=12)

        ax.set_title(f"Graph Analysis Results - Page {page_index+1}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# --- EXEMPLE D'UTILISATION ---
# Remplacez par votre fichier
FILE = "Votre_Wiring_Diagram.pdf"

extractor = GraphComponentExtractor(FILE)
extractor.visualize(0) # Visualise la page 1