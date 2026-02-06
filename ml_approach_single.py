import fitz  # PyMuPDF
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

class ClusterDebugger:
    def __init__(self, pdf_path):
        self.doc = fitz.open(pdf_path)
        
        # --- TUNING PARAMETERS (Ajustez ici !) ---
        
        # 1. Distance de fusion (Pixels)
        # 15-20 est souvent bon. Si ça coupe trop, essayez 25.
        self.EPSILON = 15 
        
        # 2. Taille Max d'un trait (Pixels)
        # Tout trait plus long que ça est considéré comme un "Long Fil" et ignoré.
        self.MAX_ELEMENT_SIZE = 300 
        
        # 3. Minimum de traits pour faire un objet
        self.MIN_SAMPLES = 2 

    def get_raw_lines(self, page):
        """
        Extracts all small vector lines/curves/rects as individual segments.
        Returns a list of dicts: {'bbox': [x1,y1,x2,y2], 'center': [cx, cy]}
        """
        paths = page.get_drawings()
        elements = [] 

        for path in paths:
            for item in path["items"]:
                p1, p2 = None, None
                
                # Handle different vector types
                if item[0] == "l": # Line
                    p1, p2 = item[1], item[2]
                elif item[0] == "c": # Curve (Bezier) -> Approximated by start-end
                    p1, p2 = item[1], item[-1] 
                elif item[0] == "re": # Rectangle -> Decomposed into 4 lines
                    r = item[1]
                    pts = [(r[0], r[1]), (r[2], r[1]), (r[2], r[3]), (r[0], r[3])]
                    for i in range(4):
                        sp1 = pts[i]
                        sp2 = pts[(i+1)%4]
                        length = math.hypot(sp2[0]-sp1[0], sp2[1]-sp1[1])
                        # Keep small sides of rectangles
                        if length < self.MAX_ELEMENT_SIZE:
                            cx = (sp1[0] + sp2[0]) / 2
                            cy = (sp1[1] + sp2[1]) / 2
                            elements.append({
                                'bbox': [min(sp1[0], sp2[0]), min(sp1[1], sp2[1]), 
                                         max(sp1[0], sp2[0]), max(sp1[1], sp2[1])],
                                'center': [cx, cy]
                            })
                    continue 

                if p1 and p2:
                    # Logic: Only keep "short" lines (ink marks), discard long wires
                    length = math.hypot(p2.x-p1.x, p2.y-p1.y)
                    if length < self.MAX_ELEMENT_SIZE:
                        cx = (p1.x + p2.x) / 2
                        cy = (p1.y + p2.y) / 2
                        elements.append({
                            'bbox': [min(p1.x, p2.x), min(p1.y, p2.y), 
                                     max(p1.x, p2.x), max(p1.y, p2.y)],
                            'center': [cx, cy]
                        })
        return elements

    def visualize_clusters(self, page_index=0):
        print(f"Analyzing Page {page_index+1}...")
        page = self.doc[page_index]
        
        # 1. Get the "Soup of Lines"
        elements = self.get_raw_lines(page)
        if not elements:
            print("No elements found (page might be empty or only long wires).")
            return

        centers = np.array([e['center'] for e in elements])

        # 2. Run DBSCAN Clustering
        print(f"Clustering {len(centers)} elements...")
        clustering = DBSCAN(eps=self.EPSILON, min_samples=self.MIN_SAMPLES).fit(centers)
        labels = clustering.labels_
        
        # 3. Render Background Image (72 DPI for 1:1 match with coords)
        pix = page.get_pixmap(dpi=72)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        
        # 4. Plotting
        fig, ax = plt.subplots(figsize=(20, 14))
        ax.imshow(img_array)
        ax.set_title(f"DBSCAN Clustering (Eps={self.EPSILON}) - {len(set(labels)) - (1 if -1 in labels else 0)} Objects Detected")

        unique_labels = set(labels)
        colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels))) # Rainbow colors

        detected_count = 0
        
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Noise (Black points) -> These are isolated lines
                # Uncomment next line to see noise in black
                # ax.plot(centers[labels == k, 0], centers[labels == k, 1], 'k.', markersize=2, alpha=0.3)
                continue

            # Get all parts of this component
            class_member_mask = (labels == k)
            indices = np.where(class_member_mask)[0]
            cluster_bboxes = [elements[i]['bbox'] for i in indices]

            # Calculate Bounding Box of the Cluster
            x1 = min([b[0] for b in cluster_bboxes])
            y1 = min([b[1] for b in cluster_bboxes])
            x2 = max([b[2] for b in cluster_bboxes])
            y2 = max([b[3] for b in cluster_bboxes])
            
            w = x2 - x1
            h = y2 - y1
            
            # Draw Rectangle
            rect = mpatches.Rectangle((x1, y1), w, h,
                                      fill=False, edgecolor=col, linewidth=2)
            ax.add_patch(rect)
            
            # Optional: Add Label ID to debug specific clusters
            # ax.text(x1, y1-2, str(k), fontsize=8, color=col)
            
            detected_count += 1

        print(f"Visualization ready. Found {detected_count} potential components.")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# --- USAGE ---
# Remplacez par votre fichier
FILE_PATH = "Chemin/Vers/Votre/Diagramme.pdf"

# Lancez le debugger
debugger = ClusterDebugger(FILE_PATH)
debugger.visualize_clusters(page_index=0)