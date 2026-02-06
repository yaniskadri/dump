import fitz  # PyMuPDF
import json
import os
import warnings
import math
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import polygonize, unary_union

# Ignore minor geometric warnings from Shapely
warnings.filterwarnings("ignore")

class VectorAnalyzer:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        
        # --- CONFIGURATION (Based on your logic) ---
        self.THRESHOLD_THIN_WIRE = 5.0
        self.THRESHOLD_BUSBAR = 40.0
        self.MIN_AREA = 100  # Minimum area to consider valid

    def analyze_page(self, page_index=0):
        """
        Extracts vector shapes and classifies them without rendering images.
        """
        page = self.doc[page_index]
        paths = page.get_drawings()
        raw_lines = []

        # 1. Vector Extraction
        for path in paths:
            for item in path["items"]:
                if item[0] == "l":  # Line
                    raw_lines.append(LineString([item[1], item[2]]))
                elif item[0] == "c":  # Curve
                    raw_lines.append(LineString([item[1], item[-1]]))
                elif item[0] == "re":  # Rectangle
                    r = item[1]
                    # Convert rect to 4 lines
                    p1, p2, p3, p4 = (r[0], r[1]), (r[2], r[1]), (r[2], r[3]), (r[0], r[3])
                    raw_lines.extend([
                        LineString([p1, p2]), LineString([p2, p3]), 
                        LineString([p3, p4]), LineString([p4, p1])
                    ])

        # 2. Polygonization (Reconstructing topology)
        try:
            network = unary_union(raw_lines)
            polygons = list(polygonize(network))
        except Exception as e:
            print(f"Topology error on page {page_index}: {e}")
            return []

        detected_objects = []

        # 3. Classification Logic
        for i, poly in enumerate(polygons):
            # Geometric cleaning
            poly_clean = poly.buffer(0)
            
            if poly_clean.area < self.MIN_AREA:
                continue

            # Calculate Bounding Box and Rotated Rectangle
            box_rot = poly_clean.minimum_rotated_rectangle
            if box_rot.is_empty:
                continue
                
            # Calculate Thickness
            x, y = box_rot.exterior.coords.xy
            edge1 = Point(x[0], y[0]).distance(Point(x[1], y[1]))
            edge2 = Point(x[1], y[1]).distance(Point(x[2], y[2]))
            thickness = min(edge1, edge2)

            # Calculate Shape Ratios
            poly_env = Polygon(poly_clean.exterior).simplify(0.5)
            box_env = poly_env.minimum_rotated_rectangle
            
            ratio_rect = 0
            if box_env.area > 0:
                ratio_rect = poly_env.area / box_env.area

            perimeter = poly_env.length
            circularity = 0
            if perimeter > 0:
                circularity = (4 * math.pi * poly_env.area) / (perimeter ** 2)

            ratio_density = 1.0
            if poly_env.area > 0:
                ratio_density = poly_clean.area / poly_env.area

            # --- DECISION TREE (Your logic translated) ---
            category = "Unknown"
            confidence = 1.0

            # 1. Thin Wires (Ignore)
            if thickness < self.THRESHOLD_THIN_WIRE:
                continue 
            
            # 2. Circles
            elif circularity > 0.85:
                category = "Circle_Component"
            
            # 3. Hexagons / Symbols
            elif 0.70 <= ratio_rect <= 0.82 and circularity > 0.6:
                category = "Hex_Symbol"

            # 4. Rectangular Components
            elif ratio_rect > 0.70:
                if thickness < self.THRESHOLD_BUSBAR:
                    # Potential Busbar or Layout Line
                    if ratio_density > 0.5:
                        category = "Busbar_Power"
                    else:
                        continue # Layout line (empty)
                else:
                    # Large Component (> 40px)
                    if ratio_density > 0.80:
                        category = "Component_Rect"
                    elif ratio_density < 0.25:
                        continue # Empty layout frame
                    else:
                        category = "Group_Container" # Dotted lines / Containers

            # 5. Complex Shapes (L-shapes, etc.)
            elif ratio_density > 0.75:
                category = "Component_Complex"
            
            else:
                # Keep unknowns for manual sorting if they are big enough
                if poly_env.area > 200:
                    category = "Unknown_Shape"
                else:
                    continue

            # Store Data
            minx, miny, maxx, maxy = poly_env.bounds
            detected_objects.append({
                "id": i,
                "type": category,
                "bbox": [minx, miny, maxx, maxy], # PDF Points coordinates
                "thickness": thickness,
                "circularity": circularity
            })

        return detected_objects

    def export_to_json(self, output_path):
        data = {
            "source_file": os.path.basename(self.pdf_path),
            "pages": []
        }
        
        for i in range(len(self.doc)):
            print(f"Analyzing page {i+1}...")
            objects = self.analyze_page(i)
            data["pages"].append({
                "page_index": i,
                "objects": objects
            })
            
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Analysis saved to: {output_path}")

# Example Usage:
# analyzer = VectorAnalyzer("wiring_diagram.pdf")
# analyzer.export_to_json("analysis_data.json")