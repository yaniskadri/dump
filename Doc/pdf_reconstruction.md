# PDF Vector Reconstruction & Analysis Module (AI-Generated)

## 1. Overview
This module provides an advanced algorithm to parse, reconstruct, and classify vector objects within technical PDF drawings (specifically electrical diagrams). 

Unlike standard PDF text extractors, this tool works on the **geometric layer**. It converts a "soup" of unstructured lines and curves into semantic objects such as **Wires**, **Busbars**, **Components**, and **Logical Groups**.

## 2. Key Features
* **Geometric Reconstruction:** Reassembles disconnected line segments ($l$, $c$, $re$) into closed polygons using graph topology.
* **Topological Repair:** Automatically fixes "bow-tie" geometries and shapes touching at single points ($buffer(0)$ fix).
* **Advanced Classification:** Distinguishes objects based on:
    * **Thickness:** Control wires vs. Power busbars vs. Components.
    * **Geometry:** Rectangularity, Circularity, and Slenderness.
    * **Density:** Solid objects vs. Containers (Groups) vs. Layout zones.
* **Complex Shape Support:** Detects L-shapes, Triangles, and "Donut" shapes (shapes with holes).
* **Visual Debugging:** Generates a semantic overlay with color-coded categories and internal ratio metrics (Geometry/Density) for troubleshooting.

## 3. Dependencies
The module requires the following Python libraries:

$$$bash
pip install pymupdf shapely matplotlib
$$$

* **PyMuPDF (fitz):** Fast extraction of raw vector paths.
* **Shapely:** Computational geometry (unions, polygonization, simplification, measurements).
* **Matplotlib:** Visualization and debug plotting.

## 4. How It Works (The Algorithm)

### Phase 1: Extraction & Nodalization
The script iterates through the PDF paths. All primitives (Lines, Curves, Rectangles) are converted into $Shapely.LineString$ objects. Curves are linearized (approximated by their chord) to simplify topological analysis.

### Phase 2: Polygonization
Using $unary_union$, the script "welds" all intersecting lines together. It then applies $polygonize$ to find all closed cycles in the graph, converting wires and contours into $Shapely.Polygon$ objects.

### Phase 3: Geometry Cleaning
Before analysis, every polygon undergoes:
1.  **Topological Fix:** $poly.buffer(0)$ resolves self-intersections.
2.  **Smoothing:** $simplify(0.5)$ removes micro-segments and noise.

### Phase 4: Metric Calculation
For every object, three key metrics are calculated:
1.  **Thickness:** The minimum dimension of the Oriented Bounding Box.
2.  **G-Ratio (Geometry):** How much the object fills its bounding box (Area_poly / Area_box).
    * 1.0 = Perfect Rectangle.
    * ~0.78 = Circle/Square.
    * ~0.5 = Triangle or L-Shape.
3.  **D-Ratio (Density):** The ratio of "Matter" vs "Envelope" (Area_matter / Area_filled_envelope).
    * 1.0 = Solid object.
    * < 0.2 = Empty Frame / Layout.
    * ~0.5 = Group / Container.

### Phase 5: The Decision Tree (Classification)
Objects are categorized based on a strict hierarchy:
1.  **Wires:** Thickness < $SEUIL_FIL_FIN$ (Yellow).
2.  **Busbars:** Thickness < $SEUIL_BUSBAR$ AND Rectangular (Light Green).
3.  **Components:** Large Rectangles, Circles, Hexagons (Dark Green, Magenta, Orange).
4.  **Complex Shapes:** Objects with low G-Ratio but high D-Ratio (e.g., L-Shapes) (Cyan).
5.  **Groups:** Rectangular envelopes with low internal density (Blue Dashed).

## 5. Configuration (Thresholds)

You can tune these constants at the top of the script to adapt to different diagram styles.

| Constant | Value (Default) | Description |
| :--- | :--- | :--- |
| $SEUIL_FIL_FIN$ | **5.0 px** | Objects thinner than this are **Control Wires** (Yellow). |
| $SEUIL_BUSBAR$ | **40.0 px** | Objects between 5px and 40px are **Power Busbars** (Light Green). Above 40px are Components. |
| $SEUIL_RECT_RATIO$ | **0.70** | Minimum G-Ratio to be considered "Rectangular". Lower this if your rectangles are distorted. |
| $SEUIL_DENSITY_SOLID$ | **0.80** | Minimum D-Ratio to be considered a "Solid Object". |
| $SEUIL_DENSITY_EMPTY$ | **0.25** | Objects below this density are considered **Layout/Empty Frames**. |

## 6. Output Legend & Interpretation

The visual output uses a semantic color code:

| Color | Label | Meaning |
| :--- | :--- | :--- |
| 🟨 **Yellow** | **Wire (Cmd)** | Thin control connections (< 5px). |
| 🟩 **Light Green** | **Busbar** | Power cables or busbars (rectangular mesh). |
| 🌲 **Dark Green** | **Component** | Solid rectangular devices (PLCs, Drives, etc.). |
| 🟦 **Cyan** | **Complex** | "Renegade" shapes: Triangles, L-shapes, or cropped rectangles (High density, irregular geometry). |
| 🟧 **Orange** | **Symbol** | Hexagons (often used for off-page connectors). |
| 🟪 **Magenta** | **Circle** | Motors, lights, or circular terminals. |
| 🔵 **Blue (Dashed)** | **Group** | Logical containers grouping other components together. |
| ⬜ **Gray** | **Layout** | Empty frames or viewports (ignored). |
| 🟥 **Red** | **Unknown** | Error in classification. Check the **G** and **D** values printed on the object to debug. |

## 7. Troubleshooting

### Problem: "My components are detected as Busbars"
* **Cause:** Your components are too thin or your $SEUIL_BUSBAR$ is too high.
* **Fix:** Lower $SEUIL_BUSBAR$ (e.g., from 40.0 to 20.0).

### Problem: "A solid rectangle is Red (Unknown)"
* **Cause:** It might have a small protrusion or a slightly "cut" corner, dropping its G-Ratio below 0.70.
* **Fix:** Read the $G:0.xx$ value on the image. Lower $ratio_rect > 0.70$ to $0.65$ in the code.

### Problem: "Wires are detected as Rectangles (Green)"
* **Cause:** The wires are drawn as rectangles in the PDF and are thicker than $SEUIL_FIL_FIN$.
* **Fix:** Increase $SEUIL_FIL_FIN$ (e.g., to 8.0 or 10.0).

### Problem: "The diagram is a mess of Red shapes"
* **Cause:** The PDF vector quality is poor (lines not touching).
* **Fix:** Increase the dilation in the cleaning phase: change $poly.buffer(0)$ to $poly.buffer(0.5)$ to force connections.