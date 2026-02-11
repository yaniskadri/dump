import fitz  # PyMuPDF
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

# ==========================================
# 1. GEOMETRY & HELPER FUNCTIONS (FIXED)
# ==========================================

def get_bounding_box(positions):
    """Calculate min/max x and y from a list of positions."""
    if len(positions) == 0:
        return (0, 0, 0, 0)
    positions = np.array(positions)
    return (np.min(positions[:,0]), np.min(positions[:,1]), 
            np.max(positions[:,0]), np.max(positions[:,1]))

def is_rectangle(positions, tolerance=0.1):
    """Check if 4 points form a rectangle (simplified: axis aligned check)."""
    # A robust check would verify dot products of vectors. 
    # For PDF diagrams, axis-aligned is common.
    bbox = get_bounding_box(positions)
    min_x, min_y, max_x, max_y = bbox
    
    # Check if all points lie close to the bbox borders
    matches = 0
    for p in positions:
        on_vert = abs(p[0] - min_x) < tolerance or abs(p[0] - max_x) < tolerance
        on_horiz = abs(p[1] - min_y) < tolerance or abs(p[1] - max_y) < tolerance
        if on_vert and on_horiz:
            matches += 1
    return matches >= 3 # Allow for slight imperfection

def has_parallel_lines(orientations, positions):
    """Check for parallel line pattern (common in capacitors)."""
    if len(orientations) != 2: return False
    # Check if angles are close (modulo pi)
    diff = abs(orientations[0] - orientations[1]) % np.pi
    return diff < 0.1 or abs(diff - np.pi) < 0.1

def has_cross_pattern(subgraph):
    """Check for intersecting lines forming a plus shape."""
    # Simplified: A node with degree 4 often implies a cross/junction
    degrees = [d for n, d in subgraph.degree()]
    return 4 in degrees

def has_arrow_pattern(subgraph):
    """Placeholder for triangle/arrow detection."""
    # This requires complex shape matching. 
    # Returning False safely to prevent crashes.
    return False

def get_bbox_center(bbox):
    return ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)

def associate_labels(components, text_blocks):
    """Associate text labels with the nearest component."""
    # text_blocks structure from PyMuPDF dict: "blocks" -> "lines" -> "spans"
    labels = []
    
    # Flatten text structure
    for block in text_blocks.get('blocks', []):
        if 'lines' not in block: continue
        for line in block['lines']:
            for span in line['spans']:
                text_bbox = span['bbox'] # (x0, y0, x1, y1)
                text_center = get_bbox_center(text_bbox)
                labels.append({'text': span['text'], 'center': text_center, 'bbox': text_bbox})

    # Assign label to nearest component
    for comp in components:
        comp_center = get_bbox_center(comp['bbox'])
        best_label = None
        min_dist = float('inf')
        
        for label in labels:
            dist = np.linalg.norm(np.array(comp_center) - np.array(label['center']))
            if dist < 50: # Threshold distance
                if dist < min_dist:
                    min_dist = dist
                    best_label = label['text']
        
        comp['label'] = best_label if best_label else "?"
        
    return components

def find_bbox_intersection(wire_segment, comp_bbox):
    """Find where a line segment intersects a box."""
    # Simplified: return the end of the line segment that is closest to the box center
    p1 = wire_segment['start']
    p2 = wire_segment['end']
    center = np.array(get_bbox_center(comp_bbox))
    
    d1 = np.linalg.norm(p1 - center)
    d2 = np.linalg.norm(p2 - center)
    
    return p1 if d1 < d2 else p2

def get_bbox_side(point, bbox):
    """Determine which side of the bbox a point is on."""
    x, y = point
    x0, y0, x1, y1 = bbox
    
    # Calculate distance to each side
    d_left = abs(x - x0)
    d_right = abs(x - x1)
    d_top = abs(y - y0)
    d_bottom = abs(y - y1)
    
    m = min(d_left, d_right, d_top, d_bottom)
    if m == d_left: return "left"
    if m == d_right: return "right"
    if m == d_top: return "top"
    return "bottom"

# ==========================================
# 2. VISUALIZATION FUNCTIONS
# ==========================================

def viz_step_1_raw_lines(lines):
    plt.figure(figsize=(10, 8))
    plt.title(f"Step 1: Raw Extracted Lines ({len(lines)} segments)")
    for line in lines:
        p1, p2 = line['start'], line['end']
        color = line.get('color', (0,0,0))
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=1, alpha=0.7)
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.show()

def viz_step_2_graph(G):
    plt.figure(figsize=(10, 8))
    plt.title(f"Step 2: Connectivity Graph ({G.number_of_nodes()} nodes)")
    pos = {n: G.nodes[n]['pos'] for n in G.nodes()}
    nx.draw_networkx_nodes(G, pos, node_size=10, node_color='blue')
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.show()

def viz_step_3_classification(classified):
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.title("Step 3: Component vs Wire Classification")
    
    # Draw Wires (Green)
    for wire in classified['wires']:
        if 'path' in wire:
            pts = np.array(wire['path'])
            ax.plot(pts[:,0], pts[:,1], color='green', linewidth=2, label='Wire')

    # Draw Components (Red Boxes)
    for comp in classified['components']:
        x0, y0, x1, y1 = comp['bbox']
        w, h = x1 - x0, y1 - y0
        rect = patches.Rectangle((x0, y0), w, h, linewidth=2, edgecolor='red', facecolor='none', label='Component')
        ax.add_patch(rect)
        ax.text(x0, y0-5, comp.get('shape_type', 'unk'), color='red', fontsize=8)

    # De-duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.show()

def viz_step_final_circuit(circuit_graph, components):
    plt.figure(figsize=(10, 8))
    plt.title("Step 7: Final Logical Circuit Graph")
    
    # Use spring layout but seed with actual positions if available
    pos_dict = {}
    labels = {}
    for i, comp in enumerate(components):
        node_id = f"C{i}"
        center = get_bbox_center(comp['bbox'])
        pos_dict[node_id] = np.array(center)
        labels[node_id] = f"{comp.get('label','?')}\n({comp['shape_type']})"
        
    # Draw logic graph
    pos = nx.spring_layout(circuit_graph, pos=pos_dict, fixed=pos_dict.keys(), k=0.5)
    
    nx.draw(circuit_graph, pos, with_labels=True, labels=labels, 
            node_size=2000, node_color='lightblue', font_size=8)
    plt.show()

# ==========================================
# 3. CORE PROCESSING LOGIC
# ==========================================

def extract_all_lines(pdf_path):
    """Extract every line segment from the PDF"""
    try:
        doc = fitz.open(pdf_path)
        page = doc[0]
        drawings = page.get_drawings()
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return []
    
    lines = []
    
    for drawing in drawings:
        for item in drawing['items']:
            if item[0] == 'l':  # straight line
                p1, p2 = item[1], item[2]
                line = {
                    'start': np.array(p1),
                    'end': np.array(p2),
                    'color': drawing.get('color', (0, 0, 0)),
                    'width': drawing.get('width', 1.0),
                    'length': np.linalg.norm(np.array(p2) - np.array(p1))
                }
                lines.append(line)
            
            elif item[0] == 'c':  # curve - sample it into line segments
                points = item[1]
                sampled = sample_bezier_curve(points, num_samples=10)
                for i in range(len(sampled)-1):
                    line = {
                        'start': sampled[i],
                        'end': sampled[i+1],
                        'color': drawing.get('color', (0, 0, 0)),
                        'width': drawing.get('width', 1.0),
                        'is_curve': True,
                        'length': np.linalg.norm(sampled[i+1] - sampled[i])
                    }
                    lines.append(line)
    return lines

def sample_bezier_curve(control_points, num_samples=10):
    """Robust bezier sampler handling 3 or 4 points."""
    t_values = np.linspace(0, 1, num_samples)
    points = []
    cps = [np.array(p) for p in control_points]
    
    if len(cps) == 4:
        # Cubic Bezier
        p0, p1, p2, p3 = cps
        for t in t_values:
            point = ((1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3)
            points.append(point)
    elif len(cps) == 3:
        # Quadratic Bezier
        p0, p1, p2 = cps
        for t in t_values:
            point = ((1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2)
            points.append(point)
    else:
        # Fallback: Just return start and end if malformed
        if len(cps) >= 2:
            points = [cps[0], cps[-1]]
        elif len(cps) == 1:
            points = [cps[0], cps[0]]
            
    return points

def build_line_graph(lines, tolerance=2.0):
    G = nx.Graph()
    endpoint_to_node = {}
    node_counter = 0
    
    for i, line in enumerate(lines):
        # Rounding helps snap close points together
        start_key = tuple(np.round(line['start'], 1))
        end_key = tuple(np.round(line['end'], 1))
        
        if start_key not in endpoint_to_node:
            endpoint_to_node[start_key] = node_counter
            G.add_node(node_counter, pos=line['start'])
            node_counter += 1
        
        if end_key not in endpoint_to_node:
            endpoint_to_node[end_key] = node_counter
            G.add_node(node_counter, pos=line['end'])
            node_counter += 1
        
        start_node = endpoint_to_node[start_key]
        end_node = endpoint_to_node[end_key]
        
        G.add_edge(start_node, end_node, 
                   line_id=i,
                   length=line['length'])
    
    return G, endpoint_to_node

def classify_line_groups(G, lines):
    # Find all connected components in the graph
    connected_components = list(nx.connected_components(G))
    
    classified = {'components': [], 'wires': [], 'junctions': []}
    
    for cc in connected_components:
        subgraph = G.subgraph(cc)
        analysis = analyze_line_group(subgraph, lines)
        
        if analysis['type'] == 'component':
            classified['components'].append({
                'nodes': list(cc),
                'subgraph': subgraph,
                'shape_type': analysis['shape_type'],
                'bbox': analysis['bbox'],
                'line_ids': analysis['line_ids']
            })
        elif analysis['type'] == 'wire':
            classified['wires'].append({
                'nodes': list(cc),
                'path': analysis['path'],
                'line_ids': analysis['line_ids']
            })
        elif analysis['type'] == 'junction':
            classified['junctions'].append({
                'position': analysis['position'],
                'connected_lines': analysis['line_ids']
            })
    
    return classified

def analyze_line_group(subgraph, lines):
    num_nodes = subgraph.number_of_nodes()
    num_edges = subgraph.number_of_edges()
    
    positions = np.array([data['pos'] for node, data in subgraph.nodes(data=True)])
    bbox = get_bounding_box(positions)
    
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    bbox_area = w * h
    bbox_aspect = w / max(h, 1e-6)
    
    total_length = sum(data['length'] for _, _, data in subgraph.edges(data=True))
    line_ids = [data['line_id'] for _, _, data in subgraph.edges(data=True)]
    
    # 1. Junction
    if num_nodes == 1 or (num_nodes <= 3 and num_edges >= 3):
        return {'type': 'junction', 'position': np.mean(positions, axis=0), 'line_ids': line_ids}
    
    # 2. Simple Wire
    if is_simple_path(subgraph) and num_edges <= 5:
        return {'type': 'wire', 'path': extract_path(subgraph), 'line_ids': line_ids}
    
    # 3. Long Wire
    if bbox_aspect > 5 or bbox_aspect < 0.2:
        if total_length / max(w, h) > 0.8:
            return {'type': 'wire', 'path': extract_path(subgraph), 'line_ids': line_ids}
    
    # 4. Closed Component
    cycles = nx.cycle_basis(subgraph)
    if len(cycles) > 0:
        shape_type = classify_closed_shape(subgraph, cycles)
        return {'type': 'component', 'shape_type': shape_type, 'bbox': bbox, 'line_ids': line_ids}
    
    # 5. Component Symbol (Cluster)
    if bbox_area < 2000 and num_edges >= 4:
        shape_type = classify_component_symbol(subgraph, lines, line_ids)
        return {'type': 'component', 'shape_type': shape_type, 'bbox': bbox, 'line_ids': line_ids}
    
    # Default
    return {'type': 'wire', 'path': extract_path(subgraph), 'line_ids': line_ids}

def is_simple_path(graph):
    degrees = dict(graph.degree())
    non_two_degree = sum(1 for d in degrees.values() if d != 2)
    return non_two_degree <= 2

def extract_path(graph):
    if len(graph) == 0: return []
    # Find endpoints
    endpoints = [n for n, d in graph.degree() if d == 1]
    start = endpoints[0] if endpoints else list(graph.nodes())[0]
    
    # Simple DFS traversal for path
    path = [start]
    visited = {start}
    current = start
    
    while True:
        neighbors = [n for n in graph.neighbors(current) if n not in visited]
        if not neighbors: break
        current = neighbors[0]
        path.append(current)
        visited.add(current)
        
    return [graph.nodes[n]['pos'] for n in path]

def classify_closed_shape(subgraph, cycles):
    main_cycle = max(cycles, key=len)
    num_sides = len(main_cycle)
    
    positions = [subgraph.nodes[n]['pos'] for n in main_cycle]
    
    if num_sides == 4:
        return 'rectangle' if is_rectangle(positions) else 'quadrilateral'
    elif num_sides == 3: return 'triangle'
    elif num_sides >= 8: return 'circle'
    return f'polygon_{num_sides}'

def classify_component_symbol(subgraph, all_lines, line_ids):
    positions = np.array([data['pos'] for node, data in subgraph.nodes(data=True)])
    
    orientations = []
    for lid in line_ids:
        line = all_lines[lid]
        angle = np.arctan2(line['end'][1]-line['start'][1], line['end'][0]-line['start'][0])
        orientations.append(angle)
    
    if has_zigzag_pattern(subgraph): return 'resistor'
    if has_parallel_lines(orientations, positions): return 'capacitor'
    if has_cross_pattern(subgraph): return 'terminal'
    return 'symbol'

def has_zigzag_pattern(graph):
    if graph.number_of_edges() < 4: return False
    path = extract_path(graph)
    if len(path) < 5: return False
    
    angles = []
    path_arr = np.array(path)
    for i in range(len(path_arr) - 2):
        v1 = path_arr[i+1] - path_arr[i]
        v2 = path_arr[i+2] - path_arr[i+1]
        a1 = np.arctan2(v1[1], v1[0])
        a2 = np.arctan2(v2[1], v2[0])
        angles.append(a2 - a1)
        
    sign_changes = sum(1 for i in range(len(angles)-1) if angles[i] * angles[i+1] < 0)
    return sign_changes >= 2

def separate_components_from_wires(classified, lines, tolerance=3.0):
    components = classified['components']
    wires = classified['wires']
    refined_wires = []
    
    for wire in wires:
        wire_line_ids = wire['line_ids']
        # Simply check if wire overlaps strictly with any component bbox?
        # A Better approach used previously: check segments.
        
        # NOTE: For simplicity in this fix, we will treat the wire object as is
        # but add connection info if it touches a component.
        connections = []
        
        if 'path' not in wire or not wire['path']: continue
            
        p_start = wire['path'][0]
        p_end = wire['path'][-1]
        
        for comp in components:
            bbox = comp['bbox']
            # Check Start
            if is_point_on_bbox_boundary(p_start, bbox, tolerance):
                connections.append({'comp': comp, 'at': 'start'})
            # Check End
            if is_point_on_bbox_boundary(p_end, bbox, tolerance):
                connections.append({'comp': comp, 'at': 'end'})
        
        refined_wires.append({
            'original_wire': wire,
            'connections': connections,
            'path': wire['path'] # Preserve path
        })
        
    return refined_wires

def is_point_on_bbox_boundary(point, bbox, tolerance):
    x, y = point
    x0, y0, x1, y1 = bbox
    # Check if inside expanded box but outside contracted box?
    # Simple proximity check:
    on_vert = (abs(x - x0) < tolerance or abs(x - x1) < tolerance) and (y0-tolerance <= y <= y1+tolerance)
    on_horiz = (abs(y - y0) < tolerance or abs(y - y1) < tolerance) and (x0-tolerance <= x <= x1+tolerance)
    return on_vert or on_horiz

def identify_component_ports(components, refined_wires):
    # This step was largely merged into separate_components_from_wires for this fixed version
    # But let's ensure components know their connections too.
    for comp in components:
        comp['ports'] = []
        
    for r_wire in refined_wires:
        for conn in r_wire['connections']:
            comp = conn['comp']
            comp['ports'].append({'connected_wire': r_wire})
            
    return components

# ==========================================
# 4. MAIN PIPELINE
# ==========================================

def analyze_line_soup_diagram(pdf_path):
    print(f"--- Processing {pdf_path} ---")
    
    # 1. Extract
    lines = extract_all_lines(pdf_path)
    if not lines: return
    print(f"Step 1: Extracted {len(lines)} lines")
    viz_step_1_raw_lines(lines)
    
    # 2. Graph
    line_graph, _ = build_line_graph(lines)
    print(f"Step 2: Built graph with {line_graph.number_of_nodes()} nodes")
    viz_step_2_graph(line_graph)
    
    # 3. Classify
    classified = classify_line_groups(line_graph, lines)
    print(f"Step 3: Found {len(classified['components'])} components, {len(classified['wires'])} wires")
    viz_step_3_classification(classified)
    
    # 4. Refine & Connect
    refined_wires = separate_components_from_wires(classified, lines)
    
    # 5. Labels
    doc = fitz.open(pdf_path)
    text_dict = doc[0].get_text("dict")
    components = associate_labels(classified['components'], text_dict)
    
    # 6. Final Logic Graph
    components = identify_component_ports(components, refined_wires)
    
    circuit = nx.Graph()
    # Add Component Nodes
    comp_map = {} # map object id to node id
    for i, comp in enumerate(components):
        node_id = f"C{i}"
        comp_map[id(comp)] = node_id
        circuit.add_node(node_id, type=comp['shape_type'], label=comp.get('label','?'))
        
    # Add Wire Edges
    for wire in refined_wires:
        conns = wire['connections']
        if len(conns) == 2:
            # Wire connects two components
            c1 = comp_map.get(id(conns[0]['comp']))
            c2 = comp_map.get(id(conns[1]['comp']))
            if c1 and c2:
                circuit.add_edge(c1, c2)
        elif len(conns) == 1:
            # Dangling wire or connected to rail?
            pass
            
    print(f"Step 7: Final circuit has {circuit.number_of_edges()} connections")
    viz_step_final_circuit(circuit, components)
    
    return circuit

if __name__ == "__main__":
    # Create a dummy test file if none exists
    import os
    file_path = "example_circuit.pdf"
    
    if not os.path.exists(file_path):
        print(f"File {file_path} not found. Please provide a path to a valid PDF.")
    else:
        analyze_line_soup_diagram(file_path)