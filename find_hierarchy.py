from shapely.geometry import Polygon

def find_hierarchy(polygons_coords):
    # 1. Convertir en objets Shapely et calculer l'aire
    polys = []
    for p in polygons_coords:
        if len(p) >= 3:
            sh_poly = Polygon(p)
            polys.append({'poly': sh_poly, 'area': sh_poly.area, 'coords': p})
    
    # 2. Trier par aire décroissante (les plus gros en premier)
    polys.sort(key=lambda x: x['area'], reverse=True)
    
    hierarchy = []
    for i, parent in enumerate(polys):
        children_points = []
        for j, child in enumerate(polys):
            if i != j and parent['poly'].contains(child['poly']):
                # On récupère le centre du composant interne pour le point négatif
                centroid = child['poly'].centroid
                children_points.append([centroid.x, centroid.y])
        
        hierarchy.append({
            'box': list(parent['poly'].bounds), # [xmin, ymin, xmax, ymax]
            'center': [parent['poly'].centroid.x, parent['poly'].centroid.y],
            'internal_points': children_points
        })
    return hierarchy