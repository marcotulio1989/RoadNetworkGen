import numpy as np
import shapely.geometry as sg
import shapely.ops as so
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

def get_default_parameters():
    """
    Returns a dictionary containing the default parameters for city generation.
    """
    params = {
        'Seed': 7,
        'NumSeeds': 280,
        'LloydIters': 3,
        'RoadW': {'Arterial': 14, 'Main': 9, 'Local': 6},
        'SidewalkW': 3,
        'ZoneTarget': {'Commercial': 0.18, 'Industrial': 0.10, 'Residential': 0.46, 'Park': 0.16},
        'Setback': {'Commercial': 6, 'Industrial': 8, 'Residential': 5, 'Park': 3},
        'Lot': {
            'Commercial': (36, 28),
            'Industrial': (50, 40),
            'Residential': (22, 18),
            'Park': (40, 40)
        },
        'BuildingFill': {'Commercial': 0.80, 'Industrial': 0.70, 'Residential': 0.55, 'Park': 0.05},
        'BuildingInset': {'Commercial': 2.5, 'Industrial': 3.5, 'Residential': 2.0, 'Park': 1.0},
        'Heights': {
            'Commercial': {'min': 12, 'max': 45},
            'Industrial': {'min': 8, 'max': 18},
            'Residential': {'min': 6, 'max': 15},
            'Park': {'min': 0, 'max': 0}
        },
        'MinBuildingArea': 25,
        'ZoneColor': {
            'Commercial': (0.95, 0.80, 0.60),
            'Industrial': (0.85, 0.75, 0.55),
            'Residential': (0.75, 0.85, 0.95),
            'Park': (0.78, 0.90, 0.78)
        }
    }
    return params

def sample_in_poly(poly, n):
    """
    Generates n random points inside a shapely Polygon using rejection sampling.
    """
    min_x, min_y, max_x, max_y = poly.bounds
    points = []
    while len(points) < n:
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        if poly.contains(sg.Point(x, y)):
            points.append((x, y))
    return np.array(points)

def voronoi_in_boundary(points, boundary):
    """
    Generates Voronoi cells and clips them to a boundary polygon.
    Returns the Voronoi object and the list of clipped cell polygons.
    """
    vor = Voronoi(points)
    regions = []
    for region_indices in vor.regions:
        if not region_indices or -1 in region_indices:
            continue

        polygon_vertices = vor.vertices[region_indices]
        poly = sg.Polygon(polygon_vertices)

        if poly.is_valid:
            clipped_poly = boundary.intersection(poly)
            if not clipped_poly.is_empty:
                regions.append(clipped_poly)

    return vor, regions

def safe_buffer(poly, dist):
    """Buffers a polygon, returning the original on error."""
    try:
        buffered = poly.buffer(dist)
        return buffered if buffered.is_valid else poly
    except:
        return poly

def grid_lots(poly, lot_mean):
    """Tiles a clipped grid over a polygon with jittered cell size."""
    if poly.is_empty:
        return []
    xlim, ylim = poly.bounds[0], poly.bounds[1]
    dx = lot_mean[0] * (0.8 + 0.4 * np.random.rand())
    dy = lot_mean[1] * (0.8 + 0.4 * np.random.rand())
    xs = np.arange(xlim, poly.bounds[2], dx)
    ys = np.arange(ylim, poly.bounds[3], dy)

    lots = []
    for ix in range(len(xs) - 1):
        for iy in range(len(ys) - 1):
            x1, x2 = xs[ix], xs[ix+1]
            y1, y2 = ys[iy], ys[iy+1]
            lot_poly = sg.box(x1, y1, x2, y2)
            c = poly.intersection(lot_poly)
            if not c.is_empty and c.area > 0:
                lots.append(c)
    return lots

def generate_city():
    """
    Main function to generate and draw the city.
    """
    params = get_default_parameters()

    # Set seed for reproducibility
    np.random.seed(params['Seed'])

    # 1. City Limits
    city_boundary = sg.Polygon([(0, 0), (1200, 60), (1250, 800), (500, 950), (-100, 700), (0, 0)])

    # 2. Seeds & Centroidal Voronoi (Lloyd's Algorithm)
    points = sample_in_poly(city_boundary, params['NumSeeds'])

    for i in range(params['LloydIters']):
        vor, voronoi_cells = voronoi_in_boundary(points, city_boundary)
        centroids = np.array([cell.centroid.coords[0] for cell in voronoi_cells if not cell.is_empty])
        if len(centroids) < params['NumSeeds'] * 0.9:
            break
        points = centroids

    vor, final_voronoi_cells = voronoi_in_boundary(points, city_boundary)

    # 3. Roads (from Voronoi ridge lines)
    road_segments = []
    for ridge_vertices in vor.ridge_vertices:
        if -1 in ridge_vertices: continue
        p1 = vor.vertices[ridge_vertices[0]]
        p2 = vor.vertices[ridge_vertices[1]]
        line = sg.LineString([p1, p2])
        if city_boundary.contains(line.interpolate(0.5, normalized=True)):
            road_segments.append(line)

    lengths = np.array([seg.length for seg in road_segments])
    tA, tM = np.percentile(lengths, 80), np.percentile(lengths, 40)

    roads_local = sg.MultiLineString([s for s, l in zip(road_segments, lengths) if l < tM])
    roads_main = sg.MultiLineString([s for s, l in zip(road_segments, lengths) if tM <= l < tA])
    roads_arterial = sg.MultiLineString([s for s, l in zip(road_segments, lengths) if l >= tA])

    road_poly_local = roads_local.buffer(params['RoadW']['Local'], cap_style=2)
    road_poly_main = roads_main.buffer(params['RoadW']['Main'], cap_style=2)
    road_poly_arterial = roads_arterial.buffer(params['RoadW']['Arterial'], cap_style=2)
    all_roads = so.unary_union([road_poly_local, road_poly_main, road_poly_arterial])

    walk_ext = all_roads.buffer(params['SidewalkW'], cap_style=2)
    sidewalks = walk_ext.difference(all_roads)

    # 4. Blocks & Zoning
    blocks_poly = city_boundary.difference(walk_ext)
    blocks = list(blocks_poly.geoms) if isinstance(blocks_poly, sg.MultiPolygon) else [blocks_poly]
    block_areas = np.array([b.area for b in blocks])

    zone_names = list(params['ZoneTarget'].keys())
    target_areas = {name: city_boundary.area * params['ZoneTarget'][name] for name in zone_names}

    assignments = [''] * len(blocks)
    sorted_block_indices = np.argsort(-block_areas)

    for i in sorted_block_indices:
        best_zone = max(target_areas, key=target_areas.get)
        assignments[i] = best_zone
        target_areas[best_zone] -= block_areas[i]

    # 5. Buildings & Green
    buildings_list = []
    greens_list = []

    for i, block in enumerate(blocks):
        zone = assignments[i]
        buildable = safe_buffer(block, -params['Setback'][zone])
        if buildable.area <= 0:
            continue
        lots = grid_lots(buildable, params['Lot'][zone])
        for lot in lots:
            if np.random.rand() < params['BuildingFill'][zone]:
                building = safe_buffer(lot, -params['BuildingInset'][zone])
                if building.area > params['MinBuildingArea']:
                    buildings_list.append(building)
            else:
                greens_list.append(lot)

    all_buildings = so.unary_union(buildings_list)
    all_greens = so.unary_union(greens_list)

    # --- Visualization ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12), gridspec_kw={'width_ratios': [3, 1]})

    def plot_polygon(ax, poly, facecolor, edgecolor='none', **kwargs):
        if poly.is_empty: return
        if isinstance(poly, sg.Polygon):
            ax.fill(*poly.exterior.xy, facecolor=facecolor, edgecolor=edgecolor, **kwargs)
            for i in poly.interiors: ax.fill(*i.xy, facecolor=ax.get_facecolor(), edgecolor=edgecolor, **kwargs)
        elif isinstance(poly, sg.MultiPolygon):
            for p in poly.geoms:
                ax.fill(*p.exterior.xy, facecolor=facecolor, edgecolor=edgecolor, **kwargs)
                for i in p.interiors: ax.fill(*i.xy, facecolor=ax.get_facecolor(), edgecolor=edgecolor, **kwargs)

    # Plot 1: Plan View
    ax1.set_aspect('equal', adjustable='box'); ax1.set_facecolor('#F6F6F6')
    ax1.set_title('Procedural City (Plan)'); ax1.set_xticks([]); ax1.set_yticks([])
    plot_polygon(ax1, city_boundary, facecolor='#F6F6F6', edgecolor='#333333', linewidth=1.5)
    plot_polygon(ax1, all_greens, facecolor='#D0E8D0')
    plot_polygon(ax1, all_buildings, facecolor='#D9D9E8', edgecolor='#505070')
    plot_polygon(ax1, sidewalks, facecolor='#DDDDDD')
    plot_polygon(ax1, road_poly_arterial, facecolor='#353537')
    plot_polygon(ax1, road_poly_main, facecolor='#404042')
    plot_polygon(ax1, road_poly_local, facecolor='#454547')

    # Plot 2: Zoning View
    ax2.set_aspect('equal', adjustable='box'); ax2.set_facecolor('w')
    ax2.set_title('Zoning'); ax2.set_xticks([]); ax2.set_yticks([])
    for block, zone in zip(blocks, assignments):
        plot_polygon(ax2, block, facecolor=params['ZoneColor'][zone], alpha=0.9)
    plot_polygon(ax2, city_boundary, facecolor='none', edgecolor='#333333', linewidth=1)

    plt.tight_layout(pad=2)
    plt.savefig("city_final.png", dpi=300, bbox_inches='tight')
    print(f"Final city plot saved to city_final.png")
    print(f"Blocks: {len(blocks)} | Buildings: {len(buildings_list)} | Green regions: {len(greens_list)}")


if __name__ == "__main__":
    generate_city()
