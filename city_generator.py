# Procedural Modeling of Cities (Parish & Müller, 2001) – Python Prototype
# This script is a Python translation of the MATLAB prototype provided by the user.
# Pipeline: maps -> HIGHWAYS (global goal) -> STREETS (local fill, constraints)
#         -> BLOCKS -> recursive LOTS -> BUILDINGS (zone + height rules)

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from skimage.measure import find_contours
from shapely.geometry import Polygon, LineString, MultiLineString, MultiPolygon
from shapely.ops import unary_union, split
from mpl_toolkits.mplot3d import art3d

def pm_city_demo():
    """
    Main function to generate and plot the city.
    This is a Python translation of the pm_city_demo MATLAB script.
    """
    # =========================================================================
    # %% ------------------ USER MAPS (replace with your rasters if you have them)
    # =========================================================================
    np.random.seed(42)                      # reproducible
    W, H = 1200, 900                        # map size (meters-ish)

    # Create a grid of coordinates
    x_coords = np.linspace(0, W, 600)
    y_coords = np.linspace(0, H, 450)
    X, Y = np.meshgrid(x_coords, y_coords)

    # Population-density map (peaks seed HIGHWAYS) (paper §2, image maps)
    # Synthesize a few Gaussian “centers”
    pd = (0.4 * np.exp(-((X - 300)**2 + (Y - 200)**2) / (2 * 120**2)) +
          0.6 * np.exp(-((X - 880)**2 + (Y - 650)**2) / (2 * 180**2)) +
          0.3 * np.exp(-((X - 950)**2 + (Y - 200)**2) / (2 * 120**2)) +
          0.2 * np.random.rand(*X.shape) * 0.05)

    # Zoning map: 1=Residential, 2=Commercial, 3=Industrial, 4=Park (paper §2)
    # Simple “rings” around density centers
    zone = np.ones(X.shape)                 # default Residential
    zone[pd > 0.45] = 2                     # core Commercial
    zone[pd < 0.12] = 4                     # outer Park
    zone[(pd > 0.28) & (pd <= 0.45) & (X < 750)] = 3  # an Industrial wedge

    # Height-limit map (paper §4: zoning/height constraints)
    # Normalize population density to range [0, 1] to create height map
    pd_normalized = (pd - pd.min()) / (pd.max() - pd.min())
    hlim = 12 + 40 * gaussian_filter(pd_normalized, sigma=6) # 12–52 m roughly

    # Forbidden/water mask for local constraints (avoid roads/buildings there)
    water = (Y < 80) | (Y > 840) | (X < 60) | (X > 1140)  # perimeter “water”
    water_body = np.exp(-((X - 600)**2 + (Y - 450)**2) / (2 * 180**2)) > 0.5
    water = water | (water_body & (Y < 450))
    forbid = water                          # drawable constraint

    # City boundary polygon
    B = Polygon([(0, 0), (W, 0), (W, H), (0, H)])
    water_polys = mask_to_polygons(water, x_coords, y_coords)
    if not water_polys.is_empty:
        B = B.difference(water_polys)

    print("Step 1/4: Maps generated.")

    # =========================================================================
    # %% ------------------ ROADS: extended L-system (paper §3)
    # =========================================================================
    # Setup interpolators for map sampling
    density_interp = RegularGridInterpolator((y_coords, x_coords), pd)
    forbid_interp = RegularGridInterpolator((y_coords, x_coords), forbid.astype(float), method='nearest')

    # 1) Seed highway “turtles” at top-K density maxima
    K = 3
    pks = find_local_max(pd, K, X, Y)

    hwy_opts = {'step': 20, 'turn_std': np.deg2rad(10), 'len': 260, 'snap': 8, 'bias_to_peak': 0.9}
    Hwy = grow_roads(pks, density_interp, forbid_interp, B, x_coords, y_coords, hwy_opts)

    # 2) Seed street “turtles” along highways and at additional density ridges
    street_seeds = downsample_polyline(Hwy, 80)
    if not street_seeds.any():
        street_seeds = find_local_max(pd, 8, X, Y)
    else:
        street_seeds = np.vstack([street_seeds, find_local_max(pd, 8, X, Y)])

    str_opts = {'step': 10, 'turn_std': np.deg2rad(18), 'len': 120, 'snap': 6, 'grid_bias': 0.65, 'bias_to_peak': 0.35}
    Str = grow_roads(street_seeds, density_interp, forbid_interp, B, x_coords, y_coords, str_opts)

    # Buffer polylines into rights-of-way and sidewalks
    PR_hwy = Hwy.buffer(12, cap_style=2)
    PR_st = Str.buffer(6, cap_style=2)
    Roads = unary_union([PR_hwy, PR_st])
    Walks = Roads.buffer(3).difference(Roads)

    print("Step 2/4: Road network generated.")

    # =========================================================================
    # %% ------------------ BLOCKS -> LOTS -> BUILDINGS (paper §4)
    # =========================================================================
    CityLand = B.difference(Roads.buffer(3))

    # Decompose the resulting MultiPolygon into individual Polygons (Blocks)
    if isinstance(CityLand, MultiPolygon):
        Blocks = [p for p in CityLand.geoms if p.area > 2500]
    elif CityLand.area > 2500:
        Blocks = [CityLand]
    else:
        Blocks = []

    # Recursive subdivision of blocks into lots
    Lots = []
    for block in Blocks:
        Lots.extend(subdiv_lots(block, 900, 0.15))

    # Discard lots without street access or that are too small
    Lots = [lot for lot in Lots if lot.area > 120]
    Lots = [lot for lot in Lots if lot.buffer(1).intersects(Roads)]

    # Zone each lot and determine height capacity
    zone_interp = RegularGridInterpolator((y_coords, x_coords), zone, method='nearest')
    hlim_interp = RegularGridInterpolator((y_coords, x_coords), hlim)

    Z = [int(zone_interp(p.centroid.coords[0][::-1])) for p in Lots]
    Hcap = [hlim_interp(p.centroid.coords[0][::-1])[0] for p in Lots]

    # Generate building footprints based on zone-dependent coverage
    Blds = []
    Heights = []

    cov_res = lambda A: 0.45 + 0.15 * np.random.rand()
    cov_com = lambda A: 0.65 + 0.10 * np.random.rand()
    cov_ind = lambda A: 0.55 + 0.10 * np.random.rand()
    cov_par = lambda A: 0.10 + 0.05 * np.random.rand()

    for i, lot in enumerate(Lots):
        A = lot.area
        z = Z[i]

        if z == 1: # Residential
            cov = cov_res(A)
            hin = [6, 18]
        elif z == 2: # Commercial
            cov = cov_com(A)
            hin = [12, 60]
        elif z == 3: # Industrial
            cov = cov_ind(A)
            hin = [8, 24]
        else: # Park
            cov = cov_par(A)
            hin = [0, 0]

        fp = shrink_to_coverage(lot, cov)
        Blds.append(fp)

        height = min(hin[0] + np.random.rand() * (hin[1] - hin[0]), Hcap[i]) if hin[1] > hin[0] else 0
        Heights.append(height)

    # Greenspace is the area within lots not covered by buildings
    Greens = unary_union(Lots).difference(unary_union(Blds))

    print("Step 3/4: Blocks, lots, and buildings generated.")

    # =========================================================================
    # %% ------------------ PLOTS
    # =========================================================================
    plot_city_isometric(B, Roads, Walks, Greens, Blds, Heights)

    print("Step 4/4: Isometric city plot generated and saved.")


# =========================================================================
# %% ====================== helpers ===========================
# =========================================================================

def plot_city_isometric(boundary, roads, walks, greens, buildings, heights):
    """Generates and saves an isometric plot of the city."""
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    # Plot flat features by manually creating 3D vertices with Z=0
    plot_polygon_3d(ax, boundary, facecolor='#f0f0f0', edgecolor='#333333', z=0)
    plot_polygon_3d(ax, roads, facecolor='#444444', z=0)
    plot_polygon_3d(ax, walks, facecolor='#bbbbbb', z=0)
    plot_polygon_3d(ax, greens, facecolor='#88cc88', z=0)

    # Plot extruded buildings
    for b, h in zip(buildings, heights):
        if b.is_empty or h <= 0:
            continue

        # Plot roof by manually creating 3D vertices with Z=h
        plot_polygon_3d(ax, b, facecolor='#cccccc', edgecolor='#444444', z=h)

        # Walls
        wall_verts = []
        # Exterior walls
        for i in range(len(b.exterior.coords) - 1):
            p1, p2 = b.exterior.coords[i], b.exterior.coords[i+1]
            wall_verts.append(list(zip([p1[0], p2[0], p2[0], p1[0]], [p1[1], p2[1], p2[1], p1[1]], [0, 0, h, h])))
        # Interior walls (for holes in buildings)
        for interior in b.interiors:
            for i in range(len(interior.coords) - 1):
                p1, p2 = interior.coords[i], interior.coords[i+1]
                wall_verts.append(list(zip([p1[0], p2[0], p2[0], p1[0]], [p1[1], p2[1], p2[1], p1[1]], [0, 0, h, h])))

        if wall_verts:
            ax.add_collection3d(art3d.Poly3DCollection(wall_verts, facecolors='#aaaaaa', linewidths=0))

    # Set view and style
    ax.view_init(elev=40, azim=-65, roll=0)
    ax.set_box_aspect([1, (boundary.bounds[3]-boundary.bounds[1])/(boundary.bounds[2]-boundary.bounds[0]), 0.3])
    ax.set_axis_off()

    # Set limits to bound the city tightly
    bounds = boundary.bounds
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_zlim(0, max(heights) * 1.5 if heights else 100)

    plt.savefig("city_isometric.png", dpi=300, bbox_inches='tight', pad_inches=0.1, transparent=True)
    plt.close(fig)

def plot_polygon_3d(ax, geom, facecolor, edgecolor=None, z=0):
    """Helper to plot a shapely Polygon or MultiPolygon in 3D at a fixed z."""
    if geom.is_empty:
        return

    geoms = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]

    for p in geoms:
        if p.is_empty: continue

        # Manually construct 3D vertices
        verts_3d = []
        xs, ys = p.exterior.xy
        verts_3d.append(list(zip(xs, ys, [z] * len(xs))))

        for inter in p.interiors:
            ixs, iys = inter.xy
            verts_3d.append(list(zip(ixs, iys, [z] * len(ixs))))

        edge_kwargs = {'linewidths': 0.5, 'edgecolors': edgecolor} if edgecolor else {}
        ax.add_collection3d(art3d.Poly3DCollection(verts_3d, facecolors=facecolor, **edge_kwargs))

def subdiv_lots(polygon, target_area, anisotropy):
    """Recursively split a polygon until lots are near the target area."""
    lots = []
    todo = [polygon]

    while todo:
        p = todo.pop(0)
        if p.area <= 1: continue

        # Accept the lot if it's close to the target area
        if p.area <= target_area * (0.7 + 0.6 * np.random.rand()):
            lots.append(p)
            continue

        # Split along the "longest" side with some randomness
        bounds = p.bounds
        w, h = bounds[2] - bounds[0], bounds[3] - bounds[1]

        if w > h:
            direction = 0  # horizontal
        else:
            direction = np.pi / 2  # vertical

        # Add random perturbation to the split angle
        direction += anisotropy * (np.random.rand() - 0.5) * np.pi / 2

        # Create a long splitting line through the centroid
        c = p.centroid
        dx, dy = np.cos(direction), np.sin(direction)
        line = LineString([(c.x - 10000 * dx, c.y - 10000 * dy),
                           (c.x + 10000 * dx, c.y + 10000 * dy)])

        try:
            split_result = split(p, line)
            todo.extend(g for g in split_result.geoms if g.area > 1)
        except Exception:
            # If split fails, just add the polygon back and hope for the best
            lots.append(p)

    return lots

def shrink_to_coverage(lot, coverage):
    """Shrinks a lot polygon to a target area coverage via binary search on buffer distance."""
    if lot.area <= 0: return Polygon()

    # Binary search for the right inset distance
    d0, d1 = 0, np.sqrt(lot.area) / 2 # Search range for inset

    for _ in range(10): # 10 iterations are enough for good precision
        dm = (d0 + d1) / 2
        shrunk_poly = lot.buffer(-dm)
        if shrunk_poly.is_empty: # Inset is too large
            d1 = dm
            continue

        current_coverage = shrunk_poly.area / lot.area
        if current_coverage > coverage:
            d0 = dm  # Need to shrink more
        else:
            d1 = dm  # Shrunk too much

    return lot.buffer(-d0)

def find_local_max(M, K, X, Y):
    """Finds the K largest local maxima in a 2D matrix M."""
    # Smooth the matrix to avoid picking up noisy peaks
    S = gaussian_filter(M, sigma=3)

    # Find regional maxima
    # A point is a regional maxima if it is not smaller than any of its neighbours
    from scipy.ndimage import maximum_filter
    regional_max_mask = (S == maximum_filter(S, footprint=np.ones((3, 3))))

    # Get the coordinates and values of the maxima
    r, c = np.where(regional_max_mask)
    vals = S[r, c]

    # Sort descending and take the top K
    sorted_indices = np.argsort(vals)[::-1]
    top_k_indices = sorted_indices[:K]

    r_top, c_top = r[top_k_indices], c[top_k_indices]

    # Return the (x, y) coordinates
    return np.vstack([X[r_top, c_top], Y[r_top, c_top]]).T

def downsample_polyline(multiline, ds):
    """Downsamples a shapely MultiLineString to a series of points."""
    points = []
    if multiline.is_empty:
        return np.array(points)

    for line in multiline.geoms:
        v = np.array(line.coords)
        d = np.cumsum(np.hypot(np.diff(v[:, 0]), np.diff(v[:, 1])))
        d = np.insert(d, 0, 0)

        s = np.arange(0, d[-1], ds)

        from scipy.interpolate import interp1d
        # interp1d requires that d does not have duplicates
        d_unique, idx_unique = np.unique(d, return_index=True)
        v_unique = v[idx_unique]

        if len(d_unique) < 2:
            continue

        interp_fx = interp1d(d_unique, v_unique[:, 0], bounds_error=False, fill_value="extrapolate")
        interp_fy = interp1d(d_unique, v_unique[:, 1], bounds_error=False, fill_value="extrapolate")

        points.extend(np.vstack([interp_fx(s), interp_fy(s)]).T)

    return np.array(points)

def grad_at(p, M_interp, x_coords, y_coords):
    """Calculates the gradient of matrix M at point p using an interpolator."""
    # Smooth the data before taking the gradient, as in the original MATLAB
    S = gaussian_filter(M_interp.values, sigma=4)

    # The gradient of the interpolator is not directly available.
    # Instead, we compute the gradient of the smoothed underlying grid and interpolate that.
    gy, gx = np.gradient(S, np.mean(np.diff(y_coords)), np.mean(np.diff(x_coords)))

    # Create interpolators for the gradient components
    gx_interp = RegularGridInterpolator((y_coords, x_coords), gx)
    gy_interp = RegularGridInterpolator((y_coords, x_coords), gy)

    # Sample the gradient at point p
    grad_x = gx_interp(p)
    grad_y = gy_interp(p)

    return grad_x[0], grad_y[0]

def wrap_to_pi(angle):
    """Wrap an angle to the range [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def grow_roads(seeds, dens_interp, forbid_interp, boundary, x_coords, y_coords, opt):
    """Grows a road network using an extended L-system."""
    lines = []
    all_vertices = []

    dirs = np.deg2rad([0, 90, 180, 270]) # grid bias directions

    if seeds.ndim == 1: seeds = np.array([seeds]) # Handle single seed case
    if seeds.size == 0: return MultiLineString() # Handle empty seeds

    for seed in seeds:
        pos = np.array(seed)
        ang = np.random.rand() * 2 * np.pi

        n_steps = round(opt['len'] / opt['step'])

        for _ in range(n_steps):
            # Global goal: drift toward density maxima
            gx, gy = grad_at(pos[::-1], dens_interp, x_coords, y_coords) # interpolator expects (y,x)
            goal = np.arctan2(gy, gx)
            if not np.isfinite(goal): goal = ang

            # Optional grid bias
            if 'grid_bias' in opt and np.random.rand() < opt['grid_bias']:
                k = np.argmin(np.abs(wrap_to_pi(ang - dirs)))
                goal = dirs[k]

            # Update angle with randomness and bias
            ang += (opt['turn_std'] * np.random.randn() +
                    opt['bias_to_peak'] * wrap_to_pi(goal - ang))

            nxt = pos + opt['step'] * np.array([np.cos(ang), np.sin(ang)])

            # Local constraints: check boundary and forbidden zones
            if not boundary.contains(LineString([pos, nxt])):
                break
            if forbid_interp(nxt[::-1]) > 0.5:
                break

            # Snapping to nearby vertices
            if all_vertices:
                from scipy.spatial import cKDTree
                tree = cKDTree(all_vertices)
                dist, idx = tree.query(nxt)
                if dist < opt['snap']:
                    nxt = tree.data[idx]

            lines.append(LineString([pos, nxt]))
            all_vertices.extend([pos, nxt])
            pos = nxt

    return unary_union(lines) if lines else MultiLineString()

def mask_to_polygons(mask, x_coords, y_coords):
    """
    Convert a boolean mask to a shapely Polygon or MultiPolygon.
    The mask is defined on a grid given by x_coords and y_coords.
    """
    # Find contours at a level of 0.5
    contours = find_contours(mask, 0.5)
    polygons = []
    for contour in contours:
        # Contour vertices are in (row, col) index space, convert to (x, y)
        # We need to interpolate to get the coordinates.
        y_contour = np.interp(contour[:, 0], np.arange(len(y_coords)), y_coords)
        x_contour = np.interp(contour[:, 1], np.arange(len(x_coords)), x_coords)
        poly = Polygon(zip(x_contour, y_contour))
        if poly.is_valid:
            polygons.append(poly)

    if not polygons:
        return Polygon()

    return unary_union(polygons)

if __name__ == "__main__":
    pm_city_demo()
