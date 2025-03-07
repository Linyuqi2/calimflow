import numpy as np
from typing import Tuple, List, Optional
from scipy.spatial import ConvexHull
from ..utils.geometry import rotation_matrix

def spiral_sample_sphere(n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Produce approximately uniform sampling of a unit sphere using spiral method."""
    # Golden ratio and angle
    gr = (1 + np.sqrt(5)) / 2
    ga = 2 * np.pi * (1 - 1/gr)
    
    # Generate indices and calculate coordinates
    i = np.arange(n_samples)
    lat = np.arccos(1 - 2*i/(n_samples-1))  # latitude
    lon = i * ga  # longitude
    
    # Convert to Cartesian coordinates
    x = np.sin(lat) * np.cos(lon)
    y = np.sin(lat) * np.sin(lon)
    z = np.cos(lat)
    vertices = np.column_stack([x, y, z])
    
    # Generate triangulation using ConvexHull
    hull = ConvexHull(vertices)
    triangulation = hull.simplices[:, ::-1]  # Flip to match MATLAB orientation
    
    return vertices, triangulation

def voxelize_internal(
    test_x: np.ndarray,
    test_y: np.ndarray,
    test_z: np.ndarray,
    mesh_xyz: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Internal function for ray-casting based point-in-mesh test.
    Based on MATLAB's VOXELISEinternal function.
    
    Args:
        test_x, test_y, test_z: Coordinates of test points
        mesh_xyz: Triangle mesh coordinates (n_faces x 3 x 3)
        
    Returns:
        output: Binary array indicating inside/outside
        correction_list: Indices of uncertain points
    """
    # Initialize output and correction list
    output = np.zeros(len(test_x), dtype=bool)
    correction_list = []
    
    # Get mesh bounds
    mesh_z_min = np.min(mesh_xyz[:,:,2])
    mesh_z_max = np.max(mesh_xyz[:,:,2])
    
    # Process each test point
    for i in range(len(test_x)):
        # Skip if point is outside mesh bounds
        if test_z[i] < mesh_z_min or test_z[i] > mesh_z_max:
            continue
            
        # Find facets that the ray potentially intersects
        facet_cross_list = []
        
        for f in range(len(mesh_xyz)):
            # Get facet coordinates
            x1, x2, x3 = mesh_xyz[f,:,0]
            y1, y2, y3 = mesh_xyz[f,:,1]
            
            # Check if ray intersects triangle projection in xy plane
            if (((y1 > test_y[i]) != (y2 > test_y[i])) and 
                (test_x[i] < (x2-x1) * (test_y[i]-y1) / (y2-y1) + x1)) or \
               (((y2 > test_y[i]) != (y3 > test_y[i])) and 
                (test_x[i] < (x3-x2) * (test_y[i]-y2) / (y3-y2) + x2)) or \
               (((y3 > test_y[i]) != (y1 > test_y[i])) and 
                (test_x[i] < (x1-x3) * (test_y[i]-y3) / (y1-y3) + x3)):
                facet_cross_list.append(f)
        
        if not facet_cross_list:
            continue
            
        # Find z-coordinates of intersections
        z_crosses = []
        for f in facet_cross_list:
            # Calculate plane equation coefficients (Ax + By + Cz + D = 0)
            x = mesh_xyz[f,:,0]
            y = mesh_xyz[f,:,1]
            z = mesh_xyz[f,:,2]
            
            A = (y[0]*(z[1]-z[2]) + y[1]*(z[2]-z[0]) + y[2]*(z[0]-z[1]))
            B = (z[0]*(x[1]-x[2]) + z[1]*(x[2]-x[0]) + z[2]*(x[0]-x[1]))
            C = (x[0]*(y[1]-y[2]) + x[1]*(y[2]-y[0]) + x[2]*(y[0]-y[1]))
            D = (-x[0]*(y[1]*z[2]-y[2]*z[1]) - 
                  x[1]*(y[2]*z[0]-y[0]*z[2]) - 
                  x[2]*(y[0]*z[1]-y[1]*z[0]))
            
            # Skip if plane is parallel to ray
            if abs(C) < 1e-14:
                continue
                
            # Calculate z intersection
            z_cross = (-D - A*test_x[i] - B*test_y[i]) / C
            
            # Check if intersection is within mesh bounds
            if mesh_z_min-1e-12 <= z_cross <= mesh_z_max+1e-12:
                z_crosses.append(z_cross)
        
        # Round to remove numerical errors and get unique values
        if z_crosses:
            z_crosses = np.unique(np.round(z_crosses, decimals=10))
            
            # Even number of crossings allows inside/outside determination
            if len(z_crosses) % 2 == 0:
                for j in range(0, len(z_crosses), 2):
                    if z_crosses[j] < test_z[i] < z_crosses[j+1]:
                        output[i] = True
                        break
            else:
                correction_list.append(i)
    
    return output, np.array(correction_list)

def in_triangulation(vertices: np.ndarray, faces: np.ndarray, testp: np.ndarray, heavytest: int = 0) -> np.ndarray:
    """
    Optimized and direct port of MATLAB's intriangulation function.
    """
    # Input validation and initialization
    if vertices.shape[1] != 3 or faces.shape[1] != 3 or testp.shape[1] != 3:
        raise ValueError("All input matrices must have three columns")
    if faces.max() >= len(vertices) or faces.min() < 0:
        raise ValueError("Face indices out of bounds")

    in_return = np.zeros(len(testp), dtype=np.int8)
    VER = vertices.copy()
    TESTP = testp.copy()

    # Pre-compute mesh faces for efficiency
    mesh_xyz = np.zeros((len(faces), 3, 3))
    for i in range(3):
        mesh_xyz[:, i, :] = vertices[faces[:, i]]

    # Get mesh bounds
    mesh_min = np.min(vertices, axis=0) - 1e-12
    mesh_max = np.max(vertices, axis=0) + 1e-12

    # Main loop for multiple tests with random rotations
    for n in range(heavytest + 1):
        # Apply random rotation if not first test
        if n > 0:
            v = np.random.rand(3)
            v = v / np.linalg.norm(v)
            angle = np.random.rand() * np.pi
            D = rotation_matrix(v, angle)
            vertices = VER @ D
            testp = TESTP @ D
            # Update mesh_xyz after rotation
            for i in range(3):
                mesh_xyz[:, i, :] = vertices[faces[:, i]]
        
        # Quick bounds check
        valid_points = np.all((testp >= mesh_min) & (testp <= mesh_max), axis=1)
        if not np.any(valid_points):
            continue

        # Process each coordinate direction
        for axis in range(3):
            # Get indices of remaining points to test
            uncertain_points = valid_points & (in_return == 0)
            if not np.any(uncertain_points):
                break

            # Roll coordinates for this axis test
            coords = np.roll(testp[uncertain_points], axis, axis=1)
            mesh_coords = np.roll(mesh_xyz, axis, axis=2)
            
            # Find intersections for each point
            intersections = np.zeros(np.sum(uncertain_points), dtype=bool)
            
            for i, point in enumerate(coords):
                # Quick triangle filtering
                xmin = np.min(mesh_coords[:, :, 0], axis=1)
                xmax = np.max(mesh_coords[:, :, 0], axis=1)
                ymin = np.min(mesh_coords[:, :, 1], axis=1)
                ymax = np.max(mesh_coords[:, :, 1], axis=1)
                
                possible_tris = (xmin <= point[0]) & (xmax >= point[0]) & \
                               (ymin <= point[1]) & (ymax >= point[1])
                
                if not np.any(possible_tris):
                    continue

                tris = mesh_coords[possible_tris]
                cross_count = 0

                # Test each potential triangle
                for tri in tris:
                    # Edge 1
                    y_pred = tri[1, 1] - ((tri[1, 1] - tri[2, 1]) * 
                             (tri[1, 0] - point[0]) / (tri[1, 0] - tri[2, 0] + np.finfo(float).eps))
                    if ((y_pred > tri[0, 1] and point[1] > tri[0, 1]) or 
                        (y_pred < tri[0, 1] and point[1] < tri[0, 1])):
                        # Edge 2
                        y_pred = tri[2, 1] - ((tri[2, 1] - tri[0, 1]) * 
                                 (tri[2, 0] - point[0]) / (tri[2, 0] - tri[0, 0] + np.finfo(float).eps))
                        if ((y_pred > tri[1, 1] and point[1] > tri[1, 1]) or 
                            (y_pred < tri[1, 1] and point[1] < tri[1, 1])):
                            # Edge 3
                            y_pred = tri[0, 1] - ((tri[0, 1] - tri[1, 1]) * 
                                     (tri[0, 0] - point[0]) / (tri[0, 0] - tri[1, 0] + np.finfo(float).eps))
                            if ((y_pred > tri[2, 1] and point[1] > tri[2, 1]) or 
                                (y_pred < tri[2, 1] and point[1] < tri[2, 1])):
                                cross_count += 1

                intersections[i] = (cross_count % 2) == 1

            # Update results for this axis test
            uncertain_indices = np.where(uncertain_points)[0]
            in_return[uncertain_indices[intersections]] = 1

        # Combine results from multiple rotations
        if n > 0:  # Additional tests
            if n == 1:  # First additional test
                base_results = in_return.copy()
            else:  # If point is inside in any test, consider it inside
                inside_any = (in_return == 1) | (base_results == 1)
                base_results[inside_any] = 1

    return in_return
