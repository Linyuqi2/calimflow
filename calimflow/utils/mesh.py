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

def in_triangulation(
    vertices: np.ndarray,
    faces: np.ndarray,
    test_points: np.ndarray,
    heavy_test: int = 0
) -> np.ndarray:
    """
    Test whether points are inside or outside a closed triangulation.
    Based on MATLAB's intriangulation function.
    """
    # Input validation
    if vertices.shape[1] != 3 or faces.shape[1] != 3 or test_points.shape[1] != 3:
        raise ValueError("All input matrices must have three columns")
        
    if np.max(faces) >= len(vertices) or np.min(faces) < 0:
        raise ValueError("Face indices are invalid")
    
    # Initialize results
    in_return = np.zeros(len(test_points), dtype=np.int8)
    orig_vertices = vertices.copy()
    orig_test_points = test_points.copy()
    
    # Perform multiple tests with random rotations
    for n in range(heavy_test + 1):
        if n > 0:
            # Random rotation for additional tests
            v = np.random.rand(3)
            v = v / np.linalg.norm(v)
            angle = np.random.rand() * np.pi
            D = rotation_matrix(v, angle)
            vertices = orig_vertices @ D
            test_points = orig_test_points @ D
        
        # Get mesh coordinates for each triangle
        mesh_xyz = np.zeros((len(faces), 3, 3))
        for i in range(3):
            mesh_xyz[:, :, i] = vertices[faces[:, i]]
        
        # Test in z-direction
        in_points, correction_list = voxelize_internal(
            test_points[:, 0],
            test_points[:, 1],
            test_points[:, 2],
            mesh_xyz
        )
        
        # Test uncertain points in x-direction
        if len(correction_list) > 0:
            mesh_xyz_x = np.roll(mesh_xyz, 1, axis=1)
            in_points_x, correction_list_x = voxelize_internal(
                test_points[correction_list, 1],
                test_points[correction_list, 2],
                test_points[correction_list, 0],
                mesh_xyz_x
            )
            in_return[correction_list[in_points_x]] = 1
            correction_list = correction_list[correction_list_x]
        
        # Test remaining uncertain points in y-direction
        if len(correction_list) > 0:
            mesh_xyz_y = np.roll(mesh_xyz, 2, axis=1)
            in_points_y, correction_list_y = voxelize_internal(
                test_points[correction_list, 2],
                test_points[correction_list, 0],
                test_points[correction_list, 1],
                mesh_xyz_y
            )
            in_return[correction_list[in_points_y]] = 1
            correction_list = correction_list[correction_list_y]
        
        # Mark remaining uncertain points
        in_return[correction_list] = -1
        
        if n > 0:
            # If AT LEAST ONCE inside, use as inside
            mask = (in_return != in_points)
            in_return[mask & in_points] = 1
    
    return in_return
