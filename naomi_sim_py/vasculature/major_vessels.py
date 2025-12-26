"""
Major vessels generation for blood vessel simulation.

This module implements the growth of major blood vessels including:
- Source nodes placement
- Vertical vessel growth
- Surface vessel growth
- Node branching logic

Based on MATLAB growMajorVessels.m implementation.
"""

import numpy as np
from typing import List, Tuple, Optional
from .data_structures import VascNode, VascNetwork, NodeParams, VascParams, create_vasc_node


def gen_node(num: Optional[int] = None,
             root: Optional[int] = None,
             conn: Optional[List[int]] = None,
             pos: Optional[np.ndarray] = None,
             node_type: Optional[str] = None,
             misc: Optional[np.ndarray] = None) -> VascNode:
    """
    Create a vascular node (equivalent to MATLAB gennode function).

    Args:
        num: Node identifying number
        root: Root node index (0 for no root)
        conn: List of connected node indices
        pos: 3D position [x, y, z]
        node_type: Node type ('edge', 'surf', 'vert', 'sfvt', 'capp')
        misc: Miscellaneous parameters

    Returns:
        New VascNode instance
    """
    return create_vasc_node(
        num=num if num is not None else 0,
        root=root if root is not None else 0,
        conn=conn if conn is not None else [],
        pos=pos if pos is not None else np.zeros(3),
        node_type=node_type if node_type is not None else '',
        misc=misc if misc is not None else np.array([])
    )


def grow_major_vessels(nv: VascNetwork, np_obj: NodeParams, vp: VascParams) -> Tuple[List[VascNode], VascNetwork]:
    """
    Grow major vessels (source, vertical, surface vessels).

    This function replicates the MATLAB growMajorVessels.m logic.

    Args:
        nv: Vascular network parameters
        np_obj: Node placement parameters
        vp: Vascular parameters

    Returns:
        Tuple of (nodes, updated_nv)
    """
    nodes: List[VascNode] = []

    # Step 1: Place source nodes at volume edges
    nodes, source_nodes = _place_source_nodes(nv, vp)

    # Step 2: Grow vertical vessels from source nodes
    nodes, vertical_nodes = _grow_vertical_vessels(nodes, source_nodes, nv, np_obj, vp)

    # Step 3: Grow surface vessels
    nodes, surface_nodes = _grow_surface_vessels(nodes, nv, vp)

    # Step 4: Convert surface nodes to sfvt nodes (MATLAB logic)
    # Convert single-connection surface nodes to sfvt (surface-to-vertical)
    neur_vert = np.zeros(nv.size[:2], dtype=bool)

    # First pass: convert surface nodes with single connections to sfvt
    for i in range(len(nodes)):
        if (hasattr(nodes[i], 'type') and nodes[i].type == 'surf' and
            len(nodes[i].conn) == 1):
            # Check if position is not occupied in neur_vert
            x, y = nodes[i].pos[0] - 1, nodes[i].pos[1] - 1  # Convert to 0-based
            if (0 <= x < neur_vert.shape[0] and 0 <= y < neur_vert.shape[1] and
                not neur_vert[x, y]):
                nodes[i].type = 'sfvt'
                # Mark as occupied (simplified dilation)
                neur_vert[x, y] = True

    # Second pass: ensure we have enough sfvt nodes (MATLAB: while loop)
    sfvt_count = sum(1 for node in nodes if hasattr(node, 'type') and node.type == 'sfvt')
    surf_nodes_available = [i for i, node in enumerate(nodes)
                           if (hasattr(node, 'type') and node.type == 'surf' and
                               not neur_vert[node.pos[0]-1, node.pos[1]-1])]

    while sfvt_count < nv.nvert and surf_nodes_available:
        # Randomly select from available surface nodes
        idx = np.random.choice(surf_nodes_available)
        nodes[idx].type = 'sfvt'

        # Mark as occupied
        x, y = nodes[idx].pos[0] - 1, nodes[idx].pos[1] - 1
        neur_vert[x, y] = True

        # Remove from available list
        surf_nodes_available.remove(idx)
        sfvt_count += 1

    # Step 5: Grow diving vessels from sfvt nodes to bottom
    vert_nodes_created = 0
    sfvt_indices = [i for i, node in enumerate(nodes) if hasattr(node, 'type') and node.type == 'sfvt']

    for sfvt_idx in sfvt_indices:
        sfvt_node = nodes[sfvt_idx]
        current_pos = sfvt_node.pos.copy()

        # Grow vertical vessel downward to bottom of volume
        while current_pos[2] < nv.size[2]:
            # Create new vertical node
            current_pos[2] = min(current_pos[2] + vp.mindists[2], nv.size[2])

            new_node = gen_node(
                num=len(nodes) + 1,
                root=sfvt_node.num,
                conn=[len(nodes)],  # Connect to previous node
                pos=current_pos.copy().astype(int),
                node_type='vert'
            )

            nodes.append(new_node)
            vert_nodes_created += 1

            # Update the connection of the previous node
            if len(nodes) > 1:
                prev_node = nodes[-2]  # Second to last (the one we just connected to)
                prev_node.conn.append(new_node.num)

            # Stop if we reached the bottom
            if current_pos[2] >= nv.size[2]:
                break

    # Update network counters
    nv.nvert = len(sfvt_indices)
    nv.nvertconn = vert_nodes_created
    nv.nnodes = len(nodes)

    # Update network parameters
    nv.nlinks = len(nodes) - nv.nsource

    return nodes, nv


def _place_source_nodes(nv: VascNetwork, vp: VascParams) -> Tuple[List[VascNode], List[VascNode]]:
    """
    Place source nodes at volume edges.

    Args:
        nv: Vascular network parameters
        vp: Vascular parameters

    Returns:
        Tuple of (all_nodes, source_nodes)
    """
    nodes: List[VascNode] = []
    source_nodes: List[VascNode] = []

    for i in range(nv.nsource):
        # Randomly decide which edges to place the source node
        # Based on MATLAB logic: TMPIDX = (rand(1,2)>=[ratio 0.5])
        ratio = nv.size[1] / (nv.size[0] + nv.size[1])  # x vs y+ x ratio
        rand_vals = np.random.rand(2)
        tmpidx = np.array([rand_vals[0] >= ratio, rand_vals[1] >= 0.5])

        # Determine position based on edge selection
        if tmpidx[0] and tmpidx[1]:  # Top edge
            pos = np.array([
                np.ceil(nv.size[0] * np.random.rand()),
                1,
                vp.depth_surf
            ])
        elif tmpidx[0] and not tmpidx[1]:  # Bottom edge
            pos = np.array([
                np.ceil(nv.size[0] * np.random.rand()),
                nv.size[1],
                vp.depth_surf
            ])
        elif not tmpidx[0] and tmpidx[1]:  # Left edge
            pos = np.array([
                1,
                np.ceil(nv.size[1] * np.random.rand()),
                vp.depth_surf
            ])
        else:  # Right edge
            pos = np.array([
                nv.size[0],
                np.ceil(nv.size[1] * np.random.rand()),
                vp.depth_surf
            ])

        # Create source node
        node = gen_node(
            num=i+1,
            root=0,
            conn=[],
            pos=pos.astype(int),
            node_type='edge',
            misc=tmpidx.astype(int)
        )

        nodes.append(node)
        source_nodes.append(node)

    return nodes, source_nodes


def _grow_vertical_vessels(nodes: List[VascNode], source_nodes: List[VascNode],
                          nv: VascNetwork, np_obj: NodeParams, vp: VascParams) -> Tuple[List[VascNode], List[VascNode]]:
    """
    Grow vertical vessels from source nodes.

    Args:
        nodes: Current node list
        source_nodes: List of source nodes
        nv: Vascular network parameters
        np_obj: Node placement parameters
        vp: Vascular parameters

    Returns:
        Tuple of (updated_nodes, vertical_nodes)
    """
    vertical_nodes: List[VascNode] = []

    # Initialize surface mask for branching logic
    neur_surf = np.zeros(nv.size[:2], dtype=bool)

    for i, source_node in enumerate(source_nodes):
        # Determine branching direction based on edge position
        misc = source_node.misc
        if misc[0] and misc[1]:  # Top edge
            rand_dir = 0.5 * np.pi + np.random.randn() * np_obj.dirvar
        elif misc[0] and not misc[1]:  # Bottom edge
            rand_dir = 1.5 * np.pi + np.random.randn() * np_obj.dirvar
        elif not misc[0] and misc[1]:  # Left edge
            rand_dir = 0.0 * np.pi + np.random.randn() * np_obj.dirvar
        else:  # Right edge
            rand_dir = 1.0 * np.pi + np.random.randn() * np_obj.dirvar

        # Grow branches from this source node
        nodes, neur_surf = _branch_grow_nodes(nodes, neur_surf, np_obj, i, rand_dir)

    vertical_nodes = nodes[len(source_nodes):]  # Nodes added after source nodes

    # Round positions for vertical nodes
    for node in vertical_nodes:
        node.pos = np.round(node.pos).astype(int)
        node.pos[2] = vp.depth_surf  # Ensure z-coordinate is at surface

    return nodes, vertical_nodes


def _grow_surface_vessels(nodes: List[VascNode], nv: VascNetwork, vp: VascParams) -> Tuple[List[VascNode], List[VascNode]]:
    """
    Grow surface vessels across the volume surface.

    Args:
        nodes: Current node list (including source and vertical nodes)
        nv: Vascular network parameters
        vp: Vascular parameters

    Returns:
        Tuple of (updated_nodes, surface_nodes)
    """
    surface_nodes: List[VascNode] = []

    # TODO: Implement surface vessel growth
    # This involves:
    # 1. Pseudo-random sampling of surface positions
    # 2. Dijkstra path finding for connections
    # 3. Creating surface vessel nodes

    # Placeholder implementation - create some dummy surface nodes
    for i in range(nv.nsurf):
        # Random position on surface
        pos = np.array([
            np.random.randint(1, nv.size[0] + 1),
            np.random.randint(1, nv.size[1] + 1),
            vp.depth_surf
        ])

        node = gen_node(
            num=len(nodes) + i + 1,
            root=1,  # Connect to first source node (placeholder)
            conn=[1],
            pos=pos.astype(int),
            node_type='surf'
        )

        nodes.append(node)
        surface_nodes.append(node)

    return nodes, surface_nodes


def _branch_grow_nodes(nodes: List[VascNode], neur_surf: np.ndarray,
                      np_obj: NodeParams, source_idx: int, rand_dir: float) -> Tuple[List[VascNode], np.ndarray]:
    """
    Grow branches from a source node (simplified implementation).

    Args:
        nodes: Current node list
        neur_surf: Surface mask for branching
        np_obj: Node placement parameters
        source_idx: Index of source node in nodes list
        rand_dir: Random direction for branching

    Returns:
        Tuple of (updated_nodes, updated_neur_surf)
    """
    # TODO: Implement proper branch growing logic
    # This should create a tree structure of connected nodes

    # Placeholder: just add a few connected nodes
    source_node = nodes[source_idx]
    current_pos = source_node.pos.copy()

    # Create a few connected nodes in the branching direction
    for i in range(3):  # Create 3 connected nodes
        # Move in the random direction
        dx = np.cos(rand_dir) * np_obj.lensc
        dy = np.sin(rand_dir) * np_obj.lensc

        current_pos[0] += dx
        current_pos[1] += dy

        # Keep within bounds
        current_pos[0] = np.clip(current_pos[0], 1, neur_surf.shape[0])
        current_pos[1] = np.clip(current_pos[1], 1, neur_surf.shape[1])

        # Create new node
        new_node = gen_node(
            num=len(nodes) + 1,
            root=source_node.num,
            conn=[source_node.num],
            pos=current_pos.copy().astype(int),
            node_type='vert'
        )

        # Update source node's connections
        source_node.conn.append(new_node.num)

        nodes.append(new_node)

        # Update surface mask (simplified)
        x, y = int(current_pos[0] - 1), int(current_pos[1] - 1)  # Convert to 0-based indexing
        if 0 <= x < neur_surf.shape[0] and 0 <= y < neur_surf.shape[1]:
            neur_surf[x, y] = True

    return nodes, neur_surf


# TODO: Implement additional helper functions:
# - pseudoRandSample2D (for surface vessel positioning)
# - vessel_dijkstra (for path finding)
# - pos2dists (distance calculations)
# - imdilate equivalent for surface dilation
