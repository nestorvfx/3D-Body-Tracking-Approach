"""Dense surface-keypoint sampling for the MPFB basemesh.

The MPFB basemesh has a fixed vertex topology at subdiv_levels=0 (the
same vertex indices mean the same body location across every character
the pipeline ever renders).  CameraHMR (3DV 2025) showed that adding
100+ surface keypoints on top of the 17 COCO joints — distributed
uniformly across the mesh — supplies a stronger shape signal than
joints alone, and meaningfully improves 3D-pose body-shape
generalisation.

We pick 100 vertex indices with an even geometric spread by projecting
rest-pose vertex world positions onto axis buckets and greedy-
selecting the farthest-point-style subsample.  The indices are
DETERMINISTIC across runs because they're derived purely from the
canonical MPFB basemesh topology.

At training/label time the vertices are forward-projected through the
same camera the COCO joints use; the 2D + 3D coords are written into
labels.jsonl as ``surface_kps_2d`` and ``surface_kps_3d_cam``.
"""
from __future__ import annotations

import math
from typing import Sequence

import bpy          # type: ignore
import mathutils    # type: ignore


NUM_SURFACE_KPS: int = 100


def _find_basemesh_object(name_prefix: str) -> object | None:
    """Return the mesh object that is the MPFB body base (has many vertices
    and a name starting with the character's subject_xxxx prefix).  We
    pick the mesh with the LARGEST vertex count among the character's
    objects — heuristically the basemesh (~7k-14k verts depending on
    subdiv) vs clothes/hair (smaller).
    """
    candidates = [
        o for o in bpy.data.objects
        if o.type == "MESH" and o.name.startswith(name_prefix)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda o: len(o.data.vertices))


def select_surface_vertex_indices(mesh_obj, num: int = NUM_SURFACE_KPS,
                                   seed: int = 0) -> list[int]:
    """Return ``num`` vertex indices spread evenly over the mesh.

    IMPORTANT: indices are into the EVALUATED mesh (post-modifiers,
    including MPFB's subdiv surface).  At compute time we re-evaluate
    and index into the same evaluated mesh to avoid topology drift.

    Strategy: farthest-point subsample in rest-pose object space of
    the evaluated mesh.  Deterministic for given (mesh, num, seed).
    """
    import random as _random
    dg = bpy.context.evaluated_depsgraph_get()
    mesh_eval = mesh_obj.evaluated_get(dg)
    verts = mesh_eval.data.vertices
    if len(verts) == 0:
        return []
    num = min(num, len(verts))
    rng = _random.Random(seed)
    start = rng.randrange(len(verts))
    picked: list[int] = [start]
    # Cache vertex coords into tuples — accessing verts[i].co repeatedly
    # through the depsgraph wrapper is slow.
    coords = [(v.co.x, v.co.y, v.co.z) for v in verts]
    p0 = coords[start]
    min_dist2 = [(c[0] - p0[0]) ** 2 + (c[1] - p0[1]) ** 2
                   + (c[2] - p0[2]) ** 2 for c in coords]
    for _ in range(num - 1):
        best_i = 0
        best_d = -1.0
        for i, d in enumerate(min_dist2):
            if d > best_d:
                best_d = d
                best_i = i
        picked.append(best_i)
        p = coords[best_i]
        for i, c in enumerate(coords):
            d2 = (c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2 + (c[2] - p[2]) ** 2
            if d2 < min_dist2[i]:
                min_dist2[i] = d2
    return picked


def compute_surface_kps(mesh_obj, vertex_indices: Sequence[int],
                         image_wh: tuple[int, int], camera,
                         scene) -> dict:
    """Forward-project a fixed set of surface vertices through the camera.

    Uses the evaluated mesh (so armature deformation / shape-key state is
    respected).  Returns:
      surface_kps_2d   [K, 3] — u px, v px, visibility (0 = outside image, 2 = inside)
      surface_kps_3d_cam  [K, 3] — camera-frame metric coords in metres

    Coordinate conventions match ``keypoints_2d`` / ``keypoints_3d_cam``
    in build_dataset.py's main record, so downstream loaders can reuse
    the same unprojection helpers.
    """
    from bpy_extras.object_utils import world_to_camera_view
    W, H = image_wh
    if mesh_obj is None or not vertex_indices:
        return {"surface_kps_2d": [], "surface_kps_3d_cam": []}

    # Evaluate the deformed mesh so surface points track the animated
    # armature + shape keys.  This is the same pattern the COCO keypoint
    # forward-projection uses for posed joints.
    dg = bpy.context.evaluated_depsgraph_get()
    mw = mesh_obj.matrix_world
    mesh_eval = mesh_obj.evaluated_get(dg)
    verts = mesh_eval.data.vertices

    # Extrinsics matrices for world -> camera_frame.
    cam_mw = camera.matrix_world
    R_w2c = cam_mw.to_3x3().inverted()
    t_w2c = -(R_w2c @ cam_mw.to_translation())
    # Flip Blender's RHS camera convention (camera looks down -Z, Y up) to
    # the OpenCV convention (camera looks +Z, Y down) — same as the
    # existing compute_extrinsics helper used for COCO keypoints.
    flip = mathutils.Matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    R_cv = flip @ R_w2c
    t_cv = flip @ t_w2c

    kps_2d: list[list[float]] = []
    kps_3d: list[list[float]] = []
    for vi in vertex_indices:
        if vi < 0 or vi >= len(verts):
            kps_2d.append([0.0, 0.0, 0])
            kps_3d.append([0.0, 0.0, 0.0])
            continue
        v_world = mw @ verts[vi].co
        # 2D projection.
        ndc = world_to_camera_view(scene, camera, v_world)
        u = ndc.x * W
        v = (1.0 - ndc.y) * H      # flip Y to image (top-left origin)
        inside = ndc.z > 0.0 and 0.0 <= ndc.x <= 1.0 and 0.0 <= ndc.y <= 1.0
        vis = 2 if inside else 1
        kps_2d.append([float(u), float(v), vis])
        # 3D cam-frame.
        p_cam = R_cv @ v_world + t_cv
        kps_3d.append([float(p_cam.x), float(p_cam.y), float(p_cam.z)])
    return {"surface_kps_2d": kps_2d, "surface_kps_3d_cam": kps_3d}


__all__ = [
    "NUM_SURFACE_KPS",
    "select_surface_vertex_indices",
    "compute_surface_kps",
    "_find_basemesh_object",
]
