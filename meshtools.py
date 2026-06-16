"""Small mesh helper module used by the example 3D femur pipeline.

This file contains the functions needed to run the manuscript/example
workflow.
"""

from __future__ import annotations

import numpy as np
import trimesh as tri
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


def splitmesh(mesh: tri.Trimesh, alpha: float = 0.0, cap: bool = True) -> tuple[tri.Trimesh, tri.Trimesh]:
    """Split a mesh into two end regions along the z axis.

    alpha controls how much of the middle part is skipped. alpha=0 splits at
    the middle; alpha=0.5 gives the outer quarter regions.
    """
    min_z = mesh.bounds[0, 2]
    max_z = mesh.bounds[1, 2]
    mid_z = 0.5 * (min_z + max_z)
    length_z = max_z - min_z

    plane1_origin = [0, 0, mid_z + length_z * alpha / 2.0]
    plane2_origin = [0, 0, mid_z - length_z * alpha / 2.0]
    normal = np.array([0, 0, 1], dtype=float)

    side1 = mesh.slice_plane(plane1_origin, normal, cap=cap)
    side2 = mesh.slice_plane(plane2_origin, -normal, cap=cap)
    return side1, side2


def savemesh(mesh: tri.Trimesh, filename: str) -> None:
    """Save a mesh without changing its geometry."""
    mesh.export(filename)


def align_principal_axes(mesh: tri.Trimesh) -> tri.Trimesh:
    """Align a mesh with its principal inertia axes and centre of mass."""
    mesh.apply_transform(mesh.principal_inertia_transform)
    mesh.apply_transform(tri.transformations.translation_matrix(-mesh.center_mass))
    return mesh


def translation_matrix(translation: np.ndarray | list[float]) -> np.ndarray:
    """Return a 4x4 homogeneous translation matrix."""
    matrix = np.eye(4)
    matrix[:3, 3] = np.asarray(translation, dtype=float)
    return matrix


def decompose_affine_transformation(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split an affine transform into rigid and non-rigid parts."""
    linear = matrix[:3, :3]
    translation = matrix[:3, 3]
    u, _, vt = np.linalg.svd(linear)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt

    rigid = np.eye(4)
    rigid[:3, :3] = rotation
    rigid[:3, 3] = translation

    soft = np.eye(4)
    soft[:3, :3] = np.linalg.inv(rotation) @ linear
    return rigid, soft


def decompose_affine_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decompose a 4x4 affine matrix into scale, rotation vector, shear, translation."""
    translation = matrix[:3, 3].copy()
    linear = matrix[:3, :3].copy()

    q, r = np.linalg.qr(linear)
    for i in range(3):
        if r[i, i] < 0:
            r[i, :] *= -1
            q[:, i] *= -1
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1
        r[-1, :] *= -1

    scaling = np.array([r[0, 0], r[1, 1], r[2, 2]], dtype=float)
    eps = 1e-15
    safe_scale_y = scaling[1] if abs(scaling[1]) > eps else eps
    safe_scale_z = scaling[2] if abs(scaling[2]) > eps else eps

    shear = np.array(
        [r[0, 1] / safe_scale_y, r[0, 2] / safe_scale_z, r[1, 2] / safe_scale_z],
        dtype=float,
    )
    rotation_vector = Rotation.from_matrix(q).as_rotvec()
    return scaling, rotation_vector, shear, translation


def recompose_affine_matrix(
    scaling: np.ndarray | list[float],
    rotation_vector: np.ndarray | list[float],
    shearing_vector: np.ndarray | list[float],
    translation: np.ndarray | list[float],
) -> np.ndarray:
    """Rebuild a 4x4 affine matrix from scale, rotation vector, shear, translation."""
    scaling = np.asarray(scaling, dtype=float)
    rotation_vector = np.asarray(rotation_vector, dtype=float)
    shearing_vector = np.asarray(shearing_vector, dtype=float)
    translation = np.asarray(translation, dtype=float)

    rotation = Rotation.from_rotvec(rotation_vector).as_matrix()
    shear = np.eye(3)
    shear[0, 1] = shearing_vector[0] * scaling[1]
    shear[0, 2] = shearing_vector[1] * scaling[2]
    shear[1, 2] = shearing_vector[2] * scaling[2]

    matrix = np.eye(4)
    matrix[:3, :3] = rotation @ (np.diag(scaling) @ shear)
    matrix[:3, 3] = translation
    return matrix


def coordinate_rotation_vectors(v1: np.ndarray, v2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Choose equivalent rotation-vector branches that are close to each other."""
    v1 = np.asarray(v1, dtype=float).copy()
    v2 = np.asarray(v2, dtype=float).copy()
    eps = 1e-12

    def shortest(v: np.ndarray) -> np.ndarray:
        angle = np.linalg.norm(v)
        if angle < eps:
            return np.zeros(3, dtype=float)
        if angle > np.pi:
            v = v - 2.0 * np.pi * v / angle
        return v

    v1 = shortest(v1)
    v2 = shortest(v2)
    angle1 = np.linalg.norm(v1)
    angle2 = np.linalg.norm(v2)

    if angle1 < eps and angle2 < eps:
        return np.zeros(3), np.zeros(3)
    if angle1 < eps:
        return np.zeros(3), v2
    if angle2 < eps:
        return v1, np.zeros(3)

    step1 = 2.0 * np.pi * v1 / angle1
    step2 = 2.0 * np.pi * v2 / angle2
    base1 = v1 - step1
    base2 = v2 - step2

    distances = np.zeros((3, 3), dtype=float)
    for i in range(3):
        candidate1 = base1 + i * step1
        for j in range(3):
            candidate2 = base2 + j * step2
            distances[i, j] = np.linalg.norm(candidate1 - candidate2)

    i_min, j_min = np.unravel_index(np.argmin(distances), distances.shape)
    return base1 + i_min * step1, base2 + j_min * step2


def interpolate_rotation_vectors(rotvec1: np.ndarray, rotvec2: np.ndarray, alpha: float) -> np.ndarray:
    """Linearly interpolate two branch-coordinated rotation vectors."""
    u1, u2 = coordinate_rotation_vectors(rotvec1, rotvec2)
    return (1.0 - alpha) * u1 + alpha * u2


def interpolate_affine_matrices(matrix1: np.ndarray, matrix2: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate two affine transforms component-wise."""
    s1, r1, h1, t1 = decompose_affine_matrix(matrix1)
    s2, r2, h2, t2 = decompose_affine_matrix(matrix2)

    scaling = (1.0 - alpha) * s1 + alpha * s2
    rotation = interpolate_rotation_vectors(r1, r2, alpha)
    shear = (1.0 - alpha) * h1 + alpha * h2
    translation = (1.0 - alpha) * t1 + alpha * t2
    return recompose_affine_matrix(scaling, rotation, shear, translation)


def registration_mesh_other(*args, **kwargs):
    """Wrapper around trimesh.registration.mesh_other."""
    return tri.registration.mesh_other(*args, **kwargs)


def registration_icp(*args, **kwargs):
    """Wrapper around trimesh.registration.icp."""
    result = tri.registration.icp(*args, **kwargs)
    if isinstance(result, np.ndarray):
        return result,
    if isinstance(result, (tuple, list)) and len(result) >= 1:
        return result
    raise ValueError("trimesh.registration.icp failed")


def squared_sum_of_distances(mesh1: tri.Trimesh, mesh2: tri.Trimesh) -> float:
    """Squared distance sum between corresponding vertices of two equal-topology meshes."""
    if mesh1.vertices.shape != mesh2.vertices.shape:
        raise ValueError("Meshes must have the same number of vertices")
    diff = mesh1.vertices - mesh2.vertices
    return float(np.sum(diff * diff))


def apply_affine_transformation(
    mesh: tri.Trimesh,
    scaling: np.ndarray,
    rotation_vector: np.ndarray,
    shearing_vector: np.ndarray,
    translation: np.ndarray,
) -> tri.Trimesh:
    """Apply an affine transform described by components to a mesh copy."""
    matrix = recompose_affine_matrix(scaling, rotation_vector, shearing_vector, translation)
    transformed = mesh.copy()
    transformed.apply_transform(matrix)
    return transformed


def _objective_function(params: np.ndarray, source: tri.Trimesh, target: tri.Trimesh) -> float:
    transformed = apply_affine_transformation(
        source,
        params[0:3],
        params[3:6],
        params[6:9],
        params[9:12],
    )
    return squared_sum_of_distances(transformed, target)


def optimize_affine_transformation(source: tri.Trimesh, target: tri.Trimesh):
    """Fit an affine transform from source to a target mesh with matching topology."""
    if source.vertices.shape != target.vertices.shape:
        raise ValueError("Meshes must have the same number of vertices")
    initial = np.zeros(12, dtype=float)
    initial[:3] = 1.0
    return minimize(_objective_function, initial, args=(source, target), method="L-BFGS-B")


def compute_normalized_inertia_tensor_eigenvalues(mesh: tri.Trimesh) -> np.ndarray:
    """Compute scale-normalized principal inertia components."""
    if not mesh.is_watertight:
        raise ValueError("Mesh must be watertight to compute inertia tensor")
    return mesh.principal_inertia_components / (mesh.mass ** (5.0 / 3.0))


def nth_moment_about_center_of_mass(mesh: tri.Trimesh, order: int) -> float:
    """Compute a scalar moment about the centre of mass using surface triangles."""
    if not mesh.is_watertight:
        raise ValueError("Mesh must be watertight to compute volume moments")
    if order < 1:
        raise ValueError("Moment order must be at least 1")

    centre = mesh.center_mass
    moment = 0.0
    for face in mesh.faces:
        triangle = mesh.vertices[face]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        area = np.linalg.norm(normal) / 2.0
        if area == 0:
            continue
        unit_normal = normal / np.linalg.norm(normal)
        r = triangle.mean(axis=0) - centre
        vector_field = r * (np.linalg.norm(r) ** (order - 1))
        moment += np.dot(vector_field, unit_normal) * area
    return float(moment / (3.0 * order))


def nth_moment_about_center_of_mass_normalized(mesh: tri.Trimesh, order: int) -> float:
    """Scale-normalized scalar moment about the centre of mass."""
    return nth_moment_about_center_of_mass(mesh, order) / (mesh.volume ** (order / 3.0 + 1.0))


def compute_3d_central_moments(mesh: tri.Trimesh, order: int = 3) -> np.ndarray:
    """Compute central moments up to the requested order."""
    centre = mesh.center_mass
    moments = np.zeros((order + 1, order + 1, order + 1), dtype=float)

    for face in mesh.faces:
        triangle = mesh.vertices[face]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        area = np.linalg.norm(normal) / 2.0
        if area == 0:
            continue
        normal = normal / np.linalg.norm(normal)
        x, y, z = triangle.mean(axis=0) - centre

        vector_field = np.zeros(3, dtype=float)
        for i in range(order + 1):
            for j in range(order + 1):
                for k in range(order + 1):
                    vector_field[0] = (x ** (i + 1)) * (y ** j) * (z ** k) / (i + 1)
                    vector_field[1] = (x ** i) * (y ** (j + 1)) * (z ** k) / (j + 1)
                    vector_field[2] = (x ** i) * (y ** j) * (z ** (k + 1)) / (k + 1)
                    moments[i, j, k] += np.dot(vector_field, normal) * area
    return moments


def normalize_moments(central_moments: np.ndarray) -> np.ndarray:
    """Normalize 3D central moments for scale."""
    mu000 = central_moments[0, 0, 0]
    order = central_moments.shape[0] - 1
    normalized = np.zeros_like(central_moments)

    for i in range(order + 1):
        for j in range(order + 1):
            for k in range(order + 1):
                power = 1.0 + (i + j + k) / 3.0
                normalized[i, j, k] = central_moments[i, j, k] / (mu000 ** power)
    return normalized


def compute_hu_moments_3d(mesh: tri.Trimesh) -> np.ndarray:
    """Compute the three Hu-like 3D invariants used by the orientation step."""
    normalized = normalize_moments(compute_3d_central_moments(mesh, order=3))
    mu200 = normalized[2, 0, 0]
    mu020 = normalized[0, 2, 0]
    mu002 = normalized[0, 0, 2]
    mu110 = normalized[1, 1, 0]
    mu101 = normalized[1, 0, 1]
    mu011 = normalized[0, 1, 1]
    mu300 = normalized[3, 0, 0]
    mu030 = normalized[0, 3, 0]
    mu003 = normalized[0, 0, 3]

    invariant1 = mu200 + mu020 + mu002
    invariant2 = mu200 * mu020 + mu020 * mu002 + mu002 * mu200 - mu110**2 - mu101**2 - mu011**2
    invariant3 = mu300**2 + mu030**2 + mu003**2
    return np.array([invariant1, invariant2, invariant3], dtype=float)
