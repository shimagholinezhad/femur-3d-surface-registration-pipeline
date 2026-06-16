"""Functions used to identify the proximal and distal femur ends.

"""

from __future__ import annotations

import numpy as np
import pandas as pd
import trimesh as tri

import meshtools as mt


def compute_end_params(mesh: tri.Trimesh, femur: str = "") -> pd.DataFrame:
    """Compute shape descriptors for the two ends of a femur mesh.

    The femur is split along its long axis after principal-axis alignment.
    The returned rows correspond to the two ends. A pre-trained orientation
    vector is then used elsewhere to decide whether the mesh needs to be
    flipped so the proximal end is consistently oriented.

    Parameters
    ----------
    mesh:
        Watertight femur surface mesh.
    femur:
        Optional row-name prefix.

    Returns
    -------
    pandas.DataFrame
        Rows ``<femur>1`` and ``<femur>2`` with columns:
        I2, I3, I4, H0, H1, H2, E0, E1, E2.
    """
    if not mesh.is_watertight:
        raise ValueError("compute_end_params requires a watertight mesh")

    integral_moments = [2, 3, 4]
    hu_moments = [0, 1, 2]
    inertia_moments = [0, 1, 2]

    columns = [f"I{i}" for i in integral_moments]
    columns += [f"H{i}" for i in hu_moments]
    columns += [f"E{i}" for i in inertia_moments]

    index = [f"{femur}1", f"{femur}2"]
    params = pd.DataFrame(index=index, columns=columns, dtype=float)

    end1, end2 = mt.splitmesh(mesh, 0.33)

    for i in integral_moments:
        params.loc[index[0], f"I{i}"] = mt.nth_moment_about_center_of_mass_normalized(end1, i)
        params.loc[index[1], f"I{i}"] = mt.nth_moment_about_center_of_mass_normalized(end2, i)

    hu1 = mt.compute_hu_moments_3d(end1)
    hu2 = mt.compute_hu_moments_3d(end2)
    for i in hu_moments:
        params.loc[index[0], f"H{i}"] = hu1[i]
        params.loc[index[1], f"H{i}"] = hu2[i]

    inertia1 = mt.compute_normalized_inertia_tensor_eigenvalues(end1)
    inertia2 = mt.compute_normalized_inertia_tensor_eigenvalues(end2)
    for i in inertia_moments:
        params.loc[index[0], f"E{i}"] = inertia1[i]
        params.loc[index[1], f"E{i}"] = inertia2[i]

    return params
