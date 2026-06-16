#!/usr/bin/env python3
"""Run the 3D femur registration pipeline for one example sample.

Input files are four STL meshes:

    pre_left.stl
    post_left.stl
    pre_right.stl
    post_right.stl

The reference side is given on the command line.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import trimesh as tri
from scipy.spatial.transform import Rotation

import meshtools as mt
from bone_ends import compute_end_params


@dataclass(frozen=True)
class ModelSpec:
    """Description of one femur model in the single-sample dataset."""

    label: str
    stage: str  # "Pre" or "Post"
    side: str   # "L" or "R"
    filename: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run preregistration, distal/proximal registration, and summary for one femur sample."
    )
    parser.add_argument("--sample-id", default="example", help="Identifier written to output tables.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Folder containing the four STL files.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder where outputs will be written.")
    parser.add_argument(
        "--reference-side",
        required=True,
        choices=["L", "R", "l", "r"],
        help="Baseline non-operated/reference side for this sample (L or R).",
    )
    parser.add_argument(
        "--orientation-vector",
        required=True,
        type=Path,
        help="JSON file containing the proximal/distal orientation vector.",
    )
    parser.add_argument("--distal-size", type=float, default=0.25, help="Distal fraction of femur length to analyse.")
    parser.add_argument("--random-seed", type=int, default=42, help="Seed used before registration calls.")

    # Canonical public filenames. These can be overridden without editing code.
    parser.add_argument("--pre-left", default="pre_left.stl")
    parser.add_argument("--post-left", default="post_left.stl")
    parser.add_argument("--pre-right", default="pre_right.stl")
    parser.add_argument("--post-right", default="post_right.stl")
    return parser.parse_args()


def model_specs(args: argparse.Namespace) -> list[ModelSpec]:
    return [
        ModelSpec("Pre_L", "Pre", "L", args.pre_left),
        ModelSpec("Post_L", "Post", "L", args.post_left),
        ModelSpec("Pre_R", "Pre", "R", args.pre_right),
        ModelSpec("Post_R", "Post", "R", args.post_right),
    ]


def _as_trimesh(loaded, path: Path) -> tri.Trimesh:
    """Return a Trimesh, also when trimesh reads a file as a Scene."""
    if isinstance(loaded, tri.Scene):
        geometries = [g for g in loaded.geometry.values()]
        if not geometries:
            raise ValueError(f"No mesh geometry found in {path}")
        loaded = tri.util.concatenate(geometries)
    if not isinstance(loaded, tri.Trimesh):
        raise TypeError(f"Expected a Trimesh from {path}, got {type(loaded)!r}")
    return loaded


def load_mesh(path: Path) -> tri.Trimesh:
    """Load a mesh using trimesh processing, matching the original workflow."""
    if not path.exists():
        raise FileNotFoundError(f"Missing mesh file: {path}")

    # The original analysis used trimesh.load_mesh(...) with the default
    # processing enabled. Some STL exports have duplicate vertices or small
    # topological issues in the raw file; trimesh processing can merge and
    # clean these during loading before the aligned OBJ is written.
    raw_mesh = _as_trimesh(tri.load_mesh(path, process=False), path)
    mesh = _as_trimesh(tri.load_mesh(path, process=True), path)

    if not raw_mesh.is_watertight and mesh.is_watertight:
        print(f"[mesh note] {path.name}: raw file is not watertight, but trimesh processing made it watertight.")
    elif not raw_mesh.is_watertight:
        print(f"[mesh warning] {path.name}: raw file is not watertight and processing did not fully close it.")

    if not mesh.is_watertight:
        raise ValueError(
            f"Mesh is not watertight after trimesh processing: {path}. "
            "Please repair/export a closed surface before analysis."
        )
    return mesh


def load_orientation_vector(path: Path) -> pd.Series:
    """Load the JSON vector used to identify the proximal/distal end."""
    if path.suffix.lower() != ".json":
        raise ValueError("The public example expects the orientation vector as a JSON file.")

    if not path.exists():
        raise FileNotFoundError(f"Missing orientation vector: {path}")

    data = json.loads(path.read_text())
    vector = pd.Series(data, dtype=float)
    if vector.empty:
        raise ValueError("Orientation vector is empty.")
    return vector.astype(float)

def mirror_x(mesh: tri.Trimesh) -> None:
    """Mirror a left femur into the right-sided coordinate convention."""
    matrix = np.array(
        [
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    mesh.apply_transform(matrix)


def pre_align_femur(mesh: tri.Trimesh, direction: pd.Series) -> None:
    """Align the long axis and orient the proximal end consistently."""
    mt.align_principal_axes(mesh)

    params = compute_end_params(mesh)
    # E0 was found to be confounding in the original workflow and is absent
    # from the final orientation vector.
    params = params.drop(columns=["E0"], errors="ignore")
    params = params.reindex(columns=direction.index)

    if params.isna().any().any():
        missing = sorted(set(direction.index) - set(params.columns))
        raise ValueError(f"Could not compute all orientation features. Missing: {missing}")

    projection = params @ direction

    # If the first half projects above the second half, the femur is upside-down
    # relative to the trained anatomical convention; flip Y/Z to invert the long axis.
    if projection.iloc[0] > projection.iloc[1]:
        flip = np.array(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )
        mesh.apply_transform(flip)


def rigid_end_registration(
    base_mesh: tri.Trimesh,
    target_mesh: tri.Trimesh,
    direction: pd.Series,
    mirror_target: bool,
) -> tri.Trimesh:
    """Rigidly register a target femur to the pre-aligned base femur.

    The target is optionally mirrored into the right-sided coordinate convention,
    pre-aligned, split into proximal/distal halves, and registered to the base.
    The average rigid transformation from the two ends is applied to the whole
    target mesh and returned.
    """
    target = target_mesh.copy()
    if mirror_target:
        mirror_x(target)

    pre_align_femur(target, direction)

    base_prox, base_dist = mt.splitmesh(base_mesh)
    target_prox, target_dist = mt.splitmesh(target)

    a_prox, _ = mt.registration_mesh_other(target_prox, base_prox, scale=True, icp_first=100)
    a_prox_rigid, _ = mt.decompose_affine_transformation(a_prox)

    tmp_dist = target_dist.copy()
    tmp_dist.apply_transform(a_prox)
    initial_translation = mt.recompose_affine_matrix(
        [1, 1, 1], [0, 0, 0], [0, 0, 0], base_dist.center_mass - tmp_dist.center_mass
    )
    initial_dist = initial_translation @ a_prox

    a_dist_result = mt.registration_icp(target_dist.vertices, base_dist, scale=True, initial=initial_dist)
    a_dist = a_dist_result[0]
    a_dist_rigid, _ = mt.decompose_affine_transformation(a_dist)

    pre_transform = mt.interpolate_affine_matrices(a_prox_rigid, a_dist_rigid, 0.5)
    target.apply_transform(pre_transform)
    return target


def preregister_models(
    specs: Iterable[ModelSpec],
    input_dir: Path,
    aligned_dir: Path,
    reference_side: str,
    direction: pd.Series,
) -> dict[str, Path]:
    """Pre-align and rigidly register all four models to the Pre reference side."""
    aligned_dir.mkdir(parents=True, exist_ok=True)
    specs = list(specs)

    base_spec = next(s for s in specs if s.stage == "Pre" and s.side == reference_side)
    base_mesh = load_mesh(input_dir / base_spec.filename)
    pre_align_femur(base_mesh, direction)
    if base_spec.side == "L":
        mirror_x(base_mesh)

    aligned_paths: dict[str, Path] = {}
    base_out = aligned_dir / f"{base_spec.label}.obj"
    mt.savemesh(base_mesh, str(base_out))
    aligned_paths[base_spec.label] = base_out

    for spec in specs:
        if spec.label == base_spec.label:
            continue
        target = load_mesh(input_dir / spec.filename)
        registered = rigid_end_registration(
            base_mesh=base_mesh,
            target_mesh=target,
            direction=direction,
            mirror_target=(spec.side == "L"),
        )
        out_path = aligned_dir / f"{spec.label}.obj"
        mt.savemesh(registered, str(out_path))
        aligned_paths[spec.label] = out_path

    return aligned_paths


def save_split_caps(mesh: tri.Trimesh, out_dir: Path, model_label: str, distal_size: float) -> None:
    """Save the proximal and distal caps used in the registration."""
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha = max(0.0, min(1.0, 1.0 - 2.0 * float(distal_size)))
    prox, _ = mt.splitmesh(mesh, 0.5, True)
    _, dist = mt.splitmesh(mesh, alpha, True)
    tag = f"{int(round(distal_size * 100))}pct"
    mt.savemesh(prox, str(out_dir / f"{model_label}_prox_{tag}.obj"))
    mt.savemesh(dist, str(out_dir / f"{model_label}_dist_{tag}.obj"))


def end_registration(
    base_mesh: tri.Trimesh,
    target_mesh: tri.Trimesh,
    distal_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Register proximal and distal caps independently and return transforms."""
    base_prox, _ = mt.splitmesh(base_mesh, 0.5, True)
    target_prox, _ = mt.splitmesh(target_mesh, 0.5, True)

    alpha = max(0.0, min(1.0, 1.0 - 2.0 * float(distal_size)))
    _, base_dist = mt.splitmesh(base_mesh, alpha, True)
    _, target_dist = mt.splitmesh(target_mesh, alpha, True)

    # Proximal cap registration: ICP -> non-rigid ICP -> affine approximation.
    trans_prox = mt.translation_matrix(target_prox.center_mass - base_prox.center_mass)
    base_prox.apply_transform(trans_prox)
    icp_prox = tri.registration.icp(base_prox.vertices, target_prox, scale=True)
    base_prox.apply_transform(icp_prox[0])
    prox_vertices = tri.registration.nricp_amberg(base_prox, target_prox)
    prox_target = tri.Trimesh(vertices=prox_vertices, faces=base_prox.faces, process=False)
    prox_opt = mt.optimize_affine_transformation(base_prox, prox_target)
    prox_affine = mt.recompose_affine_matrix(
        prox_opt.x[0:3], prox_opt.x[3:6], prox_opt.x[6:9], prox_opt.x[9:12]
    )
    prox_transform = prox_affine @ icp_prox[0] @ trans_prox

    # Distal cap registration: same procedure.
    trans_dist = mt.translation_matrix(target_dist.center_mass - base_dist.center_mass)
    base_dist.apply_transform(trans_dist)
    icp_dist = tri.registration.icp(base_dist.vertices, target_dist, scale=True)
    base_dist.apply_transform(icp_dist[0])
    dist_vertices = tri.registration.nricp_amberg(base_dist, target_dist)
    dist_target = tri.Trimesh(vertices=dist_vertices, faces=base_dist.faces, process=False)
    dist_opt = mt.optimize_affine_transformation(base_dist, dist_target)
    dist_affine = mt.recompose_affine_matrix(
        dist_opt.x[0:3], dist_opt.x[3:6], dist_opt.x[6:9], dist_opt.x[9:12]
    )
    dist_transform = dist_affine @ icp_dist[0] @ trans_dist

    return prox_transform, dist_transform



def build_distal_sweep(
    specs: Iterable[ModelSpec],
    aligned_paths: dict[str, Path],
    split_dir: Path,
    sample_id: str,
    reference_side: str,
    distal_size: float,
) -> pd.DataFrame:
    """Run proximal/distal registration for all four aligned models."""
    specs = list(specs)
    base_spec = next(s for s in specs if s.stage == "Pre" and s.side == reference_side)
    base_mesh = load_mesh(aligned_paths[base_spec.label])

    rows: list[dict[str, float | str]] = []
    for spec in specs:
        target_mesh = load_mesh(aligned_paths[spec.label])
        save_split_caps(target_mesh, split_dir, spec.label, distal_size)

        if spec.label == base_spec.label:
            prox_transform = np.eye(4)
            dist_transform = np.eye(4)
        else:
            prox_transform, dist_transform = end_registration(base_mesh, target_mesh, distal_size)

        prox_scale, prox_rot, prox_shear, prox_trans = mt.decompose_affine_matrix(prox_transform)
        dist_scale, dist_rot, dist_shear, dist_trans = mt.decompose_affine_matrix(dist_transform)
        prox_rot, dist_rot = mt.coordinate_rotation_vectors(prox_rot, dist_rot)

        row = {
            "Sample": sample_id,
            "Model": spec.label,
            "Stage": spec.stage,
            "Side": spec.side,
            "Reference_side": reference_side,
            "dist_size": round(float(distal_size), 4),
            "ProxSx": prox_scale[0], "ProxSy": prox_scale[1], "ProxSz": prox_scale[2],
            "ProxRx": prox_rot[0], "ProxRy": prox_rot[1], "ProxRz": prox_rot[2],
            "ProxSH1": prox_shear[0], "ProxSH2": prox_shear[1], "ProxSH3": prox_shear[2],
            "ProxTx": prox_trans[0], "ProxTy": prox_trans[1], "ProxTz": prox_trans[2],
            "DistSx": dist_scale[0], "DistSy": dist_scale[1], "DistSz": dist_scale[2],
            "DistRx": dist_rot[0], "DistRy": dist_rot[1], "DistRz": dist_rot[2],
            "DistSH1": dist_shear[0], "DistSH2": dist_shear[1], "DistSH3": dist_shear[2],
            "DistTx": dist_trans[0], "DistTy": dist_trans[1], "DistTz": dist_trans[2],
        }
        rows.append(row)

    return pd.DataFrame(rows)


def _rotation_matrix_from_row(row: pd.Series, prefix: str) -> np.ndarray:
    """Build a rotation matrix from saved rotation-vector components."""
    vec = np.array(
        [row[f"{prefix}Rx"], row[f"{prefix}Ry"], row[f"{prefix}Rz"]],
        dtype=float,
    )
    return Rotation.from_rotvec(vec).as_matrix()


def _relative_rotation_delta(pre_row: pd.Series, post_row: pd.Series) -> np.ndarray:
    """Return the Post-Pre change in proximal/distal relative rotation.

    The proximal and distal caps are registered separately. The summary step
    compares their relative orientation at Pre and Post; this is where the
    proximal/distal reference correction is applied.
    """
    pre_prox = _rotation_matrix_from_row(pre_row, "Prox")
    pre_dist = _rotation_matrix_from_row(pre_row, "Dist")
    post_prox = _rotation_matrix_from_row(post_row, "Prox")
    post_dist = _rotation_matrix_from_row(post_row, "Dist")

    rel_pre = np.linalg.inv(pre_dist) @ pre_prox
    rel_post = np.linalg.inv(post_dist) @ post_prox
    delta = rel_post @ np.linalg.inv(rel_pre)
    return np.degrees(Rotation.from_matrix(delta).as_rotvec())

def summarize_sweep(
    sweep: pd.DataFrame,
    reference_side: str,
) -> pd.DataFrame:
    """Convert raw transforms into volume ratio and angular change."""
    df = sweep.copy()
    numeric_cols = [
        "ProxRx", "ProxRy", "ProxRz", "DistRx", "DistRy", "DistRz",
        "DistSx", "DistSy", "DistSz",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Base/reference rows are identity transforms. If an older table has blanks
    # for those rotation fields, treat them as exact zeros.
    rotation_cols = ["ProxRx", "ProxRy", "ProxRz", "DistRx", "DistRy", "DistRz"]
    blank_rotation_row = df[rotation_cols].isna().all(axis=1)
    df.loc[blank_rotation_row, rotation_cols] = 0.0

    df["VolumeScale"] = df["DistSx"] * df["DistSy"] * df["DistSz"]

    key_cols = ["Sample", "Side", "Reference_side", "dist_size"]
    rows: list[dict[str, float | str]] = []

    for key, group in df.groupby(key_cols, dropna=False):
        key_dict = dict(zip(key_cols, key))
        pre = group.loc[group["Stage"].astype(str).str.lower() == "pre"]
        post = group.loc[group["Stage"].astype(str).str.lower() == "post"]

        row: dict[str, float | str] = dict(key_dict)

        if pre.empty or post.empty:
            row.update({"Volume_Ratio": np.nan, "X_Delta": np.nan, "Y_Delta": np.nan, "Z_Delta": np.nan})
            rows.append(row)
            continue

        pre_row = pre.iloc[0]
        post_row = post.iloc[0]
        row["Volume_Ratio"] = post_row["VolumeScale"] / pre_row["VolumeScale"]

        delta = _relative_rotation_delta(pre_row, post_row)

        row["X_Delta"], row["Y_Delta"], row["Z_Delta"] = delta.tolist()
        rows.append(row)

    out = pd.DataFrame(rows)

    # Reporting convention used in the study: when a left reference was mirrored
    # to right-sided space, flip X and Y signs back. Z is kept as calculated.
    if reference_side.upper() == "L" and not out.empty:
        out["X_Delta"] = -out["X_Delta"]
        out["Y_Delta"] = -out["Y_Delta"]

    col_order = key_cols + ["Volume_Ratio", "X_Delta", "Y_Delta", "Z_Delta"]
    return out[col_order].sort_values(["Sample", "Side", "dist_size"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    np.random.seed(args.random_seed)

    reference_side = args.reference_side.upper()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir = args.output_dir / "aligned_obj"
    split_dir = args.output_dir / "split_meshes"
    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    specs = model_specs(args)
    direction = load_orientation_vector(args.orientation_vector)

    print("[1/3] Preregistering four STL models to the Pre reference side...")
    aligned_paths = preregister_models(specs, args.input_dir, aligned_dir, reference_side, direction)

    print("[2/3] Running proximal/distal registration for the distal sweep...")
    sweep = build_distal_sweep(
        specs=specs,
        aligned_paths=aligned_paths,
        split_dir=split_dir,
        sample_id=args.sample_id,
        reference_side=reference_side,
        distal_size=args.distal_size,
    )
    sweep_path = tables_dir / "distal_sweep.csv"
    sweep.to_csv(sweep_path, index=False)

    print("[3/3] Creating core 3D summary table...")
    summary = summarize_sweep(
        sweep,
        reference_side=reference_side,
    )
    summary_path = tables_dir / "core_3d_summary.csv"
    summary.to_csv(summary_path, index=False)

    settings_path = tables_dir / "run_settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "sample_id": args.sample_id,
                "reference_side": reference_side,
                "distal_size": args.distal_size,
            },
            indent=2,
        )
        + "\n"
    )

    print("\nDone.")
    print(f"Raw sweep table: {sweep_path}")
    print(f"Core summary table: {summary_path}")


if __name__ == "__main__":
    main()
