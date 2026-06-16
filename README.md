# Example 3D femur registration pipeline

This repository contains a minimal example version of the 3D femur surface-registration
pipeline used for the manuscript. It is limited to the functions needed to run one
example specimen and does not include the full internal research-code library.

The pipeline uses four STL surface models from the same specimen:

```text
sample_data/pre_left.stl
sample_data/post_left.stl
sample_data/pre_right.stl
sample_data/post_right.stl
```

The user chooses which baseline side is the reference with `--reference-side L` or
`--reference-side R`.

## What the pipeline does

1. Loads the STL meshes with trimesh processing enabled.
2. Aligns meshes to principal inertia axes.
3. Uses `femur_orientation_vector.json` to orient proximal/distal ends consistently.
4. Mirrors left-sided meshes into a common right-sided coordinate convention.
5. Registers all four models to the chosen Pre reference model.
6. Separately registers proximal and distal femoral regions.
7. Writes distal volume ratio and angular change outputs.

## Input files

Put the four STL files in `sample_data/` with these exact names:

```text
pre_left.stl
post_left.stl
pre_right.stl
post_right.stl
```

The folder should also contain:

```text
femur_orientation_vector.json
```

## Run in Spyder

Open `run_in_spyder.py`, set:

```python
REFERENCE_SIDE = "L"  # or "R"
```

Then press Run.

The results are written to:

```text
results/example/
```

## Run from terminal

```bash
python run_sample_pipeline.py   --sample-id example   --input-dir sample_data   --output-dir results/example   --reference-side L   --orientation-vector sample_data/femur_orientation_vector.json
```

Use `--reference-side R` instead if the right Pre model is the reference.

## Output files

The main output files are:

```text
results/example/aligned_obj/
results/example/split_meshes/
results/example/tables/distal_sweep.csv
results/example/tables/core_3d_summary.csv
results/example/tables/run_settings.json
```

`core_3d_summary.csv` contains:

```text
Sample, Side, Reference_side, dist_size, Volume_Ratio, X_Delta, Y_Delta, Z_Delta
```

## Mesh notes

Closed surfaces are required for volume, centre-of-mass, inertia, and moment
calculations. The script uses trimesh processing during loading, matching the
workflow used in the analysis. If a raw STL is not watertight but trimesh closes
it during loading, the script prints a note and continues. If the processed mesh
is still not watertight, the STL should be repaired or re-exported as a closed
surface before analysis.

## Python requirements

Install the required packages with:

```bash
pip install -r requirements.txt
```

The minimal required packages are listed in `requirements.txt`.
