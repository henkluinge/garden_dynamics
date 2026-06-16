# GTSAM Solve Implementation (Garden Dynamics)

This note documents how the current GTSAM-based position refinement is implemented and how it is used from the notebook workflow.

## Scope

The implementation replaces the previous recursive Kalman-style distance update with a batch factor-graph solve in 2D.

Main implementation file:
- `python/gtsam_position_refinement.py`

Primary notebook usage:
- `notebooks/garden_map.ipynb`

## Input Contracts

### Tree state input

`gdf_trees` is expected to be a GeoDataFrame with point geometry. If `px` and `py` are missing, they are derived using `field_parsers.gdf_trees_to_regular_pandas`.

### Range measurements

Each measurement must follow:

`{"distance": [i_A, i_B, d_measured_m, sd_m]}`

where:
- `i_A`, `i_B`: tree ids (GeoDataFrame index values)
- `d_measured_m`: measured distance in meters
- `sd_m`: standard deviation in meters

## Solve Pipeline

The notebook uses a clean explicit sequence:

1. Validate inputs.
2. Build initial values and prior factors.
3. Add range factors.
4. Run batch optimizer.
5. Apply optimized positions back to geometry.
6. Compute diagnostics.

## Factor Graph Design

## Variables

Each tree id is mapped to a GTSAM key using symbol prefix `p`:

`key = gtsam.symbol("p", tree_id)`

Each variable is a `Point2` in local Veelerveen meter coordinates.

## Priors

The graph includes priors for all points from GPS initialization:
- Non-corner trees: `sigma_gps` (default 4.0 m)
- Corner trees: `sigma_corner` (default 0.1 m)

Additional anchor policy currently used:
- Hard anchor at tree id `0` with very tight sigma (`hard_anchor_sigma`, default `1e-3`)
- Soft anchor at tree id `5` with `sigma_corner`

This reflects the project choice:
- `0_Corner` anchors the origin.
- `5_Corner` constrains direction with a softer prior.

## Range factors

Range measurements are added as `CustomFactor` terms with scalar residual:

`r = ||p_A - p_B|| - d_measured`

Jacobian blocks are analytical:
- w.r.t. `p_A`: `(p_A - p_B) / ||p_A - p_B||`
- w.r.t. `p_B`: negative of the above

If points coincide numerically (`||p_A - p_B|| < 1e-12`), Jacobians fall back to zero to avoid instability.

## Optimization

Batch optimization is done with Levenberg-Marquardt:
- `gtsam.LevenbergMarquardtOptimizer`
- max iterations configurable (`max_iterations`, default 200)

Current implementation is batch-only (no iSAM2 in this version).

## Output and Diagnostics

## Geometry write-back

Optimized `Point2` values are written back to `gdf_trees.geometry` as shapely `Point` objects.

If present, `n_corrections` is incremented by 1.

## Diagnostics table

`build_distance_diagnostics` returns a DataFrame with per-measurement:
- `d_measured`
- `d_prev` (before solve)
- `r_post` (after solve)
- `residual_prev = d_prev - d_measured`
- `residual_post = r_post - d_measured`

This is used in the notebook to annotate residual arrows on the map.

## Validation and Fail-Fast Checks

`validate_inputs` checks:
- geometry column exists
- measurement schema is valid
- positive distances and positive standard deviations
- referenced tree ids exist
- hard and soft anchor ids exist

## Known Constraints

- GTSAM must be installed in the active environment; helper raises `ImportError` otherwise.
- Model currently assumes isotropic Gaussian noise.
- Robust loss for outliers is not yet enabled.
- Orientation is constrained indirectly via priors on anchor points (no explicit bearing factor yet).

## Recommended Next Enhancements

1. Optional robust noise (Huber/Cauchy) for range outliers.
2. Optional explicit direction/bearing factor between anchors.
3. Optional incremental mode using iSAM2.
4. Residual summary metrics (RMSE before/after) as standard output.
