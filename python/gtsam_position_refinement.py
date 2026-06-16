import numpy as np
import pandas as pd
from shapely.geometry import Point

from python import field_parsers

try:
    import gtsam
except Exception:  # pragma: no cover - import checked at runtime in notebook
    gtsam = None


def _require_gtsam():
    global gtsam
    if gtsam is None:
        try:
            import gtsam as _gtsam

            gtsam = _gtsam
        except Exception:
            pass
    if gtsam is None:
        raise ImportError(
            "gtsam is not available. Install it in this environment before running refinement."
        )


def _key(tree_id):
    return gtsam.symbol("p", int(tree_id))


def validate_inputs(
    gdf_trees,
    measurements,
    hard_anchor_id=0,
    soft_anchor_id=5,
):
    if "geometry" not in gdf_trees.columns:
        raise ValueError("gdf_trees must contain a geometry column.")

    missing = []
    for m in measurements:
        if "distance" not in m or len(m["distance"]) != 4:
            raise ValueError(
                "Each measurement must contain distance=[i_A, i_B, d, sd]."
            )
        i_a, i_b, d_measured, sd_d = m["distance"]
        if i_a not in gdf_trees.index:
            missing.append(i_a)
        if i_b not in gdf_trees.index:
            missing.append(i_b)
        if d_measured <= 0:
            raise ValueError(f"Measured distance must be > 0, got {d_measured}.")
        if sd_d <= 0:
            raise ValueError(f"Measurement std dev must be > 0, got {sd_d}.")

    if missing:
        missing = sorted(set(missing))
        raise ValueError(f"Measurement references unknown tree ids: {missing}")

    if hard_anchor_id not in gdf_trees.index:
        raise ValueError(
            f"hard_anchor_id={hard_anchor_id} is not present in gdf_trees."
        )
    if soft_anchor_id not in gdf_trees.index:
        raise ValueError(
            f"soft_anchor_id={soft_anchor_id} is not present in gdf_trees."
        )


def build_initial_values_and_priors(
    gdf_trees,
    sigma_gps=4.0,
    sigma_corner=4.0,
    hard_anchor_id=0,
    soft_anchor_id=5,
    hard_anchor_sigma=1e-3,
):
    _require_gtsam()

    # This function uses pandas dataframe with px and py columns.
    if "px" not in gdf_trees.columns or "py" not in gdf_trees.columns:
        df = field_parsers.gdf_trees_to_regular_pandas(gdf_trees)
    else:
        df = gdf_trees.copy()

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    state_indices = {}
    for i_state, (tree_id, row) in enumerate(df.iterrows()):
        state_indices[tree_id] = 2 * i_state + np.array([0, 1])

        key = _key(tree_id)
        point = gtsam.Point2(float(row.px), float(row.py))
        initial.insert(key, point)

        is_corner = str(row.get("kind", "")) == "Corner"
        sigma = float(sigma_corner if is_corner else sigma_gps)
        prior_noise = gtsam.noiseModel.Isotropic.Sigma(2, sigma)
        graph.add(gtsam.PriorFactorPoint2(key, point, prior_noise))

    # hard_row = df.loc[hard_anchor_id]
    # hard_key = _key(hard_anchor_id)
    # hard_point = gtsam.Point2(float(hard_row.px), float(hard_row.py))
    # hard_noise = gtsam.noiseModel.Isotropic.Sigma(2, float(hard_anchor_sigma))
    # graph.add(gtsam.PriorFactorPoint2(hard_key, hard_point, hard_noise))

    # soft_row = df.loc[soft_anchor_id]
    # soft_key = _key(soft_anchor_id)
    # soft_point = gtsam.Point2(float(soft_row.px), float(soft_row.py))
    # soft_noise = gtsam.noiseModel.Isotropic.Sigma(2, float(sigma_corner))
    # graph.add(gtsam.PriorFactorPoint2(soft_key, soft_point, soft_noise))

    return graph, initial, state_indices


def _make_range_error_function(key_a, key_b, d_measured):
    def _error(this, values, jacobians):
        p_a = values.atPoint2(key_a)
        p_b = values.atPoint2(key_b)

        delta = np.array([p_a[0] - p_b[0], p_a[1] - p_b[1]], dtype=float)
        pred = float(np.linalg.norm(delta))

        if pred < 1e-12:
            direction = np.zeros(2, dtype=float)
        else:
            direction = delta / pred

        if jacobians is not None:
            jacobians[0] = direction.reshape(1, 2)
            jacobians[1] = (-direction).reshape(1, 2)

        return np.array([pred - float(d_measured)], dtype=float)

    return _error


def add_range_factors(graph, measurements):
    _require_gtsam()

    for measurement in measurements:
        i_a, i_b, d_measured, sd_d = measurement["distance"]
        key_a = _key(i_a)
        key_b = _key(i_b)

        noise = gtsam.noiseModel.Isotropic.Sigma(1, float(sd_d))
        factor = gtsam.CustomFactor(
            noise,
            [key_a, key_b],
            _make_range_error_function(key_a, key_b, d_measured),
        )
        graph.add(factor)

    return graph


def solve_batch(graph, initial_values, max_iterations=200):
    _require_gtsam()

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(int(max_iterations))
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, params)
    return optimizer.optimize()


def apply_solution_to_gdf(gdf_trees, result_values):
    _require_gtsam()

    gdf_trees_corrected = gdf_trees.copy()

    for tree_id in gdf_trees_corrected.index:
        key = _key(tree_id)
        point = result_values.atPoint2(key)
        gdf_trees_corrected.at[tree_id, "geometry"] = Point(
            float(point[0]), float(point[1])
        )

    if "n_corrections" in gdf_trees_corrected.columns:
        gdf_trees_corrected["n_corrections"] = gdf_trees_corrected["n_corrections"] + 1

    return gdf_trees_corrected


def build_distance_diagnostics(gdf_before, gdf_after, measurements):
    df_before = field_parsers.gdf_trees_to_regular_pandas(gdf_before)
    df_after = field_parsers.gdf_trees_to_regular_pandas(gdf_after)

    rows = []
    for i_measurement, measurement in enumerate(measurements):
        i_a, i_b, d_measured, _ = measurement["distance"]

        p_a_before = np.asarray(df_before.loc[i_a][["px", "py"]], dtype=float)
        p_b_before = np.asarray(df_before.loc[i_b][["px", "py"]], dtype=float)
        p_a_after = np.asarray(df_after.loc[i_a][["px", "py"]], dtype=float)
        p_b_after = np.asarray(df_after.loc[i_b][["px", "py"]], dtype=float)

        d_prev = float(np.linalg.norm(p_a_before - p_b_before))
        d_post = float(np.linalg.norm(p_a_after - p_b_after))

        rows.append(
            {
                "index": i_measurement,
                "i_A": i_a,
                "i_B": i_b,
                "d_measured": float(d_measured),
                "d_prev": d_prev,
                "r_post": d_post,
                "residual_prev": d_prev - float(d_measured),
                "residual_post": d_post - float(d_measured),
            }
        )

    return pd.DataFrame(rows)
