import contextily as cx
import matplotlib.pyplot as plt
import numpy as np
from plotting import plot_annotation as plots

from python import garden_elements, geo_parsers


def add_state_uncertainty_ellipse(gdf_trees, state_indices, Px, ax, **kwargs):
    """Draw error covariance on each tree.

    state_indices, Px, ax: state represenation of all trees, generated using
                ranging_correction.initialize_state_from_gps(gdf_trees)

    """
    for id_tree, d_tree in gdf_trees.iterrows():
        p_tree = d_tree.geometry.coords[0]
        state_indices_tree = state_indices[id_tree]
        Px_tree = Px[np.ix_(state_indices_tree, state_indices_tree)]

        plots.plot_cov_ellipse(Px_tree, p_tree, ax=ax, **kwargs)
    return


def draw_annotated_arrow(ax, pA, pB, s=None, **line_kwargs):
    """
    Draw a line (arrow) between two points with optional text annotation.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes object to draw on
    pA : array-like
        Starting point [x, y]
    pB : array-like
        Ending point [x, y]
    s : str, optional
        Text to display above the line
    **line_kwargs : dict
        Additional keyword arguments passed to ax.plot()
    """
    # Convert to numpy arrays if needed
    pA = np.asarray(pA)
    pB = np.asarray(pB)

    # Add text annotation if provided
    if s is not None:
        # Calculate midpoint
        mid_x = (pA[0] + pB[0]) / 2
        mid_y = (pA[1] + pB[1]) / 2

        # Calculate angle of the line
        dx = pB[0] - pA[0]
        dy = pB[1] - pA[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # Add text at midpoint, rotated to align with line
        ax.text(
            mid_x,
            mid_y,
            s,
            rotation=angle,
            rotation_mode="anchor",
            ha="center",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.4
            ),
        )
        # Draw the line
    ax.plot([pA[0], pB[0]], [pA[1], pB[1]], **line_kwargs)


def plot_trees(ax, gdf_trees, row_label="label"):
    for label, df_group in gdf_trees.groupby("kind"):
        color = garden_elements.element_colors.get(label, "grey")
        df_group.plot(ax=ax, color=color)
    geo_parsers.add_label_annotation_to_map(
        ax,
        gdf_trees,
        fontweight="normal",
        fontsize=8,
        horizontalalignment="left",
        row_label=row_label,
    )
    return ax


def plot_orchard(
    gdf_kadaster,
    db_info_percelen,
    gdf_trees,
    tree_label_column="label",
    add_basemap=True,
    gdf_ditch=None,
    crs_veelerveen=None,
    p_origin_orchard=None,
    p_origin=None,
):
    if crs_veelerveen is None:
        crs_veelerveen = geo_parsers.get_crs_veeleerveen()

    _, ax = plt.subplots(1, 1, figsize=(12, 8), squeeze=True)

    gdf_north = geo_parsers.get_north_arrow(crs_veelerveen, p_shift_x=0, p_shift_y=0)
    gdf_north.plot(ax=ax, linewidth=4, color="k")

    ax = gdf_kadaster.plot(color="purple", edgecolor="black", alpha=1, ax=ax)
    ax = db_info_percelen.plot(
        color="orange", edgecolor="black", alpha=1, ax=ax, markersize=30
    )
    if add_basemap:
        cx.add_basemap(ax, crs=gdf_kadaster.crs, alpha=0.6)
    geo_parsers.add_label_annotation_to_map(ax, db_info_percelen)

    if gdf_ditch is not None:
        gdf_ditch.plot(ax=ax, color="lightblue")

    if p_origin is not None:
        p_origin.to_crs(crs_veelerveen).plot(ax=ax, color="black", markersize=100)

    plot_trees(ax, gdf_trees, row_label=tree_label_column)

    if p_origin_orchard is not None:
        p_origin_orchard.plot(ax=ax, color="black", markersize=100)
    ax.set_ylim(-2, 25)
    # ax.set_xlim(70, 180)
    ax.set_xlim(70, 120)
    ax.set_aspect("equal")

    garden_elements.widen_map_plot(ax, percentage_x=0.1, percentage_y=0.051)
    return ax
    return ax
