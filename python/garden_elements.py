"""Geopanda dataframes for (lines of) trees and ditches."""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon

tree_colors = {'wilgen': 'orange', 'kers': 'red', 'appel': 'darkgreen', 'walnoot': 'brown', 'pruim': 'purple', 'peer': 'lightgreen'}

def tree_geometry(crs_veelerveen, p_shift_x=0, p_shift_y=0, diameter=4):
    """Circle for tree """
    polygon = Polygon([(0, 0), (10, 10), (10, 0)])
    center = Point([0, 0])

    gdf_tree = gpd.GeoDataFrame(index=[0, ], crs=crs_veelerveen, geometry=[center,] ) 

    coords = gdf_tree.get_coordinates()
    p_shift_x, p_shift_y = coords.iloc[0].x + p_shift_x, coords.iloc[0].y + p_shift_y

    gdf_tree = gdf_tree.translate(p_shift_x, p_shift_y)
    gdf_tree =gdf_tree.buffer(distance=diameter)
    return gdf_tree


def get_tree_range(n_trees, p_row_start=[3,3], n=[0, 1], tree_distance = 5, crs_local=None,
                   p_origin_orchard=None, soort_label='wilg', df_append=None):
    """Returns geometry of line of trees.
    """
    if p_origin_orchard is None:
        p_origin_orchard = np.asarray([74, 2])
    wilgen =[]
    
    for i in range (n_trees):
        p_row_start = np.asarray(p_row_start)  # Start of row with respect to origin orchard.
        n = np.asarray(n)  # Direction of row with respect to origin orchard
        p_tree = i*tree_distance*n
        p_tree_origin_veelerveen = p_origin_orchard + p_row_start + p_tree
        gdf_tree = tree_geometry(crs_local, 
                                p_shift_x=p_tree_origin_veelerveen[0], p_shift_y=p_tree_origin_veelerveen[1], diameter=1)
        wilgen.append(gdf_tree)
    wilgen = pd.concat([d.to_frame() for d in wilgen])
    wilgen['label'] = soort_label

    if df_append is not None:
        wilgen = pd.concat([df_append, wilgen])

    return wilgen


def ditch_geometry(crs_veelerveen, p_start=[0,0], p_end=[10, 10], width=2):
    
    p_start = np.asarray(p_start)
    p_end = np.asarray(p_end)
    polygon = LineString([p_start, p_end])

    gdf_ditch = gpd.GeoDataFrame(index=[0, ], crs=crs_veelerveen, geometry=[polygon,] ) 

    # coords = gdf_ditch.get_coordinates()
    # p_shift_x, p_shift_y = coords.iloc[0].x + p_shift_x, coords.iloc[0].y + p_shift_y

    # gdf_ditch = gdf_ditch.translate(p_shift_x, p_shift_y)
    gdf_ditch =gdf_ditch.buffer(distance=width)
    gdf_ditch = gdf_ditch.to_frame()
    gdf_ditch['label'] = 'ditch'
    return gdf_ditch




def _add_point_number(ax, points):
    """Point annotation on plot to check data."""
    # points = np.asarray(mapping(gdf.iloc[1]['geometry'].envelope)['coordinates'][0])
    for i in range(points.shape[0]):
        label = f'{i}'
        ax.annotate(label,  (points[i,0], points[i,1]), horizontalalignment='center', fontweight='bold')

    return