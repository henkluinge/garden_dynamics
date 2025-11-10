"""Geopanda dataframes for (lines of) trees and ditches."""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from python import geo_parsers

element_colors = {'Corner': 'darkgrey', 'Hole': 'Black', 'Fruit': 'fuchsia', 'Walnoot': 'purple', 'Beukenboom': 'saddlebrown',
                  'Kersenboom': 'red', 'Pruimenboom': 'purple', 'Appelboom': 'lightgreen', 'Perenboom': 'yellow', 
                  'Druivenstruik': 'violet', 'Vijgenboom': 'blue', 'Kaki': 'cyan', 'Abrikoos': 'gold', 'Amandelboom': 'salmon',
                  'Kweepeer': 'lime',  'Wilg': 'orange', 'Populier': 'darkorange', 
                  }

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


def widen_map_plot(ax, percentage_x=0.2, percentage_y=0.3):


    xlim = ax.get_xlim()
    m = np.mean(xlim)
    d = np.diff(xlim)
    xlim = [p + percentage_x*(p-m) for p in xlim ]
    ax.set_xlim(xlim)

    ylim = ax.get_ylim()
    m = np.mean(ylim)
    d = np.diff(ylim)
    ylim = [p + percentage_y*(p-m) for p in ylim ]
    ax.set_ylim(ylim)

    return 


def set_point_coordinates(gdf, label, x=None, y=None, z=None, type=None):
    """Set the coordinates of a point in a geodataframe based on its label."""
    d = gdf[gdf['label'] == label].copy()
    if d.empty:
        print(f"Label {label} not found.")
        return gdf
    if len(d) > 1:
        print(f"Multiple entries found for label {label}. Please ensure labels are unique.")
        print('Taking the first one')
        return gdf
    d = d.iloc[0]
    # print(d)
    idx = d.name

    coords = d.geometry.coords[0]
    new_x = x if x is not None else coords[0]
    new_y = y if y is not None else coords[1]
    new_z = z if z is not None and len(coords) == 3 else (coords[2] if len(coords) == 3 else None)
    if new_z is not None:
        gdf.at[idx, 'geometry'] = Point(new_x, new_y, new_z)
    else:
        gdf.at[idx, 'geometry'] = Point(new_x, new_y)

    if type is not None:
        gdf.at[idx, 'type'] = type
    return gdf


def get_veelerveen_ditches():
    """ Very specific for veelerveen
    """
    p_origin_orchard = np.asarray([73.9996065 ,  1.99961984])    
    crs_veelerveen = geo_parsers.get_crs_veeleerveen()
    gdf_ditch_bottom = ditch_geometry(crs_veelerveen, width=1.4,
                            p_start=p_origin_orchard, p_end=np.asarray((185,0)))
    gdf_ditch_up = ditch_geometry(crs_veelerveen, width=1.4,
                            p_start=p_origin_orchard, p_end=p_origin_orchard+np.asarray((0, 21)))
    gdf_ditch_top = ditch_geometry(crs_veelerveen, width=1.4,
                            p_start=[0, 24.2], p_end=[200,22])


    gdf_ditch = pd.concat([gdf_ditch_bottom, gdf_ditch_up, gdf_ditch_top])
    return gdf_ditch