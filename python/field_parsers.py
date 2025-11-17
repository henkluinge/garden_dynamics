"""
parsers for example notebooks.
"""
import toml
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from python import geo_parsers

def _num_to_toml(x):
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        # keep a compact, human-readable float representation
        return format(x, ".15g")
    return f"\"{str(x)}\""

def _list_to_toml(lst):
    return "[" + ", ".join(_num_to_toml(v) for v in lst) + "]"

def write_range_measurements_toml(measurements, fname="range_list.toml"):
    out_lines = []
    for entry in measurements:
        out_lines.append("[[ranges]]")
        for k, v in entry.items():
            if isinstance(v, (list, tuple)):
                out_lines.append(f"{k} = {_list_to_toml(v)}")
            else:
                out_lines.append(f"{k} = {_num_to_toml(v)}")
        out_lines.append("")  # blank line between entries

    with open(fname, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))


def read_range_measurements_toml(fname="measurements.toml"):
    """
    Read range measurements from a TOML file. File is generated using the gardemap notebook.
    """
    try:
        import tomllib as _toml  # Python 3.11+
        with open(fname, "rb") as _f:
            data = _toml.load(_f)
    except Exception:
        try:
            import toml as _toml  # third-party package fallback
            with open(fname, "r", encoding="utf-8") as _f:
                data = _toml.load(_f)
        except Exception as e:
            raise RuntimeError("No TOML parser available (install toml or use Python 3.11+)") from e

    measurements = data.get("ranges", [])
    normalized = []
    for entry in measurements:
        ent = {}
        for k, v in entry.items():
            if isinstance(v, (list, tuple)):
                ent[k] = list(v)
            else:
                ent[k] = v
        normalized.append(ent)
    return normalized

def _feather_coordinates(row):
    coords = row.geometry.coords[0]
    if len(coords) == 2:
        return pd.Series(data=coords, index=['px', 'py'])
    elif len(coords) == 3:
        return pd.Series(data=coords, index=['px', 'py','pz'])
    
def to_regular_pandas(gdf):
    """ Convert a GeoDataFrame to a regular DataFrame by extracting coordinates from geometry.
    """
    gdf = gdf.copy()
    # coords = gdf.apply(lambda row: pd.Series(data=row.geometry.coords[0], index=['px', 'py','pz']), axis=1)
    pxpypz = gdf.apply(_feather_coordinates, axis=1)
    gdf[['px', 'py', 'pz']] = pxpypz
    gdf = gdf.drop(columns=['geometry'])
    return pd.DataFrame(gdf)
        
def write_gps_coordinates_to_toml(gdf_trees, filename='gdf_trees.toml', verbose=False):
    """ Export thre trees Geodataframe using in the garden_map notebook to a TOML file.
    """
    # Parse the shapely geometry to extract coordinates.
    # Get coordinates from geometry and add as columns

    
    gdf_trees = gdf_trees.copy()
    # coords = gdf_trees.apply(lambda row: pd.Series(data=row.geometry.coords[0], index=['px', 'py','pz']), axis=1)
    pxpypz = gdf_trees.apply(_feather_coordinates, axis=1)
    gdf_trees[['px', 'py', 'pz']] = pxpypz
    gdf_trees = gdf_trees.drop(columns=['geometry'])

    # Prepare records as plain Python types
    records = gdf_trees.to_dict(orient="records")

    # Create dictionary with plain types
    def _to_plain(v):
        if pd.isna(v):
            return None
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        try:
            if isinstance(v, str) and v.isdigit():
                return int(v)
        except Exception:
            pass
        return v

    records = [
        {k: _to_plain(v) for k, v in rec.items()}
        for rec in records
    ]
    toml_data = {"trees": records}

    # 
    with open(filename, "w", encoding="utf-8") as f:
        toml.dump(toml_data, f)
    if verbose:
        print(f"Wrote {len(records)} tree entries to {filename}")


def load_gps_coordinates_to_toml(filename="gdf_trees.toml", crs=None, as_geodataframe=False):
    """
    Load gdf_trees previously written to TOML by the notebook and return a GeoDataFrame
    with columns (type, label, px, py, pz, ...) and a Point geometry built from px, py.
    Uses the `toml` module if available, otherwise falls back to a simple parser.
    """
    # Prefer toml if available in the notebook environment
    try:
        data = toml.load(filename)
        records = data.get("trees", data if isinstance(data, list) else [])
    except Exception:
        # Fallback simple parser for files written as repeated [[trees]] blocks
        def _parse_value(v):
            v = v.strip()
            # None / empty string written as ''
            if v in ("''", '""'):
                return None
            # quoted strings
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                return v[1:-1]
            # booleans
            lv = v.lower()
            if lv == "true":
                return True
            if lv == "false":
                return False
            # numbers
            try:
                if "." in v:
                    return float(v)
                return int(v)
            except Exception:
                return v

        with open(fname, "r", encoding="utf-8") as f:
            content = f.read()
        blocks = [b.strip() for b in content.split("[[trees]]") if b.strip()]
        records = []
        for blk in blocks:
            rec = {}
            for line in blk.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                rec[k.strip()] = _parse_value(v)
            records.append(rec)

    # Build DataFrame
    df = pd.DataFrame(records)

    # Normalize types where possible
    if "label" in df.columns:
        try:
            df["label"] = df["label"].astype(int)
        except Exception:
            pass
    for col in ("px", "py", "pz"):
        if col in df.columns:
            try:
                df[col] = df[col].astype(float)
            except Exception:
                pass

    # Create geometry from px, py (if present)
    if "px" in df.columns and "py" in df.columns:
        df["geometry"] = df.apply(lambda r: Point(r["px"], r["py"]), axis=1)
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=crs)
    else:
        # If no px/py, try to keep any geometry column already present
        gdf = gpd.GeoDataFrame(df, geometry=df.columns[df.dtypes == "geometry"][0] if any(df.dtypes == "geometry") else None, crs=crs)

    # Keep column order similar to original notebook
    cols = [c for c in ["type", "label", "px", "py", "pz", "geometry"] if c in gdf.columns]
    gdf = gdf[cols]
    if as_geodataframe:
        gdf = gdf.drop(columns=["px", "py", "pz"], errors="ignore")
    else:
        # As normal DataFrame, drop geometry
        gdf = gdf.drop(columns=["geometry"], errors="ignore")
    return gdf


def add_row_to_tree_list(gdf, px=None, py=None, pz=None, new_id=None, base_id=0, replace=False, **kwargs):
    """Add point to tree list.
     
        gdf: geodataframe containing trees. Geometry is assumed to be points only.

        base_id: index of row that is the basis for changing, then adding or replacing. 
        new_id: New index of the row to be adder or replaced. If None, the row will be added.

    """  
    
    df = gdf.copy()   # Make sure not to modify original.
    if base_id is None:
        base_id = df.index[0]
    if new_id is None:
        if replace:
            new_id = base_id
        else:
            new_id = df.index.max() + 1

    # Separate crs, geometry and regular pandas fields.
    crs = df.crs
    d = df.loc[base_id].drop(labels=['geometry'])
    d.name = new_id
    g = gpd.GeoSeries([df.geometry.loc[base_id]], index=[new_id], crs=crs)
    # idx = d.name

    # Make adaptations to fields (do geometry later)
    for k, v in kwargs.items():
        d[k] = v

    # Adjust coordinates if provided
    if px is not None or py is not None:
        point = g.iloc[0]
        x_new = px if px is not None else point.x
        y_new = py if py is not None else point.y
        if point.has_z:
            z_new = pz if pz is not None else point.z
        g = gpd.GeoSeries([Point(x_new, y_new)], index=[new_id], crs=crs)

    # remove any existing row with the same index, then append the new row
    if new_id in df.index:
        df = df.drop(new_id)

    gdf_adapted = gpd.GeoDataFrame(d.to_frame().T, geometry=g, index=[new_id], crs=crs)
    df = pd.concat([df, gdf_adapted], ignore_index=False)
    df.sort_index(inplace=True)

    return df

def read_gps_coordinates_recorded_with_app(fname = 'datasets/boomgaard_plant_coordinaten/20251005-091825.kml',
                                           crs_veelerveen=None):
    """Easrly october, I did a recording of GPS coordintates of the trees. This function reads and cleans the specifica dataset.
    """
    gdf_trees = gpd.read_file(fname)
    gdf_trees = gdf_trees.drop(columns=['Description', 'timestamp', 'begin', 'end', 'altitudeMode', 'tessellate', 'extrude', 'visibility', 'drawOrder', 'icon'],
                            errors='ignore')
    gdf_trees = gdf_trees[~gdf_trees.Name.str.startswith('Track ')]
    gdf_trees.rename(columns={'Name': 'kind'}, inplace=True)

    # Some tree was hallucinated.
    gdf_trees = gdf_trees[gdf_trees.index!=8]

    # Clean up
    labels_to_drop = ['Woonkamer', 'Living room', ]
    gdf_trees = gdf_trees[~gdf_trees.kind.isin(labels_to_drop)]
    pioniers_bomen = ['Wilg',]
    wilde_struiken = ['Klein eikeboompje', 'Kleine eikeboom',]

    # Get consistent naming. In this case translate from English to Dutch. Use 'boom' suffix for trees.
    d_translate = {'Apple': 'Appelboom',
                'Beuk': 'Beukenboom',
                'Kers': 'Kersenboom'}
    gdf_trees['kind'] = gdf_trees['kind'].replace(dict(d_translate))

    # Filter only on larger elements.
    production_trees = ['Fruit', 'Walnoot', 'Beukenboom', 'Kersenboom', 'Pruimenboom', 'Appelboom', 'Perenboom', 'Moerbei', 'Druivenstruik', 'Vijgenboom', 'Kaki', 'Mispoes', 'Abrikoos', 'Amandelboom', 'Kweepeer', 'Kweepeertje', 'Sleedoorn']
    pioneering_trees = ['Wilg', 'Populier', 'Es', 'Linde', 'Els', 'Meidoorn', 'Iep', 'Lijsterbes']
    landmarks = ['Corner', 'Hole']

    mask = gdf_trees['kind'].isin(production_trees + landmarks + pioneering_trees)
    gdf_trees = gdf_trees[mask]
    gdf_trees.reset_index(drop=True, inplace=True)
    gdf_trees.index.kind = 'ID'

    # Convert to veelerveen coordinates.
    if crs_veelerveen is None:
        crs_veelerveen = geo_parsers.get_crs_veeleerveen()
    gdf_trees = gdf_trees.to_crs(crs_veelerveen)

    # Add walnut trees since they were not measrued with GPS.
    gdf_trees = add_row_to_tree_list(gdf_trees, kind='Walnut', px=86.0, py=6.0, new_id=None, base_id=2)
    
    # Make available to sensor_fusion_examples/notebooks/ranging_measurements.ipynb
    write_gps_coordinates_to_toml(gdf_trees, filename='./data_cache/examples_gdf_trees.toml', verbose=True)
    
    return gdf_trees