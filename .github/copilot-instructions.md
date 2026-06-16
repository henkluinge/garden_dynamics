# AI Coding Agent Instructions for garden_dynamics

## Project Overview
**garden_dynamics** is a geospatial food forest/orchard mapping and positioning system that combines Dutch cadastral (land registry) data with GPS measurements and precision positioning refinement. The goal is to create accurate maps and optimize tree positions using ranging measurements.

## Architecture: Core Data Flows

### Three Main Components
1. **Geospatial Parsers** (`python/geo_parsers.py`)
   - Reads Dutch cadastral data (GML files from PDOK)
   - Handles coordinate system transforms between `Amersfoort` (EPSG:28992), `WGS84`, and custom `Veelerveen` local coordinates
   - `Veelerveen CRS` is a site-specific oblique Mercator projection (rotated ~21°) centered at the orchard with origin at lower-left corner
   - **Key insight**: All coordinates are converted to `Veelerveen` for local calculations; understand the CRS before modifying geometry operations

2. **Field Elements** (`python/field_parsers.py` & `python/garden_elements.py`)
   - Manages trees, ditches, and landmarks as GeoDataFrames
   - Trees stored with `kind` (species), `label` (unique identifier), and geometry (Point)
   - Two data persistence formats: TOML files (human-editable) and GeoDataFrame serialization
   - GPS coordinates recorded via KML app, then cleaned and standardized

3. **Position Refinement** (`python/ranging_correction.py`)
   - Implements Kalman filtering to refine tree positions using distance measurements
   - State vector: `[x0, y0, x1, y1, ...]` where each tree gets 2D coordinates
   - Measurements format: `[tree_id_A, tree_id_B, distance_m, std_dev_m]`
   - Each measurement linearizes around current estimate and updates covariance

## Critical Workflow Patterns

### Data Import & Coordinate Transforms
```python
# Standard workflow when adding new parcels/trees
from python import geo_parsers, field_parsers

# Always specify CRS label or get Veelerveen automatically
gdf = geo_parsers.get_kadaster_objects(crs_label="Veelerveen")
gdf_trees = field_parsers.read_gps_coordinates_recorded_with_app(crs_veelerveen=crs)

# Remember: get_crs_veeleerveen() requires origin_lat/origin_lon to initialize
```

### Tree Data Manipulation
- GeoDataFrame columns: `kind` (species), `label` (unique ID), `geometry` (Point), `n_corrections` (tracking)
- **Never drop geometry without converting to regular DataFrame first** – use `field_parsers.gdf_trees_to_regular_pandas()`
- When modifying trees: use `add_row_to_tree_list()` to ensure proper index/geometry alignment

### Kalman Filter Integration
- State initialization: `ranging_correction.initialize_state_from_gps(gdf_trees, sigma_gps=4.0, sigma_corner=0.1)`
- Corner landmarks get lower uncertainty (`sigma_corner=0.1m`) than GPS-measured trees (`sigma_gps=4.0m`)
- Measurement update: always compute residual linearization `C` matrix and update `R` (measurement covariance)

## Project-Specific Conventions

### Coordinate System Naming
- `crs_amersfoort` = EPSG:28992 (Dutch national grid, North-up)
- `crs_global` = "WGS84" (lat/lon)
- `crs_veelerveen` = Custom OMERC projection for this site (locally North-up after rotation)
- **When adding new coordinate operations, always preserve the source CRS and convert intentionally**

### Element Types & Color Scheme
Trees are classified in `garden_elements.element_colors` dict by species (e.g., `'Appelboom'`, `'Wilg'`, `'Kersenboom'`). Landmarks include `'Corner'` and `'Hole'`. Visually distinguish on plots using these standard colors.

### TOML File Format
Configuration and cached data stored in TOML using simple schema:
```toml
[[trees]]
label = 1
kind = "Appelboom"
px = 85.5
py = 3.2
pz = 0.0
```
Use `field_parsers.load_gps_coordinates_to_toml()` and `write_gps_coordinates_to_toml()` for I/O. Falls back to basic parser if `tomllib` (Py3.11+) unavailable.

## Dependencies & Imports

**Key packages** (see `garden_map.ipynb` for full stack):
- `geopandas` / `shapely` – geometry operations (install: `pip install geopandas contextily`)
- `pyproj` – coordinate system transformations
- `folium` – interactive web maps (in `folium_tryout.py`)
- **Known issue**: `fiona==1.9.6` required; newer versions break geometry I/O (see README)

## File Organization
- **Notebooks**: `garden_map.ipynb` is the main workflow (cell-by-cell data loading, refining, plotting)
- **Python modules**: Modular functions in `python/` directory, imported with `from python import <module>`
- **Data sources**: `datasets/` contains GML files (cadastral boundaries), KML (GPS recordings), `data_cache/` stores TOML outputs

## When Modifying Code

### Adding New Trees or Measurements
- Use `field_parsers.add_row_to_tree_list()` to maintain state index alignment
- If updating measurements: validate distance format `[i_A, i_B, distance_m, std_dev_m]` in `ranging_correction.measurements`

### Fixing Coordinate Issues
- Check CRS first: `gdf.crs` should be Veelerveen after transforms
- Test with small subset of kadaster parcels before full import
- Use `geo_parsers.get_lower_left_point_in_veelerveen_rotation()` to determine true local origin

### Extending Kalman Filter
- State indices must match tree count: `state_indices = {tree_id: (x_idx, y_idx), ...}`
- Measurement matrix `C` is sparse (only non-zero at indices for tree pair A and B)
- Always update both `x` (state) and `Px` (covariance) via `kf.update()` from `inference.simple_kalman`

## Testing & Validation
No formal test suite yet. Validate changes by:
1. Plotting with `plot_orchard()` to visually inspect tree positions
2. Checking distance residuals: `df_correction` in notebook tracks `d_measured` vs `r_post` (post-correction)
3. Verifying CRS consistency: all geometries in a GeoDataFrame must share same CRS

## Notebook Authoring Rules
- Notebook JSON and cell-format conventions are documented in `.github/notebook-format-instructions.md`.
- Follow that file when editing or generating notebook content.

## GTSAM Refinement Notes
- GTSAM batch solve design and API flow are documented in `.github/gtsam-solve-implementation.md`.
