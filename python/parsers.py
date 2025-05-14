import pandas as pd
import geopandas as gpd
from datetime import datetime

from pyproj import CRS
import shapely
from shapely.geometry import Point, LineString, Polygon
from shapely import affinity


crs_amersfoort = CRS.from_epsg(28992)
crs_global = 'WGS84'
print('crs ', crs_global)


kadastrale_object_ids_own = [58270110570000, 58270073270000]

def get_crs_veeleerveen(origin_lat=53.0579, origin_lon = 7.12):
    crs_veelerveen = CRS.from_proj4(f"+proj=omerc +lat_0={origin_lat} +lonc={origin_lon} +alpha=-21 +k=1 +x_0=0 +y_0=0 +gamma=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0")
    # crs_veelerveen = CRS.from_proj4(f"+proj=omerc +lat_0={origin_lat} +lonc={origin_lon} +alpha=0 +k=1 +x_0=0 +y_0=0 +gamma=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0")
    return crs_veelerveen


def _label_to_crs(crs_label):
        # ax = db_percelen.plot(color='red')
    if isinstance(crs_label,str) and (crs_label.lower()=='veelerveen'):
        crs = get_crs_veeleerveen(origin_lat=53.0579, origin_lon = 7.12)
    elif isinstance(crs_label,str) and (crs_label.lower()=='amersfoort'):
        crs = CRS.from_epsg(28992)
    else:
        crs = crs_global # use crs defined at top of this module.
    return crs


def get_kadaster_objects(kadastrale_object_ids=kadastrale_object_ids_own, crs_label='Amersfoort',
                         file_path = r"./datasets/kadastralekaart_perceel.gml"):
    """ Read a gml file obtained from https://app.pdok.nl/kadaster/kadastralekaart/download-viewer/

        kadastrale_object_ids: list of int.
            A kadaster area can be referred to by a code (e.g. N732) or an integer id (Kadastrale objectidentificatie). Both can be found on the eigenaars informatie from mijn Overheid.

        CRS is converted to Amerfoort crs (28992) or veelerveen coordinates.

        returns
        -------
        geopandas dataframe with all relevant 'kadaster percelen'.
    """

    db = gpd.read_file(file_path)
    if kadastrale_object_ids is not None:
        db_percelen = db[db.identificatie.isin(kadastrale_object_ids)]
    else:
        db_percelen = db.copy()
    db_percelen = db_percelen.drop(columns=['beginGeldigheid', 'tijdstipRegistratie', 'waarde',
                                            r'kadastraleAanduiding|TypeKadastraleAanduiding|aKRKadastraleGemeenteCode|AKRKadastraleGemeenteCode|code',
                                            r'kadastraleAanduiding|TypeKadastraleAanduiding|aKRKadastraleGemeenteCode|AKRKadastraleGemeenteCode|waarde',
                                            r'kadastraleGrootte|TypeOppervlak|soortGrootte|SoortGrootte|code',
                                            r'kadastraleGrootte|TypeOppervlak|soortGrootte|SoortGrootte|waarde'])

    db_percelen = db_percelen.rename(columns={r'kadastraleAanduiding|TypeKadastraleAanduiding|kadastraleGemeente|KadastraleGemeente|code': 'gemeente_id',
                                            r'kadastraleAanduiding|TypeKadastraleAanduiding|kadastraleGemeente|KadastraleGemeente|waarde': 'gemeente',
                                            r'kadastraleGrootte|TypeOppervlak|waarde': 'oppervlakte'})

    crs = _label_to_crs(crs_label)
    db_percelen = db_percelen.to_crs(crs)

    # Select percelen close to own.
    p_point = db_percelen[db.identificatie.isin(kadastrale_object_ids_own)].union_all().centroid
    db_percelen = db_percelen[db_percelen.distance(p_point)<200]

    return db_percelen.set_index('identificatie')


def get_perceel_coordinates(perceel_id, crs_label=None,
                            file_path=r"C:\Users\HenkColleenNiamh\Code\cad\extract\kadastralekaart_kadastralegrens.gml",
                            as_polygon=True):
    """ Geopandas dataframe describing one kadaster plot including outline.
    """
    db = gpd.read_file(file_path)
    db = db.drop(columns=[r'typeGrens|TypeGrens|code', r'typeGrens|TypeGrens|waarde', 'volgnummer', 'code', 'waarde', 'beginGeldigheid', 'tijdstipRegistratie'])
    db = db[ (db.perceelLinks==perceel_id) | (db.perceelRechts==perceel_id) ]

    if as_polygon:
        g = shapely.polygonize(list(db.geometry))
    else:
        g = shapely.MultiLineString(list(db.geometry))

    d_perceel_geometry = gpd.GeoSeries(name=perceel_id, index=[perceel_id,], data=g)
    d_perceel = pd.Series(data=[2, ], index=['Veelerveen',], name=perceel_id).to_frame().T

    gdf = gpd.GeoDataFrame(d_perceel, geometry=d_perceel_geometry, crs=crs_amersfoort)
    
    return gdf


def add_perceel_name(ax, db_info_percelen):
    """Print kadaster label (e.g. N1105) to plot)"""
    p = db_info_percelen.get_coordinates()

    for id, row in db_info_percelen.iterrows():
        p_i = p.loc[id]
        if 'eigenaar' in row.index:
            label = row.eigenaar
        else:
            pass
        label = f'{row.sectie} {row.perceelnummer}'
        ax.annotate(label,  (p_i.x, p_i.y), horizontalalignment='center', fontweight='bold')

    return


def get_lower_left_point_in_veelerveen_rotation(gdf):
    """gdf in north up coordintate system."""

    crs_input = gdf.crs
    print(crs_input)
    gdf = gdf.copy()
    # Temporary veelerveen coordinates
    lat_min = gdf.to_crs('WGS84').get_coordinates().y.min()
    lon_min = gdf.to_crs('WGS84').get_coordinates().x.min()
    crs_veelerveen = get_crs_veeleerveen(lat_min, lon_min)

    p = gdf.to_crs(crs_veelerveen).get_coordinates()
    px, py = p.x.min(), p.y.min()
    
    origin = gpd.GeoSeries(data=Point(px, py), index=['origin',], crs=crs_veelerveen)
    return origin.to_crs(crs_input)


def get_north_arrow(crs_veelerveen, p_shift_x=0, p_shift_y=0, xfact=1, yfact=1):
    up_arrow = LineString([(0, 0), (0, 20), (0, 20), (4, 12),  (0, 20), (-4, 12)])
    letter_n = LineString([(3, 0), (3, 9), (9, 0), (9, 9)])

    geometry = [affinity.scale(g, xfact=xfact, yfact=yfact) for g in [up_arrow, letter_n]]


    gdf_north = gpd.GeoDataFrame(index=[0, 1], crs=crs_amersfoort, geometry=geometry ) # Because north up.
    gdf_north = gdf_north.to_crs(crs_veelerveen)

    coords = -gdf_north.get_coordinates()
    p_shift_x, p_shift_y = coords.iloc[0].x + p_shift_x, coords.iloc[0].y + p_shift_y

    gdf_north = gdf_north.translate(p_shift_x, p_shift_y)
    # coords = gdf_north.get_coordinates()
    return gdf_north


def get_eelde_weather_data(fname = 'eelde_2021-2030/uurgeg_280_2021-2030.txt'):
    """
    Data from https://www.knmi.nl/nederland-nu/klimatologie/uurgegevens
    """
    # Read header
    with open(fname, 'r') as f:
        for line in f:
            # Print each line
            s = line.strip()
            
            if s.startswith('#'):
                break
        cols = [si.strip() for si in s[1:].split(',')]

        # Read data
        df = pd.read_csv(f, low_memory=False, names=cols)

    # Groom data.
    df['HH'] = df['HH'].apply(lambda hr: 0 if hr==24 else hr)
    df['datetime'] = df.apply(lambda d: datetime.strptime(str(d.YYYYMMDD)+str(d.HH).zfill(2), r'%Y%m%d%H') , axis=1)
    col_names = {'DD': 'windrichting', 'FF': 'windsnelheid', 'FX':'max_wind', 'T': 'temperatuur', 'SQ': 'zonneschijn', 'RH': 'neerslag', 'U': 'rel_luchtvochtigheid'}
    df = df.set_index('datetime')[col_names.keys()].rename(columns=col_names)

    # Temperature is logged in 0.1 C
    df['temperatuur'] = 0.1*df['temperatuur']

    return df