#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 09:19:21 2022

@author: mike
"""
import copy
import pathlib
import os
import pickle
import io
from shapely.ops import unary_union
import geobuf
import base64
import orjson
import zstandard as zstd
import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
import hdf5tools
from scipy import stats
import booklet

##############################################
### Parameters

# base_path = '/media/nvme1/data/OLW/web_app'
base_path = '/home/mike/data/OLW/web_app'
# %cd '/home/mike/data/OLW/web_app'

base_path = pathlib.Path(base_path)

rec_rivers_feather = '/media/nvme1/data/NIWA/REC25_rivers/rec25_rivers_clean.feather'
rec_catch_feather = '/media/nvme1/data/NIWA/REC25_watersheds/rec25_watersheds_clean.feather'

rc_bounds_gpkg = '/media/nvme1/data/statsnz/regional-council-2023-clipped-generalised.gpkg'

nzrec_data_path = '/media/nvme1/git/nzrec/data'

segment_id_col = 'nzsegment'

output_path = base_path.joinpath('output')
output_path.mkdir(parents=True, exist_ok=True)

assets_path = output_path.joinpath('assets')
assets_path.mkdir(parents=True, exist_ok=True)

### RC boundaries
rc_bounds_gbuf = assets_path.joinpath('rc_bounds.pbf')

### Land use/cover
lc_base_path = base_path.joinpath('land_use')
lc_base_path.mkdir(parents=True, exist_ok=True)

lcdb_path = lc_base_path.joinpath('lcdb-v50-land-cover-database-version-50-mainland-new-zealand.gpkg')
lcdb_clean_path = lc_base_path.joinpath('lcdb_cleaned.feather')
lcdb_red_path = lc_base_path.joinpath('lcdb_reductions.feather')

snb_geo_path = lc_base_path.joinpath('SnB_Typologies.shp')
dairy_geo_path = lc_base_path.joinpath('Dairy_Typologies.shp')
dairy_geo_clean_path = lc_base_path.joinpath('dairy_typologies_clean.feather')
snb_geo_clean_path = lc_base_path.joinpath('snb_typologies_clean.feather')

snb_typo_path = lc_base_path.joinpath('typologies to reductions - snb.csv')
dairy_typo_path = lc_base_path.joinpath('typologies to reductions - dairy.csv')
dairy_model_typo_path = lc_base_path.joinpath('dairy_modelled_typologies.csv')

snb_dairy_red_path = lc_base_path.joinpath('snb_dairy_reductions.feather')

lc_clean_path = base_path.joinpath('land_cover_clean.feather')
lc_clean_diss_path = base_path.joinpath('land_cover_clean_dissolved.feather')
lc_clean_gpkg_path = base_path.joinpath('land_cover_clean.gpkg')
lc_red_gpkg_path = base_path.joinpath('land_cover_reductions.gpkg')
lc_red_feather_path = base_path.joinpath('land_cover_reductions.feather')


### Rivers
sites_loc_csv = base_path.joinpath('olw_river_sites_locations.csv')
sites_rec_csv = base_path.joinpath('olw_river_sites_rec.csv')
sites_names_csv = base_path.joinpath('LAWARiverSiteswithRCIDs.csv')

# conc_csv_path = base_path.joinpath('StBD3.csv')
river_errors_model_path = base_path.joinpath('rivers_errors_modelled_v02.csv')
river_errors_moni_path = base_path.joinpath('rivers_errors_monitored.csv')
river_sites_path = base_path.joinpath('olw_river_sites.gpkg')

river_flows_rec_path = assets_path.joinpath('rivers_flows_rec.blt')
river_flows_area_path = assets_path.joinpath('rivers_flows_area.blt')

river_loads_rec_path = assets_path.joinpath('rivers_loads_rec.blt')
river_loads_area_path = assets_path.joinpath('rivers_loads_area.blt')

# rec_delin_file = output_path.joinpath('rivers_reach_delineation.feather')
# major_catch_file = output_path.joinpath('rivers_major_catch.feather')
# catch_file = output_path.joinpath('rivers_catch.feather')

# Individual catchment land covers
rivers_catch_lc_dir = assets_path.joinpath('rivers_land_cover_gpkg')
rivers_catch_lc_dir.mkdir(parents=True, exist_ok=True)

rivers_catch_lc_gpkg_str = '{}_rivers_land_cover_reductions.gpkg'

river_sites_catch_path = assets_path.joinpath('rivers_sites_catchments.blt')
river_reach_mapping_path = assets_path.joinpath('rivers_reaches_mapping.blt')
river_reach_gbuf_path = assets_path.joinpath('rivers_reaches.blt')
river_catch_path = assets_path.joinpath('rivers_catchments_minor.blt')
river_catch_major_path = assets_path.joinpath('rivers_catchments_major.blt')

river_sims_path = output_path.joinpath('rivers_sims')
river_sims_path.mkdir(parents=True, exist_ok=True)

river_sims_h5_path = river_sims_path.joinpath('rivers_sims_all.h5')
# river_sims_gam_path = river_sims_path.joinpath('rivers_sims_gam.h5')
river_power_moni_path = assets_path.joinpath('rivers_reaches_power_monitored.h5')
river_power_model_path = assets_path.joinpath('rivers_reaches_power_modelled.h5')
# river_reach_loads_path = assets_path.joinpath('rivers_reaches_loads.h5')
# river_reach_loads_area_path = assets_path.joinpath('rivers_reaches_loads_area.h5')

# parcels_path = base_path.joinpath('nz-primary-land-parcels.gpkg')

## Sims params
# conc_perc = np.arange(2, 101, 2, dtype='int8')
conc_perc = np.arange(1, 101, 1, dtype='int8')
n_samples_year = [12, 26, 52, 104, 364]
n_years = [5, 10, 20, 30]

catch_lc_path = assets_path.joinpath('rivers_catch_lc.blt')
catch_lc_pbf_path = assets_path.joinpath('rivers_catch_lc_pbf.blt')

# catch_lc_clean_path = assets_path.joinpath('rivers_catch_lc.blt')


### Lakes
lakes_fenz_catch_path = base_path.joinpath('lakes_catchments_fenz.gpkg')
lakes_fenz_poly_path = base_path.joinpath('lakes_polygons_fenz.gpkg')
lakes_points_path = output_path.joinpath('lakes_points.feather')
lakes_poly_path = output_path.joinpath('lakes_poly.feather')
lakes_catch_path = output_path.joinpath('lakes_catch.feather')

lakes_points_gbuf_path = assets_path.joinpath('lakes_points.pbf')
lakes_poly_gbuf_path = assets_path.joinpath('lakes_poly.blt')
# lakes_poly_path = base_path.joinpath('lakes_locations_fenz.gpkg')
lakes_delin_points_path = base_path.joinpath('lakes_delineate_points.gpkg')
lakes_reaches_mapping_path = assets_path.joinpath('lakes_reaches_mapping.blt')
lakes_catches_major_path = assets_path.joinpath('lakes_catchments_major.blt')
lakes_catches_minor_path = assets_path.joinpath('lakes_catchments_minor.blt')
lakes_reaches_path = assets_path.joinpath('lakes_reaches.blt')

lakes_catch_lc_dir = assets_path.joinpath('lakes_land_cover_gpkg')
lakes_catch_lc_dir.mkdir(parents=True, exist_ok=True)

lakes_catch_lc_gpkg_str = '{}_lakes_land_cover_reductions.gpkg'

lakes_sims_path = output_path.joinpath('lakes_sims')
lakes_sims_path.mkdir(parents=True, exist_ok=True)

lakes_sims_h5_path = lakes_sims_path.joinpath('lakes_sims.h5')
lakes_power_combo_path = assets_path.joinpath('lakes_power_combo.h5')
lakes_power_model_path = assets_path.joinpath('lakes_power_modelled.h5')
lakes_power_moni_path = assets_path.joinpath('lakes_power_monitored.h5')

lakes_lc_path = assets_path.joinpath('lakes_catch_lc.blt')

## Model data
lakes_data_path = base_path.joinpath('lakes_wq_data.csv')
lakes_data_clean_path = base_path.joinpath('lakes_wq_data_clean.feather')
lakes_class_csv = base_path.joinpath('fenz_lakes_classification.csv')
lakes_stdev_model_path = output_path.joinpath('lakes_stdev_modelled.h5')
lakes_stdev_moni_path = output_path.joinpath('lakes_stdev_monitored.h5')


### GW

## Source data
# gw_data_path = base_path.joinpath('gw_points_data.hdf')
gw_monitoring_data_path = base_path.joinpath('gw_monitoring_data_v03.nc')

## Spatial data
# gw_points_gbuf_path = assets_path.joinpath('gw_points.pbf')
gw_points_path = output_path.joinpath('gw_points.feather')
gw_points_rc_blt = assets_path.joinpath('gw_points_rc.blt')

## Power calcs
gw_sims_path = output_path.joinpath('gw_sims')
gw_sims_path.mkdir(parents=True, exist_ok=True)

gw_power_moni_path = assets_path.joinpath('gw_power_monitored.h5')

gw_sims_h5_path = gw_sims_path.joinpath('gw_sims.h5')






#############################################
### Functions


# def geojson_to_geobuf(geojson):
#     return base64.b64encode(geobuf.encode(geojson)).decode()


def gpd_to_feather(gdf, output):
    """

    """
    gdf.to_feather(output, compression='zstd', compression_level=1)



def read_pkl_zstd(obj, unpickle=False):
    """
    Deserializer from a pickled object compressed with zstandard.

    Parameters
    ----------
    obj : bytes or str
        Either a bytes object that has been pickled and compressed or a str path to the file object.
    unpickle : bool
        Should the bytes object be unpickled or left as bytes?

    Returns
    -------
    Python object
    """
    if isinstance(obj, (str, pathlib.Path)):
        with open(obj, 'rb') as p:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(p) as reader:
                obj1 = reader.read()

    elif isinstance(obj, bytes):
        dctx = zstd.ZstdDecompressor()
        obj1 = dctx.decompress(obj)
    else:
        raise TypeError('obj must either be a str path or a bytes object')

    if unpickle:
        obj1 = pickle.loads(obj1)

    return obj1


def write_pkl_zstd(obj, file_path=None, compress_level=1, pkl_protocol=pickle.HIGHEST_PROTOCOL, retries=3):
    """
    Serializer using pickle and zstandard. Converts any object that can be pickled to a binary object then compresses it using zstandard. Optionally saves the object to disk. If obj is bytes, then it will only be compressed without pickling.

    Parameters
    ----------
    obj : any
        Any pickleable object.
    file_path : None or str
        Either None to return the bytes object or a str path to save it to disk.
    compress_level : int
        zstandard compression level.

    Returns
    -------
    If file_path is None, then it returns the byte object, else None.
    """
    if isinstance(obj, bytes):
        p_obj = obj
    else:
        p_obj = pickle.dumps(obj, protocol=pkl_protocol)

    if isinstance(file_path, (str, pathlib.Path)):

        with open(file_path, 'wb') as f:
            cctx = zstd.ZstdCompressor(level=compress_level, write_content_size=True)
            with cctx.stream_writer(f, size=len(p_obj)) as compressor:
                compressor.write(p_obj)
    else:
        cctx = zstd.ZstdCompressor(level=compress_level, write_content_size=True)
        c_obj = cctx.compress(p_obj)

        return c_obj


# def error_cats(max_error=2.7):
#     """

#     """
#     start = 0.025
#     list1 = [0.001, 0.005, 0.01, start]

#     while start < max_error:
#         if start < 0.1:
#             start = round(start*2, 3)
#         else:
#             start = round(start*1.2, 3)
#         list1.append(start)

#     return list1


def error_cats(start, end, change):
    """

    """
    list1 = [start]

    while start < end:
        delta = abs(start*change)
        if delta < 0.002:
            delta = 0.002
        start = round(start + delta, 3)
        list1.append(start)

    return list1


def log_error_cats(start, end, change):
    """

    """
    s1 = np.asarray(start).round(3)
    list1 = [s1]

    while s1 < end:
        delta = change
        s1 = round(s1 + delta, 3)
        list1.append(s1)

    return list1


def power_test(x, Y, min_p_value=0.05):
    """

    """
    n_sims, n_samples = Y.shape
    p_list = []
    append = p_list.append
    for y in Y:
        o2 = stats.linregress(x, y)
        append(o2.pvalue < min_p_value)

    power = round((sum(p_list)/n_sims) * 100)

    return power


def power_sims(error, n_years, n_samples_year, n_sims, output_path):
    """
    Power simulation function.
    Given an error (float), a number of sampling years (list of int), a number of samples per year (list of int), and the conc percentages (list of int), run n simulations (int) on all possible combinations.
    """
    print(error)

    conc_perc = np.arange(1, 101, 1, dtype='int8')

    n_samples = np.prod(hdf5tools.utils.cartesian([n_samples_year, n_years]), axis=1)
    n_samples = list(set(n_samples))
    n_samples.sort()

    filler = np.empty((1, len(n_samples), len(conc_perc)), dtype='int8')

    rng = np.random.default_rng()

    for ni, n in enumerate(n_samples):
        # print(n)

        for pi, perc in enumerate(conc_perc):
            red1 = np.log(np.interp(np.arange(n), [0, n-1], [1, perc*0.01]))

            rand_shape = (n_sims, n)
            red2 = np.tile(red1, n_sims).reshape(rand_shape)
            r2 = rng.normal(0, error, rand_shape)

            p2 = power_test(np.arange(n), red2 + r2, min_p_value=0.05)

            filler[0][ni][pi] = p2

    error1 = int(error*1000)

    props = xr.Dataset(data_vars={'power': (('error', 'n_samples', 'conc_perc'), filler)
                                  },
                       coords={'error': np.array([error1], dtype='int16'),
                               'n_samples': np.array(n_samples, dtype='int16'),
                               'conc_perc': conc_perc}
                       )

    output = os.path.join(output_path, str(error1) + '.h5')
    hdf5tools.xr_to_hdf5(props, output)

    return output


def xr_concat(datasets):
    """
    A much more efficient concat/combine of xarray datasets. It's also much safer on memory.
    """
    # Get variables for the creation of blank dataset
    coords_list = []
    chunk_dict = {}

    for chunk in datasets:
        coords_list.append(chunk.coords.to_dataset())
        for var in chunk.data_vars:
            if var not in chunk_dict:
                dims = tuple(chunk[var].dims)
                enc = chunk[var].encoding.copy()
                dtype = chunk[var].dtype
                _ = [enc.pop(d) for d in ['original_shape', 'source'] if d in enc]
                var_dict = {'dims': dims, 'enc': enc, 'dtype': dtype, 'attrs': chunk[var].attrs}
                chunk_dict[var] = var_dict

    try:
        xr3 = xr.combine_by_coords(coords_list, compat='override', data_vars='minimal', coords='all', combine_attrs='override')
    except:
        xr3 = xr.merge(coords_list, compat='override', combine_attrs='override')

    # Create the blank dataset
    for var, var_dict in chunk_dict.items():
        dims = var_dict['dims']
        shape = tuple(xr3[c].shape[0] for c in dims)
        xr3[var] = (dims, np.full(shape, np.nan, var_dict['dtype']))
        xr3[var].attrs = var_dict['attrs']
        xr3[var].encoding = var_dict['enc']

    # Update the attributes in the coords from the first ds
    for coord in xr3.coords:
        xr3[coord].encoding = datasets[0][coord].encoding
        xr3[coord].attrs = datasets[0][coord].attrs

    # Fill the dataset with data
    for chunk in datasets:
        for var in chunk.data_vars:
            if isinstance(chunk[var].variable._data, np.ndarray):
                xr3[var].loc[chunk[var].transpose(*chunk_dict[var]['dims']).coords.indexes] = chunk[var].transpose(*chunk_dict[var]['dims']).values
            elif isinstance(chunk[var].variable._data, xr.core.indexing.MemoryCachedArray):
                c1 = chunk[var].copy().load().transpose(*chunk_dict[var]['dims'])
                xr3[var].loc[c1.coords.indexes] = c1.values
                c1.close()
                del c1
            else:
                raise TypeError('Dataset data should be either an ndarray or a MemoryCachedArray.')

    return xr3


def calc_river_reach_reductions(catch_id, reductions, reduction_cols):
    """

    """
    with booklet.open(river_catch_path) as f:
        c1 = f[int(catch_id)]

    with booklet.open(river_reach_mapping_path) as f:
        branches = f[int(catch_id)]

    # TODO: Package the flow up by catch_id so that there is less work here
    # flows = {}
    # with booklet.open(utils.river_flows_rec_path) as f:
    #     for way_id in branches:
    #         flows[int(way_id)] = f[int(way_id)]

    # flows_df = pd.DataFrame.from_dict(flows, orient='index', columns=['flow'])
    # flows_df.index.name = 'nzsegment'
    # flows_df = flows_df.reset_index()

    plan1 = reductions[reduction_cols + ['geometry']]
    # plan1 = plan0.to_crs(2193)

    ## Calc reductions per nzsegment given sparse geometry input
    c2 = plan1.overlay(c1)
    c2['sub_area'] = c2.area

    c2['combo_area'] = c2.groupby('nzsegment')['sub_area'].transform('sum')

    c2b = c2.copy()

    results_list = []
    for col in reduction_cols:
        c2b['prop_reductions'] = c2b[col]*(c2b['sub_area']/c2['combo_area'])
        c3 = c2b.groupby('nzsegment')[['prop_reductions', 'sub_area']].sum()

        ## Add in missing areas and assume that they are 0 reductions
        c1['tot_area'] = c1.area

        c4 = pd.merge(c1.drop('geometry', axis=1), c3, on='nzsegment', how='left')
        c4.loc[c4['prop_reductions'].isnull(), ['prop_reductions', 'sub_area']] = 0

        c4['reduction'] = (c4['prop_reductions'] * c4['sub_area'])/c4['tot_area']

        c5 = c4[['nzsegment', 'reduction']].rename(columns={'reduction': col}).groupby('nzsegment').sum().round(2)
        results_list.append(c5)

    results = pd.concat(results_list, axis=1)

    ## Scale the reductions to the flows
    c4 = c4.merge(flows_df, on='nzsegment')

    c4['base_flow'] = c4.flow * 100
    c4['prop_flow'] = c4.flow * c4['reduction']

    c5 = c4[['nzsegment', 'base_flow', 'prop_flow']].set_index('nzsegment').copy()
    c5 = {r: list(v.values()) for r, v in c5.to_dict('index').items()}

    props_index = np.array(list(branches.keys()), dtype='int32')
    props_val = np.zeros(props_index.shape)
    for h, reach in enumerate(branches):
        branch = branches[reach]
        t_area = np.zeros(branch.shape)
        prop_area = t_area.copy()

        for i, b in enumerate(branch):
            if b in c5:
                t_area1, prop_area1 = c5[b]
                t_area[i] = t_area1
                prop_area[i] = prop_area1
            else:
                prop_area[i] = 0

        p1 = (np.sum(prop_area)/np.sum(t_area))
        if p1 < 0:
            props_val[h] = 0
        else:
            props_val[h] = p1

    props = xr.Dataset(data_vars={'reduction': (('nzsegment'), np.round(props_val*100).astype('int8')) # Round to nearest even number
                                  },
                        coords={'nzsegment': props_index}
                        ).sortby('nzsegment')

    return props





# import matplotlib.pyplot as plt
# count, bins, ignored = plt.hist(s, 100, density=True, align='mid')
# x = np.linspace(min(bins), max(bins), 10000)
# pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
#         / (x * sigma * np.sqrt(2 * np.pi)))
# plt.plot(x, pdf, linewidth=2, color='r')
# plt.axis('tight')
# plt.show()
