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
import tempfile

##############################################
### Parameters

# base_path = '/media/nvme1/data/OLW/web_app'
base_path = '/home/mike/data/OLW/web_app'
# %cd '/home/mike/data/OLW/web_app'

base_path = pathlib.Path(base_path)

rec_rivers_feather = '/home/mike/data/NIWA/REC25_rivers/rec25_rivers_clean.feather'
rec_catch_feather = '/home/mike/data/NIWA/REC25_watersheds/rec25_watersheds_clean.feather'

rc_bounds_gpkg = '/home/mike/data/statsnz/regional-council-2023-clipped-generalised.gpkg'

nzrec_data_path = '/home/mike/git/nzrec/data'

segment_id_col = 'nzsegment'

output_path = base_path.joinpath('output')
output_path.mkdir(parents=True, exist_ok=True)

assets_path = output_path.joinpath('assets')
assets_path.mkdir(parents=True, exist_ok=True)

indicators = {'rivers': ['Visual Clarity', 'E.coli', 'Dissolved reactive phosporus', 'Ammoniacal nitrogen', 'Nitrate', 'Total nitrogen', 'Total phosphorus'],
              'lakes': ['E.coli', 'Ammoniacal nitrogen', 'Total nitrogen', 'Total phosphorus', 'Chlorophyll a', 'Total Cyanobacteria', 'Secchi Depth']
              }

indicator_dict = {
    'Visual Clarity': 'sediment',
    'E.coli': 'e.coli',
    'Dissolved reactive phosporus': 'DRP',
    'Ammoniacal nitrogen': 'NNN',
    'Nitrate': 'NNN',
    'Total nitrogen': 'TN',
    'Total phosphorus': 'TP',
    'Chlorophyll a': 'e.coli',
    'Total Cyanobacteria': 'e.coli',
    'Secchi Depth': 'sediment',
    }

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

lc_red_csv_path = lc_base_path.joinpath('typology_reductions.csv')

### Rivers
sites_loc_csv = base_path.joinpath('olw_river_sites_locations.csv')
sites_rec_csv = base_path.joinpath('lawa_to_nzsegment.csv')
sites_names_csv = base_path.joinpath('LAWARiverSiteswithRCIDs.csv')

## concentrations
rivers_conc_base_path = base_path.joinpath('rivers')
rivers_conc_csv_path1 = rivers_conc_base_path.joinpath('NutrientConcsYields.csv')
rivers_conc_csv_path2 = rivers_conc_base_path.joinpath('EcoliConcsYields.csv')
rivers_conc_csv_path3 = rivers_conc_base_path.joinpath('updated-suspended-sediment-yield-estimator-and-estuarine-tra.csv')

rivers_ref_conc3_csv_path = rivers_conc_base_path.joinpath('reference_conc_rec_level_3.csv')
rivers_ref_conc2_csv_path = rivers_conc_base_path.joinpath('reference_conc_rec_level_2.csv')
rivers_ref_conc_csv_path = rivers_conc_base_path.joinpath('reference_conc_rec_clean.csv')
rivers_ref_load_csv_path = rivers_conc_base_path.joinpath('reference_load_rec_clean.csv')

# catch_break_points_gpkg = rivers_conc_base_path.joinpath('catch_management_points.gpkg')

## Errors and powers
river_errors_model_path = base_path.joinpath('rivers_errors_modelled_v02.csv')
river_errors_moni_path = base_path.joinpath('rivers_errors_monitored.csv')
river_sites_path = base_path.joinpath('olw_river_sites.gpkg')

## Flows and loads
river_flows_rec_path = output_path.joinpath('rivers_flows_rec.blt')
# river_flows_area_path = assets_path.joinpath('rivers_flows_area.blt')

river_loads_rec_path = assets_path.joinpath('rivers_loads_rec.blt')
# river_loads_area_path = assets_path.joinpath('rivers_loads_area.blt')

# rec_delin_file = output_path.joinpath('rivers_reach_delineation.feather')
# major_catch_file = output_path.joinpath('rivers_major_catch.feather')
# catch_file = output_path.joinpath('rivers_catch.feather')

# Individual catchment land covers
rivers_catch_lc_dir = assets_path.joinpath('rivers_land_cover_gpkg')
rivers_catch_lc_dir.mkdir(parents=True, exist_ok=True)

rivers_catch_lc_gpkg_str = '{}_rivers_land_cover_reductions.gpkg'
rivers_catch_lc_gpkg_path = output_path.joinpath('olw_land_cover_reductions.gpkg')
rivers_red_csv_path = output_path.joinpath('olw_rivers_reductions.csv')

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
n_samples_year = [4, 12, 26, 52, 104, 364]
n_years = [5, 10, 20, 30]

## Reductions
catch_lc_path = assets_path.joinpath('rivers_catch_lc.blt')
catch_lc_pbf_path = assets_path.joinpath('rivers_catch_lc_pbf.blt')
river_reductions_model_path = assets_path.joinpath('rivers_reductions_modelled.h5')

# catch_lc_clean_path = assets_path.joinpath('rivers_catch_lc.blt')


### Lakes
## Source data processing
lakes_source_path = base_path.joinpath('lakes')

lakes_source_data_path = lakes_source_path.joinpath('lakes_source_data.csv')
lakes_deseason_path = lakes_source_path.joinpath('lakes_deseason_data.csv')
lakes_deseason_comp_path = lakes_source_path.joinpath('lakes_deseason_comparison.csv')

lakes_fenz_catch_path = base_path.joinpath('lakes_catchments_fenz.gpkg')
lakes_fenz_poly_path = base_path.joinpath('lakes_polygons_fenz.gpkg')

## Geo processing
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
lakes_reductions_model_path = assets_path.joinpath('lakes_reductions_modelled.h5')

lakes_lc_path = assets_path.joinpath('lakes_catch_lc.blt')

lakes_loads_rec_path = assets_path.joinpath('lakes_loads_rec.blt')

## Model data
lakes_rupesh_stdev_path = base_path.joinpath('lakes_stdev_v04.csv')
lakes_data_path = base_path.joinpath('lakes_wq_data.csv')
# lakes_data_clean_path = base_path.joinpath('lakes_wq_data_clean.feather')
lakes_class_csv = base_path.joinpath('fenz_lakes_classification.csv')
lakes_stdev_model_path = output_path.joinpath('lakes_stdev_modelled_v05.h5')
lakes_stdev_moni_path = output_path.joinpath('lakes_stdev_monitored_v05.csv')

lakes_missing_3rd_path = output_path.joinpath('lakes_stdev_missing.gpkg')

### GW

## Source data
# gw_data_path = base_path.joinpath('gw_points_data.hdf')
gw_monitoring_data_path = base_path.joinpath('gw_monitoring_data_v04.nc')

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


def power_sims_gw(error, n_years, n_samples_year, n_sims, output_path):
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
            red1 = np.interp(np.arange(n), [0, n-1], [5, 5*perc*0.01])

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


def calc_river_reach_reductions(feature, catch_id, reduction_ratios=range(10, 101, 10)):
    """

    """
    print(catch_id)

    red_ratios = np.array(list(reduction_ratios), dtype='int8')
    reduction_cols = indicators[feature]

    with booklet.open(river_catch_path) as f:
        catches1 = f[int(catch_id)]

    with booklet.open(river_reach_mapping_path) as f:
        branches = f[int(catch_id)]

    with booklet.open(river_loads_rec_path) as f:
        loads = f[int(catch_id)][reduction_cols]

    with booklet.open(catch_lc_path) as f:
        reductions = f[int(catch_id)]

    plan1 = reductions[reduction_cols + ['geometry']]
    # plan1 = plan0.to_crs(2193)

    ## Calc reductions per nzsegment given sparse geometry input
    c2 = plan1.overlay(catches1)
    c2['sub_area'] = c2.area

    c2['combo_area'] = c2.groupby('nzsegment')['sub_area'].transform('sum')

    c2b = c2.copy()
    catches1['tot_area'] = catches1.area
    catches1 = catches1.drop('geometry', axis=1)

    results_list = []
    for col in reduction_cols:
        c2b['prop_reductions'] = c2b[col]*(c2b['sub_area']/c2['combo_area'])
        c3 = c2b.groupby('nzsegment')[['prop_reductions', 'sub_area']].sum()

        ## Add in missing areas and assume that they are 0 reductions
        c4 = pd.merge(catches1, c3, on='nzsegment', how='left')
        c4.loc[c4['prop_reductions'].isnull(), ['prop_reductions', 'sub_area']] = 0
        c4['reduction'] = (c4['prop_reductions'] * c4['sub_area'])/c4['tot_area']

        c5 = c4[['nzsegment', 'reduction']].rename(columns={'reduction': col}).groupby('nzsegment').sum().round(2)
        results_list.append(c5)

    results = pd.concat(results_list, axis=1)

    ## Scale the reductions
    props_index = np.array(list(branches.keys()), dtype='int32')
    props_val = np.zeros((len(red_ratios), len(props_index)))

    reach_red = {}
    for ind in reduction_cols:
        c4 = results[[ind]].merge(loads[[ind]], on='nzsegment')

        c4['base'] = c4[ind + '_y'] * 100

        for r, ratio in enumerate(red_ratios):
            c4['prop'] = c4[ind + '_y'] * c4[ind + '_x'] * ratio * 0.01
            c4b = c4[['base', 'prop']]
            c5 = {r: list(v.values()) for r, v in c4b.to_dict('index').items()}

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

                t_area_sum = np.sum(t_area)
                if t_area_sum <= 0:
                    props_val[r, h] = 0
                else:
                    p1 = np.sum(prop_area)/t_area_sum
                    props_val[r, h] = p1

            reach_red[ind] = np.round(props_val*100).astype('int8') # Round to nearest even number

    props = xr.Dataset(data_vars={ind: (('reduction_perc', 'nzsegment'), values)  for ind, values in reach_red.items()},
                       coords={'nzsegment': props_index,
                                'reduction_perc': red_ratios}
                       )

    # file1 = tempfile.NamedTemporaryFile()
    # hdf5tools.xr_to_hdf5(props, file1)
    # props = props.assign_coords(catch_id=catch_id).expand_dims('catch_id').sortby(['nzsegment', 'reduction_perc'])

    return props


def calc_lakes_reach_reductions(feature, lake_id, reduction_ratios=range(10, 101, 10)):
    """

    """
    print(lake_id)

    red_ratios = np.array(list(reduction_ratios), dtype='int8')
    reduction_cols = indicators[feature]

    with booklet.open(lakes_catches_minor_path) as f:
        catches1 = f[int(lake_id)]

    with booklet.open(lakes_reaches_mapping_path) as f:
        branches = f[int(lake_id)]

    with booklet.open(lakes_loads_rec_path) as f:
        loads = f[int(lake_id)][reduction_cols]

    with booklet.open(lakes_lc_path) as f:
        reductions = f[int(lake_id)]

    plan1 = reductions[reduction_cols + ['geometry']]
    # plan1 = plan0.to_crs(2193)

    ## Calc reductions per nzsegment given sparse geometry input
    c2 = plan1.overlay(catches1)
    c2['sub_area'] = c2.area

    c2['combo_area'] = c2.groupby('nzsegment')['sub_area'].transform('sum')

    c2b = c2.copy()
    catches1['tot_area'] = catches1.area
    catches1 = catches1.drop('geometry', axis=1)

    results_list = []
    for col in reduction_cols:
        c2b['prop_reductions'] = c2b[col]*(c2b['sub_area']/c2['combo_area'])
        c3 = c2b.groupby('nzsegment')[['prop_reductions', 'sub_area']].sum()

        ## Add in missing areas and assume that they are 0 reductions
        c4 = pd.merge(catches1, c3, on='nzsegment', how='left')
        c4.loc[c4['prop_reductions'].isnull(), ['prop_reductions', 'sub_area']] = 0
        c4['reduction'] = (c4['prop_reductions'] * c4['sub_area'])/c4['tot_area']

        c5 = c4[['nzsegment', 'reduction']].rename(columns={'reduction': col}).groupby('nzsegment').sum().round(2)
        results_list.append(c5)

    results = pd.concat(results_list, axis=1)

    ## Scale the reductions
    props_val = np.zeros((len(red_ratios)))

    reach_red = {}
    for ind in reduction_cols:
        c4 = results[[ind]].merge(loads[[ind]], on='nzsegment')

        c4['base'] = c4[ind + '_y'] * 100

        for r, ratio in enumerate(red_ratios):
            c4['prop'] = c4[ind + '_y'] * c4[ind + '_x'] * ratio * 0.01
            c4b = c4[['base', 'prop']]
            c5 = {r: list(v.values()) for r, v in c4b.to_dict('index').items()}

            branch = branches
            t_area = np.zeros(branch.shape)
            prop_area = t_area.copy()

            for i, b in enumerate(branch):
                if b in c5:
                    t_area1, prop_area1 = c5[b]
                    t_area[i] = t_area1
                    prop_area[i] = prop_area1
                else:
                    prop_area[i] = 0

            t_area_sum = np.sum(t_area)
            if t_area_sum <= 0:
                props_val[r] = 0
            else:
                p1 = np.sum(prop_area)/t_area_sum
                props_val[r] = p1

            reach_red[ind] = np.round(props_val*100).astype('int8') # Round to nearest even number

    # props = xr.Dataset(data_vars={ind: (('reduction_perc', 'LFENZID'), values)  for ind, values in reach_red.items()},
    #                    coords={'LFENZID': lake_id,
    #                             'reduction_perc': red_ratios}
    #                    )
    props = xr.Dataset(data_vars={ind: (('reduction_perc'), values)  for ind, values in reach_red.items()},
                       coords={
                                'reduction_perc': red_ratios}
                       )
    props = props.assign_coords(LFENZID=np.array(lake_id, dtype='int32')).expand_dims('LFENZID')

    # file1 = tempfile.NamedTemporaryFile()
    # hdf5tools.xr_to_hdf5(props, file1)
    # props = props.assign_coords(catch_id=catch_id).expand_dims('catch_id').sortby(['nzsegment', 'reduction_perc'])

    return props


def get_directly_upstream_ways(way_id, node_way, way, way_index):
    """

    """
    ways_up = set([way_id])

    new_ways = set(way_index[int(way_id)]).difference(ways_up)

    down_node = way[int(way_id)][-1]
    down_ways = set(node_way[down_node])

    new_ways = new_ways.difference(down_ways)

    return new_ways


def discrete_resample(df, freq_code, agg_fun, remove_inter=False, **kwargs):
    """
    Function to properly set up a resampling class for discrete data. This assumes a linear interpolation between data points.

    Parameters
    ----------
    df: DataFrame or Series
        DataFrame or Series with a time index.
    freq_code: str
        Pandas frequency code. e.g. 'D'.
    agg_fun : str
        The aggregation function to be applied on the resampling object.
    **kwargs
        Any keyword args passed to Pandas resample.

    Returns
    -------
    Pandas DataFrame or Series
    """
    if isinstance(df, (pd.Series, pd.DataFrame)):
        if isinstance(df.index, pd.DatetimeIndex):
            reg1 = pd.date_range(df.index[0].ceil(freq_code), df.index[-1].floor(freq_code), freq=freq_code)
            reg2 = reg1[~reg1.isin(df.index)]
            if isinstance(df, pd.Series):
                s1 = pd.Series(np.nan, index=reg2)
            else:
                s1 = pd.DataFrame(np.nan, index=reg2, columns=df.columns)
            s2 = pd.concat([df, s1]).sort_index()
            s3 = s2.interpolate('time')
            s4 = (s3 + s3.shift(-1))/2
            s5 = s4.resample(freq_code, **kwargs).agg(agg_fun).dropna()

            if remove_inter:
                index1 = df.index.floor(freq_code).unique()
                s6 = s5[s5.index.isin(index1)].copy()
            else:
                s6 = s5
        else:
            raise ValueError('The index must be a datetimeindex')
    else:
        raise TypeError('The object must be either a DataFrame or a Series')

    return s6


# import matplotlib.pyplot as plt
# count, bins, ignored = plt.hist(s, 100, density=True, align='mid')
# x = np.linspace(min(bins), max(bins), 10000)
# pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
#         / (x * sigma * np.sqrt(2 * np.pi)))
# plt.plot(x, pdf, linewidth=2, color='r')
# plt.axis('tight')
# plt.show()
