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

##############################################
### Parameters

base_path = '/media/nvme1/data/OLW/web_app'
# base_path = '/home/mike/data/OLW/web_app'
# %cd '/home/mike/data/OLW/web_app'

base_path = pathlib.Path(base_path)

rec_rivers_feather = '/media/nvme1/data/NIWA/REC25_rivers/rec25_rivers_clean.feather'
rec_catch_feather = '/media/nvme1/data/NIWA/REC25_watersheds/rec25_watersheds_clean.feather'

nzrec_data_path = '/media/nvme1/git/nzrec/data'

segment_id_col = 'nzsegment'

output_path = base_path.joinpath('output')
output_path.mkdir(parents=True, exist_ok=True)

assets_path = output_path.joinpath('assets')
assets_path.mkdir(parents=True, exist_ok=True)

### Rivers
conc_csv_path = base_path.joinpath('StBD3.csv')
river_sites_path = base_path.joinpath('olw_river_sites.feather')

river_flows_rec_path = assets_path.joinpath('rivers_flows_rec.blt')
river_flows_area_path = assets_path.joinpath('rivers_flows_area.blt')

river_loads_rec_path = assets_path.joinpath('rivers_loads_rec.blt')
river_loads_area_path = assets_path.joinpath('rivers_loads_area.blt')

# rec_delin_file = output_path.joinpath('rivers_reach_delineation.feather')
# major_catch_file = output_path.joinpath('rivers_major_catch.feather')
# catch_file = output_path.joinpath('rivers_catch.feather')

river_sites_catch_path = assets_path.joinpath('rivers_sites_catchments.blt')
river_reach_mapping_path = assets_path.joinpath('rivers_reaches_mapping.blt')
river_reach_gbuf_path = assets_path.joinpath('rivers_reaches.blt')
river_catch_path = assets_path.joinpath('rivers_catchments_minor.blt')
river_catch_major_path = assets_path.joinpath('rivers_catchments_major.blt')

river_sims_path = output_path.joinpath('rivers_sims')
river_sims_path.mkdir(parents=True, exist_ok=True)

river_sims_h5_path = river_sims_path.joinpath('rivers_sims.h5')
river_sims_gam_path = river_sims_path.joinpath('rivers_sims_gam.h5')
river_reach_error_path = assets_path.joinpath('rivers_reaches_error.h5')
river_reach_error_gam_path = output_path.joinpath('rivers_reaches_error_gam.h5')
river_reach_loads_path = assets_path.joinpath('rivers_reaches_loads.h5')
river_reach_loads_area_path = assets_path.joinpath('rivers_reaches_loads_area.h5')

land_cover_path = base_path.joinpath('lcdb-v50-land-cover-database-version-50-mainland-new-zealand.gpkg')
# parcels_path = base_path.joinpath('nz-primary-land-parcels.gpkg')

## Sims params
# conc_perc = np.arange(2, 101, 2, dtype='int8')
conc_perc = np.arange(1, 101, 1, dtype='int8')
n_samples_year = [12, 26, 52, 104, 364]
n_years = [5, 10, 20, 30]
start = 0.01
end = 2.72
change = 0.1

catch_lc_path = output_path.joinpath('rivers_catch_lc.blt')

land_cover_reductions = {'Exotic Forest': 0, 'High Producing Exotic Grassland': 30, 'Low Producing Grassland': 10, 'Forest - Harvested': 0, 'Orchard, Vineyard or Other Perennial Crop': 10, 'Short-rotation Cropland': 30, 'Other': 0}

catch_lc_clean_path = assets_path.joinpath('rivers_catch_lc.blt')


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

lakes_sims_path = output_path.joinpath('lakes_sims')
lakes_sims_path.mkdir(parents=True, exist_ok=True)

lakes_sims_h5_path = lakes_sims_path.joinpath('lakes_sims.h5')
lakes_error_path = assets_path.joinpath('lakes_error.h5')
lakes_lc_path = assets_path.joinpath('lakes_catch_lc.blt')

## Model data
lakes_data_path = base_path.joinpath('lakes_wq_data.csv')
lakes_data_clean_path = base_path.joinpath('lakes_wq_data_clean.feather')
lakes_class_csv = base_path.joinpath('fenz_lakes_classification.csv')
lakes_stdev_path = output_path.joinpath('lakes_stdev.h5')


### GW

## Model data
gw_data_path = base_path.joinpath('gw_points_data.hdf')

## Spatial data
gw_points_gbuf_path = assets_path.joinpath('gw_points.pbf')
gw_points_path = output_path.joinpath('gw_points.feather')

## Power calcs
gw_sims_path = output_path.joinpath('gw_sims')
gw_sims_path.mkdir(parents=True, exist_ok=True)

gw_points_error_path = assets_path.joinpath('gw_points_error.h5')

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

    """
    print(error)

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

            # red1 = np.empty((len(conc_perc), n), dtype='int16')
            # for i, v in enumerate(conc_perc):
            #     l1 = np.interp(np.arange(n), [0, n-1], [10000, v*100 ]).round().astype('int16')
            #     red1[i] = l1

            # red2 = np.tile(red1, n_sims).reshape((len(conc_perc), n_sims, n))
            red2 = np.tile(red1, n_sims).reshape(rand_shape)

            # r1 = np.log(rng.lognormal(0, error, rand_shape))
            # r1 = rng.normal(0, error, rand_shape)
            # r1[r1 < -1] = -1
            # r1 = (r1*10000).astype('int16')
            # r1 = np.tile(r1a, len(conc_perc)).reshape((len(conc_perc), n_sims, n))
            # r2 = np.log(rng.lognormal(0, error, rand_shape))
            r2 = rng.normal(0, error, rand_shape)
            # r2[r2 < -1] = -1
            # r2 = (r2*red2).astype('int16')
            # r2 = r2.astype('uint16')
            # r2 = np.tile(r2a, len(conc_perc)).reshape((len(conc_perc), n_sims, n))

            # ones = np.ones(red2.shape)*10

            # o1 = stats.ttest_ind(r1, red2 + r2, axis=1)

            p2 = power_test(np.arange(n), red2 + r2, min_p_value=0.05)

            # p1 = np.empty(dep1.shape[:2], dtype='int16')
            # for i, v in enumerate(ind1):
            #     o1 = stats.ttest_ind(v, dep1[i])
            #     p1[i] = o1.pvalue*10000

            # p2 = (((o1.pvalue < 0.05).sum()/n_sims) *100).round().astype('int8')

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

    # Run checks - requires psutil which I don't want to make it a dep yet...
    # available_memory = getattr(psutil.virtual_memory(), 'available')
    # dims_dict = dict(xr3.coords.dims)
    # size = 0
    # for var, var_dict in chunk_dict.items():
    #     dims = var_dict['dims']
    #     dtype_size = var_dict['dtype'].itemsize
    #     n_dims = np.prod([dims_dict[dim] for dim in dims])
    #     size = size + (n_dims*dtype_size)

    # if size >= available_memory:
    #     raise MemoryError('Trying to create a dataset of size {}MB, while there is only {}MB available.'.format(int(size*10**-6), int(available_memory*10**-6)))

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







# import matplotlib.pyplot as plt
# count, bins, ignored = plt.hist(s, 100, density=True, align='mid')
# x = np.linspace(min(bins), max(bins), 10000)
# pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
#         / (x * sigma * np.sqrt(2 * np.pi)))
# plt.plot(x, pdf, linewidth=2, color='r')
# plt.axis('tight')
# plt.show()
