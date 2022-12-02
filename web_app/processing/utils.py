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
from gistools import vector, rec
import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
import hdf5tools
from scipy import stats

##############################################
### Parameters

base_path = '/media/nvme1/data/OLW/web_app'
# %cd '/home/mike/data/OLW/web_app'

base_path = pathlib.Path(base_path)

rec_rivers_shp = '/media/nvme1/data/NIWA/REC25_rivers/rec25_rivers.shp'
rec_catch_shp ='/media/nvme1/data/NIWA/REC25_watersheds/rec25_watersheds.shp'

segment_id_col = 'nzsegment'

output_path = base_path.joinpath('output')

output_path.mkdir(parents=True, exist_ok=True)

rec_delin_file = 'reach_delineation.pkl.zst'
reach_mapping_file = 'reach_mapping.pkl.zst'
major_catch_file = 'major_catch.pkl.zst'
catch_file = 'catch.pkl.zst'
catch_simple_file = 'catch_simple.pkl.zst'

assets_path = output_path.joinpath('assets')
assets_path.mkdir(parents=True, exist_ok=True)

reach_gbuf_path = assets_path.joinpath('reaches')
reach_gbuf_path.mkdir(parents=True, exist_ok=True)

catch_path = assets_path.joinpath('catchments')
catch_path.mkdir(parents=True, exist_ok=True)

reach_map_path = assets_path.joinpath('reach_mappings')
reach_map_path.mkdir(parents=True, exist_ok=True)

conc_csv_path = base_path.joinpath('StBD3.csv')

error_pkl_path = assets_path.joinpath('catch_error.pkl.zst')

## Sims params
conc_perc = np.arange(2, 101, 2, dtype='int8')
n_samples_year = [12, 26, 52, 104, 364]
n_years = [5, 10, 20, 30]







#############################################
### Functions


# def geojson_to_geobuf(geojson):
#     return base64.b64encode(geobuf.encode(geojson)).decode()


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


def error_cats():
    """

    """
    start = 0.025
    list1 = [0.001, 0.005, 0.01, start]

    while start < 2.7:
        if start < 0.1:
            start = round(start*2, 3)
        else:
            start = round(start*1.2, 3)
        list1.append(start)

    return list1


def catch_sims(error, n_years, n_samples_year, n_sims, output_path):
    """

    """
    print(error)
    error1 = int(error*1000)

    n_samples = np.prod(hdf5tools.utils.cartesian([n_samples_year, n_years]), axis=1)
    n_samples = list(set(n_samples))
    n_samples.sort()

    filler = np.empty((1, len(n_samples), len(conc_perc)), dtype='int8')

    rng = np.random.default_rng()

    for ni, n in enumerate(n_samples):
        # print(n)

        for pi, perc in enumerate(conc_perc):
            red1 = np.interp(np.arange(n), [0, n-1], [10000, perc*100 ]).round().astype('int16')

            # red1 = np.empty((len(conc_perc), n), dtype='int16')
            # for i, v in enumerate(conc_perc):
            #     l1 = np.interp(np.arange(n), [0, n-1], [10000, v*100 ]).round().astype('int16')
            #     red1[i] = l1

            # red2 = np.tile(red1, n_sims).reshape((len(conc_perc), n_sims, n))
            red2 = np.tile(red1, n_sims).reshape((n_sims, n))

            rand_shape = (n_sims, n)

            r1 = rng.normal(0, error, rand_shape)
            r1[r1 < -1] = -1
            r1 = (r1*10000).astype('int16')
            # r1 = np.tile(r1a, len(conc_perc)).reshape((len(conc_perc), n_sims, n))
            r2 = rng.normal(0, error, rand_shape)
            r2[r2 < -1] = -1
            r2 = (r2*red2).astype('int16')
            # r2 = r2.astype('uint16')
            # r2 = np.tile(r2a, len(conc_perc)).reshape((len(conc_perc), n_sims, n))

            # ones = np.ones(red2.shape)*10

            ind1 = 10000 + r1
            dep1 = red2 + r2

            o1 = stats.ttest_ind(ind1, dep1, axis=1)

            # p1 = np.empty(dep1.shape[:2], dtype='int16')
            # for i, v in enumerate(ind1):
            #     o1 = stats.ttest_ind(v, dep1[i])
            #     p1[i] = o1.pvalue*10000

            p2 = (((o1.pvalue < 0.05).sum()/n_sims) *100).round().astype('int8')

            filler[0][ni][pi] = p2

    props = xr.Dataset(data_vars={'power': (('error', 'n_samples', 'conc_perc'), filler)
                                  },
                       coords={'error': np.array([error1], dtype='int16'),
                               'n_samples': np.array(n_samples, dtype='int16'),
                               'conc_perc': conc_perc}
                       )

    output = os.path.join(output_path, str(error1) + '.h5')
    hdf5tools.xr_to_hdf5(props, output)

    return output













































































