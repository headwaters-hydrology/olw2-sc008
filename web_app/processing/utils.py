#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 09:19:21 2022

@author: mike
"""
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

conc_pkl_path = assets_path.joinpath('catch_conc.pkl.zst')


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

















































































