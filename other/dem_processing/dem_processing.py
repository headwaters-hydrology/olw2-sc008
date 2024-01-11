#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 17:01:10 2022

@author: mike
"""
import os
import pathlib
import xarray as xr
from pysheds.grid import Grid
from tethysts import Tethys
# from gistools import vector, rec
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import wkb, wkt
import pickle
from shapely.ops import unary_union
import orjson
import zstandard as zstd
import rioxarray as rxr

####################################################
### Parameters

base_path = pathlib.Path('/media/nvme1/data/OLW/web_app')

dem_remote = {'bucket': 'linz-data', 'public_url': 'https://b2.tethys-ts.xyz/file', 'version': 4}

dem_ds_id = '54374801c0311a98a0f8e5ef'

catch_pkl = 'rec_catch3plus.pkl.zst'
rivers_pkl = 'rec_streams3plus.pkl.zst'
spatial_ref_nc = 'spatial_ref.nc'

nzsegment = 1000038

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

####################################################
### Get REC catchments

rec_shed = read_pkl_zstd(base_path.joinpath(catch_pkl), True)

catch1 = rec_shed.loc[nzsegment].geometry
catch1_buff = catch1.buffer(0.002)

bounds = tuple(np.round(catch1_buff.bounds, 3))

### Get DEM

t1 = Tethys([dem_remote])
stns = t1.get_stations(dem_ds_id, geometry=catch1_buff.__geo_interface__)

dem1 = t1.get_results(dem_ds_id, stns[0]['station_id']).squeeze().altitude

dem2 = dem1.where((dem1.lon > bounds[0]) & (dem1.lon < bounds[2]) & (dem1.lat > bounds[1]) & (dem1.lat < bounds[3]), drop=True)

dem3 = dem2.drop(['height', 'time']).rename({'lat': 'y', 'lon': 'x'}).to_dataset()

# rec_shed.loc[[nzsegment]].to_file('/media/nvme1/data/OLW/web_app/test.shp')

spatial_ref = xr.load_dataset(base_path.joinpath(spatial_ref_nc)).spatial_ref

dem3 = dem3.assign_coords(spatial_ref=spatial_ref)

dem3.rio.to_raster(base_path.joinpath('test.tif'))




















































































