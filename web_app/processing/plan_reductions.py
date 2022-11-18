#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 11:49:03 2022

@author: mike
"""
import os
from gistools import vector, rec
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import wkb, wkt
import pickle
import io
from shapely.ops import unary_union
import geobuf
import base64
import orjson
import zstandard as zstd
import pathlib

pd.options.display.max_columns = 10

##############################################
### Parameters

base_path = '/media/nvme1/data/OLW/web_app'
# %cd '/home/mike/data/OLW/web_app'

base_path = pathlib.Path(base_path)

plan_file_name = 'test_plan1.gpkg'
plan_file = os.path.join(base_path, plan_file_name)

catch_id = 14295077

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
    if isinstance(obj, str):
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

###############################################
### Run

c1 = read_pkl_zstd(os.path.join(base_path, 'catchments', '14295077.pkl.zst'), True).reset_index()
# c1.to_file(os.path.join(base_path, '14295077_catchments.gpkg'))

r1 = read_pkl_zstd(os.path.join(base_path, 'reaches', '14295077.pkl.zst'), True).reset_index()
# r1.to_file(os.path.join(base_path, '14295077_reaches.gpkg'))

plan1 = gpd.read_file(os.path.join(base_path, 'test_plan1.gpkg')).drop('id', axis=1)

c2 = vector.sjoin(c1, plan1, how='left').drop('index_right', axis=1)

c2.loc[c2['reduction'].isnull(), 'reduction'] = 0
c2['s_area'] = c2.area

c2['combo_area'] = c2.groupby('nzsegment')['s_area'].transform('sum')

c2['prop'] = c2['reduction']*(c2['s_area']/c2['combo_area'])

c3 = c2.groupby('nzsegment')['prop'].sum()

c4 = c1.merge(c3.reset_index(), on='nzsegment')

c4['base_area'] = c4.area * 100

c4['prop_area'] = c4.area * c4['prop']

t_area, prop_area = c4[['base_area', 'prop_area']].sum()

t_reduction = (prop_area)/t_area


up1 = rec.find_upstream(r1.nzsegment.tolist(), r1)

branches = {}
for grp, segs in up1['nzsegment'].reset_index().groupby('start')['nzsegment']:
    branches[grp] = segs.values.astype('int32')

c5 = c4[['nzsegment', 't_area', 'prop_area']].set_index('nzsegment').copy()
c5a = c5.to_dict('index')
c5b = {r: list(v.values()) for r, v in c5a.items()}


props = {}
for reach, branch in branches.items():
    t_area, prop_area = c5.loc[branches[reach]].sum()
    p1 = prop_area/t_area
    props[reach] = p1


def f1():
    props = {}
    for reach, branch in branches.items():
        t_area = []
        a_append = t_area.append
        prop_area = []
        p_append = prop_area.append

        for b in branch:
            t1 = c5a[b]
            a_append(t1['t_area'])
            p_append(t1['prop_area'])

        p1 = np.sum(prop_area)/np.sum(t_area)
        props[reach] = p1


def f2():
    props = {}
    for reach, branch in branches.items():
        t_area = np.zeros(branch.shape)
        prop_area = t_area.copy()

        for i, b in enumerate(branch):
            t1 = c5a[b]
            t_area[i] = t1['t_area']
            prop_area[i] = t1['prop_area']

        p1 = np.sum(prop_area)/np.sum(t_area)
        props[reach] = p1


def f3():
    props = {}
    for reach, branch in branches.items():
        t_area = []
        a_append = t_area.append
        prop_area = []
        p_append = prop_area.append

        for b in branch:
            t_area1, prop_area1 = c5b[b]
            a_append(t_area1)
            p_append(prop_area1)

        p1 = np.sum(prop_area)/np.sum(t_area)
        props[reach] = p1


def calc_reach_reductions(catch_id, base_path, plan_file, reduction_col='reduction'):
    """

    """
    plan1 = gpd.read_file(plan_file)[[reduction_col, 'geometry']].copy()
    c1 = read_pkl_zstd(os.path.join(base_path, 'catchments', '{}.pkl.zst'.format(catch_id)), True).reset_index()

    c2 = vector.sjoin(c1, plan1, how='left').drop('index_right', axis=1)
    c2.loc[c2['reduction'].isnull(), 'reduction'] = 0
    c2['s_area'] = c2.area

    c2['combo_area'] = c2.groupby('nzsegment')['s_area'].transform('sum')

    c2['prop'] = c2['reduction']*(c2['s_area']/c2['combo_area'])

    c3 = c2.groupby('nzsegment')['prop'].sum()
    c4 = c1.merge(c3.reset_index(), on='nzsegment')
    c4['base_area'] = c4.area * 100
    c4['prop_area'] = c4.area * c4['prop']

    c5 = c4[['nzsegment', 'base_area', 'prop_area']].set_index('nzsegment').copy()
    c5 = {r: list(v.values()) for r, v in c5.to_dict('index').items()}

    branches = read_pkl_zstd(os.path.join(base_path, 'reaches', '{}_mapping.pkl.zst'.format(catch_id)), True)

    props = {}
    for reach, branch in branches.items():
        t_area = []
        a_append = t_area.append
        prop_area = []
        p_append = prop_area.append

        for b in branch:
            t_area1, prop_area1 = c5b[b]
            a_append(t_area1)
            p_append(prop_area1)

        p1 = np.sum(prop_area)/np.sum(t_area)
        props[reach] = p1
































