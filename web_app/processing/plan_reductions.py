#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 11:49:03 2022

@author: mike
"""
import xarray as xr
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
from scipy import stats

pd.options.display.max_columns = 10

##############################################
### Parameters

base_path = '/media/nvme1/data/OLW/web_app'
# %cd '/home/mike/data/OLW/web_app'

base_path = pathlib.Path(base_path)

plan_file_name = 'test_plan1.gpkg'
plan_file = os.path.join(base_path, plan_file_name)

catch_id = 14295077

s_freq = np.array([12, 24, 52, 104, 365])
s_lens = np.array([5, 10, 20, 30])

n_years = 5
n_samples_year = 12

test_reach = 14231918

#############################################
### Functions


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
            [1, 4, 7],
            [1, 5, 6],
            [1, 5, 7],
            [2, 4, 6],
            [2, 4, 7],
            [2, 5, 6],
            [2, 5, 7],
            [3, 4, 6],
            [3, 4, 7],
            [3, 5, 6],
            [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]

    return out

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


def seasonal_sine(n_samples_year, n_years):
    """

    """
    l1 = np.concatenate((np.linspace(start=2, stop=6, num=int(n_samples_year/2), endpoint=False), np.linspace(start=6, stop=2, num=int(n_samples_year/2), endpoint=False)))

    s1 = np.sin(np.pi/l1)
    s2 = np.tile(s1, n_years)

    return s2


def calc_reach_reductions(catch_id, base_path, plan_file, reduction_col='reduction'):
    """
    This assumes that the concentration is the same throughout the entire greater catchment. If it's meant to be different, then each small catchment must be set and multiplied by the area to weight the contribution downstream.
    """
    plan1 = gpd.read_file(plan_file)[[reduction_col, 'geometry']].copy()
    c1 = read_pkl_zstd(os.path.join(base_path, 'catchments', '{}.pkl.zst'.format(catch_id)), True).reset_index()

    c2 = vector.sjoin(c1, plan1, how='left').drop('index_right', axis=1)
    c2.loc[c2[reduction_col].isnull(), reduction_col] = 0
    c2['s_area'] = c2.area

    c2['combo_area'] = c2.groupby('nzsegment')['s_area'].transform('sum')

    c2['prop'] = c2[reduction_col]*(c2['s_area']/c2['combo_area'])

    c3 = c2.groupby('nzsegment')['prop'].sum()
    c4 = c1.merge(c3.reset_index(), on='nzsegment')
    area = c4.area
    c4['base_area'] = area * 100
    c4['prop_area'] = area * c4['prop']

    c5 = c4[['nzsegment', 'base_area', 'prop_area']].set_index('nzsegment').copy()
    c5 = {r: list(v.values()) for r, v in c5.to_dict('index').items()}

    branches = read_pkl_zstd(os.path.join(base_path, 'reaches', '{}_mapping.pkl.zst'.format(catch_id)), True)

    # props = {}
    # for reach, branch in branches.items():
    #     t_area = []
    #     a_append = t_area.append
    #     prop_area = []
    #     p_append = prop_area.append

    #     for b in branch:
    #         t_area1, prop_area1 = c5[b]
    #         a_append(t_area1)
    #         p_append(prop_area1)

    #     p1 = (np.sum(prop_area)/np.sum(t_area))
    #     props[reach] = p1

    props_index = np.array(list(branches.keys()), dtype='int32')
    props_val = np.zeros(props_index.shape)
    for h, reach in enumerate(branches):
        branch = branches[reach]
        t_area = np.zeros(branch.shape)
        prop_area = t_area.copy()

        for i, b in enumerate(branch):
            t_area1, prop_area1 = c5[b]
            t_area[i] = t_area1
            prop_area[i] = prop_area1

        p1 = (np.sum(prop_area)/np.sum(t_area))
        props_val[h] = p1

    props = xr.Dataset(data_vars={'reduction': (('reach'), props_val)
                                  },
                       coords={'reach': props_index}
                       )

    return props


def t_test(props, n_samples_year, n_years, ):
    """

    """
    red1 = 1 - props.reduction.values

    red2 = np.tile(red1, (n_samples_year*n_years,1)).transpose()
    ones = np.ones(red2.shape)

    rng = np.random.default_rng()

    r1 = rng.uniform(-.1, .1, red2.shape)
    r2 = rng.uniform(-.1, .1, red2.shape)

    season1 = seasonal_sine(n_samples_year, n_years)

    o1 = stats.ttest_ind((ones+r1)*season1, (red2+r2)*season1, axis=1)

    props = props.assign(p_value=(('reach'), o1.pvalue), t_stat=(('reach'), np.abs(o1.statistic)))

    return props


def apply_filters(props, t_bins=[0, 5, 10, 20, 40, 60, 80, 100], p_cutoff=0.01, reduction_cutoff=0.01):
    """

    """
    c1 = pd.cut(props.t_stat.values, t_bins, labels=t_bins[:-1])

    props = props.assign(t_cat=(('reach'), c1.to_numpy().astype('int8')))

    props = props.where((props.p_value <= p_cutoff) & (props.reduction >= reduction_cutoff), drop=True)
    props['t_cat'] = props['t_cat'].astype('int8')

    return props


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

# t_area, prop_area = c4[['base_area', 'prop_area']].sum()

# t_reduction = (prop_area)/t_area


# up1 = rec.find_upstream(r1.nzsegment.tolist(), r1)

# branches = {}
# for grp, segs in up1['nzsegment'].reset_index().groupby('start')['nzsegment']:
#     branches[grp] = segs.values.astype('int32')

# c5 = c4[['nzsegment', 't_area', 'prop_area']].set_index('nzsegment').copy()
# c5a = c5.to_dict('index')
# c5b = {r: list(v.values()) for r, v in c5a.items()}


# props = {}
# for reach, branch in branches.items():
#     t_area, prop_area = c5.loc[branches[reach]].sum()
#     p1 = prop_area/t_area
#     props[reach] = p1


# def f1():
#     props = {}
#     for reach, branch in branches.items():
#         t_area = []
#         a_append = t_area.append
#         prop_area = []
#         p_append = prop_area.append

#         for b in branch:
#             t1 = c5a[b]
#             a_append(t1['t_area'])
#             p_append(t1['prop_area'])

#         p1 = np.sum(prop_area)/np.sum(t_area)
#         props[reach] = p1


def f2():
    props_index = np.array(list(branches.keys()), dtype='int32')
    props_val = np.zeros(props_index.shape, dtype='int16')
    for h, reach in enumerate(branches):
        branch = branches[reach]
        t_area = np.zeros(branch.shape)
        prop_area = t_area.copy()

        for i, b in enumerate(branch):
            t_area1, prop_area1 = c5[b]
            t_area[i] = t_area1
            prop_area[i] = prop_area1

        p1 = np.round((np.sum(prop_area)/np.sum(t_area)) * 10000)
        props_val[h] = p1


def f3():
    props = {}
    for reach, branch in branches.items():
        t_area = []
        a_append = t_area.append
        prop_area = []
        p_append = prop_area.append

        for b in branch:
            t_area1, prop_area1 = c5[b]
            a_append(t_area1)
            p_append(prop_area1)

        p1 = (np.sum(prop_area)/np.sum(t_area))
        props[reach] = p1


# def f4():
#     props = {}
#     for reach, branch in branches.items():
#         t_area = []
#         a_append = t_area.append
#         prop_area = []
#         p_append = prop_area.append

#         for b in branch:
#             t_area1, prop_area1 = c5[b]
#             a_append(t_area1)
#             p_append(prop_area1)

#         p1 = np.round((np.sum(prop_area)/np.sum(t_area)) * 10000).astype('int16')
#         props[reach] = p1


def calc_reach_reductions(catch_id, base_path, plan_file, reduction_col='reduction'):
    """
    This assumes that the concentration is the same throughout the entire greater catchment. If it's meant to be different, then each small catchment must be set and multiplied by the area to weight the contribution downstream.
    """
    plan1 = gpd.read_file(plan_file)[[reduction_col, 'geometry']].copy()
    c1 = read_pkl_zstd(os.path.join(base_path, 'catchments', '{}.pkl.zst'.format(catch_id)), True).reset_index()

    c2 = vector.sjoin(c1, plan1, how='left').drop('index_right', axis=1)
    c2.loc[c2[reduction_col].isnull(), reduction_col] = 0
    c2['s_area'] = c2.area

    c2['combo_area'] = c2.groupby('nzsegment')['s_area'].transform('sum')

    c2['prop'] = c2[reduction_col]*(c2['s_area']/c2['combo_area'])

    c3 = c2.groupby('nzsegment')['prop'].sum()
    c4 = c1.merge(c3.reset_index(), on='nzsegment')
    area = c4.area
    c4['base_area'] = area * 100
    c4['prop_area'] = area * c4['prop']

    c5 = c4[['nzsegment', 'base_area', 'prop_area']].set_index('nzsegment').copy()
    c5 = {r: list(v.values()) for r, v in c5.to_dict('index').items()}

    branches = read_pkl_zstd(os.path.join(base_path, 'reaches', '{}_mapping.pkl.zst'.format(catch_id)), True)

    # props = {}
    # for reach, branch in branches.items():
    #     t_area = []
    #     a_append = t_area.append
    #     prop_area = []
    #     p_append = prop_area.append

    #     for b in branch:
    #         t_area1, prop_area1 = c5[b]
    #         a_append(t_area1)
    #         p_append(prop_area1)

    #     p1 = (np.sum(prop_area)/np.sum(t_area))
    #     props[reach] = p1

    props_index = np.array(list(branches.keys()), dtype='int32')
    props_val = np.zeros(props_index.shape)
    for h, reach in enumerate(branches):
        branch = branches[reach]
        t_area = np.zeros(branch.shape)
        prop_area = t_area.copy()

        for i, b in enumerate(branch):
            t_area1, prop_area1 = c5[b]
            t_area[i] = t_area1
            prop_area[i] = prop_area1

        p1 = (np.sum(prop_area)/np.sum(t_area))
        props_val[h] = p1

    props = xr.Dataset(data_vars={'reduction': (('reach'), props_val)
                                  },
                       coords={'reach': props_index}
                       )

    return props




props = calc_reach_reductions(catch_id, base_path, plan_file)

s_combos = cartesian([s_lens, s_freq])

n_samples = s_combos.prod(axis=1)
n_samples_year = 12

p2 = props.where(props.reduction > 0, drop=True)
p3 = 1 - p2.reduction.values
p4 = np.tile(p3, (60,1)).transpose()

ones = np.ones(p4.shape)

rng = np.random.default_rng()

r1 = rng.uniform(-.1, .1, p4.shape)
r2 = rng.uniform(-.1, .1, p4.shape)

o1 = stats.ttest_ind(ones+r1, p4+r2, axis=1)

l1 = np.concatenate((np.linspace(start=2, stop=6, num=int(n_samples_year/2), endpoint=False), np.linspace(start=6, stop=2, num=int(n_samples_year/2), endpoint=False)))

s1 = np.sin(np.pi/l1)
s2 = np.tile(s1, int(60/n_samples_year))

stats.ttest_ind((ones[-1]+r1[0])*s2, (p4[-1]+r2[0])*s2)
stats.ttest_ind((ones[-1]+r1[0]), (p4[-1]+r2[0]))

stats.ttest_ind((ones[0]+r1[0])*s2, (p4[0]+r2[0])*s2)
stats.ttest_ind((ones[0]+r1[0]), (p4[0]+r2[0]))




































































