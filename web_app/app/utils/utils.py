#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:01:00 2023

@author: mike
"""
import pathlib
import zstandard as zstd
import pickle
import codecs
import numpy as np
import geopandas as gpd
import base64
import io
import booklet
import pandas as pd
import xarray as xr

# import parameters as param
import utils.parameters as param

##############################################
### Parameters

param_weights = {
    'peri': {
        'Total nitrogen': 0.4,
        'Total phosphorus': 0.4,
        'Visual Clarity': 0.2
        },
    'mci': {
        'Total nitrogen': 0.5,
        'Total phosphorus': 0.2,
        'Visual Clarity': 0.3
        },
    'sediment': {
        'Total nitrogen': 0,
        'Total phosphorus': 0,
        'Visual Clarity': 1
        },
    }

weights_encoding = {'missing_value': -99, 'dtype': 'int16', 'scale_factor': 0.1}

weights_params = list(param_weights['peri'].keys())


###############################################
### Helper Functions


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


# def encode_xr(obj: xr.Dataset):
#     """

#     """
#     i1 = io.BytesIO()
#     hdf5tools.xr_to_hdf5(obj, i1)
#     str_obj = codecs.encode(i1.read(), encoding="base64").decode()

#     return str_obj


# def decode_xr(str_obj):
#     """

#     """
#     i1 = io.BytesIO(codecs.decode(str_obj.encode(), encoding="base64"))
#     x1 = xr.load_dataset(i1)

#     return x1


def encode_obj(obj):
    """

    """
    cctx = zstd.ZstdCompressor(level=1)
    c_obj = codecs.encode(cctx.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)), encoding="base64").decode()

    return c_obj


def decode_obj(str_obj):
    """

    """
    dctx = zstd.ZstdDecompressor()
    obj1 = dctx.decompress(codecs.decode(str_obj.encode(), encoding="base64"))
    d1 = pickle.loads(obj1)

    return d1


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


def parse_gis_file(contents, filename):
    """

    """
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        plan1 = gpd.read_file(io.BytesIO(decoded))

        output = encode_obj(plan1)
    except:
        output = ['Wrong file type. It must be a GeoPackage (gpkg).']

    return output


def check_reductions_input(new_reductions, base_reductions):
    """

    """
    base_typos = base_reductions.typology.unique()
    try:
        missing_typos = np.in1d(new_reductions.typology.unique(), base_typos).all()
    except:
        missing_typos = False

    return missing_typos


def diff_reductions(new_reductions, base_reductions, reduction_cols):
    """

    """
    new_reductions1 = new_reductions.set_index('typology').sort_index()[reduction_cols]
    base_reductions1 = base_reductions.set_index('typology').sort_index()[reduction_cols]
    temp1 = new_reductions1.compare(base_reductions1, align_axis=0)

    return list(temp1.columns)


def set_default_rivers_reach_reductions(catch_id):
    """

    """
    red1 = xr.open_dataset(param.rivers_reductions_model_path, engine='h5netcdf')

    with booklet.open(param.rivers_reach_mapping_path) as f:
        branches = f[int(catch_id)][int(catch_id)]

    base_props = red1.sel(nzsegment=branches).sortby('nzsegment').copy().load()
    red1.close()
    del red1
    # print(base_props)

    data = encode_obj(base_props)

    return data


def set_default_lakes_reach_reductions(lake_id):
    """

    """
    red1 = xr.open_dataset(param.lakes_reductions_model_path, engine='h5netcdf')

    base_props = red1.sel(LFENZID=int(lake_id), drop=True).load().copy()
    red1.close()
    del red1

    data = encode_obj(base_props)

    return data


def set_default_eco_reach_weights(catch_id):
    """

    """
    red1 = xr.open_dataset(param.eco_reach_weights_path, engine='h5netcdf')

    with booklet.open(param.rivers_reach_mapping_path) as f:
        branches = f[int(catch_id)][int(catch_id)]

    base_props = red1.sel(nzsegment=branches).sortby('nzsegment').copy().load()
    red1.close()
    del red1
    # print(base_props)

    data = encode_obj(base_props)

    return data


def calc_river_reach_reductions(catch_id, new_reductions, base_reductions, diff_cols):
    """
    This assumes that the concentration is the same throughout the entire greater catchment. If it's meant to be different, then each small catchment must be set and multiplied by the area to weight the contribution downstream.
    """
    with booklet.open(param.rivers_catch_path) as f:
        catches1 = f[int(catch_id)]

    with booklet.open(param.rivers_reach_mapping_path) as f:
        branches = f[int(catch_id)]

    with booklet.open(param.rivers_loads_rec_path) as f:
        loads = f[int(catch_id)][diff_cols]

    new_reductions0 = new_reductions[diff_cols + ['geometry']]
    not_all_zeros = new_reductions0[diff_cols].sum(axis=1) > 0
    new_reductions1 = new_reductions0.loc[not_all_zeros]

    ## Calc reductions per nzsegment given sparse geometry input
    c2 = new_reductions1.overlay(catches1)
    c2['sub_area'] = c2.area

    c2['combo_area'] = c2.groupby('nzsegment')['sub_area'].transform('sum')

    c2b = c2.copy()
    catches1['tot_area'] = catches1.area
    catches1 = catches1.drop('geometry', axis=1)

    results_list = []
    for col in diff_cols:
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
    props_val = np.zeros((len(param.red_ratios), len(props_index)))

    reach_red = {}
    for ind in diff_cols:
        c4 = results[[ind]].merge(loads[[ind]], on='nzsegment')

        c4['base'] = c4[ind + '_y'] * 100

        for r, ratio in enumerate(param.red_ratios):
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

    new_props = xr.Dataset(data_vars={ind: (('reduction_perc', 'nzsegment'), values)  for ind, values in reach_red.items()},
                       coords={'nzsegment': props_index,
                                'reduction_perc': param.red_ratios}
                       )

    return new_props


def calc_river_reach_eco_weights(catch_id, new_reductions, base_reductions):
    """
    This assumes that the concentration is the same throughout the entire greater catchment. If it's meant to be different, then each small catchment must be set and multiplied by the area to weight the contribution downstream.
    """
    diff_cols = weights_params

    with booklet.open(param.rivers_catch_path) as f:
        catches1 = f[int(catch_id)]

    with booklet.open(param.rivers_reach_mapping_path) as f:
        branches = f[int(catch_id)]

    with booklet.open(param.rivers_loads_rec_path) as f:
        loads = f[int(catch_id)][diff_cols]

    new_reductions0 = new_reductions[diff_cols + ['geometry']]
    not_all_zeros = new_reductions0[diff_cols].sum(axis=1) > 0
    new_reductions1 = new_reductions0.loc[not_all_zeros]

    ## Calc reductions per nzsegment given sparse geometry input
    c2 = new_reductions1.overlay(catches1)
    c2['sub_area'] = c2.area

    c2['combo_area'] = c2.groupby('nzsegment')['sub_area'].transform('sum')

    c2b = c2.copy()
    catches1['tot_area'] = catches1.area
    catches1 = catches1.drop('geometry', axis=1)

    results_list = []
    for col in diff_cols:
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
    props_val = np.zeros((len(param.red_ratios), len(props_index)))

    reach_red = {}
    for ind in diff_cols:
        c4 = results[[ind]].merge(loads[[ind]], on='nzsegment')

        c4['base'] = c4[ind + '_y'] * 100

        for r, ratio in enumerate(param.red_ratios):
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

    new_props = xr.Dataset(data_vars={ind: (('reduction_perc', 'nzsegment'), values)  for ind, values in reach_red.items()},
                       coords={'nzsegment': props_index,
                                'reduction_perc': param.red_ratios}
                       )

    arr_list = []
    for ind, weights in param_weights.items():
        arr1 = sum([new_props[p]*w for p, w in weights.items()])
        arr1.name = ind
        arr1.encoding = weights_encoding
        arr_list.append(arr1)

    reach_weights2 = xr.merge(arr_list).sortby('nzsegment')

    return reach_weights2


def calc_lake_reach_reductions(lake_id, new_reductions, base_reductions, diff_cols):
    """
    This assumes that the concentration is the same throughout the entire greater catchment. If it's meant to be different, then each small catchment must be set and multiplied by the area to weight the contribution downstream.
    """
    with booklet.open(param.lakes_catches_minor_path, 'r') as f:
        catches1 = f[str(lake_id)]

    with booklet.open(param.lakes_reaches_mapping_path) as f:
        branches = f[int(lake_id)]

    with booklet.open(param.lakes_loads_rec_path) as f:
        loads = f[int(lake_id)][diff_cols]

    new_reductions0 = new_reductions[diff_cols + ['geometry']]
    not_all_zeros = new_reductions0[diff_cols].sum(axis=1) > 0
    new_reductions1 = new_reductions0.loc[not_all_zeros]

    ## Calc reductions per nzsegment given sparse geometry input
    c2 = new_reductions1.overlay(catches1)
    c2['sub_area'] = c2.area

    c2['combo_area'] = c2.groupby('nzsegment')['sub_area'].transform('sum')

    c2b = c2.copy()
    catches1['tot_area'] = catches1.area
    catches1 = catches1.drop('geometry', axis=1)

    results_list = []
    for col in diff_cols:
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
    props_val = np.zeros((len(param.red_ratios)))

    reach_red = {}
    for ind in diff_cols:
        c4 = results[[ind]].merge(loads[[ind]], on='nzsegment')

        c4['base'] = c4[ind + '_y'] * 100

        for r, ratio in enumerate(param.red_ratios):
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

    props = xr.Dataset(data_vars={ind: (('reduction_perc'), values)  for ind, values in reach_red.items()},
                       coords={
                                'reduction_perc': param.red_ratios}
                       )

    return props


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
















































































