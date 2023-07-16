#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 10:00:41 2021

@author: mike
"""
import io
import zstandard as zstd
import codecs
import pickle
import pandas as pd
import numpy as np
# import requests
import xarray as xr
# from shapely.geometry import shape, mapping
# import tethysts
import os
from gistools import vector
import geopandas as gpd
import base64
from dash import dcc, html
import pathlib
import booklet
import dash_leaflet as dl
from dash_extensions.javascript import assign, arrow_function
# import plotly.graph_objs as go

#####################################
#### Parameters

### Paths
assets_path = pathlib.Path(os.path.realpath(os.path.dirname(__file__))).parent.joinpath('assets')

app_base_path = pathlib.Path('/assets')

## Rivers
river_power_model_path = assets_path.joinpath('rivers_reaches_power_modelled.h5')
river_power_moni_path = assets_path.joinpath('rivers_reaches_power_monitored.h5')
rivers_reductions_model_path = assets_path.joinpath('rivers_reductions_modelled.h5')
rivers_catch_pbf_path = app_base_path.joinpath('rivers_catchments.pbf')

rivers_reach_gbuf_path = assets_path.joinpath('rivers_reaches.blt')
# rivers_loads_path = assets_path.joinpath('rivers_reaches_loads.h5')
# rivers_flows_path = assets_path.joinpath('rivers_flows_rec.blt')
rivers_lc_clean_path = assets_path.joinpath('rivers_catch_lc.blt')
rivers_catch_path = assets_path.joinpath('rivers_catchments_minor.blt')
rivers_reach_mapping_path = assets_path.joinpath('rivers_reaches_mapping.blt')
rivers_sites_path = assets_path.joinpath('rivers_sites_catchments.blt')
river_loads_rec_path = assets_path.joinpath('rivers_loads_rec.blt')

rivers_catch_lc_dir = assets_path.joinpath('rivers_land_cover_gpkg')
rivers_catch_lc_gpkg_str = '{}_rivers_land_cover_reductions.gpkg'

## Lakes
lakes_power_combo_path = assets_path.joinpath('lakes_power_combo.h5')
# lakes_power_moni_path = assets_path.joinpath('lakes_power_monitored.h5')
lakes_reductions_model_path = assets_path.joinpath('lakes_reductions_modelled.h5')

lakes_pbf_path = app_base_path.joinpath('lakes_points.pbf')
lakes_poly_gbuf_path = assets_path.joinpath('lakes_poly.blt')
lakes_catches_major_path = assets_path.joinpath('lakes_catchments_major.blt')
lakes_reach_gbuf_path = assets_path.joinpath('lakes_reaches.blt')
lakes_lc_path = assets_path.joinpath('lakes_catch_lc.blt')
lakes_reaches_mapping_path = assets_path.joinpath('lakes_reaches_mapping.blt')
lakes_catches_minor_path = assets_path.joinpath('lakes_catchments_minor.blt')

lakes_catch_lc_dir = assets_path.joinpath('lakes_land_cover_gpkg')
lakes_catch_lc_gpkg_str = '{}_lakes_land_cover_reductions.gpkg'

lakes_loads_rec_path = assets_path.joinpath('lakes_loads_rec.blt')

## GW
gw_error_path = assets_path.joinpath('gw_points_error.h5')

# gw_pbf_path = app_base_path.joinpath('gw_points.pbf')
gw_points_rc_blt = assets_path.joinpath('gw_points_rc.blt')
rc_bounds_gbuf = app_base_path.joinpath('rc_bounds.pbf')


### Layout
map_height = 700
center = [-41.1157, 172.4759]

attribution = 'Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'

freq_mapping = {12: 'monthly', 26: 'fortnightly', 52: 'weekly', 104: 'twice weekly', 364: 'daily'}
time_periods = [5, 10, 20, 30]

style = dict(weight=4, opacity=1, color='white')
classes = [0, 20, 40, 60, 80]
bins = classes.copy()
bins.append(101)
# colorscale = ['#808080', '#FED976', '#FEB24C', '#FC4E2A', '#BD0026', '#800026']
colorscale = ['#808080', '#FED976', '#FD8D3C', '#E31A1C', '#800026']
# reductions_colorscale = ['#edf8fb','#b2e2e2','#66c2a4','#2ca25f','#006d2c']
# ctg = ["{}%+".format(cls, classes[i + 1]) for i, cls in enumerate(classes[1:-1])] + ["{}%+".format(classes[-1])]
# ctg.insert(0, 'NA')
ctg = ["{}%+".format(cls, classes[i + 1]) for i, cls in enumerate(classes[:-1])] + ["{}%+".format(classes[-1])]
# ctg.insert(0, 'NA')

site_point_radius = 6

reduction_ratios = range(10, 101, 10)
red_ratios = np.array(list(reduction_ratios), dtype='int8')

## Rivers

rivers_points_hideout = {'classes': [], 'colorscale': ['#232323'], 'circleOptions': dict(fillOpacity=1, stroke=True, weight=1, color='black', radius=site_point_radius), 'colorProp': 'nzsegment'}

rivers_indicator_dict = {'BD': 'Black disk', 'EC': 'E.coli', 'DRP': 'Dissolved reactive phosporus', 'NH': 'Ammoniacal nitrogen', 'NO': 'Nitrate', 'TN': 'Total nitrogen', 'TP': 'Total phosphorus'}

rivers_reduction_cols = list(rivers_indicator_dict.values())

## Lakes
catch_style = {'fillColor': 'grey', 'weight': 2, 'opacity': 1, 'color': 'black', 'fillOpacity': 0.1}
lake_style = {'fillColor': '#A4DCCC', 'weight': 4, 'opacity': 1, 'color': 'black', 'fillOpacity': 1}
reach_style = {'weight': 2, 'opacity': 0.75, 'color': 'grey'}

lakes_indicator_dict = {'CHLA': 'Chlorophyll a', 'CYANOTOT': 'Total Cyanobacteria', 'ECOLI': 'E.coli', 'NH4N': 'Ammoniacal nitrogen', 'Secchi': 'Secchi Depth', 'TN': 'Total nitrogen', 'TP': 'Total phosphorus'}

lakes_reduction_cols = list(lakes_indicator_dict.values())


## GW
gw_points_hideout = {'classes': [], 'colorscale': ['#808080'], 'circleOptions': dict(fillOpacity=1, stroke=False, radius=site_point_radius), 'colorProp': 'tooltip'}

gw_freq_mapping = {1: 'Yearly', 4: 'Quarterly', 12: 'monthly', 26: 'fortnightly', 52: 'weekly'}

gw_indicator_dict = {'Nitrate': 'Nitrate'}

gw_reductions_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

gw_reductions_options = [{'value': v, 'label': str(v)+'%'} for v in gw_reductions_values]

### Handles

## Rivers
catch_style_handle = assign("""function style(feature) {
    return {
        fillColor: 'grey',
        weight: 2,
        opacity: 1,
        color: 'black',
        fillOpacity: 0.1
    };
}""", name='rivers_catch_style_handle')

base_reach_style_handle = assign("""function style3(feature) {
    return {
        weight: 2,
        opacity: 0.75,
        color: 'grey',
    };
}""", name='rivers_base_reach_style_handle')

reach_style_handle = assign("""function style2(feature, context){
    const {classes, colorscale, style, colorProp} = context.props.hideout;  // get props from hideout
    const value = feature.properties[colorProp];  // get value the determines the color
    for (let i = 0; i < classes.length; ++i) {
        if (value == classes[i]) {
            style.color = colorscale[i];  // set the fill color according to the class
        }
    }
    return style;
}""", name='rivers_reach_style_handle')

sites_points_handle = assign("""function rivers_sites_points_handle(feature, latlng, context){
    const {classes, colorscale, circleOptions, colorProp} = context.props.hideout;  // get props from hideout
    const value = feature.properties[colorProp];  // get value the determines the fillColor
    for (let i = 0; i < classes.length; ++i) {
        if (value == classes[i]) {
            circleOptions.fillColor = colorscale[i];  // set the color according to the class
        }
    }

    return L.circleMarker(latlng, circleOptions);
}""", name='rivers_sites_points_handle')

## Lakes
lake_style_handle = assign("""function style4(feature, context){
    const {classes, colorscale, style, colorProp} = context.props.hideout;  // get props from hideout
    const value = feature.properties[colorProp];  // get value the determines the color
    for (let i = 0; i < classes.length; ++i) {
        if (value == classes[i]) {
            style.color = colorscale[i];  // set the color according to the class
        }
    }

    return style;
}""", name='lakes_lake_style_handle')


## GW
rc_style_handle = assign("""function style(feature) {
    return {
        fillColor: 'grey',
        weight: 2,
        opacity: 1,
        color: 'black',
        fillOpacity: 0.1
    };
}""", name='gw_rc_style_handle')

gw_points_style_handle = assign("""function gw_points_style_handle(feature, latlng, context){
    const {classes, colorscale, circleOptions, colorProp} = context.props.hideout;  // get props from hideout
    const value = feature.properties[colorProp];  // get value the determines the fillColor
    for (let i = 0; i < classes.length; ++i) {
        if (value == classes[i]) {
            circleOptions.fillColor = colorscale[i];  // set the color according to the class
        }
    }

    return L.circleMarker(latlng, circleOptions);
}""", name='gw_points_style_handle')


### Colorbar
colorbar_base = dl.Colorbar(style={'opacity': 0})
base_reach_style = dict(weight=4, opacity=1, color='white')

indices = list(range(len(ctg) + 1))
colorbar_power = dl.Colorbar(min=0, max=len(ctg), classes=indices, colorscale=colorscale, tooltip=True, tickValues=[item + 0.5 for item in indices[:-1]], tickText=ctg, width=300, height=30, position="bottomright")

# mapbox_access_token = "pk.eyJ1IjoibXVsbGVua2FtcDEiLCJhIjoiY2pudXE0bXlmMDc3cTNxbnZ0em4xN2M1ZCJ9.sIOtya_qe9RwkYXj5Du1yg"


#####################################
### Functions


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


def calc_river_reach_reductions(catch_id, new_reductions, base_reductions):
    """
    This assumes that the concentration is the same throughout the entire greater catchment. If it's meant to be different, then each small catchment must be set and multiplied by the area to weight the contribution downstream.
    """
    diff_cols = diff_reductions(new_reductions, base_reductions)

    with booklet.open(rivers_catch_path) as f:
        catches1 = f[int(catch_id)]

    with booklet.open(rivers_reach_mapping_path) as f:
        branches = f[int(catch_id)]

    with booklet.open(river_loads_rec_path) as f:
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
    props_val = np.zeros((len(red_ratios), len(props_index)))

    reach_red = {}
    for ind in diff_cols:
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

    new_props = xr.Dataset(data_vars={ind: (('reduction_perc', 'nzsegment'), values)  for ind, values in reach_red.items()},
                       coords={'nzsegment': props_index,
                                'reduction_perc': red_ratios}
                       )

    return new_props


def calc_lake_reach_reductions(lake_id, new_reductions, base_reductions):
    """
    This assumes that the concentration is the same throughout the entire greater catchment. If it's meant to be different, then each small catchment must be set and multiplied by the area to weight the contribution downstream.
    """
    diff_cols = diff_reductions(new_reductions, base_reductions)

    with booklet.open(lakes_catches_minor_path, 'r') as f:
        catches1 = f[str(lake_id)]

    with booklet.open(lakes_reaches_mapping_path) as f:
        branches = f[int(lake_id)]

    with booklet.open(lakes_loads_rec_path) as f:
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
    props_val = np.zeros((len(red_ratios)))

    reach_red = {}
    for ind in diff_cols:
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

    props = xr.Dataset(data_vars={ind: (('reduction_perc'), values)  for ind, values in reach_red.items()},
                       coords={
                                'reduction_perc': red_ratios}
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



