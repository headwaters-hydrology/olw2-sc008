#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 13:37:46 2022

@author: mike
"""
import io
import xarray as xr
import dash
from dash import dcc, html, dash_table, callback, ctx
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import dash_leaflet as dl
import dash_leaflet.express as dlx
from dash_extensions.javascript import assign, arrow_function
import pandas as pd
import numpy as np
# import requests
import zstandard as zstd
import codecs
import pickle
import geopandas as gpd
from gistools import vector
import os
# import tethysts
import base64
import geobuf
import pathlib
import hdf5plugin
import booklet
import hdf5tools

# from .app import app
# from . import utils

# from app import app
# import utils

##########################################
### Parameters

# assets_path = pathlib.Path(os.path.split(os.path.realpath(os.path.dirname(__file__)))[0]).joinpath('assets')

dash.register_page(
    __name__,
    path='/land-cover',
    title='Land Cover Reductions',
    name='land_cover',
    description='Land Cover Reductions'
)

### Paths
assets_path = pathlib.Path(os.path.realpath(os.path.dirname(__file__))).parent.joinpath('assets')

app_base_path = pathlib.Path('/assets')

base_data_url = 'https://b2.tethys-ts.xyz/file/'

lc_url = '{}olw-data/olw-sc008/olw_land_cover_reductions.gpkg'.format(base_data_url)
rivers_red_url = '{}olw-data/olw-sc008/olw_rivers_reductions.csv.zip'.format(base_data_url)

rivers_reductions_model_path = assets_path.joinpath('rivers_reductions_modelled.h5')
rivers_catch_pbf_path = app_base_path.joinpath('rivers_catchments.pbf')
catch_lc_pbf_path = assets_path.joinpath('rivers_catch_lc_pbf.blt')

rivers_reach_gbuf_path = assets_path.joinpath('rivers_reaches.blt')
# rivers_loads_path = assets_path.joinpath('rivers_reaches_loads.h5')
# rivers_flows_path = assets_path.joinpath('rivers_flows_rec.blt')
rivers_lc_clean_path = assets_path.joinpath('rivers_catch_lc.blt')
rivers_reach_mapping_path = assets_path.joinpath('rivers_reaches_mapping.blt')
rivers_sites_path = assets_path.joinpath('rivers_sites_catchments.blt')
river_catch_name_path = assets_path.joinpath('rivers_catchments_names.blt')

# rivers_catch_lc_dir = assets_path.joinpath('rivers_land_cover_gpkg')
rivers_catch_lc_gpkg_str = '{base_url}olw-data/olw-sc008/rivers_land_cover_gpkg/{catch_id}_rivers_land_cover_reductions.gpkg'

### Layout
map_height = 700
center = [-41.1157, 172.4759]

attribution = 'Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'

# freq_mapping = {12: 'monthly', 26: 'fortnightly', 52: 'weekly', 104: 'twice weekly', 364: 'daily'}
# time_periods = [5, 10, 20, 30]

reach_style = dict(weight=4, opacity=1, color='white')
lc_style = dict(weight=1, opacity=0.7, color='white', dashArray='3', fillOpacity=0.7)
classes = [0, 20, 40, 60, 80]
bins = classes.copy()
bins.append(101)
# colorscale = ['#808080', '#FED976', '#FEB24C', '#FC4E2A', '#BD0026', '#800026']
# colorscale = ['#808080', '#FED976', '#FD8D3C', '#E31A1C', '#800026']
colorscale = ['#ffffd4','#fed98e','#fe9929','#d95f0e','#993404']
# reductions_colorscale = ['#edf8fb','#b2e2e2','#66c2a4','#2ca25f','#006d2c']
# ctg = ["{}%+".format(cls, classes[i + 1]) for i, cls in enumerate(classes[1:-1])] + ["{}%+".format(classes[-1])]
# ctg.insert(0, 'NA')
ctg = ["{}%+".format(cls, classes[i + 1]) for i, cls in enumerate(classes[:-1])] + ["{}%+".format(classes[-1])]
# ctg.insert(0, 'NA')

site_point_radius = 6

reduction_ratios = range(10, 101, 10)
red_ratios = np.array(list(reduction_ratios), dtype='int8')

rivers_points_hideout = {'classes': [], 'colorscale': ['#232323'], 'circleOptions': dict(fillOpacity=1, stroke=True, weight=1, color='black', radius=site_point_radius), 'colorProp': 'nzsegment'}

rivers_indicator_dict = {'BD': 'Visual Clarity', 'EC': 'E.coli', 'DRP': 'Dissolved reactive phosporus', 'NH': 'Ammoniacal nitrogen', 'NO': 'Nitrate', 'TN': 'Total nitrogen', 'TP': 'Total phosphorus'}

rivers_reduction_cols = list(rivers_indicator_dict.values())

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

lc_style_handle = assign("""function style2(feature, context){
    const {classes, colorscale, style, colorProp} = context.props.hideout;  // get props from hideout
    const value = feature.properties[colorProp];  // get value the determines the color
    for (let i = 0; i < classes.length; ++i) {
        if (value >= classes[i]) {
            style.fillColor = colorscale[i];  // set the fill color according to the class
        }
    }
    return style;
}""", name='rivers_lc_style_handle')

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


### Colorbar
# colorbar_base = dl.Colorbar(style={'opacity': 0})
# base_reach_style = dict(weight=4, opacity=1, color='white')

indices = list(range(len(ctg) + 1))
colorbar_power = dl.Colorbar(min=0, max=len(ctg), classes=indices, colorscale=colorscale, tooltip=True, tickValues=[item + 0.5 for item in indices[:-1]], tickText=ctg, width=300, height=30, position="bottomright")

reach_hideout = {'colorscale': colorscale, 'classes': classes, 'style': reach_style, 'colorProp': 'nzsegment'}
lc_hideout = {'colorscale': colorscale, 'classes': classes, 'style': lc_style, 'colorProp': 'Nitrate'}

marks = []
for i in range(0, 101, 10):
    if (i % 20) == 0:
        marks.append({'label': str(i) + '%', 'value': i})
    else:
        marks.append({'value': i})


# catch_id = 3076139

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

###############################################
### Initial processing

with booklet.open(rivers_reach_gbuf_path, 'r') as f:
    catches = [int(c) for c in f]

catches.sort()
indicators = list(rivers_indicator_dict.keys())
indicators.sort()

###############################################
### App layout


def layout():
    layout = dmc.Container(
        fluid=True,
        # size='xl',
        # px=0,
        # py=0,
        # my=0,
        # mx=0,
        # ml=0,
        # pl=0,
        # style={
        #     'margin': 0,
        #     'padding': 0
        #     },
        children=[
            dmc.Grid(
                columns=7,
                # grow=True,
                # justify="space-between",
                # align='stretch',
                # style={
                #     'margin': 0,
                #     'padding': 0
                #     },
                children=[
                    dmc.Col(
                        span=3,
                        # style={
                        #     # 'margin-top': 20,
                        #     'margin': 0,
                        #     'padding': 0,
                        #     },
                        children=dmc.Accordion(
                            value="1",
                            chevronPosition='left',
                            children=[
                            dmc.AccordionItem([
                                # html.H5('(1) Catchment selection', style={'font-weight': 'bold'}),
                                dmc.AccordionControl('(1) Catchment Selection', style={'font-size': 18}),
                                dmc.AccordionPanel([

                                    html.Label('(1a) Select a catchment on the map:'),
                                    dmc.Text(id='catch_name_lc', weight=700, style={'margin-top': 10}),
                                    ]
                                    )
                                ],
                                value='1'
                                ),

                            dmc.AccordionItem([
                                dmc.AccordionControl('(2) Query Options', style={'font-size': 18}),
                                dmc.AccordionPanel([
                                    dmc.Text('(2a) Select Indicator:'),
                                    dcc.Dropdown(options=[{'label': rivers_indicator_dict[d], 'value': d} for d in indicators], id='indicator_lc', optionHeight=40, clearable=False),

                                    dmc.Text('(2b) Change the percent of the reductions applied. 100% is the max realistic reduction (This option only applies to the river segments):', style={'margin-top': 20}),
                                    dmc.Slider(id='Reductions_slider_lc',
                                                value=100,
                                                mb=35,
                                                step=10,
                                                # min=10,
                                                showLabelOnHover=True,
                                                disabled=False,
                                                marks=marks
                                                ),
                                    dmc.Text('NOTE', weight=700, underline=True, style={'margin-top': 20}),
                                    dmc.Text('The river segments can be added to the map via the layer button on the top right corner of the map.')
                                    ],
                                    )
                                ],
                                value='2'
                                ),

                            dmc.AccordionItem([
                                dmc.AccordionControl('(3) Download Results', style={'font-size': 18}),
                                dmc.AccordionPanel([
                                    dmc.Text('(3a) Download land cover reductions for the selected catchment (gpkg):'),
                                    dcc.Loading(
                                    type="default",
                                    children=[
                            dmc.Anchor(dmc.Button('Download land cover'), href='', id='lc_dl1')],
                                    ),
                                    ],
                                    ),
                                dmc.AccordionPanel([
                                    dmc.Text('(3b) Download river reach reductions for the selected catchment (csv):'),
                                    dcc.Loading(
                                    type="default",
                                    children=[
                            dmc.Button('Download reaches', id='reach_dl_btn'),
                            dcc.Download(id='reach_dl1')],
                                    ),
                                    ],
                                    ),
                                dmc.AccordionPanel([
                                    dmc.Text('(3c) Download land cover reductions for all NZ (gpkg):'),
                                    dcc.Loading(
                                    type="default",
                                    children=[
                            dmc.Anchor(dmc.Button('Download land cover NZ-wide'), href=lc_url, id='lc_dl2')],
                                    ),
                                    ],
                                    ),
                                dmc.AccordionPanel([
                                    dmc.Text('(3d) Download river reach reductions for all NZ (csv):'),
                                    dcc.Loading(
                                    type="default",
                                    children=[
                            dmc.Anchor(dmc.Button('Download reaches NZ-wide'), href=rivers_red_url, id='reach_dl2')],
                                    ),
                                    ],
                                    )
                                ],
                                value='3'
                                ),

                            ],
                            # style={
                            #     'margin': 0,
                            #     'padding': 0
                            #     },
                            # className='four columns', style={'margin': 10}
                            )
                        ),
                    dmc.Col(
                        span=4,
                        # style={
                        #     'margin-top': 20
                        #     },
                        children=html.Div([
                            dl.Map(center=center, zoom=6, children=[
                                dl.LayersControl([
                                    dl.BaseLayer(dl.TileLayer(attribution=attribution, opacity=0.7), checked=True, name='OpenStreetMap'),
                                    dl.BaseLayer(dl.TileLayer(url='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', attribution='Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap (CC-BY-SA)', opacity=0.6), checked=False, name='OpenTopoMap'),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(url=str(rivers_catch_pbf_path), format="geobuf", id='catch_map_lc', zoomToBoundsOnClick=True, zoomToBounds=False, options=dict(style=catch_style_handle))), name='Catchments', checked=True),
                                    # dl.GeoJSON(url='', format="geobuf", id='base_reach_map', options=dict(style=base_reaches_style_handle)),

                                    # dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='sites_points', options=dict(pointToLayer=sites_points_handle), hideout={'circleOptions': dict(fillOpacity=1, stroke=False, radius=5, color='black')})), name='Monitoring sites', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='reductions_poly_lc', hoverStyle=arrow_function(dict(weight=5, color='#666', dashArray='')), options=dict(style=lc_style_handle), hideout={})), name='Land cover', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='reach_map_lc', options={}, hideout={}, hoverStyle=arrow_function(dict(weight=8, color='black', dashArray='')))), name='Rivers', checked=False),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='sites_points_lc', options=dict(pointToLayer=sites_points_handle), hideout=rivers_points_hideout)), name='Monitoring sites', checked=False),
                                    ], id='layers_lc'),
                                colorbar_power,
                                # html.Div(id='colorbar', children=colorbar_base),
                                # dmc.Group(id='colorbar', children=colorbar_base),
                                dcc.Markdown(id="info_lc", className="info", style={"position": "absolute", "top": "10px", "right": "160px", "z-index": "1000"})
                                                ], style={'width': '100%', 'height': 700, 'margin': "auto", "display": "block"}, id="map2_lc"),

                            ],
                            # className='five columns', style={'margin': 10}
                            ),
                        ),
                    ]
                    ),
            dcc.Store(id='catch_id_lc', data=''),
            dcc.Store(id='powers_obj_lc', data=''),
            dcc.Store(id='reaches_obj_lc', data=''),
            dcc.Store(id='base_reductions_obj_lc', data=''),
            ]
        )

    return layout


###############################################
### Callbacks

@callback(
    Output('catch_id_lc', 'data'),
    [Input('catch_map_lc', 'click_feature')]
    )
def update_catch_id(feature):
    """

    """
    # print(ds_id)
    catch_id = ''
    if feature is not None:
        if not feature['properties']['cluster']:
            catch_id = str(feature['id'])

    return catch_id


@callback(
    Output('catch_name_lc', 'children'),
    [Input('catch_id_lc', 'data')]
    )
def update_catch_name(catch_id):
    """

    """
    # print(ds_id)
    if catch_id != '':
        with booklet.open(river_catch_name_path) as f:
            catch_name = f[int(catch_id)]

        return catch_name


@callback(
        Output('reach_map_lc', 'data'),
        Input('catch_id_lc', 'data'),
        )
# @cache.memoize()
def update_reaches(catch_id):
    if catch_id != '':
        with booklet.open(rivers_reach_gbuf_path, 'r') as f:
            data = base64.b64encode(f[int(catch_id)]).decode()

    else:
        data = ''

    return data


@callback(
        Output('reductions_poly_lc', 'data'),
        Input('catch_id_lc', 'data'),
        )
# @cache.memoize()
def update_lc_map(catch_id):
    if catch_id != '':
        with booklet.open(catch_lc_pbf_path, 'r') as f:
            data = base64.b64encode(f[int(catch_id)]).decode()

    else:
        data = ''

    return data


@callback(
        Output('sites_points_lc', 'data'),
        Input('catch_id_lc', 'data'),
        )
# @cache.memoize()
def update_monitor_sites(catch_id):
    if catch_id != '':
        with booklet.open(rivers_sites_path, 'r') as f:
            data = base64.b64encode(f[int(catch_id)]).decode()

    else:
        data = ''

    return data


@callback(
        Output('reach_map_lc', 'options'),
        Input('reach_map_lc', 'hideout'),
        Input('catch_id_lc', 'data')
        )
# @cache.memoize()
def update_reaches_option(hideout, catch_id):
    trig = ctx.triggered_id

    if (len(hideout) == 0) or (trig == 'catch_id_lc'):
        options = dict(style=base_reach_style_handle)
    else:
        options = dict(style=reach_style_handle)

    return options


@callback(
        Output('base_reductions_obj_lc', 'data'),
        Input('catch_id_lc', 'data'),
        prevent_initial_call=True
        )
# @cache.memoize()
def update_base_reductions_obj(catch_id):
    data = ''

    if catch_id != '':
        with booklet.open(rivers_lc_clean_path, 'r') as f:
            data = encode_obj(f[int(catch_id)])

    return data


@callback(
    Output('reaches_obj_lc', 'data'),
    Input('base_reductions_obj_lc', 'data'),
    [
      State('catch_id_lc', 'data'),
      ],
    prevent_initial_call=True)
def update_reach_reductions(base_reductions_obj, catch_id):
    """

    """
    if catch_id != '':
        # print('trigger')
        red1 = xr.open_dataset(rivers_reductions_model_path)

        with booklet.open(rivers_reach_mapping_path) as f:
            branches = f[int(catch_id)][int(catch_id)]

        base_props = red1.sel(nzsegment=branches).sortby('nzsegment').load().copy()
        red1.close()
        del red1
        # print(base_props)

        data = encode_obj(base_props)
    else:
        data = ''

    return data


@callback(
    Output('reach_map_lc', 'hideout'),
    [Input('reaches_obj_lc', 'data'),
     Input('indicator_lc', 'value'),
     Input('Reductions_slider_lc', 'value')],
    prevent_initial_call=True
    )
def update_reach_hideout(reaches_obj, indicator, prop_red):
    """

    """
    if (reaches_obj != '') and (reaches_obj is not None) and isinstance(indicator, str):
        ind_name = rivers_indicator_dict[indicator]

        props = decode_obj(reaches_obj)[[ind_name]].sel(reduction_perc=prop_red, drop=True).rename({ind_name: 'reduction'})

        ## Modelled
        color_arr = pd.cut(props.reduction.values, bins, labels=colorscale, right=False).tolist()

        hideout = {'colorscale': color_arr, 'classes': props.nzsegment.values.tolist(), 'style': reach_style, 'colorProp': 'nzsegment'}

    else:
        hideout = {}

    return hideout


@callback(
    Output('reductions_poly_lc', 'hideout'),
    [
     Input('indicator_lc', 'value'),
     ],
    prevent_initial_call=True
    )
def update_lc_hideout(indicator):
    """

    """
    if isinstance(indicator, str):
        ind_name = rivers_indicator_dict[indicator]

        hideout = {'colorscale': colorscale, 'classes': classes, 'style': lc_style, 'colorProp': ind_name}

    else:
        hideout = {}

    return hideout


@callback(
    Output("info_lc", "children"),
    [Input('reaches_obj_lc', 'data'),
      Input("reach_map_lc", "click_feature"),
      Input('reductions_poly_lc', 'click_feature'),
      Input('indicator_lc', 'value'),
      Input('Reductions_slider_lc', 'value')],
    )
def update_map_info(reaches_obj, reach_feature, lc_feature, indicator, prop_red):
    """

    """
    info = """"""

    trig = ctx.triggered_id
    # print(trig)

    if isinstance(indicator, str):
        ind_name = rivers_indicator_dict[indicator]

        if trig == 'reach_map_lc':
            props = decode_obj(reaches_obj)[[ind_name]].sel(reduction_perc=prop_red, drop=True).rename({ind_name: 'reduction'})
            # print(reach_feature)
            feature_id = int(reach_feature['id'])

            if feature_id in props.nzsegment:

                reach_data = props.sel(nzsegment=feature_id)

                info_str = """**nzsegment**: {seg}\n\n**Reduction**: {red}%""".format(red=int(reach_data.reduction), seg=feature_id)

                info = info + info_str

            else:
                info = info + """Click on a reach to see info"""

        elif trig == 'reductions_poly_lc':
            feature = lc_feature['properties']
            # print(feature)

            info_str = """**Typology**: {typo}\n\n**Land Cover**: {lc}\n\n**Reduction**: {red}%""".format(red=int(feature[ind_name]), typo=feature['typology'], lc=feature['land_cover'])

            info = info + info_str

        else:
            info = info + """Click on a reach/polygon to see info"""

    return info


@callback(
    Output("lc_dl1", "href"),
    # Input('indicator_lc', 'value'),
    Input('catch_id_lc', 'data'),
    prevent_initial_call=True,
    )
def download_catch_lc(catch_id):

    if catch_id != '':
        url = rivers_catch_lc_gpkg_str.format(base_url=base_data_url, catch_id=catch_id)

        return url


@callback(
    Output("reach_dl1", "data"),
    Input("reach_dl_btn", "n_clicks"),
    State('catch_id_lc', 'data'),
    State('reaches_obj_lc', 'data'),
    prevent_initial_call=True,
    )
def download_power(n_clicks, catch_id, reaches_obj):

    if catch_id != '':
        props = decode_obj(reaches_obj)

        df1 = props.to_dataframe().reset_index()
        for col in df1.columns:
            df1[col] = df1[col].astype(int)

        df2 = df1.set_index(['reduction_perc', 'nzsegment'])

        return dcc.send_data_frame(df2.to_csv, f"rivers_reductions_{catch_id}.csv")

