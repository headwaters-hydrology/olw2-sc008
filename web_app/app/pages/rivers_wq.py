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
    path='/rivers-wq',
    title='Rivers Water Quality',
    name='rivers_wq',
    description='Rivers Water Quality'
)

### Paths
assets_path = pathlib.Path(os.path.realpath(os.path.dirname(__file__))).parent.joinpath('assets')

app_base_path = pathlib.Path('/assets')

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

rivers_points_hideout = {'classes': [], 'colorscale': ['#232323'], 'circleOptions': dict(fillOpacity=1, stroke=True, weight=1, color='black', radius=site_point_radius), 'colorProp': 'nzsegment'}

rivers_indicator_dict = {'BD': 'Black disk', 'EC': 'E.coli', 'DRP': 'Dissolved reactive phosporus', 'NH': 'Ammoniacal nitrogen', 'NO': 'Nitrate', 'TN': 'Total nitrogen', 'TP': 'Total phosphorus'}

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
colorbar_base = dl.Colorbar(style={'opacity': 0})
base_reach_style = dict(weight=4, opacity=1, color='white')

indices = list(range(len(ctg) + 1))
colorbar_power = dl.Colorbar(min=0, max=len(ctg), classes=indices, colorscale=colorscale, tooltip=True, tickValues=[item + 0.5 for item in indices[:-1]], tickText=ctg, width=300, height=30, position="bottomright")


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
                                    dcc.Dropdown(options=[{'label': d, 'value': d} for d in catches], id='catch_id', optionHeight=40, clearable=False,
                                                  style={'margin-top': 10}
                                                  ),
                                    ]
                                    )
                                ],
                                value='1'
                                ),

                            dmc.AccordionItem([
                                # html.H5('Optional (2) Customise Reductions Layer', style={'font-weight': 'bold', 'margin-top': 20}),
                                dmc.AccordionControl('(2 - Optional) Customise Reductions Layer', style={'font-size': 18}),
                                dmc.AccordionPanel([
                                    html.Label('(2a) Download reductions polygons as GPKG:'),
                                    dcc.Loading(
                                    id="loading-2",
                                    type="default",
                                    children=[html.Div(dmc.Button("Download reductions",
                                                                  id='dl_btn',
                                                                  # className='btn btn-primary'
                                                                  ),
                                                       # className="me-1",
                                                       style={'margin-top': 10}),
                            dcc.Download(id="dl_poly")],
                                    ),
                                    html.Label('NOTE: Only modify existing values. Do not add additional columns; they will be ignored.', style={
                                        'margin-top': 10
                                    }
                                        ),
                                    html.Label('(2b) Upload modified reductions polygons as GPKG:', style={
                                        'margin-top': 20
                                    }
                                        ),
                                    dcc.Loading(
                                        children=[
                                            dcc.Upload(
                                                id='upload_data_rivers',
                                                children=dmc.Button('Upload reductions',
                                                                     # className="me-1"
                                                                      # style={
                                                                      #     'width': '50%',
                                                                      #     }
                                                ),
                                                style={
                                                    'margin-top': 10
                                                },
                                                multiple=False
                                            ),
                                            ]
                                        ),
                                    dcc.Markdown('', style={
                                        'textAlign': 'left',
                                                    }, id='upload_error_text'),
                                    html.Label('(2c) Process the reductions layer and route the reductions downstream:', style={
                                        'margin-top': 20
                                    }
                                        ),
                                    dcc.Loading(
                                    id="loading-1",
                                    type="default",
                                    children=html.Div([dmc.Button('Process reductions', id='process_reductions_rivers',
                                                                  # className="me-1",
                                                                  n_clicks=0),
                                                        html.Div(id='process_text')],
                                                      style={'margin-top': 10, 'margin-bottom': 10}
                                                      )
                                    ),
                                    ]
                                    )
                                ],
                                value='2'
                                ),

                            dmc.AccordionItem([
                                dmc.AccordionControl('(3) Sampling Options', style={'font-size': 18}),
                                dmc.AccordionPanel([
                                    dmc.Text('(3a) Select Indicator:'),
                                    dcc.Dropdown(options=[{'label': rivers_indicator_dict[d], 'value': d} for d in indicators], id='indicator_rivers', optionHeight=40, clearable=False),
                                    dmc.Text('(3b) Select sampling length (years):', style={'margin-top': 20}),
                                    dmc.SegmentedControl(data=[{'label': d, 'value': str(d)} for d in time_periods],
                                                         id='time_period',
                                                         value='5',
                                                         fullWidth=True,
                                                         color=1,
                                                         ),
                                    dmc.Text('(3c) Select sampling frequency:', style={'margin-top': 20}),
                                    dmc.SegmentedControl(data=[{'label': v, 'value': str(k)} for k, v in freq_mapping.items()],
                                                         id='freq',
                                                         value='12',
                                                         fullWidth=True,
                                                         color=1
                                                         ),
                                    html.Label('(3d) Change the percent of the reductions applied (100% is the max realistic reduction):', style={'margin-top': 20}),
                                    dmc.Slider(id='Reductions_slider',
                                               value=100,
                                               mb=35,
                                               step=10,
                                               # min=10,
                                               showLabelOnHover=True,
                                               disabled=False,
                                               marks=[{'label': str(d) + '%', 'value': d} for d in range(0, 101, 20)]
                                               ),
                                    # dcc.Dropdown(options=[{'label': d, 'value': d} for d in time_periods], id='time_period', clearable=False, value=5),
                                    # html.Label('Select sampling frequency:'),
                                    # dcc.Dropdown(options=[{'label': v, 'value': k} for k, v in freq_mapping.items()], id='freq', clearable=False, value=12),

                                    ],
                                    )
                                ],
                                value='3'
                                ),

                            dmc.AccordionItem([
                                dmc.AccordionControl('(4) Download Results', style={'font-size': 18}),
                                dmc.AccordionPanel([
                                    dmc.Text('(4a) Download power results given the prior sampling options (csv):'),
                                    dcc.Loading(
                                    type="default",
                                    children=[html.Div(dmc.Button("Download power results", id='dl_btn_power_rivers'), style={'margin-bottom': 20, 'margin-top': 10}),
                            dcc.Download(id="dl_power_rivers")],
                                    ),
                                    ],
                                    )
                                ],
                                value='4'
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
                                    dl.BaseLayer(dl.TileLayer(id='tile_layer', attribution=attribution), checked=True, name='OpenStreetMap'),
                                    dl.BaseLayer(dl.TileLayer(url='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', id='opentopo', attribution='Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap (CC-BY-SA)'), checked=False, name='OpenTopoMap'),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(url=str(rivers_catch_pbf_path), format="geobuf", id='catch_map', zoomToBoundsOnClick=True, zoomToBounds=False, options=dict(style=catch_style_handle))), name='Catchments', checked=True),
                                    # dl.GeoJSON(url='', format="geobuf", id='base_reach_map', options=dict(style=base_reaches_style_handle)),

                                    # dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='sites_points', options=dict(pointToLayer=sites_points_handle), hideout={'circleOptions': dict(fillOpacity=1, stroke=False, radius=5, color='black')})), name='Monitoring sites', checked=True),
                                    # dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='reductions_poly')), name='Land use reductions', checked=False),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='reach_map', options={}, hideout={}, hoverStyle=arrow_function(dict(weight=10, color='black', dashArray='')))), name='Rivers', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='sites_points', options=dict(pointToLayer=sites_points_handle), hideout=rivers_points_hideout)), name='Monitoring sites', checked=True),
                                    ], id='layers_rivers'),
                                colorbar_power,
                                # html.Div(id='colorbar', children=colorbar_base),
                                # dmc.Group(id='colorbar', children=colorbar_base),
                                dcc.Markdown(id="info", className="info", style={"position": "absolute", "top": "10px", "right": "160px", "z-index": "1000"})
                                                ], style={'width': '100%', 'height': 700, 'margin': "auto", "display": "block"}, id="map2"),

                            ],
                            # className='five columns', style={'margin': 10}
                            ),
                        ),
                    ]
                    ),
            dcc.Store(id='powers_obj', data=''),
            dcc.Store(id='reaches_obj', data=''),
            dcc.Store(id='custom_reductions_obj', data=''),
            dcc.Store(id='base_reductions_obj', data=''),
            ]
        )

    return layout


###############################################
### Callbacks

@callback(
    Output('catch_id', 'value'),
    [Input('catch_map', 'click_feature')]
    )
def update_catch_id(feature):
    """

    """
    # print(ds_id)
    catch_id = None
    if feature is not None:
        if not feature['properties']['cluster']:
            catch_id = feature['id']

    return catch_id


@callback(
        Output('reach_map', 'data'),
        Input('catch_id', 'value'),
        )
# @cache.memoize()
def update_reaches(catch_id):
    if (catch_id is not None):
        with booklet.open(rivers_reach_gbuf_path, 'r') as f:
            data = base64.b64encode(f[int(catch_id)]).decode()

    else:
        data = ''

    return data


@callback(
        Output('sites_points', 'data'),
        Input('catch_id', 'value'),
        )
# @cache.memoize()
def update_monitor_sites(catch_id):
    if (catch_id is not None):
        with booklet.open(rivers_sites_path, 'r') as f:
            data = base64.b64encode(f[int(catch_id)]).decode()

    else:
        data = ''

    return data


@callback(
        Output('reach_map', 'options'),
        Input('reach_map', 'hideout'),
        Input('catch_id', 'value')
        )
# @cache.memoize()
def update_reaches_option(hideout, catch_id):
    trig = ctx.triggered_id

    if (len(hideout) == 0) or (trig == 'catch_id'):
        options = dict(style=base_reach_style_handle)
    else:
        options = dict(style=reach_style_handle)

    return options


@callback(
        Output('base_reductions_obj', 'data'),
        Input('catch_id', 'value'),
        prevent_initial_call=True
        )
# @cache.memoize()
def update_base_reductions_obj(catch_id):
    data = ''

    if catch_id is not None:
        with booklet.open(rivers_lc_clean_path, 'r') as f:
            data = encode_obj(f[int(catch_id)])

    return data


@callback(
    Output("dl_poly", "data"),
    Input("dl_btn", "n_clicks"),
    State('catch_id', 'value'),
    prevent_initial_call=True,
    )
def download_lc(n_clicks, catch_id):
    if isinstance(catch_id, str):
        path = rivers_catch_lc_dir.joinpath(rivers_catch_lc_gpkg_str.format(catch_id))

        return dcc.send_file(path)


@callback(
        Output('custom_reductions_obj', 'data'), Output('upload_error_text', 'children'),
        Input('upload_data_rivers', 'contents'),
        State('upload_data_rivers', 'filename'),
        State('catch_id', 'value'),
        prevent_initial_call=True
        )
# @cache.memoize()
def update_land_reductions(contents, filename, catch_id):
    data = None
    error_text = ''

    if catch_id is not None:
        if contents is not None:
            data = parse_gis_file(contents, filename)

            if isinstance(data, list):
                error_text = data[0]
                data = None

    return data, error_text


@callback(
    Output('reaches_obj', 'data'), Output('process_text', 'children'),
    Input('process_reductions_rivers', 'n_clicks'),
    Input('base_reductions_obj', 'data'),
    [
      State('catch_id', 'value'),
      State('custom_reductions_obj', 'data'),
      ],
    prevent_initial_call=True)
def update_reach_reductions(click, base_reductions_obj, catch_id, reductions_obj):
    """

    """
    trig = ctx.triggered_id

    if (trig == 'process_reductions_rivers'):
        if isinstance(catch_id, str) and (reductions_obj != '') and (reductions_obj is not None):
            red1 = xr.open_dataset(rivers_reductions_model_path)

            with booklet.open(rivers_reach_mapping_path) as f:
                branches = f[int(catch_id)][int(catch_id)]

            base_props = red1.sel(nzsegment=branches)

            new_reductions = decode_obj(reductions_obj)
            base_reductions = decode_obj(base_reductions_obj)

            new_props = calc_river_reach_reductions(catch_id, new_reductions, base_reductions)
            new_props1 = new_props.combine_first(base_props).sortby('nzsegment').load().copy()
            red1.close()
            del red1
            base_props.close()
            del base_props

            data = encode_obj(new_props1)
            text_out = 'Routing complete'
        else:
            data = ''
            text_out = 'Not all inputs have been selected'
    else:
        if isinstance(catch_id, str):
            # print('trigger')
            red1 = xr.open_dataset(rivers_reductions_model_path)

            with booklet.open(rivers_reach_mapping_path) as f:
                branches = f[int(catch_id)][int(catch_id)]

            base_props = red1.sel(nzsegment=branches).sortby('nzsegment').load().copy()
            red1.close()
            del red1
            # print(base_props)

            data = encode_obj(base_props)
            text_out = ''
        else:
            data = ''
            text_out = ''

    return data, text_out


@callback(
    Output('powers_obj', 'data'),
    [Input('reaches_obj', 'data'), Input('indicator_rivers', 'value'), Input('time_period', 'value'), Input('freq', 'value'), Input('Reductions_slider', 'value')],
    [State('catch_id', 'value')]
    )
def update_powers_data(reaches_obj, indicator, n_years, n_samples_year, prop_red, catch_id):
    """

    """
    if (reaches_obj != '') and (reaches_obj is not None) and isinstance(indicator, str):
        # print('triggered')

        ind_name = rivers_indicator_dict[indicator]

        ## Modelled
        props = decode_obj(reaches_obj)[[ind_name]].sel(reduction_perc=prop_red, drop=True).rename({ind_name: 'reduction'})

        n_samples = int(n_samples_year)*int(n_years)

        power_data = xr.open_dataset(river_power_model_path, engine='h5netcdf')

        with booklet.open(rivers_reach_mapping_path) as f:
            branches = f[int(catch_id)][int(catch_id)]

        power_data1 = power_data.sel(indicator=indicator, nzsegment=branches, n_samples=n_samples, drop=True).load().sortby('nzsegment').copy()
        power_data.close()
        del power_data

        conc_perc = 100 - props.reduction

        new_powers = props.assign(power_modelled=(('nzsegment'), power_data1.sel(conc_perc=conc_perc).power.values.astype('int8')))
        new_powers['nzsegment'] = new_powers['nzsegment'].astype('int32')
        new_powers['reduction'] = new_powers['reduction'].astype('int8')

        ## Monitored
        power_data = xr.open_dataset(river_power_moni_path, engine='h5netcdf')
        sites = power_data.nzsegment.values[power_data.nzsegment.isin(branches)].astype('int32')
        sites.sort()
        if len(sites) > 0:
            conc_perc1 = conc_perc.sel(nzsegment=sites)
            power_data1 = power_data.sel(indicator=indicator, nzsegment=sites, n_samples=n_samples, drop=True).load().sortby('nzsegment').copy()
            power_data1 = power_data1.rename({'power': 'power_monitored'})
            power_data.close()
            del power_data

            power_data2 = power_data1.sel(conc_perc=conc_perc1).drop('conc_perc')

            new_powers = xr_concat([new_powers, power_data2])
        else:
            new_powers = new_powers.assign(power_monitored=(('nzsegment'), xr.full_like(new_powers.reduction, np.nan, dtype='float32').values))

        data = encode_obj(new_powers)
    else:
        data = ''

    return data


# @callback(
#     Output('colorbar', 'children'),
#     Input('powers_obj', 'data'),
#     prevent_initial_call=True
#     )
# def update_colorbar(powers_obj):
#     """

#     """
#     if (powers_obj != '') and (powers_obj is not None):
#         # print('trigger')
#         return colorbar_power
#     else:
#         return colorbar_base


@callback(
    Output('reach_map', 'hideout'),
    Output('sites_points', 'hideout'),
    [Input('powers_obj', 'data')],
    prevent_initial_call=True
    )
def update_hideout(powers_obj):
    """

    """
    if (powers_obj != '') and (powers_obj is not None):
        props = decode_obj(powers_obj)

        ## Modelled
        color_arr = pd.cut(props.power_modelled.values, bins, labels=colorscale, right=False).tolist()

        hideout_model = {'colorscale': color_arr, 'classes': props.nzsegment.values.tolist(), 'style': style, 'colorProp': 'nzsegment'}

        ## Monitored
        props_moni = props.dropna('nzsegment')
        if len(props_moni.nzsegment) > 0:
            # print(props_moni)
            color_arr2 = pd.cut(props_moni.power_monitored.values, bins, labels=colorscale, right=False).tolist()

            hideout_moni = {'classes': props_moni.nzsegment.values.astype(int), 'colorscale': color_arr2, 'circleOptions': dict(fillOpacity=1, stroke=True, color='black', weight=1, radius=site_point_radius), 'colorProp': 'nzsegment'}

            # hideout_moni = {'colorscale': color_arr2, 'classes': props_moni.nzsegment.values.astype(int).tolist(), 'style': style, 'colorProp': 'nzsegment'}
        else:
            hideout_moni = rivers_points_hideout
    else:
        hideout_model = {}
        hideout_moni = rivers_points_hideout

    return hideout_model, hideout_moni


@callback(
    Output("info", "children"),
    [Input('powers_obj', 'data'),
      # Input('reductions_obj', 'data'),
      # Input('map_checkboxes_rivers', 'value'),
      Input("reach_map", "click_feature"),
      Input('sites_points', 'click_feature')],
    )
def update_map_info(powers_obj, reach_feature, sites_feature):
    """

    """
    # info = """###### Likelihood of observing a reduction (%)"""
    info = """"""

    # if (reductions_obj != '') and (reductions_obj is not None) and ('reductions_poly' in map_checkboxes):
    #     info = info + """\n\nHover over the polygons to see reduction %"""

    trig = ctx.triggered_id
    # print(trig)

    if (powers_obj != '') and (powers_obj is not None):

        props = decode_obj(powers_obj)

        if trig == 'reach_map':
            # print(reach_feature)
            feature_id = int(reach_feature['id'])

            if feature_id in props.nzsegment:

                reach_data = props.sel(nzsegment=feature_id)

                info_str = """\n\n**nzsegment**: {seg}\n\n**Reduction**: {red}%\n\n**Likelihood of observing a reduction (power)**: {t_stat}%""".format(red=int(reach_data.reduction), t_stat=int(reach_data.power_modelled), seg=feature_id)

                info = info + info_str

            else:
                info = info + """\n\nClick on a reach to see info"""
        elif trig == 'sites_points':
            feature_id = int(sites_feature['properties']['nzsegment'])
            # print(sites_feature)

            if feature_id in props.nzsegment:

                reach_data = props.sel(nzsegment=feature_id)

                info_str = """\n\n**nzsegment**: {seg}\n\n**Site name**: {site}\n\n**Reduction**: {red}%\n\n**Likelihood of observing a reduction (power)**: {t_stat}%""".format(red=int(reach_data.reduction), t_stat=int(reach_data.power_monitored), seg=feature_id, site=sites_feature['id'])

                info = info + info_str

        else:
            info = info + """\n\nClick on a reach to see info"""

    return info


@callback(
    Output("dl_power_rivers", "data"),
    Input("dl_btn_power_rivers", "n_clicks"),
    State('catch_id', 'value'),
    State('powers_obj', 'data'),
    State('indicator_rivers', 'value'),
    State('time_period', 'value'),
    State('freq', 'value'),
    prevent_initial_call=True,
    )
def download_power(n_clicks, catch_id, powers_obj, indicator, n_years, n_samples_year):

    if isinstance(catch_id, str) and (powers_obj != '') and (powers_obj is not None):
        power_data = decode_obj(powers_obj)

        df1 = power_data.to_dataframe().reset_index()
        df1['indicator'] = rivers_indicator_dict[indicator]
        df1['n_years'] = n_years
        df1['n_samples_per_year'] = n_samples_year

        df2 = df1.set_index(['indicator', 'n_years', 'n_samples_per_year', 'nzsegment']).sort_index()

        return dcc.send_data_frame(df2.to_csv, f"river_power_{catch_id}.csv")
