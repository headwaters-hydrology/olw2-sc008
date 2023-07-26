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

# from .app import app

# from app import app
# import utils
# from . import utils

##########################################
### Parameters

dash.register_page(
    __name__,
    path='/lakes-wq',
    title='Lakes Water Quality',
    name='lakes_wq',
    description='Lakes and Lagoons Water Quality'
)

### Paths
assets_path = pathlib.Path(os.path.realpath(os.path.dirname(__file__))).parent.joinpath('assets')

app_base_path = pathlib.Path('/assets')

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

catch_style = {'fillColor': 'grey', 'weight': 2, 'opacity': 1, 'color': 'black', 'fillOpacity': 0.1}
lake_style = {'fillColor': '#A4DCCC', 'weight': 4, 'opacity': 1, 'color': 'black', 'fillOpacity': 1}
reach_style = {'weight': 2, 'opacity': 0.75, 'color': 'grey'}

lakes_indicator_dict = {'CHLA': 'Chlorophyll a', 'CYANOTOT': 'Total Cyanobacteria', 'ECOLI': 'E.coli', 'NH4N': 'Ammoniacal nitrogen', 'Secchi': 'Secchi Depth', 'TN': 'Total nitrogen', 'TP': 'Total phosphorus'}

lakes_reduction_cols = list(lakes_indicator_dict.values())

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

catch_style_handle = assign("""function style(feature) {
    return {
        fillColor: 'grey',
        weight: 2,
        opacity: 1,
        color: 'black',
        fillOpacity: 0.1
    };
}""", name='rivers_catch_style_handle')

### Colorbar
colorbar_base = dl.Colorbar(style={'opacity': 0})
base_reach_style = dict(weight=4, opacity=1, color='white')

indices = list(range(len(ctg) + 1))
colorbar_power = dl.Colorbar(min=0, max=len(ctg), classes=indices, colorscale=colorscale, tooltip=True, tickValues=[item + 0.5 for item in indices[:-1]], tickText=ctg, width=300, height=30, position="bottomright")


# base_reach_style = dict(weight=4, opacity=1, color='white')

# lake_id = 48177

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



###############################################
### Initial processing

# with booklet.open(lakes_catches_major_path, 'r') as f:
#     lakes = list(f.keys())

# lakes.sort()

with open(assets_path.joinpath('lakes_points.pbf'), 'rb') as f:
    geodict = geobuf.decode(f.read())

lakes_options = [{'value': int(f['id']), 'label': ' '.join(f['properties']['name'].split())} for f in geodict['features']]

lakes_data = {int(f['id']): f['properties'] for f in geodict['features']}

indicators = [{'value': k, 'label': v} for k, v in lakes_indicator_dict.items()]

###############################################
### App layout


def layout():
    layout = dmc.Container(
        fluid=True,
        # size='xl',
        px=0,
        py=0,
        my=0,
        mx=0,
        ml=0,
        pl=0,
        children=[
            dmc.Grid(
                columns=7,
                children=[
                    dmc.Col(
                        span=3,
                        children=dmc.Accordion(
                            value="1",
                            chevronPosition='left',
                            children=[
                            dmc.AccordionItem([
                                dmc.AccordionControl('(1) Lake Selection'),
                                dmc.AccordionPanel([

                                    html.Label('(1a) Select a lake/lagoon on the map:'),
                                    dcc.Dropdown(options=[v for v in lakes_options], id='lake_id', optionHeight=40, clearable=False,
                                                  style={'margin-top': 10}
                                                  ),
                                    ]
                                    )
                                ],
                                value='1'
                                ),

                            dmc.AccordionItem([
                                dmc.AccordionControl('(2 - Optional) Customise Reductions Layer'),
                                dmc.AccordionPanel([
                                    html.Label('(2a) Download reductions polygons as GPKG:'),
                                    dcc.Loading(
                                    type="default",
                                    children=[html.Div(dmc.Button("Download reductions",
                                                                  id='dl_btn_lakes',
                                                                  ),
                                                       style={'margin-top': 10}),
                            dcc.Download(id="dl_poly_lakes")],
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
                                                id='upload_data_lakes',
                                                children=dmc.Button('Upload reductions',
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
                                                    }, id='upload_error_text_lakes'),
                                    html.Label('(2c) Process the reductions layer and route the reductions downstream:', style={
                                        'margin-top': 20
                                    }
                                        ),
                                    dcc.Loading(
                                    type="default",
                                    children=html.Div([dmc.Button('Process reductions', id='process_reductions_lakes',
                                                                  # className="me-1",
                                                                  n_clicks=0),
                                                        html.Div(id='process_text_lakes')],
                                                      style={'margin-top': 10, 'margin-bottom': 10}
                                                      )
                                    ),
                                    ]
                                    )
                                ],
                                value='2'
                                ),

                            dmc.AccordionItem([
                                dmc.AccordionControl('(3) Sampling Options'),
                                dmc.AccordionPanel([
                                    dmc.Text('(3a) Select Indicator:'),
                                    dcc.Dropdown(options=indicators, id='indicator_lakes', optionHeight=40, clearable=False),
                                    dmc.Text('(3b) Select sampling length (years):', style={'margin-top': 20}),
                                    dmc.SegmentedControl(data=[{'label': d, 'value': str(d)} for d in time_periods],
                                                         id='time_period_lakes',
                                                         value='5',
                                                         fullWidth=True,
                                                         color=1,
                                                         ),
                                    dmc.Text('(3c) Select sampling frequency:', style={'margin-top': 20}),
                                    dmc.SegmentedControl(data=[{'label': v, 'value': str(k)} for k, v in freq_mapping.items()],
                                                         id='freq_lakes',
                                                         value='12',
                                                         fullWidth=True,
                                                         color=1
                                                         ),
                                    html.Label('(3d) Change the percent of the reductions applied (100% is the max realistic reduction):', style={'margin-top': 20}),
                                    dmc.Slider(id='reductions_slider_lakes',
                                               value=100,
                                               mb=35,
                                               step=10,
                                               # min=10,
                                               showLabelOnHover=True,
                                               disabled=False,
                                               marks=[{'label': str(d) + '%', 'value': d} for d in range(0, 101, 20)]
                                               ),
                                    ],
                                    )
                                ],
                                value='3'
                                ),

                            dmc.AccordionItem([
                                dmc.AccordionControl('(4) Download Results'),
                                dmc.AccordionPanel([
                                    dmc.Text('(4a) Download power results given the prior sampling options (csv):'),
                                    dcc.Loading(
                                    type="default",
                                    children=[html.Div(dmc.Button("Download power results", id='dl_btn_power_lakes'), style={'margin-bottom': 20, 'margin-top': 10}),
                            dcc.Download(id="dl_power_lakes")],
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
                                    dl.BaseLayer(dl.TileLayer(attribution=attribution), checked=True, name='OpenStreetMap'),
                                    dl.BaseLayer(dl.TileLayer(url='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', attribution='Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap (CC-BY-SA)'), checked=False, name='OpenTopoMap'),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(url=str(lakes_pbf_path), format="geobuf", id='lake_points', zoomToBoundsOnClick=True, cluster=True)), name='Lake points', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='catch_map_lakes', zoomToBoundsOnClick=True, zoomToBounds=False, options=dict(style=catch_style_handle))), name='Catchments', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='reach_map_lakes', options=dict(style=reach_style))), name='Rivers', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='lake_poly', options=dict(style=lake_style_handle), hideout={'classes': [''], 'colorscale': ['#808080'], 'style': lake_style, 'colorProp': 'name'})), name='Lakes', checked=True),
                                    # dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='sites_points_lakes', options=dict(pointToLayer=sites_points_handle), hideout=rivers_points_hideout)), name='Monitoring sites', checked=True),
                                    ], id='layers_gw'),
                                colorbar_power,
                                # html.Div(id='colorbar', children=colorbar_base),
                                # dmc.Group(id='colorbar', children=colorbar_base),
                                dcc.Markdown(id="info_lakes", className="info", style={"position": "absolute", "top": "10px", "right": "160px", "z-index": "1000"})
                                                ], style={'width': '100%', 'height': 700, 'margin': "auto", "display": "block"}, id="map2"),

                            ],
                            ),
                        ),
                    ]
                    ),
            dcc.Store(id='powers_obj_lakes', data=''),
            dcc.Store(id='reaches_obj_lakes', data=''),
            dcc.Store(id='custom_reductions_obj_lakes', data=''),
            dcc.Store(id='base_reductions_obj_lakes', data=''),
            ]
        )

    return layout


# def layout():
#     layout = html.Div(children=[
#         # html.Div([html.H1('Lake/Lagoon Water Quality')]),
#         html.Div([
#             html.H3('(1) Reductions routing'),

#             html.Label('Select a lake/lagoon on the map:'),
#             dcc.Dropdown(options=lakes_options, id='lake_id', optionHeight=40, clearable=False),

#             dcc.Upload(
#                 id='upload_data_lakes',
#                 children=html.Button('Upload reductions polygons gpkg', style={
#                     'width': '100%',
#                 }),
#                 style={
#                     'width': '100%',
#                     'height': '50%',
#                     'textAlign': 'left',
#                     'margin-top': 20
#                 },
#                 multiple=False
#             ),
#             dcc.Markdown('''###### **Or**''', style={
#                 'textAlign': 'center',
#                             }),
#             html.Button('Use land cover for reductions', id='demo_data_lakes',
#                         style={
#                             'width': '100%',
#                             'height': '50%',
#                             'textAlign': 'left',
#                             'margin-top': 20
#                         }),

#             html.Label('Select a reductions column in the GIS file:', style={'margin-top': 20}),
#             dcc.Dropdown(options=[], id='col_name_lakes', optionHeight=40, clearable=False),
#             dcc.Loading(
#             type="default",
#             children=[html.Div(html.Button("Download reductions polygons", id='dl_btn_lakes'), style={'margin-top': 10}),
#     dcc.Download(id="dl_poly_lakes")],
#             ),
#             dcc.Loading(
#             type="default",
#             children=html.Div([html.Button('Process reductions', id='process_lakes', n_clicks=0),
#                                html.Div(id='process_text_lakes')],
#                               style={'margin-top': 20, 'margin-bottom': 10}
#                               )
#         ),
#             ], className='two columns', style={'margin': 10}),

#     html.Div([
#         html.H3('(2) Sampling options'),
#         html.Label('Select Indicator:'),
#         dcc.Dropdown(options=indicators, id='indicator_lakes', optionHeight=40, clearable=False),
#         html.Label('Select sampling length (years):', style={'margin-top': 20}),
#         dcc.Dropdown(options=[{'label': d, 'value': d} for d in time_periods], id='time_period_lakes', clearable=False, value=5),
#         html.Label('Select sampling frequency:'),
#         dcc.Dropdown(options=[{'label': v, 'value': k} for k, v in freq_mapping.items()], id='freq_lakes', clearable=False, value=12),

#         html.H4(children='Map Layers', style={'margin-top': 20}),
#         dcc.Checklist(
#                options=[
#                    {'label': 'Reductions polygons', 'value': 'reductions_poly'},
#                     # {'label': 'Lake polygon', 'value': 'lake_poly'},
#                    {'label': 'River reaches', 'value': 'reach_map'}
#                ],
#                value=['reductions_poly', 'reach_map'],
#                id='map_checkboxes_lakes',
#                style={'padding': 5, 'margin-bottom': 50}
#             ),
#         dcc.Loading(
#         type="default",
#         children=[html.Div(html.Button("Download power results csv", id='dl_btn_power_lakes'), style={'margin-bottom': 20}),
# dcc.Download(id="dl_power_lakes")],
#         ),
#         dcc.Markdown('', style={
#             'textAlign': 'left',
#                         }, id='red_disclaimer_lakes'),

#         ], className='two columns', style={'margin': 10}),

#     html.Div([
#         dl.Map(center=center, zoom=7, children=[
#             dl.TileLayer(id='tile_layer_lakes', attribution=attribution),
#             dl.GeoJSON(url=str(lakes_pbf_path), format="geobuf", id='lake_points', zoomToBoundsOnClick=True, zoomToBounds=True, cluster=True),
#             dl.GeoJSON(data='', format="geobuf", id='catch_map_lakes', zoomToBoundsOnClick=True, options=dict(style=catch_style)),
#             dl.GeoJSON(data='', format="geobuf", id='reach_map_lakes', options=dict(style=reach_style)),
#             dl.GeoJSON(data='', format="geobuf", id='lake_poly', options=dict(style=lake_style_handle), hideout={'classes': [''], 'colorscale': ['#808080'], 'style': lake_style, 'colorProp': 'name'}),
#             dl.GeoJSON(data='', format="geobuf", id='reductions_poly_lakes'),
#             colorbar_power,
#             info
#                             ], style={'width': '100%', 'height': 700, 'margin': "auto", "display": "block"})
#     ], className='five columns', style={'margin': 10}),

#     dcc.Store(id='props_obj_lakes', data=''),
#     dcc.Store(id='reaches_obj_lakes', data=''),
#     dcc.Store(id='reductions_obj_lakes', data=''),
# ], style={'margin':0})

#     return layout

###############################################
### Callbacks

@callback(
    Output('lake_id', 'value'),
    [Input('lake_points', 'click_feature')]
    )
def update_lake_id(feature):
    """

    """
    # print(ds_id)
    lake_id = None
    if feature is not None:
        # print(feature)
        if not feature['properties']['cluster']:
            lake_id = feature['id']

    return lake_id


@callback(
        Output('reach_map_lakes', 'data'),
        Input('lake_id', 'value'),
        )
# @cache.memoize()
def update_reaches_lakes(lake_id):
    if (lake_id is not None):
        with booklet.open(lakes_reach_gbuf_path, 'r') as f:
            data = base64.b64encode(f[int(lake_id)]).decode()

    else:
        data = ''

    return data


# @callback(
#         Output('sites_points', 'data'),
#         Input('lake_id', 'value'),
#         )
# # @cache.memoize()
# def update_monitor_sites(lake_id):
#     if (lake_id is not None):
#         with booklet.open(lakes_sites_path, 'r') as f:
#             data = base64.b64encode(f[int(lake_id)]).decode()

#     else:
#         data = ''

#     return data


@callback(
        Output('catch_map_lakes', 'data'),
        Input('lake_id', 'value'),
        )
# @cache.memoize()
def update_catch_lakes(lake_id):
    if isinstance(lake_id, str):
        with booklet.open(lakes_catches_major_path, 'r') as f:
            data = base64.b64encode(f[int(lake_id)]).decode()
    else:
        data = ''

    return data


@callback(
        Output('lake_poly', 'data'),
        Input('lake_id', 'value'),
        )
# @cache.memoize()
def update_lake(lake_id):
    if isinstance(lake_id, str):
        with booklet.open(lakes_poly_gbuf_path, 'r') as f:
            data = base64.b64encode(f[int(lake_id)]).decode()
    else:
        data = ''

    return data


@callback(
        Output('base_reductions_obj_lakes', 'data'),
        Input('lake_id', 'value'),
        prevent_initial_call=True
        )
# @cache.memoize()
def update_base_reductions_obj(lake_id):
    data = ''

    if lake_id is not None:
        with booklet.open(lakes_lc_path, 'r') as f:
            data = encode_obj(f[int(lake_id)])

    return data


@callback(
    Output("dl_poly_lakes", "data"),
    Input("dl_btn_lakes", "n_clicks"),
    State('lake_id', 'value'),
    prevent_initial_call=True,
    )
def download_lc(n_clicks, lake_id):
    if isinstance(lake_id, str):
        path = lakes_catch_lc_dir.joinpath(lakes_catch_lc_gpkg_str.format(lake_id))

        return dcc.send_file(path)


@callback(
        Output('custom_reductions_obj_lakes', 'data'), Output('upload_error_text_lakes', 'children'),
        Input('upload_data_lakes', 'contents'),
        State('upload_data_lakes', 'filename'),
        State('lake_id', 'value'),
        prevent_initial_call=True
        )
# @cache.memoize()
def update_land_reductions(contents, filename, lake_id):
    data = None
    error_text = ''

    if lake_id is not None:
        if contents is not None:
            data = parse_gis_file(contents, filename)

            if isinstance(data, list):
                error_text = data[0]
                data = None

    return data, error_text


@callback(
    Output('reaches_obj_lakes', 'data'), Output('process_text_lakes', 'children'),
    Input('process_reductions_lakes', 'n_clicks'),
    Input('base_reductions_obj_lakes', 'data'),
    [
      State('lake_id', 'value'),
      State('custom_reductions_obj_lakes', 'data'),
      ],
    prevent_initial_call=True)
def update_reach_reductions(click, base_reductions_obj, lake_id, reductions_obj):
    """

    """
    trig = ctx.triggered_id

    if (trig == 'process_reductions_lakes'):
        if isinstance(lake_id, str) and (reductions_obj != '') and (reductions_obj is not None):
            red1 = xr.open_dataset(lakes_reductions_model_path)

            base_props = red1.sel(LFENZID=int(lake_id), drop=True)

            new_reductions = decode_obj(reductions_obj)
            base_reductions = decode_obj(base_reductions_obj)

            new_props = calc_lake_reach_reductions(lake_id, new_reductions, base_reductions)
            new_props1 = new_props.combine_first(base_props).load().copy()
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
        if isinstance(lake_id, str):
            red1 = xr.open_dataset(lakes_reductions_model_path)

            base_props = red1.sel(LFENZID=int(lake_id), drop=True).load().copy()
            red1.close()
            del red1

            data = encode_obj(base_props)
            text_out = ''
        else:
            data = ''
            text_out = ''

    return data, text_out


# @callback(
#         Output('red_disclaimer_lakes', 'children'),
#         Input('upload_data_lakes', 'contents'),
#         Input('demo_data_lakes', 'n_clicks'),
#         Input('map_checkboxes_lakes', 'value'),
#         prevent_initial_call=True
#         )
# def update_reductions_diclaimer(contents, n_clicks, map_checkboxes):
#     if (n_clicks is None) or (contents is not None):
#         return ''
#     elif 'reductions_poly' in map_checkboxes:
#         return '''* Areas on the map without polygon reductions are considered to have 0% reductions.'''
#     else:
#         return ''


# @callback(
#         Output('reductions_poly_lakes', 'data'),
#         Input('reductions_obj_lakes', 'data'),
#         Input('map_checkboxes_lakes', 'value'),
#         Input('col_name_lakes', 'value'),
#         )
# def update_reductions_poly_lakes(reductions_obj, map_checkboxes, col_name):
#     # print(reductions_obj)
#     # print(col_name)
#     if (reductions_obj != '') and (reductions_obj is not None) and ('reductions_poly' in map_checkboxes):

#         data = decode_obj(reductions_obj).to_crs(4326)

#         if isinstance(col_name, str):
#             data[col_name] = data[col_name].astype(str).str[:] + '% reduction'
#             data.rename(columns={col_name: 'tooltip'}, inplace=True)

#         gbuf = dlx.geojson_to_geobuf(data.__geo_interface__)

#         return gbuf
#     else:
#         return ''


@callback(
    Output('powers_obj_lakes', 'data'),
    [Input('reaches_obj_lakes', 'data'), Input('indicator_lakes', 'value'), Input('time_period_lakes', 'value'), Input('freq_lakes', 'value'), Input('reductions_slider_lakes', 'value')],
    [State('lake_id', 'value')]
    )
def update_powers_data_lakes(reaches_obj, indicator, n_years, n_samples_year, prop_red, lake_id):
    """

    """
    if (reaches_obj != '') and (reaches_obj is not None) and isinstance(n_years, str) and isinstance(n_samples_year, str) and isinstance(indicator, str):
        ind_name = lakes_indicator_dict[indicator]

        props = int(decode_obj(reaches_obj)[ind_name].sel(reduction_perc=prop_red, drop=True))
        # print(props)

        conc_perc = 100 - props

        lake_data = lakes_data[int(lake_id)]

        ## Lake residence time calcs
        if indicator in ['TP', 'CHLA', 'Secchi']:
            if lake_data['max_depth'] > 7.5:
                b = 1 + (0.44*(lake_data['residence_time']**0.13))
                conc_perc = int(round(((conc_perc*0.01)**(1/b)) * 100))
        elif indicator == 'TN':
            conc_perc = int(round(((conc_perc*0.01)**0.54) * 100))
        if indicator in ['CHLA', 'Secchi']:
            conc_perc = int(round(((conc_perc*0.01)**1.25) * 100))
        if indicator == 'Secchi':
            if lake_data['max_depth'] > 20:
                conc_perc = int(round(((conc_perc*0.01)**(0.9)) * 100))
            else:
                conc_perc = int(round(((conc_perc*0.01)**(0.9)) * 100))

        if conc_perc < 1:
            conc_perc = 1
        elif conc_perc > 100:
            conc_perc = 100

        ## Lookup power
        n_samples = int(n_samples_year)*int(n_years)

        power_data = xr.open_dataset(lakes_power_combo_path, engine='h5netcdf')
        try:
            power_data1 = power_data.sel(indicator=indicator, LFENZID=int(lake_id), n_samples=n_samples, conc_perc=conc_perc).load()

            power_data2 = [int(power_data1.power_modelled.values), float(power_data1.power_monitored.values)]
        except:
            power_data2 = [0, np.nan]
        power_data.close()
        del power_data

        data = encode_obj({'reduction': props, 'power': power_data2, 'lake_id': lake_id})
    else:
        data = ''

    return data


# @callback(
#         Output('lake_poly', 'options'),
#         Input('lake_poly', 'hideout'),
#         # State('lake_id', 'value')
#         )
# # @cache.memoize()
# def update_lakes_option(hideout):
#     trig = ctx.triggered_id
#     # print(trig)
#     # print(len(hideout))

#     if (len(hideout) == 1) or (trig == 'lake_id'):
#         options = dict(style=lake_style)
#     else:
#         options = dict(style=lake_style_handle)

#     return options


@callback(
    Output('lake_poly', 'hideout'),
    [Input('powers_obj_lakes', 'data')],
    Input('lake_id', 'value'),
    prevent_initial_call=True
    )
def update_hideout_lakes(powers_obj, lake_id):
    """

    """
    if (powers_obj != '') and (powers_obj is not None):
        # print('trigger')
        props = decode_obj(powers_obj)
        # print(props)
        # print(type(lake_id))

        if props['lake_id'] == lake_id:

            color_arr = pd.cut([props['power'][0]], bins, labels=colorscale, right=False).tolist()
            # print(color_arr)
            # print(props['lake_id'])

            hideout = {'classes': [props['lake_id']], 'colorscale': color_arr, 'style': lake_style, 'colorProp': 'LFENZID'}
        else:
            hideout = {'classes': [lake_id], 'colorscale': ['#808080'], 'style': lake_style, 'colorProp': 'LFENZID'}
    else:
        hideout = {'classes': [lake_id], 'colorscale': ['#808080'], 'style': lake_style, 'colorProp': 'LFENZID'}

    return hideout


@callback(
    Output("info_lakes", "children"),
    [Input('powers_obj_lakes', 'data'),
      Input("lake_poly", "click_feature")],
    )
def update_map_info_lakes(powers_obj, feature):
    """

    """
    info = """"""

    # if (reductions_obj != '') and (reductions_obj is not None) and ('reductions_poly' in map_checkboxes):
    #     info = info + """\n\nHover over the polygons to see reduction %"""

    if (powers_obj != '') and (powers_obj is not None):
        if feature is not None:
            props = decode_obj(powers_obj)

            if np.isnan(props['power'][1]):
                moni1 = 'NA'
            else:
                moni1 = str(int(props['power'][1])) + '%'

            info_str = """\n\n**Reduction**: {red}%\n\n**Likelihood of observing a reduction (power)**:\n\n&nbsp;&nbsp;&nbsp;&nbsp;**Modelled**: {t_stat1}%\n\n&nbsp;&nbsp;&nbsp;&nbsp;**Monitored**: {t_stat2}""".format(red=int(props['reduction']), t_stat1=int(props['power'][0]), t_stat2=moni1)

            info = info + info_str

        else:
            info = info + """\n\nClick on a lake to see info"""

    return info


# @callback(
#     Output("dl_poly_lakes", "data"),
#     Input("dl_btn_lakes", "n_clicks"),
#     State('lake_id', 'value'),
#     State('reductions_obj_lakes', 'data'),
#     prevent_initial_call=True,
#     )
# def download_lc(n_clicks, lake_id, reductions_obj):
#     # data = decode_obj(reductions_obj)
#     # io1 = io.BytesIO()
#     # data.to_file(io1, driver='GPKG')
#     # io1.seek(0)

#     if isinstance(lake_id, str) and (reductions_obj != '') and (reductions_obj is not None):
#         path = lakes_catch_lc_dir.joinpath(lakes_catch_lc_gpkg_str.format(lake_id))

#         return dcc.send_file(path)


@callback(
    Output("dl_power_lakes", "data"),
    Input("dl_btn_power_lakes", "n_clicks"),
    State('lake_id', 'value'),
    State('powers_obj_lakes', 'data'),
    State('indicator_lakes', 'value'),
    State('time_period_lakes', 'value'),
    State('freq_lakes', 'value'),
    prevent_initial_call=True,
    )
def download_power(n_clicks, lake_id, powers_obj, indicator, n_years, n_samples_year):

    if isinstance(lake_id, str) and (powers_obj != '') and (powers_obj is not None):
        power_data = decode_obj(powers_obj)

        df1 = pd.DataFrame([power_data['power']], columns=['modelled', 'monitored'])
        df1['indicator'] = lakes_indicator_dict[indicator]
        df1['n_years'] = n_years
        df1['n_samples_per_year'] = n_samples_year
        df1['LFENZID'] = lake_id
        df1['reduction'] = power_data['reduction']

        df2 = df1.set_index(['indicator', 'n_years', 'n_samples_per_year', 'reduction', 'LFENZID']).sort_index()

        return dcc.send_data_frame(df2.to_csv, f"lake_power_{lake_id}.csv")
