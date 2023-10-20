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
    path='/rivers-eco',
    title='Ecology Reaches',
    name='rivers_eco',
    description='River Ecology Reaches'
)

### Paths
assets_path = pathlib.Path(os.path.realpath(os.path.dirname(__file__))).parent.joinpath('assets')

app_base_path = pathlib.Path('/assets')

base_data_url = 'https://b2.tethys-ts.xyz/file/'

eco_power_catch_path = assets_path.joinpath('eco_reaches_power_modelled.h5')
# eco_power_moni_path = assets_path.joinpath('eco_reaches_power_monitored.h5')
eco_reach_weights_path = assets_path.joinpath('eco_reach_weights.h5')

rivers_catch_pbf_path = app_base_path.joinpath('rivers_catchments.pbf')

rivers_reach_gbuf_path = assets_path.joinpath('rivers_reaches.blt')
rivers_lc_clean_path = assets_path.joinpath('rivers_catch_lc.blt')
rivers_catch_path = assets_path.joinpath('rivers_catchments_minor.blt')
river_loads_rec_path = assets_path.joinpath('rivers_loads_rec.blt')
river_catch_name_path = assets_path.joinpath('rivers_catchments_names.blt')
rivers_reach_mapping_path = assets_path.joinpath('rivers_reaches_mapping.blt')

eco_sites_path = assets_path.joinpath('eco_sites_catchments.blt')
river_marae_path = assets_path.joinpath('rivers_catchments_marae.blt')

rivers_catch_lc_gpkg_str = '{base_url}olw-data/olw-sc008/rivers_land_cover_gpkg/{catch_id}_rivers_land_cover_reductions.gpkg'

### Layout
map_height = 700
center = [-41.1157, 172.4759]

attribution = 'Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'

freq_mapping = {1: 'yearly', 4: 'quarterly', 12: 'monthly'}
time_periods = [5, 10, 20, 30]
n_sites = [5, 10, 20, 30]

style = dict(weight=4, opacity=1, color='white')

site_point_radius = 6

# reduction_ratios = range(10, 101, 10)
# red_ratios = np.array(list(reduction_ratios), dtype='int8')

eco_reductions_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

eco_reductions_options = [{'value': v, 'label': str(v)+'%'} for v in eco_reductions_values]

eco_freq_dict = {'mci': 1, 'peri': 12, 'sediment': 12}

rivers_points_hideout = {'classes': [], 'colorscale': ['#232323'], 'circleOptions': dict(fillOpacity=1, stroke=True, weight=1, color='black', radius=site_point_radius), 'colorProp': 'nzsegment'}

eco_indicator_dict = {'peri': 'Periphyton', 'mci': 'MCI', 'sediment': 'Percent deposited sediment'}

eco_reduction_cols = list(eco_indicator_dict.values())

catch_style_handle = assign("""function style(feature) {
    return {
        fillColor: 'grey',
        weight: 2,
        opacity: 1,
        color: 'black',
        fillOpacity: 0.1
    };
}""", name='eco_catch_style_handle')

base_reach_style_handle = assign("""function style3(feature) {
    return {
        weight: 2,
        opacity: 0.75,
        color: 'grey',
    };
}""", name='eco_base_reach_style_handle')

reach_style_handle = assign("""function style2(feature, context){
    const {classes, colorscale, style, colorProp} = context.props.hideout;  // get props from hideout
    const value = feature.properties[colorProp];  // get value the determines the color
    for (let i = 0; i < classes.length; ++i) {
        if (value == classes[i]) {
            style.color = colorscale[i];  // set the fill color according to the class
        }
    }
    return style;
}""", name='eco_reach_style_handle')

sites_points_handle = assign("""function rivers_sites_points_handle(feature, latlng, context){
    const {classes, colorscale, circleOptions, colorProp} = context.props.hideout;  // get props from hideout
    const value = feature.properties[colorProp];  // get value the determines the fillColor
    for (let i = 0; i < classes.length; ++i) {
        if (value == classes[i]) {
            circleOptions.fillColor = colorscale[i];  // set the color according to the class
        }
    }

    return L.circleMarker(latlng, circleOptions);
}""", name='eco_sites_points_handle')

draw_marae = assign("""function(feature, latlng){
const flag = L.icon({iconUrl: '/assets/nzta-marae.svg', iconSize: [20, 30]});
return L.marker(latlng, {icon: flag});
}""", name='eco_marae_handle')

### Colorbars
colorbar_base = dl.Colorbar(style={'opacity': 0})
base_reach_style = dict(weight=4, opacity=1, color='white')

bins_weights = [0, 10, 30, 101]
colorscale_weights = ['#edf8b1','#7fcdbb','#2c7fb8']
ctg_weights = ['Low', 'Moderate', 'High']

indices = list(range(len(ctg_weights) + 1))
colorbar_weights = dl.Colorbar(min=0, max=len(ctg_weights), classes=indices, colorscale=colorscale_weights, tooltip=True, tickValues=[item + 0.5 for item in indices[:-1]], tickText=ctg_weights, width=300, height=30, position="bottomright")

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

###############################################
### Initial processing

# with booklet.open(eco_sites_path, 'r') as f:
#     catches = [int(c) for c in f]

# catches.sort()
indicators = list(eco_indicator_dict.keys())
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
                        children=[dmc.Accordion(
                            value="1",
                            chevronPosition='left',
                            children=[
                            dmc.AccordionItem([
                                # html.H5('(1) Catchment selection', style={'font-weight': 'bold'}),
                                dmc.AccordionControl('(1) Catchment Selection', style={'font-size': 18}),
                                dmc.AccordionPanel([

                                    html.Label('(1a) Select a catchment on the map:'),
                                    dmc.Text(id='catch_name_eco', weight=700, style={'margin-top': 10}),
                                    ]
                                    )
                                ],
                                value='1'
                                ),

                            dmc.AccordionItem([
                                dmc.AccordionControl('(2 - Optional) Customise Improvements Layer', style={'font-size': 18}),
                                dmc.AccordionPanel([
                                    html.Label('(2a) Download improvements polygons as GPKG:'),
                                    dcc.Loading(
                                    id="loading-2",
                                    type="default",
                                    children=[dmc.Anchor(dmc.Button('Download land cover'), href='', id='dl_poly', style={'margin-top': 10})],
                                    ),
                                    html.Label('NOTE: Only modify existing values. Do not add additional columns; they will be ignored.', style={
                                        'margin-top': 10
                                    }
                                        ),
                                    html.Label('(2b) Upload modified improvements polygons as GPKG:', style={
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
                                    html.Label('(2c) Process the improvements layer and route the improvements downstream:', style={
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
                                dmc.AccordionControl('(3) Query Options', style={'font-size': 18}),
                                dmc.AccordionPanel([
                                    dmc.Text('(3a) Select Indicator:'),
                                    dcc.Dropdown(options=[{'label': eco_indicator_dict[d], 'value': d} for d in indicators], id='indicator_eco', optionHeight=40, clearable=False, style={'margin-bottom': 20}),
                                    html.Label('(3b) Select a percent improvement for the overall catchment:'),
                                    dmc.Slider(id='reductions_slider_eco',
                                               value=25,
                                               mb=35,
                                               step=5,
                                               min=5,
                                               max=50,
                                               showLabelOnHover=True,
                                               disabled=False,
                                               marks=eco_reductions_options
                                               ),
                                    dmc.Text('(3c) Select sampling length (years):', style={'margin-top': 20}),
                                    dmc.SegmentedControl(data=[{'label': d, 'value': str(d)} for d in time_periods],
                                                         id='time_period_eco',
                                                         value='5',
                                                         fullWidth=True,
                                                         color=1,
                                                         ),
                                    # dmc.Text('(3c) Select sampling frequency (monitoring site power):', style={'margin-top': 20}),
                                    # dmc.SegmentedControl(data=[{'label': v, 'value': str(k)} for k, v in freq_mapping.items()],
                                    #                       id='freq_eco',
                                    #                       value='12',
                                    #                       fullWidth=True,
                                    #                       color=1
                                    #                       ),
                                    dmc.Text('(3d) Select the number of sites per catchment:', style={'margin-top': 20}),
                                    dmc.SegmentedControl(data=[{'label': k, 'value': str(k)} for k in n_sites],
                                                          id='n_sites_eco',
                                                          value='10',
                                                          fullWidth=True,
                                                          color=1
                                                          ),
                                    # html.Label('(3d) Change the percent of the reductions applied (100% is the max realistic reduction):', style={'margin-top': 20}),
                                    # dmc.Slider(id='Reductions_slider',
                                    #            value=100,
                                    #            mb=35,
                                    #            step=10,
                                    #            # min=10,
                                    #            showLabelOnHover=True,
                                    #            disabled=False,
                                    #            marks=marks
                                    #            ),
                                    # dcc.Dropdown(options=[{'label': d, 'value': d} for d in time_periods], id='time_period', clearable=False, value=5),
                                    # html.Label('Select sampling frequency:'),
                                    # dcc.Dropdown(options=[{'label': v, 'value': k} for k, v in freq_mapping.items()], id='freq', clearable=False, value=12),

                                    ],
                                    )
                                ],
                                value='3'
                                ),

                            # dmc.AccordionItem([
                            #     dmc.AccordionControl('(4) Download Results', style={'font-size': 18}),
                            #     dmc.AccordionPanel([
                            #         dmc.Text('(4a) Download power results given the prior query options (csv):'),
                            #         dcc.Loading(
                            #         type="default",
                            #         children=[html.Div(dmc.Button("Download power results", id='dl_btn_power_eco'), style={'margin-bottom': 20, 'margin-top': 10}),
                            # dcc.Download(id="dl_power_eco")],
                            #         ),
                            #         ],
                            #         )
                            #     ],
                            #     value='4'
                            #     ),

                            ],
                            # style={
                            #     'margin': 0,
                            #     'padding': 0
                            #     },
                            # className='four columns', style={'margin': 10}
                            ),
                        dcc.Markdown("""* The rivers colored with *Low*, *Moderate*, and *High* are the qualitative monitoring priorities as there  is too much uncertainty in estimating the powers per reach.""")]
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
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(url=str(rivers_catch_pbf_path), format="geobuf", id='catch_map_eco', zoomToBoundsOnClick=True, zoomToBounds=False, options=dict(style=catch_style_handle))), name='Catchments', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='marae_map_eco', zoomToBoundsOnClick=False, zoomToBounds=False, options=dict(pointToLayer=draw_marae))), name='Marae', checked=False),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='reach_map_eco', options={}, hideout={})), name='Rivers', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='sites_points_eco', options=dict(pointToLayer=sites_points_handle), hideout=rivers_points_hideout)), name='Monitoring sites', checked=True),
                                    ], 
                                    id='layers_eco'
                                    ),
                                colorbar_weights,
                                dcc.Markdown(id="info_eco", className="info", style={"position": "absolute", "top": "10px", "right": "160px", "z-index": "1000"})
                                ], 
                                style={'width': '100%', 'height': '100vh', 'margin': "auto", "display": "block"}
                                ),

                            ],
                            # className='five columns', style={'margin': 10}
                            ),
                        ),
                    ]
                    ),
            dcc.Store(id='catch_id_eco', data=''),
            dcc.Store(id='reaches_obj_eco', data=''),
            dcc.Store(id='catch_power_obj_eco', data=''),
            ]
        )

    return layout


###############################################
### Callbacks

@callback(
    Output('catch_id_eco', 'data'),
    [Input('catch_map_eco', 'click_feature')]
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
    Output('catch_name_eco', 'children'),
    [Input('catch_id_eco', 'data')]
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
        Output('reach_map_eco', 'data'),
        Input('catch_id_eco', 'data'),
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
        Output('sites_points_eco', 'data'),
        Input('catch_id_eco', 'data'),
        )
# @cache.memoize()
def update_monitor_sites(catch_id):
    if catch_id != '':
        with booklet.open(eco_sites_path, 'r') as f:
            data = base64.b64encode(f[int(catch_id)]).decode()

    else:
        data = ''

    return data


@callback(
        Output('marae_map_eco', 'data'),
        Input('catch_id_eco', 'data'),
        )
def update_marae(catch_id):
    if catch_id != '':
        with booklet.open(river_marae_path, 'r') as f:
            data = base64.b64encode(f[int(catch_id)]).decode()

    else:
        data = ''

    return data


@callback(
        Output('reach_map_eco', 'options'),
        Input('reach_map_eco', 'hideout'),
        Input('catch_id_eco', 'data')
        )
# @cache.memoize()
def update_reaches_option(hideout, catch_id):
    trig = ctx.triggered_id

    if (len(hideout) == 0) or (trig == 'catch_id_eco'):
        options = dict(style=base_reach_style_handle)
    else:
        options = dict(style=reach_style_handle)

    return options


# @callback(
#     Output("freq_eco", "value"),
#     Input('indicator_eco', 'value'),
#     prevent_initial_call=True,
#     )
# def update_freq(indicator):

#     if isinstance(indicator, str):
#         n_samples_year = eco_freq_dict[indicator]

#         return str(n_samples_year)


@callback(
    Output('reaches_obj_eco', 'data'),
    Input('reductions_slider_eco', 'value'),
    Input('indicator_eco', 'value'),
    State('catch_id_eco', 'data'),
    prevent_initial_call=True
    )
def update_reaches_obj(reductions, indicator, catch_id):
    """

    """
    if isinstance(reductions, (str, int)) and isinstance(indicator, str) and (catch_id != ''):
        power_data = xr.open_dataset(eco_reach_weights_path, engine='h5netcdf')
        with booklet.open(rivers_reach_mapping_path, 'r') as f:
            reaches = f[int(catch_id)][int(catch_id)]
        power_data1 = power_data.sel(reduction_perc=int(reductions), nzsegment=reaches, drop=True).rename({indicator: 'weights'}).load().copy()
        power_data.close()
        del power_data

        data = encode_obj(power_data1)
        return data
    else:
        raise dash.exceptions.PreventUpdate


@callback(
    Output('catch_power_obj_eco', 'data'),
    [Input('reductions_slider_eco', 'value'), Input('indicator_eco', 'value'), Input('time_period_eco', 'value'), Input('n_sites_eco', 'value')],
    [State('catch_id_eco', 'data')],
    prevent_initial_call=True
    )
def update_catch_power_obj(reductions, indicator, n_years, n_sites, catch_id):
    """

    """
    if isinstance(reductions, (str, int)) and isinstance(n_years, str) and isinstance(indicator, str) and isinstance(n_sites, str) and (catch_id != ''):
        n_samples = int(n_sites)*int(n_years)

        power_data = xr.open_dataset(eco_power_catch_path, engine='h5netcdf')
        power_data1 = int(power_data.sel(nzsegment=int(catch_id), indicator=indicator, n_samples=n_samples, conc_perc=100-int(reductions), drop=True).power.values)
        power_data.close()
        del power_data

        data = str(power_data1)
        return data
    else:
        raise dash.exceptions.PreventUpdate


@callback(
    Output('reach_map_eco', 'hideout'),
    Input('reaches_obj_eco', 'data'),
    prevent_initial_call=True
    )
def update_reach_hideout(reaches_obj):
    """

    """
    if (reaches_obj != '') and (reaches_obj is not None):
        props = decode_obj(reaches_obj)

        ## Modelled
        values = props.weights.values
        bins_weights = np.quantile(values, [0, 0.50, 0.75, 1])
        bins_weights[-1] += 0.01
        color_arr = pd.cut(values, bins_weights, labels=colorscale_weights, right=False).tolist()

        hideout_model = {'colorscale': color_arr, 'classes': props.nzsegment.values.astype(int), 'style': style, 'colorProp': 'nzsegment'}

    else:
        hideout_model = {}

    return hideout_model


@callback(
    Output("info_eco", "children"),
    [
      Input('catch_power_obj_eco', 'data'),
      ],
    [State("info_eco", "children"),
     State('n_sites_eco', 'value')
     ]
    )
def update_map_info(catch_power_obj, old_info, n_sites):
    """

    """
    info = """"""

    if (catch_power_obj != '') and (catch_power_obj is not None):

        catch_power = int(catch_power_obj)

        info += """##### Catchment:

            \n\n**Likelihood of observing an improvement (power)**: {power}%\n\n**Number of sites**: {n_sites}\n\n""".format(power = catch_power, n_sites=n_sites)

    if info == """""":
        info = old_info

    return info


# @callback(
#     Output("dl_power_eco", "data"),
#     Input("dl_btn_power_eco", "n_clicks"),
#     State('catch_id_eco', 'data'),
#     State('indicator_eco', 'value'),
#     State('time_period_eco', 'value'),
#     prevent_initial_call=True,
#     )
# def download_power(n_clicks, catch_id, powers_obj, indicator, n_years):

#     if (catch_id != '') and (powers_obj != '') and (powers_obj is not None) and isinstance(n_samples_year, str):
#         power_data = decode_obj(powers_obj)

#         df1 = power_data.to_dataframe().reset_index()
#         df1['indicator'] = eco_indicator_dict[indicator]
#         df1['n_years'] = n_years
#         df1['n_samples_per_year'] = n_samples_year

#         df2 = df1.set_index(['indicator', 'n_years', 'n_samples_per_year', 'nzsegment']).sort_index()

#         return dcc.send_data_frame(df2.to_csv, f"ecoology_sites_power_{catch_id}.csv")
