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
from . import utils

##########################################
### Parameters

dash.register_page(
    __name__,
    path='/lakes-wq',
    title='Lakes Water Quality',
    name='lakes_wq',
    description='Lakes and Lagoons Water Quality'
)



# base_reach_style = dict(weight=4, opacity=1, color='white')

# lake_id = 48177

###############################################
### Helper Functions



###############################################
### Initial processing

# with booklet.open(lakes_catches_major_path, 'r') as f:
#     lakes = list(f.keys())

# lakes.sort()

with open(utils.assets_path.joinpath('lakes_points.pbf'), 'rb') as f:
    geodict = geobuf.decode(f.read())

lakes_options = [{'value': int(f['id']), 'label': ' '.join(f['properties']['name'].split())} for f in geodict['features']]

lakes_data = {int(f['id']): f['properties'] for f in geodict['features']}

indicators = [{'value': k, 'label': v} for k, v in utils.lakes_indicator_dict.items()]

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
                                dmc.AccordionControl(html.Div('(1) Lake Selection', style={'font-size': 22})),
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
                                dmc.AccordionControl(html.Div('(2 - Optional) Customise Reductions Layer', style={'font-size': 22})),
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
                                dmc.AccordionControl(html.Div('(3) Sampling Options', style={'font-size': 22})),
                                dmc.AccordionPanel([
                                    dmc.Text('(3a) Select Indicator:'),
                                    dcc.Dropdown(options=indicators, id='indicator_lakes', optionHeight=40, clearable=False),
                                    dmc.Text('(3b) Select sampling length (years):', style={'margin-top': 20}),
                                    dmc.SegmentedControl(data=[{'label': d, 'value': str(d)} for d in utils.time_periods],
                                                         id='time_period_lakes',
                                                         value='5',
                                                         fullWidth=True,
                                                         color=1,
                                                         ),
                                    dmc.Text('(3c) Select sampling frequency:', style={'margin-top': 20}),
                                    dmc.SegmentedControl(data=[{'label': v, 'value': str(k)} for k, v in utils.freq_mapping.items()],
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
                                dmc.AccordionControl(html.Div('(4) Download Results', style={'font-size': 22})),
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
                            dl.Map(center=utils.center, zoom=6, children=[
                                dl.LayersControl([
                                    dl.BaseLayer(dl.TileLayer(attribution=utils.attribution), checked=True, name='OpenStreetMap'),
                                    dl.BaseLayer(dl.TileLayer(url='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', attribution='Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap (CC-BY-SA)'), checked=False, name='OpenTopoMap'),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(url=str(utils.lakes_pbf_path), format="geobuf", id='lake_points', zoomToBoundsOnClick=True, cluster=True)), name='Lake points', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='catch_map_lakes', zoomToBoundsOnClick=True, zoomToBounds=False, options=dict(style=utils.catch_style_handle))), name='Catchments', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='reach_map_lakes', options=dict(style=utils.reach_style))), name='Rivers', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='lake_poly', options=dict(style=utils.lake_style_handle), hideout={'classes': [''], 'colorscale': ['#808080'], 'style': utils.lake_style, 'colorProp': 'name'})), name='Lakes', checked=True),
                                    # dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='sites_points_lakes', options=dict(pointToLayer=utils.sites_points_handle), hideout=utils.rivers_points_hideout)), name='Monitoring sites', checked=True),
                                    ], id='layers_gw'),
                                utils.colorbar_power,
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
        with booklet.open(utils.lakes_reach_gbuf_path, 'r') as f:
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
#         with booklet.open(utils.lakes_sites_path, 'r') as f:
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
        with booklet.open(utils.lakes_catches_major_path, 'r') as f:
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
        with booklet.open(utils.lakes_poly_gbuf_path, 'r') as f:
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
        with booklet.open(utils.lakes_lc_path, 'r') as f:
            data = utils.encode_obj(f[int(lake_id)])

    return data


@callback(
    Output("dl_poly_lakes", "data"),
    Input("dl_btn_lakes", "n_clicks"),
    State('lake_id', 'value'),
    prevent_initial_call=True,
    )
def download_lc(n_clicks, lake_id):
    if isinstance(lake_id, str):
        path = utils.lakes_catch_lc_dir.joinpath(utils.lakes_catch_lc_gpkg_str.format(lake_id))

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
            data = utils.parse_gis_file(contents, filename)

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
            red1 = xr.open_dataset(utils.lakes_reductions_model_path)

            base_props = red1.sel(LFENZID=int(lake_id), drop=True)

            new_reductions = utils.decode_obj(reductions_obj)
            base_reductions = utils.decode_obj(base_reductions_obj)

            new_props = utils.calc_lake_reach_reductions(lake_id, new_reductions, base_reductions)
            new_props1 = new_props.combine_first(base_props)
            data = utils.encode_obj(new_props1)
            text_out = 'Routing complete'
        else:
            data = ''
            text_out = 'Not all inputs have been selected'
    else:
        if isinstance(lake_id, str):
            red1 = xr.open_dataset(utils.lakes_reductions_model_path)

            base_props = red1.sel(LFENZID=int(lake_id), drop=True)

            data = utils.encode_obj(base_props)
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
        ind_name = utils.lakes_indicator_dict[indicator]

        props = int(utils.decode_obj(reaches_obj)[ind_name].sel(reduction_perc=prop_red, drop=True))
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

        power_data = xr.open_dataset(utils.lakes_power_combo_path, engine='h5netcdf')
        try:
            power_data1 = power_data.sel(indicator=indicator, LFENZID=int(lake_id), n_samples=n_samples, conc_perc=conc_perc)

            power_data2 = [int(power_data1.power_modelled.values), float(power_data1.power_monitored.values)]
        except:
            power_data2 = [0, np.nan]
        power_data.close()
        del power_data

        data = utils.encode_obj({'reduction': props, 'power': power_data2, 'lake_id': lake_id})
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
        props = utils.decode_obj(powers_obj)
        # print(props)
        # print(type(lake_id))

        if props['lake_id'] == lake_id:

            color_arr = pd.cut([props['power'][0]], utils.bins, labels=utils.colorscale, right=False).tolist()
            # print(color_arr)
            # print(props['lake_id'])

            hideout = {'classes': [props['lake_id']], 'colorscale': color_arr, 'style': utils.lake_style, 'colorProp': 'LFENZID'}
        else:
            hideout = {'classes': [lake_id], 'colorscale': ['#808080'], 'style': utils.lake_style, 'colorProp': 'LFENZID'}
    else:
        hideout = {'classes': [lake_id], 'colorscale': ['#808080'], 'style': utils.lake_style, 'colorProp': 'LFENZID'}

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
            props = utils.decode_obj(powers_obj)

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
#         path = utils.lakes_catch_lc_dir.joinpath(utils.lakes_catch_lc_gpkg_str.format(lake_id))

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
        power_data = utils.decode_obj(powers_obj)

        df1 = pd.DataFrame([power_data['power']], columns=['modelled', 'monitored'])
        df1['indicator'] = utils.lakes_indicator_dict[indicator]
        df1['n_years'] = n_years
        df1['n_samples_per_year'] = n_samples_year
        df1['LFENZID'] = lake_id
        df1['reduction'] = power_data['reduction']

        df2 = df1.set_index(['indicator', 'n_years', 'n_samples_per_year', 'reduction', 'LFENZID']).sort_index()

        return dcc.send_data_frame(df2.to_csv, f"lake_power_{lake_id}.csv")
