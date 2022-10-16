#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 09:55:56 2021

@author: mike
"""
# import dash_core_components as dcc
# import dash_html_components as html
# import dash_table
import xarray as xr
from dash import dcc, html, dash_table
import pandas as pd
import numpy as np
import requests
import dash_leaflet as dl
import dash_leaflet.express as dlx
import copy
import os
# import pathlib
import tethysts
import base64
import geobuf
import orjson
import dash_leaflet.express as dlx
from dash_extensions.javascript import assign

# from . import utils
import utils

##########################################
### Parameters

base_path = os.path.realpath(os.path.dirname(__file__))

reach_geobuf_zst = 'reach_geobuf.pbf.zst'
catch_geobuf_zst = 'catch_geobuf.pbf.zst'
catch_pbf = os.path.join(base_path, 'catch_geobuf.pbf')
sel_data_h5 = os.path.join(base_path, 'selection_data.h5')

reach_dict = tethysts.utils.read_pkl_zstd(os.path.join(base_path, reach_geobuf_zst), True)
catch_gbuf = base64.b64encode(tethysts.utils.read_pkl_zstd(os.path.join(base_path, catch_geobuf_zst))).decode()
# catch_gbuf = tethysts.utils.read_pkl_zstd(os.path.join(base_path, catch_geobuf_zst))

# catch1 = orjson.loads(orjson.dumps(geobuf.decode(catch_gbuf)))
# catch_features = []

# for feature in catch1['features']:
#     if feature['geometry']['type'] == 'Polygon':
#         catch_features.append(feature)

# catch2 = {'type': catch1['type'], 'features': catch_features}

# with open(os.path.join(base_path, 'catch.geojson'), 'wb') as f:
#     f.write(orjson.dumps(catch1))

# catch_gbuf = geobuf.decode(tethysts.utils.read_pkl_zstd(os.path.join(base_path, catch_geobuf_zst)))

map_height = 700

lat1 = -45.74
lon1 = 168.25
zoom1 = 8

# mapbox_access_token = "pk.eyJ1IjoibXVsbGVua2FtcDEiLCJhIjoiY2pudXE0bXlmMDc3cTNxbnZ0em4xN2M1ZCJ9.sIOtya_qe9RwkYXj5Du1yg"

tabs_styles = {
    'height': '40px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '5px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '5px'
}

attribution = 'Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'

style_handle = assign("""function style(feature) {
    return {
        fillColor: 'grey',
        weight: 2,
        opacity: 1,
        color: 'black',
        fillOpacity: 0.1
    };
}""")

###############################################
### Initial processing

sel1 = xr.open_dataset(sel_data_h5, engine='h5netcdf')

freqs = sel1['frequency'].values
indicators = sel1['indicator'].values
indicators.sort()
# nzsegments = sel1['nzsegment'].values
percent_changes = sel1['percent_change'].values
time_periods = sel1['time_period'].values
catches = list(reach_dict.keys())




###############################################
### App layout


def layout1():


    ### Dash layout
    layout = html.Div(children=[
        html.Div([
            # html.P(children='Select species:'),
            html.Label('Please select a catchment on the map:'),
            dcc.Dropdown(options=[{'label': d, 'value': d} for d in catches], id='indicator', optionHeight=40, clearable=True),
            html.Label('Select Indicator:'),
            dcc.Dropdown(options=[{'label': d, 'value': d} for d in indicators], id='catch_id', optionHeight=40, clearable=True),
            html.Label('Select expected percent improvement:'),
            dcc.Dropdown(options=[{'label': d, 'value': d} for d in percent_changes], id='percent_change', clearable=True),
            html.Label('Select sampling length (years):'),
            dcc.Dropdown(options=[{'label': d, 'value': d} for d in time_periods], id='time_period', clearable=True),
            html.Label('Select sampling frequency (obsevations per year):'),
            dcc.Dropdown(options=[{'label': d, 'value': d} for d in freqs], id='freq', clearable=True),
            # dcc.Link(html.Img(src=os.path.join('assets', 'es-logo.svg')), href='https://www.es.govt.nz/')
            ], className='two columns', style={'margin': 10}),

    html.Div([
        dl.Map(children=[
            dl.TileLayer(id='tile_layer', attribution=attribution),
            dl.GeoJSON(url="/assets/catch.pbf", format="geobuf", id='catch_map', zoomToBoundsOnClick=True, zoomToBounds=True, options=dict(style=style_handle))
                            ], style={'width': '100%', 'height': 780, 'margin': "auto", "display": "block"}, id="map2")
    ], className='fourish columns', style={'margin': 10}),

    html.Div([
        dcc.Loading(
                id="loading-tabs",
                type="default",
                children=[dcc.Tabs(id='plot_tabs', value='info_tab', style=tabs_styles, children=[
                            dcc.Tab(label='Info', value='info_tab', style=tab_style, selected_style=tab_selected_style),
                            dcc.Tab(label='Habitat Suitability', value='hs_tab', style=tab_style, selected_style=tab_selected_style),
                            ]
                        ),
                    html.Div(id='plots')
                    ]
                ),

    ], className='fourish columns', style={'margin': 10}),

    # dcc.Store(id='tethys_obj', data=utils.encode_obj(tethys)),
    # dcc.Store(id='hsc_obj', data=utils.encode_obj(hsc_dict)),
    # dcc.Store(id='stns_obj', data=utils.encode_obj(stns_dict3)),
    # dcc.Store(id='result_obj', data='')
], style={'margin':0})

    return layout
