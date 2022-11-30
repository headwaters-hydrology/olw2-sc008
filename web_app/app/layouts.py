#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 09:55:56 2021

@author: mike
"""
import xarray as xr
from dash import dcc, html, dash_table
import pandas as pd
import numpy as np
# import requests
import dash_leaflet as dl
import dash_leaflet.express as dlx
import copy
import os
# import tethysts
import base64
import geobuf
import orjson
import dash_leaflet.express as dlx
from dash_extensions.javascript import assign, arrow_function
import pathlib
import hdf5plugin

# from . import utils
import utils

##########################################
### Parameters

base_path = pathlib.Path(os.path.realpath(os.path.dirname(__file__))).joinpath('assets')

app_base_path = pathlib.Path('/assets')

catch_pbf = 'catchments.pbf'
# sel_data_h5 = 'selection_data.h5'
base_reaches_path = 'reaches'

map_height = 700

center = [-41.1157, 172.4759]

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

catch_style_handle = assign("""function style(feature) {
    return {
        fillColor: 'grey',
        weight: 2,
        opacity: 1,
        color: 'black',
        fillOpacity: 0.1
    };
}""")

base_reaches_style_handle = assign("""function style3(feature) {
    return {
        weight: 2,
        opacity: 0.75,
        color: 'grey',
    };
}""")

reach_style_handle = assign("""function style2(feature, context){
    const {classes, colorscale, style, colorProp} = context.props.hideout;  // get props from hideout
    const value = feature.properties[colorProp];  // get value the determines the color
    for (let i = 0; i < classes.length; ++i) {
        if (value == classes[i]) {
            style.color = colorscale[i];  // set the fill color according to the class
        }
    }
    return style;
}""")

freq_mapping = {12: 'once a month', 26: 'once a fortnight', 52: 'once a week', 104: 'twice a week', 364: 'once a day'}
time_periods = [5, 10, 20, 30]

classes = [0, 5, 20, 40, 60, 80]
colorscale = ['#808080', '#FED976', '#FEB24C', '#FC4E2A', '#BD0026', '#800026']
ctg = ["{}%+".format(cls, classes[i + 1]) for i, cls in enumerate(classes[1:-1])] + ["{}%+".format(classes[-1])]
ctg.insert(0, 'NA')
# colorbar = dlx.categorical_colorbar(categories=ctg, colorscale=colorscale, width=300, height=30, position="bottomleft")
indices = list(range(len(ctg) + 1))
colorbar = dl.Colorbar(min=0, max=len(ctg), classes=indices, colorscale=colorscale, tooltip=True, tickValues=[item + 0.5 for item in indices[:-1]], tickText=ctg, width=300, height=30, position="bottomright")

base_reach_style = dict(weight=4, opacity=1, color='white')

info = html.Div(id="info", className="info", style={"position": "absolute", "top": "10px", "right": "10px", "z-index": "1000"})

conc_dict = utils.read_pkl_zstd(base_path.joinpath('catch_conc.pkl.zst'), True)

###############################################
### Initial processing

# sel1 = xr.open_dataset(base_path.joinpath(sel_data_h5), engine='h5netcdf')

with open(base_path.joinpath(catch_pbf), 'rb') as f:
    catch1 = geobuf.decode(f.read())

# freqs = sel1['frequency'].values
indicators = list(conc_dict.keys())
indicators.sort()
# nzsegments = sel1['nzsegment'].values
# percent_changes = sel1['percent_change'].values
# time_periods = sel1['time_period'].values
catches = [int(c['id']) for c in catch1['features']]

# sel1.close()
# del sel1
del catch1

# catch_reaches = utils.read_pkl_zstd(str(base_path.joinpath(catch_reaches_file)), True)


###############################################
### App layout


def layout1():


    ### Dash layout
    layout = html.Div(children=[
        html.Div([
            html.Label('Select Indicator:'),
            dcc.Dropdown(options=[{'label': d, 'value': d} for d in indicators], id='indicator', optionHeight=40, clearable=False),
            html.Label('Select a catchment on the map:', style={'margin-top': 40}),
            dcc.Dropdown(options=[{'label': d, 'value': d} for d in catches], id='catch_id', optionHeight=40, clearable=False),

            dcc.Upload(
                id='upload-data',
                children=html.Button('Upload reductions polygons gpkg'),
                style={
                    'width': '100%',
                    'height': '60px',
                    'textAlign': 'center',
                    'margin-top': 40
                },
                multiple=False
            ),

            html.Label('Select a reductions column in the GIS file:', style={'margin-top': 0}),
            dcc.Dropdown(options=[], id='col_name', optionHeight=40, clearable=False),

            # html.Label('Select Indicator:'),
            # dcc.Dropdown(options=[{'label': d, 'value': d} for d in indicators], id='indicator', optionHeight=40, clearable=False, value='NH4'),
            # html.Label('Select expected percent improvement:'),
            # dcc.Dropdown(options=[{'label': d, 'value': d} for d in percent_changes], id='percent_change', clearable=False, value=10),
            html.Label('Select sampling length (years):', style={'margin-top': 40}),
            dcc.Dropdown(options=[{'label': d, 'value': d} for d in time_periods], id='time_period', clearable=False, value=5),
            html.Label('Select sampling frequency:'),
            dcc.Dropdown(options=[{'label': v, 'value': k} for k, v in freq_mapping.items()], id='freq', clearable=False, value=12),

            html.H4(children='Map Layers', style={'margin-top': 20}),
            dcc.Checklist(
                   options=[
                       {'label': 'Reductions polygons', 'value': 'reductions_poly'},
                       # {'label': 'River reach reductions', 'value': 'reach_map'},
                       {'label': 'River reaches', 'value': 'reach_map'}
                   ],
                   value=['reductions_poly', 'reach_map'],
                   id='map_checkboxes',
                   style={'padding': 5, 'margin-bottom': 120}
                ),

            dcc.Link(html.Img(src=str(app_base_path.joinpath('our-land-and-water-logo.svg'))), href='https://ourlandandwater.nz/')
            ], className='two columns', style={'margin': 10}),

    html.Div([
        dl.Map(children=[
            dl.TileLayer(id='tile_layer', attribution=attribution),
            dl.GeoJSON(url=str(app_base_path.joinpath(catch_pbf)), format="geobuf", id='catch_map', zoomToBoundsOnClick=True, zoomToBounds=True, options=dict(style=catch_style_handle)),
            # dl.GeoJSON(url='', format="geobuf", id='base_reach_map', options=dict(style=base_reaches_style_handle)),
            dl.GeoJSON(url='', format="geobuf", id='reach_map', options={}, hideout={}, hoverStyle=arrow_function(dict(weight=10, color='black', dashArray=''))),
            dl.GeoJSON(data='', format="geobuf", id='reductions_poly'),
            colorbar,
            info
                            ], style={'width': '100%', 'height': 780, 'margin': "auto", "display": "block"}, id="map2")
    ], className='six columns', style={'margin': 10}),

    html.Div([
        dcc.Loading(
                id="loading-tabs",
                type="default",
                children=[dcc.Tabs(id='plot_tabs', value='info_tab', style=tabs_styles, children=[
                            dcc.Tab(label='Info', value='info_tab', style=tab_style, selected_style=tab_selected_style),
                            # dcc.Tab(label='Habitat Suitability', value='hs_tab', style=tab_style, selected_style=tab_selected_style),
                            ]
                        ),
                    html.Div(id='plots')
                    ]
                ),

    ], className='three columns', style={'margin': 10}),

    dcc.Store(id='conc_obj', data=utils.encode_obj(conc_dict)),
    dcc.Store(id='props_obj', data=''),
    dcc.Store(id='reaches_obj', data=''),
    dcc.Store(id='reductions_obj', data='')
], style={'margin':0})

    return layout
