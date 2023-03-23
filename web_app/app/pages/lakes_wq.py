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
import pandas as pd
import numpy as np
# import requests
import zstandard as zstd
import codecs
import pickle
import dash_leaflet as dl
import dash_leaflet.express as dlx
import geopandas as gpd
from gistools import vector
import os
# import tethysts
import base64
import geobuf
import dash_leaflet.express as dlx
from dash_extensions.javascript import assign, arrow_function
import pathlib
import hdf5plugin
import booklet

# from .app import app
# from . import utils

# from app import app
# import utils

##########################################
### Parameters

assets_path = pathlib.Path(os.path.split(os.path.realpath(os.path.dirname(__file__)))[0]).joinpath('assets')

dash.register_page(
    __name__,
    path='/lakes-wq',
    title='Lakes and Lagoons Water Quality',
    name='Lakes and Lagoons Water Quality'
)

app_base_path = pathlib.Path('/assets')

lakes_error_path = assets_path.joinpath('lakes_error.h5')

lakes_pbf_path = app_base_path.joinpath('lakes_points.pbf')
lakes_poly_gbuf_path = assets_path.joinpath('lakes_poly.blt')
lakes_catches_major_path = assets_path.joinpath('lakes_catchments_major.blt')
lakes_reach_gbuf_path = assets_path.joinpath('lakes_reaches.blt')
lakes_lc_path = assets_path.joinpath('lakes_catch_lc.blt')
lakes_reaches_mapping_path = assets_path.joinpath('lakes_reaches_mapping.blt')
lakes_catches_minor_path = assets_path.joinpath('lakes_catchments_minor.blt')

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

# catch_style_handle = assign("""function style(feature) {
#                             return {
#                                 fillColor: 'grey',
#                                 weight: 2,
#                                 opacity: 1,
#                                 color: 'black',
#                                 fillOpacity: 0.1
#                             };
#                         }""")

# lake_style_handle = assign("""function style4(feature) {
#                             return {
#                                 fillColor: '#A4DCCC',
#                                 weight: 2,
#                                 opacity: 1,
#                                 color: 'black',
#                                 fillOpacity: 1
#                             };
#                         }""")

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

freq_mapping = {12: 'once a month', 26: 'once a fortnight', 52: 'once a week', 104: 'twice a week', 364: 'once a day'}
time_periods = [5, 10, 20, 30]

catch_style = {'fillColor': 'grey', 'weight': 2, 'opacity': 1, 'color': 'black', 'fillOpacity': 0.1}
lake_style = {'fillColor': '#A4DCCC', 'weight': 4, 'opacity': 1, 'color': 'black', 'fillOpacity': 1}
reach_style = {'weight': 2, 'opacity': 0.75, 'color': 'grey'}
# lake_style2 = dict(weight=4, opacity=1, color='white', fillColor='#A4DCCC', fillOpacity=1)
# style = dict(weight=4, opacity=1, color='white')
classes = [0, 20, 40, 60, 80]
bins = classes.copy()
bins.append(101)
# colorscale = ['#808080', '#FED976', '#FEB24C', '#FC4E2A', '#BD0026', '#800026']
colorscale = ['#808080', '#FED976', '#FD8D3C', '#E31A1C', '#800026']
# ctg = ["{}%+".format(cls, classes[i + 1]) for i, cls in enumerate(classes[1:-1])] + ["{}%+".format(classes[-1])]
# ctg.insert(0, 'NA')
ctg = ["{}%+".format(cls, classes[i + 1]) for i, cls in enumerate(classes[:-1])] + ["{}%+".format(classes[-1])]
# ctg.insert(0, 'NA')
# colorbar = dlx.categorical_colorbar(categories=ctg, colorscale=colorscale, width=300, height=30, position="bottomleft")
indices = list(range(len(ctg) + 1))
colorbar = dl.Colorbar(min=0, max=len(ctg), classes=indices, colorscale=colorscale, tooltip=True, tickValues=[item + 0.5 for item in indices[:-1]], tickText=ctg, width=300, height=30, position="bottomright")

base_reach_style = dict(weight=4, opacity=1, color='white')

info = dcc.Markdown(id="info_lakes", className="info", style={"position": "absolute", "top": "10px", "right": "10px", "z-index": "1000"})
# info = html.Div(id="info", className="info", style={"position": "absolute", "top": "10px", "right": "10px", "z-index": "1000"})

indicator_dict = {'CHLA': 'Chlorophyll a', 'CYANOTOT': 'Total Cyanobacteria', 'ECOLI': 'E.coli', 'NH4N': 'Ammoniacal nitrogen', 'Secchi': 'Secchi depth', 'TN': 'Total nitrogen', 'TP': 'Total phosphorus', 'pH': 'pH'}

###############################################
### Helper Functions

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
    if '.gpkg' in filename:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        plan1 = gpd.read_file(io.BytesIO(decoded))

        output = encode_obj(plan1)
    elif contents is None:
        output = None
    else:
        output = html.Div(['Wrong file type. It must be a GeoPackage (gpkg).'])

    return output


def calc_lake_reach_reductions(lake_id, plan_file, reduction_col='reduction'):
    """
    This assumes that the concentration is the same throughout the entire greater catchment. If it's meant to be different, then each small catchment must be set and multiplied by the area to weight the contribution downstream.
    """
    with booklet.open(lakes_catches_minor_path, 'r') as f:
        c1 = f[str(lake_id)]

    # with shelflet.open(lakes_reaches_mapping_path, 'r') as f:
    #     branches = f[str(lake_id)]

    plan1 = plan_file[[reduction_col, 'geometry']].to_crs(2193)
    # c1 = read_pkl_zstd(os.path.join(base_path, 'catchments', '{}.pkl.zst'.format(catch_id)), True)

    c2 = plan1.overlay(c1)
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

    c6 = c5.sum()
    props = (np.round((c6.prop_area/c6.base_area)*100*0.5)*2).astype('int8')

    return props



###############################################
### Initial processing

# sel1 = xr.open_dataset(base_path.joinpath(sel_data_h5), engine='h5netcdf')

# with booklet.open(lakes_catches_major_path, 'r') as f:
#     lakes = list(f.keys())

# lakes.sort()

with open(assets_path.joinpath('lakes_points.pbf'), 'rb') as f:
    geodict = geobuf.decode(f.read())

lakes = [{'value': f['id'], 'label': ' '.join(f['properties']['name'].split())} for f in geodict['features']]
# freqs = sel1['frequency'].values
# x1 = xr.open_dataset(lakes_error_path, engine='h5netcdf')
# indicators = x1.indicator.values
# indicators.sort()
# nzsegments = sel1['nzsegment'].values
# percent_changes = sel1['percent_change'].values
# time_periods = sel1['time_period'].values

# x1.close()
# del x1

indicators = [{'value': k, 'label': v} for k, v in indicator_dict.items()]

###############################################
### App layout


def layout():
    layout = html.Div(children=[
        html.Div([html.H1('Lake/Lagoon Water Quality')]),
        html.Div([
            html.H3('(1) Reductions routing'),

            html.Label('Select a lake/lagoon on the map:'),
            dcc.Dropdown(options=lakes, id='lake_id', optionHeight=40, clearable=False),

            dcc.Upload(
                id='upload_data_lakes',
                children=html.Button('Upload reductions polygons gpkg'),
                style={
                    'width': '100%',
                    'height': '60px',
                    'textAlign': 'center',
                    'margin-top': 40
                },
                multiple=False
            ),
            dcc.Markdown('''##### **Or**''', style={
                'textAlign': 'center',
                            }),
            html.Button('Use land cover for reductions', id='demo_data_lakes',
                        style={
                            'width': '100%',
                            'height': '50%',
                            'textAlign': 'center',
                            'margin-top': 20
                        }),

            html.Label('Select a reductions column in the GIS file:', style={'margin-top': 20}),
            dcc.Dropdown(options=[], id='col_name_lakes', optionHeight=40, clearable=False),
            dcc.Loading(
            id="loading-2",
            type="default",
            children=html.Div([html.Button('Process reductions', id='process_lakes', n_clicks=0),
                               html.Div(id='process_text_lakes')],
                              style={'margin-top': 20, 'margin-bottom': 100}
                              )
        ),

            # html.Label('Select Indicator:'),
            # dcc.Dropdown(options=[{'label': d, 'value': d} for d in indicators], id='indicator', optionHeight=40, clearable=False, value='NH4'),
            # html.Label('Select expected percent improvement:'),
            # dcc.Dropdown(options=[{'label': d, 'value': d} for d in percent_changes], id='percent_change', clearable=False, value=10),

            # dcc.Link(html.Img(src=str(app_base_path.joinpath('our-land-and-water-logo.svg'))), href='https://ourlandandwater.nz/')
            ], className='two columns', style={'margin': 10}),

    html.Div([
        html.H3('(2) Sampling options'),
        html.Label('Select Indicator:'),
        dcc.Dropdown(options=indicators, id='indicator_lakes', optionHeight=40, clearable=False),
        html.Label('Select sampling length (years):', style={'margin-top': 20}),
        dcc.Dropdown(options=[{'label': d, 'value': d} for d in time_periods], id='time_period_lakes', clearable=False, value=5),
        html.Label('Select sampling frequency:'),
        dcc.Dropdown(options=[{'label': v, 'value': k} for k, v in freq_mapping.items()], id='freq_lakes', clearable=False, value=12),

        html.H4(children='Map Layers', style={'margin-top': 20}),
        dcc.Checklist(
               options=[
                   {'label': 'Reductions polygons', 'value': 'reductions_poly'},
                    # {'label': 'Lake polygon', 'value': 'lake_poly'},
                   {'label': 'River reaches', 'value': 'reach_map'}
               ],
               value=['reductions_poly', 'reach_map'],
               id='map_checkboxes_lakes',
               style={'padding': 5, 'margin-bottom': 330}
            ),
        dcc.Link(html.Img(src=str(app_base_path.joinpath('our-land-and-water-logo.svg'))), href='https://ourlandandwater.nz/')
        ], className='two columns', style={'margin': 10}),

    html.Div([
        dl.Map(center=center, zoom=7, children=[
            dl.TileLayer(id='tile_layer_lakes', attribution=attribution),
            dl.GeoJSON(url=str(lakes_pbf_path), format="geobuf", id='lake_points', zoomToBoundsOnClick=True, zoomToBounds=True, cluster=True),
            dl.GeoJSON(data='', format="geobuf", id='catch_map_lakes', zoomToBoundsOnClick=True, options=dict(style=catch_style)),
            # dl.GeoJSON(url='', format="geobuf", id='base_reach_map', options=dict(style=base_reaches_style_handle)),
            dl.GeoJSON(data='', format="geobuf", id='reach_map_lakes', options=dict(style=reach_style), hideout={}),
            dl.GeoJSON(data='', format="geobuf", id='lake_poly', options=dict(style=lake_style_handle), hideout={'classes': [''], 'colorscale': ['#808080'], 'style': lake_style, 'colorProp': 'name'}),
            dl.GeoJSON(data='', format="geobuf", id='reductions_poly_lakes'),
            colorbar,
            info
                            ], style={'width': '100%', 'height': 780, 'margin': "auto", "display": "block"})
    ], className='five columns', style={'margin': 10}),

    # html.Div([
    #     dcc.Loading(
    #             id="loading-tabs",
    #             type="default",
    #             children=[dcc.Tabs(id='plot_tabs', value='info_tab', style=tabs_styles, children=[
    #                         dcc.Tab(label='Info', value='info_tab', style=tab_style, selected_style=tab_selected_style),
    #                         # dcc.Tab(label='Habitat Suitability', value='hs_tab', style=tab_style, selected_style=tab_selected_style),
    #                         ]
    #                     ),
    #                 html.Div(id='plots')
    #                 ]
    #             ),

    # ], className='three columns', style={'margin': 10}),

    dcc.Store(id='props_obj_lakes', data=''),
    dcc.Store(id='reaches_obj_lakes', data=''),
    dcc.Store(id='reductions_obj_lakes', data=''),
], style={'margin':0})

    return layout

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
        lake_id = feature['id']

    return lake_id


@callback(
        Output('reach_map_lakes', 'data'),
        Input('lake_id', 'value'),
        Input('map_checkboxes_lakes', 'value'),
        )
# @cache.memoize()
def update_reaches_lakes(lake_id, map_checkboxes):
    if isinstance(lake_id, str) and ('reach_map' in map_checkboxes):
        with booklet.open(lakes_reach_gbuf_path, 'r') as f:
            data = base64.b64encode(f[int(lake_id)]).decode()
    else:
        data = ''

    return data


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
        Output('reductions_obj_lakes', 'data'), Output('col_name_lakes', 'value'),
        Input('upload_data_lakes', 'contents'),
        Input('demo_data_lakes', 'n_clicks'),
        Input('lake_id', 'value'),
        State('upload_data_lakes', 'filename'),
        prevent_initial_call=True
        )
# @cache.memoize()
def update_reductions_obj_lakes(contents, n_clicks, lake_id, filename):
    if n_clicks is None:
        if contents is not None:
            data = parse_gis_file(contents, filename)

            if isinstance(data, str):
                return data, None
        else:
            return '', None
    elif isinstance(lake_id, str):
        with booklet.open(lakes_lc_path, 'r') as f:
            data = encode_obj(f[int(lake_id)])

        return data, 'reduction'
    else:
        return '', None


@callback(
        Output('reductions_poly_lakes', 'data'),
        Input('reductions_obj_lakes', 'data'),
        Input('map_checkboxes_lakes', 'value'),
        Input('col_name_lakes', 'value'),
        )
# @cache.memoize()
def update_reductions_poly_lakes(reductions_obj, map_checkboxes, col_name):
    # print(reductions_obj)
    # print(col_name)
    if (reductions_obj != '') and (reductions_obj is not None) and ('reductions_poly' in map_checkboxes):

        data = decode_obj(reductions_obj).to_crs(4326)

        if isinstance(col_name, str):
            data[col_name] = data[col_name].astype(str).str[:] + '% reduction'
            data.rename(columns={col_name: 'tooltip'}, inplace=True)

        gbuf = dlx.geojson_to_geobuf(data.__geo_interface__)

        return gbuf
    else:
        return ''


@callback(
        Output('col_name_lakes', 'options'),
        Input('reductions_obj_lakes', 'data')
        )
# @cache.memoize()
def update_column_options_lakes(reductions_obj):
    # print(reductions_obj)
    if (reductions_obj != '') and (reductions_obj is not None):
        data = decode_obj(reductions_obj)
        cols = [{'label': col, 'value': col} for col in data.columns if col not in ['geometry', 'id']]

        return cols
    else:
        return []


@callback(
    Output('reaches_obj_lakes', 'data'), Output('process_text_lakes', 'children'),
    Input('process_lakes', 'n_clicks'),
    [State('lake_id', 'value'), State('reductions_obj_lakes', 'data'), State('col_name_lakes', 'value')],
    prevent_initial_call=True)
def update_reach_data_lakes(click, lake_id, reductions_obj, col_name):
    """

    """
    if isinstance(lake_id, str) and (reductions_obj != '') and (reductions_obj is not None) and isinstance(col_name, str):
        plan_file = decode_obj(reductions_obj)
        props = calc_lake_reach_reductions(lake_id, plan_file, reduction_col=col_name)
        data = encode_obj(props)
        text_out = 'Routing complete'
    else:
        data = ''
        text_out = 'Not all inputs have been selected'

    return data, text_out


@callback(
    Output('props_obj_lakes', 'data'),
    [Input('reaches_obj_lakes', 'data'), Input('indicator_lakes', 'value'), Input('time_period_lakes', 'value'), Input('freq_lakes', 'value')],
    [State('lake_id', 'value')]
    )
def update_props_data_lakes(reaches_obj, indicator, n_years, n_samples_year, lake_id):
    """

    """
    if (reaches_obj != '') and (reaches_obj is not None) and isinstance(n_years, int) and isinstance(n_samples_year, int) and isinstance(indicator, str):
        props = decode_obj(reaches_obj)
        # print(props)

        n_samples = n_samples_year*n_years

        power_data = xr.open_dataset(lakes_error_path, engine='h5netcdf')
        try:
            power_data1 = int(power_data.sel(indicator=indicator, LFENZID=int(lake_id), n_samples=n_samples, conc_perc=100-props).power.values)
        except:
            power_data1 = 0
        power_data.close()
        del power_data

        data = encode_obj({'reduction': props, 'power': power_data1, 'lake_id': lake_id})
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
    [Input('props_obj_lakes', 'data')],
    Input('lake_id', 'value'),
    prevent_initial_call=True
    )
def update_hideout_lakes(props_obj, lake_id):
    """

    """
    if (props_obj != '') and (props_obj is not None):
        # print('trigger')
        props = decode_obj(props_obj)
        # print(props)
        # print(type(lake_id))

        if props['lake_id'] == lake_id:

            color_arr = pd.cut([props['power']], bins, labels=colorscale, right=False).tolist()
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
    [Input('props_obj_lakes', 'data'),
      Input('reductions_obj_lakes', 'data'),
      Input('map_checkboxes_lakes', 'value'),
      Input("lake_poly", "click_feature")],
    )
def update_map_info_lakes(props_obj, reductions_obj, map_checkboxes, feature):
    """

    """
    info = """###### Likelihood of observing a reduction (%)"""

    if (reductions_obj != '') and (reductions_obj is not None) and ('reductions_poly' in map_checkboxes):
        info = info + """\n\nHover over the polygons to see reduction %"""

    if (props_obj != '') and (props_obj is not None):
        if feature is not None:
            props = decode_obj(props_obj)

            info_str = """\n\nReduction: {red}%\n\nLikelihood of observing a reduction (power): {t_stat}%""".format(red=int(props['reduction']), t_stat=int(props['power']))

            info = info + info_str

        else:
            info = info + """\n\nClick on a lake to see info"""

    return info








# with shelflet.open(lakes_lc_path, 'r') as f:
#     plan_file = f[str(lake_id)]
