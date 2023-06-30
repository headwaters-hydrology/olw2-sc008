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
    path='/gw-wq',
    title='Groundwater Quality',
    name='gw_wq',
    description='Groundwater Quality'
)

app_base_path = pathlib.Path('/assets')

gw_error_path = assets_path.joinpath('gw_points_error.h5')

# gw_pbf_path = app_base_path.joinpath('gw_points.pbf')
gw_points_rc_blt = assets_path.joinpath('gw_points_rc.blt')
rc_bounds_gbuf = app_base_path.joinpath('rc_bounds.pbf')

# gw_poly_gbuf_path = assets_path.joinpath('gw_poly.blt')
# gw_catches_major_path = assets_path.joinpath('gw_catchments_major.blt')
# gw_reach_gbuf_path = assets_path.joinpath('gw_reaches.blt')
# gw_lc_path = assets_path.joinpath('gw_catch_lc.blt')
# gw_reaches_mapping_path = assets_path.joinpath('gw_reaches_mapping.blt')
# gw_catches_minor_path = assets_path.joinpath('gw_catchments_minor.blt')

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

# base_rc_style_handle = assign("""function style3(feature) {
#     return {
#         weight: 2,
#         opacity: 0.75,
#         color: 'grey',
#     };
# }""", name='gw_base_rc_style_handle')

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

point_radius = 6

gw_points_hideout = {'classes': [], 'colorscale': ['#808080'], 'circleOptions': dict(fillOpacity=1, stroke=False, radius=point_radius), 'colorProp': 'tooltip'}

# freq_mapping = {12: 'once a month', 26: 'once a fortnight', 52: 'once a week', 104: 'twice a week', 364: 'once a day'}
freq_mapping = {1: 'once a year', 4: 'once a quarter', 12: 'once a month', 26: 'once a fortnight', 52: 'once a week'}
time_periods = [5, 10, 20, 30]

# catch_style = {'fillColor': 'grey', 'weight': 2, 'opacity': 1, 'color': 'black', 'fillOpacity': 0.1}
# gw_style = {'fillColor': '#A4DCCC', 'weight': 4, 'opacity': 1, 'color': 'black', 'fillOpacity': 1}
# reach_style = {'weight': 2, 'opacity': 0.75, 'color': 'grey'}
# gw_style2 = dict(weight=4, opacity=1, color='white', fillColor='#A4DCCC', fillOpacity=1)
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

# base_reach_style = dict(weight=4, opacity=1, color='white')

info = dcc.Markdown(id="gw_info", className="info", style={"position": "absolute", "top": "10px", "right": "10px", "z-index": "1000"})
# info = html.Div(id="info", className="info", style={"position": "absolute", "top": "10px", "right": "10px", "z-index": "1000"})

# indicator_dict = {'CHLA': 'Chlorophyll a', 'CYANOTOT': 'Total Cyanobacteria', 'ECOLI': 'E.coli', 'NH4N': 'Ammoniacal nitrogen', 'Secchi': 'Secchi depth', 'TN': 'Total nitrogen', 'TP': 'Total phosphorus', 'pH': 'pH'}

indicator_dict = {'Nitrate': 'Nitrate'}

reductions_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

reductions_options = [{'value': v, 'label': str(v)+'%'} for v in reductions_values]



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


# def calc_gw_reach_reductions(gw_id, plan_file, reduction_col='reduction'):
#     """
#     This assumes that the concentration is the same throughout the entire greater catchment. If it's meant to be different, then each small catchment must be set and multiplied by the area to weight the contribution downstream.
#     """
#     with booklet.open(gw_catches_minor_path, 'r') as f:
#         c1 = f[str(gw_id)]

#     plan1 = plan_file[[reduction_col, 'geometry']].to_crs(2193)

#     c2 = plan1.overlay(c1)
#     c2.loc[c2[reduction_col].isnull(), reduction_col] = 0
#     c2['s_area'] = c2.area

#     c2['combo_area'] = c2.groupby('nzsegment')['s_area'].transform('sum')

#     c2['prop'] = c2[reduction_col]*(c2['s_area']/c2['combo_area'])

#     c3 = c2.groupby('nzsegment')['prop'].sum()
#     c4 = c1.merge(c3.reset_index(), on='nzsegment')
#     area = c4.area
#     c4['base_area'] = area * 100
#     c4['prop_area'] = area * c4['prop']

#     c5 = c4[['nzsegment', 'base_area', 'prop_area']].set_index('nzsegment').copy()

#     c6 = c5.sum()
#     props = (np.round((c6.prop_area/c6.base_area)*100*0.5)*2).astype('int8')

#     return props



###############################################
### Initial processing

# sel1 = xr.open_dataset(base_path.joinpath(sel_data_h5), engine='h5netcdf')

# with booklet.open(gw_catches_major_path, 'r') as f:
#     lakes = list(f.keys())

# lakes.sort()

with booklet.open(gw_points_rc_blt, 'r') as f:
    rcs = list(f.keys())

rcs.sort()

# with open(assets_path.joinpath('gw_points.pbf'), 'rb') as f:
#     geodict = geobuf.decode(f.read())

# lakes = [{'value': f['id'], 'label': ' '.join(f['properties']['name'].split())} for f in geodict['features']]
# lakes = [{'value': f['id'], 'label': f['id']} for f in geodict['features']]
# gw_refs = [f['id'] for f in geodict['features']]
# freqs = sel1['frequency'].values
# x1 = xr.open_dataset(gw_error_path, engine='h5netcdf')
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
        # html.Div([html.H2('Groundwater Quality')]),
        html.Div([
            html.H3('(1) Reductions'),

            html.Label('Select a Regional Council on the map:'),
            dcc.Dropdown(options=[{'label': d, 'value': d} for d in rcs], id='rc_id', optionHeight=40, clearable=False, style={'margin-bottom': 20}),

            # dcc.Upload(
            #     id='upload_data_gw',
            #     children=html.Button('Upload reductions polygons gpkg'),
            #     style={
            #         'width': '100%',
            #         'height': '60px',
            #         'textAlign': 'center',
            #         'margin-top': 40
            #     },
            #     multiple=False
            # ),
            # dcc.Markdown('''##### **Or**''', style={
            #     'textAlign': 'center',
            #                 }),
            # html.Button('Use land cover for reductions', id='demo_data_lakes',
            #             style={
            #                 'width': '100%',
            #                 'height': '50%',
            #                 'textAlign': 'center',
            #                 'margin-top': 20
            #             }),

            html.Label('Select a reduction %:'),
            dcc.Dropdown(options=reductions_options, id='gw_reductions', optionHeight=40, clearable=False),

        #     html.Label('Select a reductions column in the GIS file:', style={'margin-top': 20}),
        #     dcc.Dropdown(options=[], id='col_name_lakes', optionHeight=40, clearable=False),
        #     dcc.Loading(
        #     id="loading-2",
        #     type="default",
        #     children=html.Div([html.Button('Process reductions', id='process_lakes', n_clicks=0),
        #                        html.Div(id='process_text_lakes')],
        #                       style={'margin-top': 20, 'margin-bottom': 100}
        #                       )
        # ),

            # html.Label('Select Indicator:'),
            # dcc.Dropdown(options=[{'label': d, 'value': d} for d in indicators], id='indicator', optionHeight=40, clearable=False, value='NH4'),
            # html.Label('Select expected percent improvement:'),
            # dcc.Dropdown(options=[{'label': d, 'value': d} for d in percent_changes], id='percent_change', clearable=False, value=10),

            # dcc.Link(html.Img(src=str(app_base_path.joinpath('our-land-and-water-logo.svg'))), href='https://ourlandandwater.nz/')
            ], className='two columns', style={'margin': 10}),

    html.Div([
        html.H3('(2) Sampling options'),
        html.Label('Select Indicator:'),
        dcc.Dropdown(options=indicators, id='gw_indicator', optionHeight=40, clearable=False, value='Nitrate'),
        html.Label('Select sampling length (years):', style={'margin-top': 20}),
        dcc.Dropdown(options=[{'label': d, 'value': d} for d in time_periods], id='gw_time_period', clearable=False, value=5),
        html.Label('Select sampling frequency:'),
        dcc.Dropdown(options=[{'label': v, 'value': k} for k, v in freq_mapping.items()], id='gw_freq', clearable=False, value=12, style={'margin-bottom': 460}),

        # html.H4(children='Map Layers', style={'margin-top': 20}),
        # dcc.Checklist(
        #        options=[
        #            {'label': 'Reductions polygons', 'value': 'reductions_poly'},
        #             # {'label': 'Lake polygon', 'value': 'gw_poly'},
        #            {'label': 'River reaches', 'value': 'reach_map'}
        #        ],
        #        value=['reductions_poly', 'reach_map'],
        #        id='map_checkboxes_lakes',
        #        style={'padding': 5, 'margin-bottom': 330}
        #     ),
        # dcc.Link(html.Img(src=str(app_base_path.joinpath('our-land-and-water-logo.svg'))), href='https://ourlandandwater.nz/')
        ], className='two columns', style={'margin': 10}),

    html.Div([
        dl.Map(center=center, zoom=7, children=[
            dl.TileLayer(id='gw_tile_layer', attribution=attribution),
            dl.GeoJSON(url=str(rc_bounds_gbuf), format="geobuf", id='rc_map', zoomToBoundsOnClick=True, zoomToBounds=True, options=dict(style=rc_style_handle),  hideout={}),
            dl.GeoJSON(data='', format="geobuf", id='gw_points', zoomToBounds=True, zoomToBoundsOnClick=True, cluster=False, options=dict(pointToLayer=gw_points_style_handle), hideout=gw_points_hideout),
            # dl.GeoJSON(data='', format="geobuf", id='catch_map_lakes', zoomToBoundsOnClick=True, options=dict(style=catch_style)),
            # dl.GeoJSON(url='', format="geobuf", id='base_reach_map', options=dict(style=base_reaches_style_handle)),
            # dl.GeoJSON(data='', format="geobuf", id='reach_map_lakes', options=dict(style=reach_style), hideout={}),
            # dl.GeoJSON(data='', format="geobuf", id='gw_poly', options=dict(style=gw_style_handle), hideout={'classes': [''], 'colorscale': ['#808080'], 'style': gw_style, 'colorProp': 'ref'}),
            # dl.GeoJSON(data='', format="geobuf", id='reductions_poly_lakes'),
            colorbar,
            info
                            ], style={'width': '100%', 'height': 700, 'margin': "auto", "display": "block"})
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

    dcc.Store(id='gw_props_obj', data=None),
    dcc.Store(id='gw_points_ids', data=None),
    # dcc.Store(id='reaches_obj_lakes', data=''),
    # dcc.Store(id='reductions_obj_lakes', data=''),
], style={'margin':0})

    return layout

###############################################
### Callbacks


@callback(
    Output('rc_id', 'value'),
    [Input('rc_map', 'click_feature')]
    )
def update_rc_id(feature):
    """

    """
    # print(ds_id)

    if feature is not None:
        rc_id = feature['id']
    else:
        rc_id = None

    return rc_id


@callback(
        Output('gw_points', 'data'),
        Output('gw_points_ids', 'data'),
        Input('rc_id', 'value'),
        )
# @cache.memoize()
def update_gw_points(rc_id):
    if (rc_id is not None):
        with booklet.open(gw_points_rc_blt, 'r') as f:
            data0 = f[rc_id]
            geo1 = geobuf.decode(data0)
            gw_points = [s['id'] for s in geo1['features']]
            gw_points_encode = encode_obj(gw_points)
            data = base64.b64encode(data0).decode()
    else:
        data = None
        gw_points_encode = None

    return data, gw_points_encode


# @callback(
#         Output('sites_points', 'data'),
#         Input('catch_id', 'value'),
#         # Input('map_checkboxes', 'value'),
#         )
# # @cache.memoize()
# def update_monitor_sites(catch_id):
#     if (catch_id is not None):
#         with booklet.open(rivers_sites_path, 'r') as f:
#             data = base64.b64encode(f[int(catch_id)]).decode()

#     else:
#         data = ''

#     return data


# @callback(
#         Output('rc_map', 'options'),
#         Input('rc_map', 'hideout'),
#         Input('rc_id', 'value')
#         )
# # @cache.memoize()
# def update_rcs_option(hideout, rc_id):
#     trig = ctx.triggered_id

#     if (len(hideout) == 0) or (trig == 'rc_id'):
#         options = dict(style=base_rc_style_handle)
#     else:
#         options = dict(style=rc_style_handle)

#     return options


@callback(
    Output('gw_props_obj', 'data'),
    [Input('gw_reductions', 'value'), Input('gw_indicator', 'value'), Input('gw_time_period', 'value'), Input('gw_freq', 'value')],
    # [State('gw_id', 'value')]
    )
def update_props_data_gw(reductions, indicator, n_years, n_samples_year):
    """

    """
    if isinstance(reductions, int) and isinstance(n_years, int) and isinstance(n_samples_year, int) and isinstance(indicator, str):
        n_samples = n_samples_year*n_years

        power_data = xr.open_dataset(gw_error_path, engine='h5netcdf')
        power_data1 = power_data.sel(indicator=indicator, n_samples=n_samples, conc_perc=100-reductions, drop=True).to_dataframe().reset_index()
        power_data.close()
        del power_data

        data = encode_obj(power_data1)
        return data
    else:
        raise dash.exceptions.PreventUpdate


# @callback(
#         Output('gw_poly', 'options'),
#         Input('gw_poly', 'hideout'),
#         # State('gw_id', 'value')
#         )
# # @cache.memoize()
# def update_gw_option(hideout):
#     trig = ctx.triggered_id
#     # print(trig)
#     # print(len(hideout))

#     if (len(hideout) == 1) or (trig == 'gw_id'):
#         options = dict(style=gw_style)
#     else:
#         options = dict(style=gw_style_handle)

#     return options


@callback(
    Output('gw_points', 'hideout'),
    Input('gw_props_obj', 'data'),
    Input('gw_points_ids', 'data'),
    prevent_initial_call=True
    )
def update_hideout_gw_points(props_obj, gw_points_encode):
    """

    """
    if (props_obj != '') and (props_obj is not None):

        # print('trigger')
        props = decode_obj(props_obj)
        # print(props)
        # print(type(gw_id))

        color_arr = pd.cut(props.power.values, bins, labels=colorscale, right=False).tolist()
        # print(color_arr)
        # print(props['gw_id'])

        hideout = {'classes': props['ref'].values, 'colorscale': color_arr, 'circleOptions': dict(fillOpacity=1, stroke=False, radius=point_radius), 'colorProp': 'tooltip'}
    elif (gw_points_encode is not None):
        # print('trigger')
        gw_refs = decode_obj(gw_points_encode)

        hideout = {'classes': gw_refs, 'colorscale': ['#808080'] * len(gw_refs), 'circleOptions': dict(fillOpacity=1, stroke=False, radius=point_radius), 'colorProp': 'tooltip'}
    else:
        hideout = gw_points_hideout

    return hideout


@callback(
    Output("gw_info", "children"),
    [Input('gw_props_obj', 'data'),
      Input('gw_reductions', 'value'),
      Input("gw_points", "click_feature")],
    State('gw_points_ids', 'data')
    )
def update_map_info_gw(props_obj, reductions, feature, gw_points_encode):
    """

    """
    info = """###### Likelihood of observing a reduction (%)"""
    info_str = """\n\nReduction: {red}%\n\nLikelihood of observing a reduction (power): {t_stat}%\n\nWell Depth (m): {depth:.1f}"""

    # if (reductions_obj != '') and (reductions_obj is not None) and ('reductions_poly' in map_checkboxes):
    #     info = info + """\n\nHover over the polygons to see reduction %"""

    if isinstance(reductions, int) and (props_obj != '') and (props_obj is not None):
        if feature is not None:
            # print(feature)
            gw_refs = decode_obj(gw_points_encode)
            if feature['id'] in gw_refs:
                props = decode_obj(props_obj)

                info2 = info_str.format(red=int(reductions), t_stat=int(props[props.ref==feature['id']].iloc[0]['power']), depth=feature['properties']['bore_depth'])

                info = info + info2

        else:
            info = info + """\n\nClick on a well to see info"""

    return info








# with shelflet.open(gw_lc_path, 'r') as f:
#     plan_file = f[str(gw_id)]
