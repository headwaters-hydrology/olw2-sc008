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
import utils

##########################################
### Parameters

assets_path = pathlib.Path(os.path.split(os.path.realpath(os.path.dirname(__file__)))[0]).joinpath('assets')

# dash.register_page(
#     __name__,
#     path='/rivers-wq',
#     title='River Water Quality',
#     name='River Water Quality'
# )

app_base_path = pathlib.Path('/assets')

river_power_model_path = assets_path.joinpath('rivers_reaches_power_modelled.h5')
river_power_moni_path = assets_path.joinpath('rivers_reaches_power_monitored.h5')

rivers_catch_pbf_path = app_base_path.joinpath('rivers_catchments.pbf')

rivers_reach_gbuf_path = assets_path.joinpath('rivers_reaches.blt')
# rivers_loads_path = assets_path.joinpath('rivers_reaches_loads.h5')
rivers_flows_path = assets_path.joinpath('rivers_flows_rec.blt')
rivers_lc_clean_path = assets_path.joinpath('rivers_catch_lc.blt')
rivers_catch_path = assets_path.joinpath('rivers_catchments_minor.blt')
rivers_reach_mapping_path = assets_path.joinpath('rivers_reaches_mapping.blt')
rivers_sites_path = assets_path.joinpath('rivers_sites_catchments.blt')

rivers_catch_lc_dir = assets_path.joinpath('rivers_land_cover_gpkg')
rivers_catch_lc_gpkg_str = '{}_rivers_land_cover_reductions.gpkg'

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

# sites_points_handle = assign("""function style_sites(feature, latlng, context){
#                              const {circleOptions} = context.props.hideout;
#                              return L.circleMarker(latlng, circleOptions);}""", name='sites_points_handle')

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

freq_mapping = {12: 'once a month', 26: 'once a fortnight', 52: 'once a week', 104: 'twice a week', 364: 'once a day'}
time_periods = [5, 10, 20, 30]

style = dict(weight=4, opacity=1, color='white')
classes = [0, 20, 40, 60, 80]
bins = classes.copy()
bins.append(101)
# colorscale = ['#232323', '#FED976', '#FEB24C', '#FC4E2A', '#BD0026', '#800026']
colorscale = ['#808080', '#FED976', '#FD8D3C', '#E31A1C', '#800026']
# ctg = ["{}%+".format(cls, classes[i + 1]) for i, cls in enumerate(classes[1:-1])] + ["{}%+".format(classes[-1])]
# ctg.insert(0, 'NA')
ctg = ["{}%+".format(cls, classes[i + 1]) for i, cls in enumerate(classes[:-1])] + ["{}%+".format(classes[-1])]
# ctg.insert(0, 'NA')
# colorbar = dlx.categorical_colorbar(categories=ctg, colorscale=colorscale, width=300, height=30, position="bottomleft")
indices = list(range(len(ctg) + 1))
colorbar = dl.Colorbar(min=0, max=len(ctg), classes=indices, colorscale=colorscale, tooltip=True, tickValues=[item + 0.5 for item in indices[:-1]], tickText=ctg, width=300, height=30, position="bottomright")

base_reach_style = dict(weight=4, opacity=1, color='white')

info = dcc.Markdown(id="info", className="info", style={"position": "absolute", "top": "10px", "right": "10px", "z-index": "1000"})
# info = html.Div(id="info", className="info", style={"position": "absolute", "top": "10px", "right": "10px", "z-index": "1000"})

indicator_dict = {'BD': 'Black disk', 'EC': 'E.coli', 'DRP': 'Dissolved reactive phosporus', 'NH': 'Ammoniacal nitrogen', 'NO': 'Nitrate', 'TN': 'Total nitrogen', 'TP': 'Total phosphorus'}

point_radius = 5

rivers_points_hideout = {'classes': [], 'colorscale': ['#232323'], 'circleOptions': dict(fillOpacity=1, stroke=False, radius=point_radius), 'colorProp': 'nzsegment'}

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


def calc_river_reach_reductions(catch_id, plan_file, reduction_col='default_reductions'):
    """
    This assumes that the concentration is the same throughout the entire greater catchment. If it's meant to be different, then each small catchment must be set and multiplied by the area to weight the contribution downstream.
    """
    with booklet.open(rivers_catch_path) as f:
        c1 = f[int(catch_id)]

    with booklet.open(rivers_reach_mapping_path) as f:
        branches = f[int(catch_id)]

    # TODO: Package the flow up by catch_id so that there is less work here
    flows = {}
    with booklet.open(rivers_flows_path) as f:
        for way_id in branches:
            flows[int(way_id)] = f[int(way_id)]

    flows_df = pd.DataFrame.from_dict(flows, orient='index', columns=['flow'])
    flows_df.index.name = 'nzsegment'
    flows_df = flows_df.reset_index()

    plan0 = plan_file[[reduction_col, 'geometry']]
    plan1 = plan0[plan0[reduction_col] > 0].to_crs(2193)

    ## Calc reductions per nzsegment given sparse geometry input
    c2 = plan1.overlay(c1)
    c2['sub_area'] = c2.area

    c2['combo_area'] = c2.groupby('nzsegment')['sub_area'].transform('sum')
    c2['prop_reductions'] = c2[reduction_col]*(c2['sub_area']/c2['combo_area'])
    c3 = c2.groupby('nzsegment')[['prop_reductions', 'sub_area']].sum()

    ## Add in missing areas and assume that they are 0 reductions
    c1['tot_area'] = c1.area

    c4 = pd.merge(c1.drop('geometry', axis=1), c3, on='nzsegment', how='left')
    c4.loc[c4['prop_reductions'].isnull(), ['prop_reductions', 'sub_area']] = 0

    c4['reduction'] = (c4['prop_reductions'] * c4['sub_area'])/c4['tot_area']

    ## Scale the reductions to the flows
    c4 = c4.merge(flows_df, on='nzsegment')

    c4['base_flow'] = c4.flow * 100
    c4['prop_flow'] = c4.flow * c4['reduction']

    c5 = c4[['nzsegment', 'base_flow', 'prop_flow']].set_index('nzsegment').copy()
    c5 = {r: list(v.values()) for r, v in c5.to_dict('index').items()}

    # branches = read_pkl_zstd(os.path.join(base_path, 'reach_mappings', '{}.pkl.zst'.format(catch_id)), True)

    # props = {}
    # for reach, branch in branches.items():
    #     t_area = []
    #     a_append = t_area.append
    #     prop_area = []
    #     p_append = prop_area.append

    #     for b in branch:
    #         t_area1, prop_area1 = c5[b]
    #         a_append(t_area1)
    #         p_append(prop_area1)

    #     p1 = (np.sum(prop_area)/np.sum(t_area))
    #     props[reach] = p1

    props_index = np.array(list(branches.keys()), dtype='int32')
    props_val = np.zeros(props_index.shape)
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

        p1 = (np.sum(prop_area)/np.sum(t_area))
        if p1 < 0:
            props_val[h] = 0
        else:
            props_val[h] = p1

    props = xr.Dataset(data_vars={'reduction': (('nzsegment'), np.round(props_val*100).astype('int8')) # Round to nearest even number
                                  },
                        coords={'nzsegment': props_index}
                        ).sortby('nzsegment')

    ## Filter out lower stream orders
    # so3 = c1.loc[c1.stream_order > 2, 'nzsegment'].to_numpy()
    # props = props.sel(reach=so3)

    return props


###############################################
### Initial processing

# sel1 = xr.open_dataset(base_path.joinpath(sel_data_h5), engine='h5netcdf')

with booklet.open(rivers_reach_gbuf_path, 'r') as f:
    catches = [int(c) for c in f]

catches.sort()
# freqs = sel1['frequency'].values
indicators = list(indicator_dict.keys())
indicators.sort()
# nzsegments = sel1['nzsegment'].values
# percent_changes = sel1['percent_change'].values
# time_periods = sel1['time_period'].values

# sel1.close()
# del sel1



###############################################
### App layout


def layout():
    layout = html.Div(children=[
        # html.Div([html.H1('River Water Quality')]),
        html.Div([
            html.H3('(1) Reductions routing'),

            html.Label('Select a catchment on the map:'),
            dcc.Dropdown(options=[{'label': d, 'value': d} for d in catches], id='catch_id', optionHeight=40, clearable=False),

            dcc.Loading(
                children=[
                    dcc.Upload(
                        id='upload_data_rivers',
                        children=html.Button('Upload reductions polygons gpkg', style={
                            'width': '100%',
                        }),
                        style={
                            'width': '100%',
                            'height': '50%',
                            'textAlign': 'left',
                            'margin-top': 20
                        },
                        multiple=False
                    ),
                    dcc.Markdown('''###### **Or**''', style={
                        'textAlign': 'center',
                                    }),
                    html.Button('Use land cover for reductions', id='demo_data_rivers',
                                style={
                                    'width': '100%',
                                    'height': '50%',
                                    'textAlign': 'left',
                                    'margin-top': 20
                                }),
                    ]
                ),

            html.Label('Select a reductions column in the GIS file:', style={'margin-top': 20}),
            dcc.Dropdown(options=[], id='col_name', optionHeight=40, clearable=False),
            dcc.Loading(
            id="loading-2",
            type="default",
            children=[html.Div(html.Button("Download reductions polygons", id='dl_btn'), style={'margin-top': 10}),
    dcc.Download(id="dl_poly")],
            ),
            dcc.Loading(
            id="loading-1",
            type="default",
            children=html.Div([html.Button('Process reductions', id='process', n_clicks=0),
                               html.Div(id='process_text')],
                              style={'margin-top': 20, 'margin-bottom': 10}
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
        dcc.Dropdown(options=[{'label': indicator_dict[d], 'value': d} for d in indicators], id='indicator', optionHeight=40, clearable=False),
        html.Label('Select sampling length (years):', style={'margin-top': 20}),
        dcc.Dropdown(options=[{'label': d, 'value': d} for d in time_periods], id='time_period', clearable=False, value=5),
        html.Label('Select sampling frequency:'),
        dcc.Dropdown(options=[{'label': v, 'value': k} for k, v in freq_mapping.items()], id='freq', clearable=False, value=12),

        html.H4(children='Type of results', style={'margin-top': 20}),
        dcc.RadioItems(id='rivers_results_types',
               options=[
                   {'label': 'Modelled', 'value': 'modelled'},
                   {'label': 'Monitored', 'value': 'monitored'},
               ],
               value='modelled'
            ),

        html.H4(children='Map Layers', style={'margin-top': 20}),
        dcc.Checklist(
               options=[
                   {'label': 'Reductions polygons', 'value': 'reductions_poly'},
                   # {'label': 'River reach reductions', 'value': 'reach_map'},
                   {'label': 'River reaches', 'value': 'reach_map'}
               ],
               value=['reductions_poly', 'reach_map'],
               id='map_checkboxes_rivers',
               style={'padding': 5, 'margin-bottom': 20}
            ),
        dcc.Markdown('', style={
            'textAlign': 'left',
                        }, id='red_disclaimer_rivers')
        # dcc.Link(html.Img(src=str(app_base_path.joinpath('our-land-and-water-logo.svg'))), href='https://ourlandandwater.nz/')
        ], className='two columns', style={'margin': 10}),

    html.Div([
        dl.Map(center=center, zoom=7, children=[
            dl.TileLayer(id='tile_layer', attribution=attribution),
            dl.GeoJSON(url=str(rivers_catch_pbf_path), format="geobuf", id='catch_map', zoomToBoundsOnClick=True, zoomToBounds=True, options=dict(style=catch_style_handle)),
            # dl.GeoJSON(url='', format="geobuf", id='base_reach_map', options=dict(style=base_reaches_style_handle)),
            dl.GeoJSON(data='', format="geobuf", id='reach_map', options={}, hideout={}, hoverStyle=arrow_function(dict(weight=10, color='black', dashArray=''))),
            dl.GeoJSON(data='', format="geobuf", id='reductions_poly'),
            dl.GeoJSON(data='', format="geobuf", id='sites_points', options=dict(pointToLayer=sites_points_handle), hideout=rivers_points_hideout),
            colorbar,
            info
                            ], style={'width': '100%', 'height': 700, 'margin': "auto", "display": "block"}, id="map2")
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

    dcc.Store(id='props_obj', data=''),
    dcc.Store(id='reaches_obj', data=''),
    dcc.Store(id='reductions_obj', data=''),
], style={'margin':0})

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
        catch_id = feature['id']

    return catch_id


@callback(
        Output('reach_map', 'data'),
        Input('catch_id', 'value'),
        Input('map_checkboxes_rivers', 'value'),
        )
# @cache.memoize()
def update_reaches(catch_id, map_checkboxes):
    if (catch_id is not None) and ('reach_map' in map_checkboxes):
        with booklet.open(rivers_reach_gbuf_path, 'r') as f:
            data = base64.b64encode(f[int(catch_id)]).decode()

    else:
        data = ''

    return data


@callback(
        Output('sites_points', 'data'),
        Input('catch_id', 'value'),
        # Input('map_checkboxes', 'value'),
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
        Output('reductions_obj', 'data'), Output('col_name', 'value'),
        Input('upload_data_rivers', 'contents'),
        Input('demo_data_rivers', 'n_clicks'),
        Input('catch_id', 'value'),
        State('upload_data_rivers', 'filename'),
        prevent_initial_call=True
        )
# @cache.memoize()
def update_reductions_obj(contents, n_clicks, catch_id, filename):
    if n_clicks is None:
        if contents is not None:
            data = parse_gis_file(contents, filename)

            if isinstance(data, str):
                return data, None
        else:
            return '', None
    elif catch_id is not None:
        with booklet.open(rivers_lc_clean_path, 'r') as f:
            data = encode_obj(f[int(catch_id)])

        return data, 'default_reductions'
    else:
        return '', None


@callback(
        Output('red_disclaimer_rivers', 'children'),
        Input('upload_data_rivers', 'contents'),
        Input('demo_data_rivers', 'n_clicks'),
        Input('map_checkboxes_rivers', 'value'),
        prevent_initial_call=True
        )
def update_reductions_diclaimer(contents, n_clicks, map_checkboxes):
    if (n_clicks is None) or (contents is not None):
        return ''
    elif 'reductions_poly' in map_checkboxes:
        return '''* Areas on the map without polygon reductions are considered to have 0% reductions.'''
    else:
        return ''


@callback(
        Output('reductions_poly', 'data'),
        Input('reductions_obj', 'data'),
        Input('map_checkboxes_rivers', 'value'),
        Input('col_name', 'value'),
        )
# @cache.memoize()
def update_reductions_poly(reductions_obj, map_checkboxes, col_name):
    # print(reductions_obj)
    # print(col_name)
    if (reductions_obj != '') and (reductions_obj is not None) and ('reductions_poly' in map_checkboxes):

        data = decode_obj(reductions_obj).to_crs(4326)

        if isinstance(col_name, str):
            data = data[data[col_name] > 0]
            data[col_name] = data[col_name].astype(str).str[:] + '% reduction'
            data.rename(columns={col_name: 'tooltip'}, inplace=True)

        gbuf = dlx.geojson_to_geobuf(data.__geo_interface__)

        return gbuf
    else:
        return ''


@callback(
        Output('col_name', 'options'),
        Input('reductions_obj', 'data')
        )
# @cache.memoize()
def update_column_options(reductions_obj):
    # print(reductions_obj)
    if (reductions_obj != '') and (reductions_obj is not None):
        data = decode_obj(reductions_obj)
        cols = [{'label': col, 'value': col} for col in data.columns if (col not in ['geometry', 'id', 'fid', 'OBJECTID']) and np.issubdtype(data[col].dtype, np.number)]

        return cols
    else:
        return []


@callback(
    Output('reaches_obj', 'data'), Output('process_text', 'children'),
    Input('process', 'n_clicks'),
    [State('catch_id', 'value'), State('reductions_obj', 'data'), State('col_name', 'value')],
    prevent_initial_call=True)
def update_reach_data(click, catch_id, reductions_obj, col_name):
    """

    """
    if isinstance(catch_id, str) and (reductions_obj != '') and (reductions_obj is not None) and isinstance(col_name, str):
        plan_file = decode_obj(reductions_obj)
        props = calc_river_reach_reductions(catch_id, plan_file, reduction_col=col_name)
        data = encode_obj(props)
        text_out = 'Routing complete'
    else:
        data = ''
        text_out = 'Not all inputs have been selected'

    return data, text_out


@callback(
    Output('props_obj', 'data'),
    [Input('reaches_obj', 'data'), Input('indicator', 'value'), Input('time_period', 'value'), Input('freq', 'value')],
    [State('catch_id', 'value')]
    )
def update_props_data(reaches_obj, indicator, n_years, n_samples_year, catch_id):
    """

    """
    if (reaches_obj != '') and (reaches_obj is not None) and isinstance(n_years, int) and isinstance(n_samples_year, int) and isinstance(indicator, str):
        props = decode_obj(reaches_obj)

        n_samples = n_samples_year*n_years

        with booklet.open(rivers_reach_mapping_path) as f:
            branches = f[int(catch_id)][int(catch_id)]

        ## Modelled
        power_data = xr.open_dataset(river_power_model_path, engine='h5netcdf')

        power_data1 = power_data.sel(indicator=indicator, nzsegment=branches, n_samples=n_samples, drop=True).load().sortby('nzsegment').copy()
        power_data.close()
        del power_data

        props['conc_perc'] = 100 - props.reduction

        props = props.assign(power_modelled=(('nzsegment'), power_data1.sel(conc_perc=props.conc_perc, drop=True).power.values.astype('int8')))

        ## Monitored
        power_data = xr.open_dataset(river_power_moni_path, engine='h5netcdf')
        sites = power_data.nzsegment.values[power_data.nzsegment.isin(branches)].astype('int32')
        sites.sort()
        if len(sites) > 0:
            conc_perc = props.sel(nzsegment=sites).conc_perc
            power_data1 = power_data.sel(indicator=indicator, nzsegment=sites, n_samples=n_samples, drop=True).load().sortby('nzsegment').copy()
            power_data1 = power_data1.rename({'power': 'power_monitored'})
            power_data.close()
            del power_data

            power_data2 = power_data1.sel(conc_perc=conc_perc, drop=True)

            props = utils.xr_concat([props, power_data2])
        else:
            props = props.assign(power_monitored=(('nzsegment'), xr.full_like(props.reduction, np.nan, dtype='float32').values))

        data = encode_obj(props)
    else:
        data = ''

    return data


@callback(
    Output('reach_map', 'hideout'),
    Output('sites_points', 'hideout'),
    [Input('props_obj', 'data'),
     Input('rivers_results_types', 'value')],
    prevent_initial_call=True
    )
def update_hideout(props_obj, rivers_results_type):
    """

    """
    if (props_obj != '') and (props_obj is not None):
        props = decode_obj(props_obj)

        if rivers_results_type == 'modelled':

            ## modelled
            color_arr1 = pd.cut(props.power_modelled.values, bins, labels=colorscale, right=False).tolist()

            hideout_model = {'colorscale': color_arr1, 'classes': props.nzsegment.values.astype(int).tolist(), 'style': style, 'colorProp': 'nzsegment'}
            hideout_moni = rivers_points_hideout

        else:
            ## Monitored
            props_moni = props.dropna('nzsegment')
            if len(props_moni.nzsegment) > 0:
                # print(props_moni)
                color_arr2 = pd.cut(props_moni.power_monitored.values, bins, labels=colorscale, right=False).tolist()

                hideout_moni = {'classes': props_moni.nzsegment.values.astype(int), 'colorscale': color_arr2, 'circleOptions': dict(fillOpacity=1, stroke=False, radius=point_radius), 'colorProp': 'nzsegment'}

                # hideout_moni = {'colorscale': color_arr2, 'classes': props_moni.nzsegment.values.astype(int).tolist(), 'style': style, 'colorProp': 'nzsegment'}
            else:
                hideout_moni = rivers_points_hideout

            model_segs = props.nzsegment.values.astype(int)
            hideout_model = {'colorscale': ['#808080'] * len(model_segs), 'classes': model_segs, 'style': style, 'colorProp': 'nzsegment'}
    else:
        hideout_model = {}
        hideout_moni = rivers_points_hideout

    return hideout_model, hideout_moni


@callback(
    Output("info", "children"),
    [Input('props_obj', 'data'),
      Input('reductions_obj', 'data'),
      Input('map_checkboxes_rivers', 'value'),
      Input("reach_map", "click_feature"),
      Input("sites_points", "click_feature")],
    State('rivers_results_types', 'value')
    )
def update_map_info(props_obj, reductions_obj, map_checkboxes, feature, sites_points, rivers_results_type):
    """

    """
    info = """###### Likelihood of observing a reduction (%)"""

    if (reductions_obj != '') and (reductions_obj is not None) and ('reductions_poly' in map_checkboxes):
        info = info + """\n\nHover over the polygons to see reduction %"""

    if (props_obj != '') and (props_obj is not None):
        if rivers_results_type == 'modelled':
            if feature is not None:
                feature_id = int(feature['id'])
                props = decode_obj(props_obj)

                if feature_id in props.nzsegment:

                    reach_data = props.sel(nzsegment=feature_id)

                    info_str = """\n\nReduction: {red}%\n\nLikelihood of observing a reduction (power): {t_stat}%""".format(red=int(reach_data.reduction), t_stat=int(reach_data.power_modelled))

                    info = info + info_str

                else:
                    info = info + """\n\nClick on a reach to see info"""

            else:
                info = info + """\n\nClick on a reach to see info"""
        else:
            if sites_points is not None:
                # print(sites_points)
                feature_id = int(sites_points['properties']['nzsegment'])
                props = decode_obj(props_obj)

                if feature_id in props.nzsegment:

                    reach_data = props.sel(nzsegment=feature_id)

                    info_str = """\n\nReduction: {red}%\n\nLikelihood of observing a reduction (power): {t_stat}%""".format(red=int(reach_data.reduction), t_stat=int(reach_data.power_monitored))

                    info = info + info_str

                else:
                    info = info + """\n\nClick on a site to see info"""

            else:
                info = info + """\n\nClick on a site to see info"""


    return info


@callback(
    Output("dl_poly", "data"),
    Input("dl_btn", "n_clicks"),
    State('catch_id', 'value'),
    State('reductions_obj', 'data'),
    prevent_initial_call=True,
    )
def download_lc(n_clicks, catch_id, reductions_obj):
    # data = decode_obj(reductions_obj)
    # io1 = io.BytesIO()
    # data.to_file(io1, driver='GPKG')
    # io1.seek(0)

    if isinstance(catch_id, str) and (reductions_obj != '') and (reductions_obj is not None):
        path = rivers_catch_lc_dir.joinpath(rivers_catch_lc_gpkg_str.format(catch_id))

        return dcc.send_file(path)