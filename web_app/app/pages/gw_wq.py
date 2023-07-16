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
    path='/gw-wq',
    title='Groundwater Quality',
    name='gw_wq',
    description='Groundwater Quality'
)




###############################################
### Helper Functions


###############################################
### Initial processing

# sel1 = xr.open_dataset(base_path.joinpath(sel_data_h5), engine='h5netcdf')

# with booklet.open(gw_catches_major_path, 'r') as f:
#     lakes = list(f.keys())

# lakes.sort()

with booklet.open(utils.gw_points_rc_blt, 'r') as f:
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

indicators = [{'value': k, 'label': v} for k, v in utils.gw_indicator_dict.items()]

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
                                dmc.AccordionControl(html.Div('(1) Select a Regional Council', style={'font-size': 22})),
                                dmc.AccordionPanel([

                                    html.Label('(1a) Select a Regional Council on the map:'),
                                    dcc.Dropdown(options=[{'label': d, 'value': d} for d in rcs], id='rc_id', optionHeight=40, clearable=False,
                                                  style={'margin-top': 10}
                                                  ),
                                    ]
                                    )
                                ],
                                value='1'
                                ),

                            dmc.AccordionItem([
                                dmc.AccordionControl(html.Div('(2) Select a reduction', style={'font-size': 22})),
                                dmc.AccordionPanel([
                                    html.Label('(2a) Select a reduction:'),
                                    dmc.Slider(id='reductions_slider_gw',
                                               value=25,
                                               mb=35,
                                               step=5,
                                               min=5,
                                               max=50,
                                               showLabelOnHover=True,
                                               disabled=False,
                                               marks=utils.gw_reductions_options
                                               ),
                                    # dcc.Dropdown(options=utils.gw_reductions_options, id='reductions_gw', optionHeight=40, clearable=False,
                                    #               style={'margin-top': 10}
                                    #               ),
                                    ]
                                    )
                                ],
                                value='2'
                                ),

                            dmc.AccordionItem([
                                dmc.AccordionControl(html.Div('(3) Sampling Options', style={'font-size': 22})),
                                dmc.AccordionPanel([
                                    dmc.Text('(3a) Select Indicator:'),
                                    dcc.Dropdown(options=indicators, id='indicator_gw', optionHeight=40, clearable=False),
                                    dmc.Text('(3b) Select sampling length (years):', style={'margin-top': 20}),
                                    dmc.SegmentedControl(data=[{'label': d, 'value': str(d)} for d in utils.time_periods],
                                                         id='time_period_gw',
                                                         value='5',
                                                         fullWidth=True,
                                                         color=1,
                                                         ),
                                    dmc.Text('(3c) Select sampling frequency:', style={'margin-top': 20}),
                                    dmc.SegmentedControl(data=[{'label': v, 'value': str(k)} for k, v in utils.freq_mapping.items()],
                                                         id='freq_gw',
                                                         value='12',
                                                         fullWidth=True,
                                                         color=1
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
                                    children=[html.Div(dmc.Button("Download power results", id='dl_btn_power_gw'), style={'margin-bottom': 20, 'margin-top': 10}),
                            dcc.Download(id="dl_power_gw")],
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
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(url=str(utils.rc_bounds_gbuf), format="geobuf", id='rc_map', zoomToBoundsOnClick=True, options=dict(style=utils.rc_style_handle),  hideout={})), name='Regional Councils', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='gw_points', zoomToBounds=True, zoomToBoundsOnClick=True, cluster=False, options=dict(pointToLayer=utils.gw_points_style_handle), hideout=utils.gw_points_hideout)), name='GW wells', checked=True),
                                    # dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='sites_points_gw', options=dict(pointToLayer=utils.sites_points_handle), hideout=utils.rivers_points_hideout)), name='Monitoring sites', checked=True),
                                    ], id='layers_gw'),
                                utils.colorbar_power,
                                # html.Div(id='colorbar', children=colorbar_base),
                                # dmc.Group(id='colorbar', children=colorbar_base),
                                dcc.Markdown(id="info_gw", className="info", style={"position": "absolute", "top": "10px", "right": "160px", "z-index": "1000"})
                                                ], style={'width': '100%', 'height': 700, 'margin': "auto", "display": "block"}, id="map2"),

                            ],
                            ),
                        ),
                    ]
                    ),
            dcc.Store(id='powers_obj_gw', data=None),
            dcc.Store(id='gw_points_ids', data=None),
            ]
        )

    return layout


# def layout():
#     layout = html.Div(children=[
#         # html.Div([html.H2('Groundwater Quality')]),
#         html.Div([
#             html.H3('(1) Reductions'),

#             html.Label('Select a Regional Council on the map:'),
#             dcc.Dropdown(options=[{'label': d, 'value': d} for d in rcs], id='rc_id', optionHeight=40, clearable=False, style={'margin-bottom': 20}),

#             # dcc.Upload(
#             #     id='upload_data_gw',
#             #     children=html.Button('Upload reductions polygons gpkg'),
#             #     style={
#             #         'width': '100%',
#             #         'height': '60px',
#             #         'textAlign': 'center',
#             #         'margin-top': 40
#             #     },
#             #     multiple=False
#             # ),
#             # dcc.Markdown('''##### **Or**''', style={
#             #     'textAlign': 'center',
#             #                 }),
#             # html.Button('Use land cover for reductions', id='demo_data_gw',
#             #             style={
#             #                 'width': '100%',
#             #                 'height': '50%',
#             #                 'textAlign': 'center',
#             #                 'margin-top': 20
#             #             }),

#             html.Label('Select a reduction %:'),
#             dcc.Dropdown(options=reductions_options, id='gw_reductions', optionHeight=40, clearable=False),

#         #     html.Label('Select a reductions column in the GIS file:', style={'margin-top': 20}),
#         #     dcc.Dropdown(options=[], id='col_name_gw', optionHeight=40, clearable=False),
#         #     dcc.Loading(
#         #     id="loading-2",
#         #     type="default",
#         #     children=html.Div([html.Button('Process reductions', id='process_gw', n_clicks=0),
#         #                        html.Div(id='process_text_gw')],
#         #                       style={'margin-top': 20, 'margin-bottom': 100}
#         #                       )
#         # ),

#             # html.Label('Select Indicator:'),
#             # dcc.Dropdown(options=[{'label': d, 'value': d} for d in indicators], id='indicator', optionHeight=40, clearable=False, value='NH4'),
#             # html.Label('Select expected percent improvement:'),
#             # dcc.Dropdown(options=[{'label': d, 'value': d} for d in percent_changes], id='percent_change', clearable=False, value=10),

#             # dcc.Link(html.Img(src=str(app_base_path.joinpath('our-land-and-water-logo.svg'))), href='https://ourlandandwater.nz/')
#             ], className='two columns', style={'margin': 10}),

#     html.Div([
#         html.H3('(2) Sampling options'),
#         html.Label('Select Indicator:'),
#         dcc.Dropdown(options=indicators, id='gw_indicator', optionHeight=40, clearable=False, value='Nitrate'),
#         html.Label('Select sampling lehtml.Label('(3d) Change the percent of the reductions applied (100% is the max realistic reduction):', style={'margin-top': 20}),
                                    # dmc.Slider(id='reductions_slider_gw',
                                    #            value=100,
                                    #            mb=35,
                                    #            step=10,
                                    #            # min=10,
                                    #            showLabelOnHover=True,
                                    #            disabled=False,
                                    #            marks=[{'label': str(d) + '%', 'value': d} for d in range(0, 101, 20)]
                                    #            ),ngth (years):', style={'margin-top': 20}),
#         dcc.Dropdown(options=[{'label': d, 'value': d} for d in time_periods], id='gw_time_period', clearable=False, value=5),
#         html.Label('Select sampling frequency:'),
#         dcc.Dropdown(options=[{'label': v, 'value': k} for k, v in freq_mapping.items()], id='gw_freq', clearable=False, value=12, style={'margin-bottom': 460}),

#         # html.H4(children='Map Layers', style={'margin-top': 20}),
#         # dcc.Checklist(
#         #        options=[
#         #            {'label': 'Reductions polygons', 'value': 'reductions_poly'},
#         #             # {'label': 'Lake polygon', 'value': 'gw_poly'},
#         #            {'label': 'River reaches', 'value': 'reach_map'}
#         #        ],
#         #        value=['reductions_poly', 'reach_map'],
#         #        id='map_checkboxes_gw',
#         #        style={'padding': 5, 'margin-bottom': 330}
#         #     ),
#         # dcc.Link(html.Img(src=str(app_base_path.joinpath('our-land-and-water-logo.svg'))), href='https://ourlandandwater.nz/')
#         ], className='two columns', style={'margin': 10}),

#     html.Div([
#         dl.Map(center=center, zoom=7, children=[
#             dl.TileLayer(id='gw_tile_layer', attribution=attribution),
#             dl.GeoJSON(url=str(rc_bounds_gbuf), format="geobuf", id='rc_map', zoomToBoundsOnClick=True, zoomToBounds=True, options=dict(style=rc_style_handle),  hideout={}),
#             dl.GeoJSON(data='', format="geobuf", id='gw_points', zoomToBounds=True, zoomToBoundsOnClick=True, cluster=False, options=dict(pointToLayer=gw_points_style_handle), hideout=gw_points_hideout),
#             # dl.GeoJSON(data='', format="geobuf", id='catch_map_gw', zoomToBoundsOnClick=True, options=dict(style=catch_style)),
#             # dl.GeoJSON(url='', format="geobuf", id='base_reach_map', options=dict(style=base_reaches_style_handle)),
#             # dl.GeoJSON(data='', format="geobuf", id='reach_map_gw', options=dict(style=reach_style), hideout={}),
#             # dl.GeoJSON(data='', format="geobuf", id='gw_poly', options=dict(style=gw_style_handle), hideout={'classes': [''], 'colorscale': ['#808080'], 'style': gw_style, 'colorProp': 'ref'}),
#             # dl.GeoJSON(data='', format="geobuf", id='reductions_poly_gw'),
#             colorbar,
#             info
#                             ], style={'width': '100%', 'height': 700, 'margin': "auto", "display": "block"})
#     ], className='five columns', style={'margin': 10}),

#     # html.Div([
#     #     dcc.Loading(
#     #             id="loading-tabs",
#     #             type="default",
#     #             children=[dcc.Tabs(id='plot_tabs', value='info_tab', style=tabs_styles, children=[
#     #                         dcc.Tab(label='Info', value='info_tab', style=tab_style, selected_style=tab_selected_style),
#     #                         # dcc.Tab(label='Habitat Suitability', value='hs_tab', style=tab_style, selected_style=tab_selected_style),
#     #                         ]
#     #                     ),
#     #                 html.Div(id='plots')
#     #                 ]
#     #             ),

#     # ], className='three columns', style={'margin': 10}),

#     dcc.Store(id='gw_props_obj', data=None),
#     dcc.Store(id='gw_points_ids', data=None),
#     # dcc.Store(id='reaches_obj_gw', data=''),
#     # dcc.Store(id='reductions_obj_gw', data=''),
# ], style={'margin':0})

#     return layout

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
        if not feature['properties']['cluster']:
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
        with booklet.open(utils.gw_points_rc_blt, 'r') as f:
            data0 = f[rc_id]
            geo1 = geobuf.decode(data0)
            gw_points = [s['id'] for s in geo1['features']]
            gw_points_encode = utils.encode_obj(gw_points)
            data = base64.b64encode(data0).decode()
    else:
        data = None
        gw_points_encode = None

    return data, gw_points_encode


@callback(
    Output('powers_obj_gw', 'data'),
    [Input('reductions_slider_gw', 'value'), Input('indicator_gw', 'value'), Input('time_period_gw', 'value'), Input('freq_gw', 'value')],
    # [State('gw_id', 'value')]
    )
def update_props_data_gw(reductions, indicator, n_years, n_samples_year):
    """

    """
    if isinstance(reductions, (str, int)) and isinstance(n_years, str) and isinstance(n_samples_year, str) and isinstance(indicator, str):
        n_samples = int(n_samples_year)*int(n_years)

        power_data = xr.open_dataset(utils.gw_error_path, engine='h5netcdf')
        power_data1 = power_data.sel(indicator=indicator, n_samples=n_samples, conc_perc=100-int(reductions), drop=True).to_dataframe().reset_index()
        power_data.close()
        del power_data

        data = utils.encode_obj(power_data1)
        return data
    else:
        raise dash.exceptions.PreventUpdate


@callback(
    Output('gw_points', 'hideout'),
    Input('powers_obj_gw', 'data'),
    Input('gw_points_ids', 'data'),
    prevent_initial_call=True
    )
def update_hideout_gw_points(powers_obj, gw_points_encode):
    """

    """
    if (powers_obj != '') and (powers_obj is not None):

        # print('trigger')
        props = utils.decode_obj(powers_obj)
        # print(props)
        # print(type(gw_id))

        color_arr = pd.cut(props.power.values, utils.bins, labels=utils.colorscale, right=False).tolist()
        # print(color_arr)
        # print(props['gw_id'])

        hideout = {'classes': props['ref'].values, 'colorscale': color_arr, 'circleOptions': dict(fillOpacity=1, stroke=False, radius=utils.site_point_radius), 'colorProp': 'tooltip'}
    elif (gw_points_encode is not None):
        # print('trigger')
        gw_refs = utils.decode_obj(gw_points_encode)

        hideout = {'classes': gw_refs, 'colorscale': ['#808080'] * len(gw_refs), 'circleOptions': dict(fillOpacity=1, stroke=False, radius=utils.site_point_radius), 'colorProp': 'tooltip'}
    else:
        hideout = utils.gw_points_hideout

    return hideout


@callback(
    Output("info_gw", "children"),
    [Input('powers_obj_gw', 'data'),
      Input('reductions_slider_gw', 'value'),
      Input("gw_points", "click_feature")],
    State('gw_points_ids', 'data')
    )
def update_map_info_gw(powers_obj, reductions, feature, gw_points_encode):
    """

    """
    info = """"""
    info_str = """\n\n**Reduction**: {red}%\n\n**Likelihood of observing a reduction (power)**: {t_stat}%\n\n**Well Depth (m)**: {depth:.1f}"""

    # if (reductions_obj != '') and (reductions_obj is not None) and ('reductions_poly' in map_checkboxes):
    #     info = info + """\n\nHover over the polygons to see reduction %"""

    if isinstance(reductions, int) and (powers_obj != '') and (powers_obj is not None):
        if feature is not None:
            # print(feature)
            gw_refs = utils.decode_obj(gw_points_encode)
            if feature['id'] in gw_refs:
                props = utils.decode_obj(powers_obj)

                info2 = info_str.format(red=int(reductions), t_stat=int(props[props.ref==feature['id']].iloc[0]['power']), depth=feature['properties']['bore_depth'])

                info = info + info2

        else:
            info = info + """\n\nClick on a well to see info"""

    return info


@callback(
    Output("dl_power_gw", "data"),
    Input("dl_btn_power_gw", "n_clicks"),
    State('powers_obj_gw', 'data'),
    State('rc_id', 'value'),
    State('indicator_gw', 'value'),
    State('time_period_gw', 'value'),
    State('freq_gw', 'value'),
    State('reductions_slider_gw', 'value'),
    prevent_initial_call=True,
    )
def download_power(n_clicks, powers_obj, rc_id, indicator, n_years, n_samples_year, reductions):

    if (powers_obj != '') and (powers_obj is not None):
        df1 = utils.decode_obj(powers_obj)

        df1['power'] = df1['power'].astype(int)

        df1['indicator'] = utils.gw_indicator_dict[indicator]
        df1['n_years'] = n_years
        df1['n_samples_per_year'] = n_samples_year
        df1['reduction'] = reductions

        df2 = df1.rename(columns={'ref': 'site_id'}).set_index(['indicator', 'n_years', 'n_samples_per_year', 'reduction', 'site_id']).sort_index()

        return dcc.send_data_frame(df2.to_csv, f"gw_power_{rc_id}.csv")





# with shelflet.open(gw_lc_path, 'r') as f:
#     plan_file = f[str(gw_id)]
