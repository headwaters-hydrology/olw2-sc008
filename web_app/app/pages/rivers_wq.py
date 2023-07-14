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

import sys
if '..' not in sys.path:
    sys.path.append('..')

# from .app import app
# from . import utils

# from app import app
import utils

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


# catch_id = 3076139

###############################################
### Helper Functions


###############################################
### Initial processing

# sel1 = xr.open_dataset(base_path.joinpath(sel_data_h5), engine='h5netcdf')

with booklet.open(utils.rivers_reach_gbuf_path, 'r') as f:
    catches = [int(c) for c in f]

catches.sort()
# freqs = sel1['frequency'].values
indicators = list(utils.rivers_indicator_dict.keys())
indicators.sort()
# nzsegments = sel1['nzsegment'].values
# percent_changes = sel1['percent_change'].values
# time_periods = sel1['time_period'].values

# sel1.close()
# del sel1



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
                                dmc.AccordionControl(html.Div('(1) Catchment Selection', style={'font-size': 22})),
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
                                dmc.AccordionControl(html.Div('(2 - Optional) Customise Reductions Layer', style={'font-size': 22})),
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
                                dmc.AccordionControl(html.Div('(3) Sampling Options', style={'font-size': 22})),
                                dmc.AccordionPanel([
                                    dmc.Text('(3a) Select Indicator:'),
                                    dcc.Dropdown(options=[{'label': utils.rivers_indicator_dict[d], 'value': d} for d in indicators], id='indicator_rivers', optionHeight=40, clearable=False),
                                    dmc.Text('(3b) Select sampling length (years):', style={'margin-top': 20}),
                                    dmc.SegmentedControl(data=[{'label': d, 'value': str(d)} for d in utils.time_periods],
                                                         id='time_period',
                                                         value='5',
                                                         fullWidth=True,
                                                         color=1,
                                                         ),
                                    dmc.Text('(3c) Select sampling frequency:', style={'margin-top': 20}),
                                    dmc.SegmentedControl(data=[{'label': v, 'value': str(k)} for k, v in utils.freq_mapping.items()],
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
                                dmc.AccordionControl(html.Div('(4) Download Results', style={'font-size': 22})),
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
                            dl.Map(center=utils.center, zoom=6, children=[
                                dl.LayersControl([
                                    dl.BaseLayer(dl.TileLayer(id='tile_layer', attribution=utils.attribution), checked=True, name='OpenStreetMap'),
                                    dl.BaseLayer(dl.TileLayer(url='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', id='opentopo', attribution='Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap (CC-BY-SA)'), checked=False, name='OpenTopoMap'),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(url=str(utils.rivers_catch_pbf_path), format="geobuf", id='catch_map', zoomToBoundsOnClick=True, zoomToBounds=False, options=dict(style=utils.catch_style_handle))), name='Catchments', checked=True),
                                    # dl.GeoJSON(url='', format="geobuf", id='base_reach_map', options=dict(style=base_reaches_style_handle)),

                                    # dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='sites_points', options=dict(pointToLayer=sites_points_handle), hideout={'circleOptions': dict(fillOpacity=1, stroke=False, radius=5, color='black')})), name='Monitoring sites', checked=True),
                                    # dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='reductions_poly')), name='Land use reductions', checked=False),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='reach_map', options={}, hideout={}, hoverStyle=arrow_function(dict(weight=10, color='black', dashArray='')))), name='Rivers', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='sites_points', options=dict(pointToLayer=utils.sites_points_handle), hideout=utils.rivers_points_hideout)), name='Monitoring sites', checked=True),
                                    ]),
                                utils.colorbar_power,
                                # html.Div(id='colorbar', children=colorbar_base),
                                # dmc.Group(id='colorbar', children=colorbar_base),
                                utils.info
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
        catch_id = feature['id']

    return catch_id


@callback(
        Output('reach_map', 'data'),
        Input('catch_id', 'value'),
        )
# @cache.memoize()
def update_reaches(catch_id):
    if (catch_id is not None):
        with booklet.open(utils.rivers_reach_gbuf_path, 'r') as f:
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
        with booklet.open(utils.rivers_sites_path, 'r') as f:
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
        options = dict(style=utils.base_reach_style_handle)
    else:
        options = dict(style=utils.reach_style_handle)

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
        with booklet.open(utils.rivers_lc_clean_path, 'r') as f:
            data = utils.encode_obj(f[int(catch_id)])

    return data


@callback(
    Output("dl_poly", "data"),
    Input("dl_btn", "n_clicks"),
    State('catch_id', 'value'),
    prevent_initial_call=True,
    )
def download_lc(n_clicks, catch_id):
    if isinstance(catch_id, str):
        path = utils.rivers_catch_lc_dir.joinpath(utils.rivers_catch_lc_gpkg_str.format(catch_id))

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
            data = utils.parse_gis_file(contents, filename)

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
            red1 = xr.open_dataset(utils.rivers_reductions_model_path)

            with booklet.open(utils.rivers_reach_mapping_path) as f:
                branches = f[int(catch_id)][int(catch_id)]

            base_props = red1.sel(nzsegment=branches)

            new_reductions = utils.decode_obj(reductions_obj)
            base_reductions = utils.decode_obj(base_reductions_obj)

            new_props = utils.calc_river_reach_reductions(catch_id, new_reductions, base_reductions)
            new_props1 = new_props.combine_first(base_props).sortby('nzsegment')
            data = utils.encode_obj(new_props1)
            text_out = 'Routing complete'
        else:
            data = ''
            text_out = 'Not all inputs have been selected'
    else:
        if isinstance(catch_id, str):
            red1 = xr.open_dataset(utils.rivers_reductions_model_path)

            with booklet.open(utils.rivers_reach_mapping_path) as f:
                branches = f[int(catch_id)][int(catch_id)]

            base_props = red1.sel(nzsegment=branches).sortby('nzsegment')

            data = utils.encode_obj(base_props)
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

        ind_name = utils.rivers_indicator_dict[indicator]

        ## Modelled
        props = utils.decode_obj(reaches_obj)[[ind_name]].sel(reduction_perc=prop_red, drop=True).rename({ind_name: 'reduction'})

        n_samples = int(n_samples_year)*int(n_years)

        power_data = xr.open_dataset(utils.river_power_model_path, engine='h5netcdf')

        with booklet.open(utils.rivers_reach_mapping_path) as f:
            branches = f[int(catch_id)][int(catch_id)]

        power_data1 = power_data.sel(indicator=indicator, nzsegment=branches, n_samples=n_samples, drop=True).load().sortby('nzsegment').copy()
        power_data.close()
        del power_data

        conc_perc = 100 - props.reduction

        new_powers = props.assign(power_modelled=(('nzsegment'), power_data1.sel(conc_perc=conc_perc).power.values.astype('int8')))
        new_powers['nzsegment'] = new_powers['nzsegment'].astype('int32')
        new_powers['reduction'] = new_powers['reduction'].astype('int8')

        ## Monitored
        power_data = xr.open_dataset(utils.river_power_moni_path, engine='h5netcdf')
        sites = power_data.nzsegment.values[power_data.nzsegment.isin(branches)].astype('int32')
        sites.sort()
        if len(sites) > 0:
            conc_perc1 = conc_perc.sel(nzsegment=sites)
            power_data1 = power_data.sel(indicator=indicator, nzsegment=sites, n_samples=n_samples, drop=True).load().sortby('nzsegment').copy()
            power_data1 = power_data1.rename({'power': 'power_monitored'})
            power_data.close()
            del power_data

            power_data2 = power_data1.sel(conc_perc=conc_perc1).drop('conc_perc')

            new_powers = utils.xr_concat([new_powers, power_data2])
        else:
            new_powers = new_powers.assign(power_monitored=(('nzsegment'), xr.full_like(new_powers.reduction, np.nan, dtype='float32').values))

        data = utils.encode_obj(new_powers)
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
        props = utils.decode_obj(powers_obj)

        ## Modelled
        color_arr = pd.cut(props.power_modelled.values, utils.bins, labels=utils.colorscale, right=False).tolist()

        hideout_model = {'colorscale': color_arr, 'classes': props.nzsegment.values.tolist(), 'style': utils.style, 'colorProp': 'nzsegment'}

        ## Monitored
        props_moni = props.dropna('nzsegment')
        if len(props_moni.nzsegment) > 0:
            # print(props_moni)
            color_arr2 = pd.cut(props_moni.power_monitored.values, utils.bins, labels=utils.colorscale, right=False).tolist()

            hideout_moni = {'classes': props_moni.nzsegment.values.astype(int), 'colorscale': color_arr2, 'circleOptions': dict(fillOpacity=1, stroke=True, color='black', weight=1, radius=utils.site_point_radius), 'colorProp': 'nzsegment'}

            # hideout_moni = {'colorscale': color_arr2, 'classes': props_moni.nzsegment.values.astype(int).tolist(), 'style': style, 'colorProp': 'nzsegment'}
        else:
            hideout_moni = utils.rivers_points_hideout
    else:
        hideout_model = {}
        hideout_moni = utils.rivers_points_hideout

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

        props = utils.decode_obj(powers_obj)

        if trig == 'reach_map':
            # print(reach_feature)
            feature_id = int(reach_feature['id'])

            if feature_id in props.nzsegment:

                reach_data = props.sel(nzsegment=feature_id)

                info_str = """\n\nnzsegment: {seg}\n\nReduction: {red}%\n\nLikelihood of observing a reduction (power): {t_stat}%""".format(red=int(reach_data.reduction), t_stat=int(reach_data.power_modelled), seg=feature_id)

                info = info + info_str

            else:
                info = info + """\n\nClick on a reach to see info"""
        elif trig == 'sites_points':
            feature_id = int(sites_feature['properties']['nzsegment'])
            # print(sites_feature)

            if feature_id in props.nzsegment:

                reach_data = props.sel(nzsegment=feature_id)

                info_str = """\n\nnzsegment: {seg}\n\nSite name: {site}\n\nReduction: {red}%\n\nLikelihood of observing a reduction (power): {t_stat}%""".format(red=int(reach_data.reduction), t_stat=int(reach_data.power_monitored), seg=feature_id, site=sites_feature['id'])

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
        power_data = utils.decode_obj(powers_obj)

        df1 = power_data.to_dataframe().reset_index()
        df1['indicator'] = utils.rivers_indicator_dict[indicator]
        df1['n_years'] = n_years
        df1['n_samples_per_year'] = n_samples_year

        df2 = df1.set_index(['indicator', 'n_years', 'n_samples_per_year', 'nzsegment']).sort_index()

        return dcc.send_data_frame(df2.to_csv, f"river_power_{catch_id}.csv")



