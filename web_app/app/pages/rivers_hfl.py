#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 13:37:46 2022

@author: mike
"""
import xarray as xr
import dash
from dash import dcc, html, callback, ctx
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import dash_leaflet as dl
import dash_leaflet.express as dlx
from dash_extensions.javascript import assign, arrow_function
import pandas as pd
import numpy as np
import base64
import booklet
import geobuf
from time import sleep

# from .utils import parameters as param
# from . import utils

# from app import app
# import utils

from utils import parameters as param
from utils import components as gc
from utils import utils

##########################################
### Parameters

dash.register_page(
    __name__,
    path='/rivers-hfl',
    title='High Res Flow',
    name='rivers_hfl',
    description='River High Res Flow'
)

### Handles
catch_style_handle = assign("""function style(feature) {
    return {
        fillColor: 'grey',
        weight: 2,
        opacity: 1,
        color: 'black',
        fillOpacity: 0.1
    };
}""", name='rivers_catch_style_handle_hfl')

base_reach_style_handle = assign("""function style3(feature) {
    return {
        weight: 2,
        opacity: 0.75,
        color: 'grey',
    };
}""", name='rivers_base_reach_style_handle_hfl')

reach_style_handle = assign("""function style2(feature, context){
    const {classes, colorscale, style, colorProp} = context.props.hideout;  // get props from hideout
    const value = feature.properties[colorProp];  // get value the determines the color
    for (let i = 0; i < classes.length; ++i) {
        if (value == classes[i]) {
            style.color = colorscale[i];  // set the fill color according to the class
        }
    }
    return style;
}""", name='rivers_reach_style_handle_hfl')

sites_points_handle = assign("""function rivers_sites_points_handle(feature, latlng, context){
    const {classes, colorscale, circleOptions, colorProp} = context.props.hideout;  // get props from hideout
    const value = feature.properties[colorProp];  // get value the determines the fillColor
    for (let i = 0; i < classes.length; ++i) {
        if (value == classes[i]) {
            circleOptions.fillColor = colorscale[i];  // set the color according to the class
        }
    }

    return L.circleMarker(latlng, circleOptions);
}""", name='rivers_sites_points_handle_hfl')

draw_marae = assign("""function(feature, latlng){
const flag = L.icon({iconUrl: '/assets/nzta-marae.svg', iconSize: [20, 30]});
return L.marker(latlng, {icon: flag});
}""", name='rivers_marae_handle_hfl')

# reach_hover_style = arrow_function(dict(weight=10, color='black', dashArray=''))
reach_hover_style = None

# catch_id = 3076139

###############################################
### Initial processing

with booklet.open(param.rivers_reach_gbuf_path, 'r') as f:
    catches = [int(c) for c in f]

catches.sort()
indicators = list(param.hfl_indicator_dict.keys())
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
                        children=dmc.Accordion(
                            value="1",
                            chevronPosition='left',
                            children=[
                            dmc.AccordionItem([
                                # html.H5('(1) Catchment selection', style={'font-weight': 'bold'}),
                                dmc.AccordionControl('(1) Catchment Selection', style={'font-size': 18}),
                                dmc.AccordionPanel([

                                    html.Label('(1a) Select a catchment on the map:'),
                                    dmc.Text(id='catch_name_hfl', weight=700, style={'margin-top': 10}),
                                    # dcc.Dropdown(options=[{'label': d, 'value': d} for d in catches], id='catch_id', optionHeight=40, clearable=False,
                                    #               style={'margin-top': 10}
                                    #               ),
                                    ]
                                    )
                                ],
                                value='1'
                                ),

                            dmc.AccordionItem([
                                dmc.AccordionControl('(2) Select Indicator', style={'font-size': 18}),
                                dmc.AccordionPanel([
                                    dmc.Text('(2a) Select Indicator:'),
                                    dcc.Dropdown(options=[{'label': param.hfl_indicator_dict[d], 'value': d} for d in indicators], id='indicator_hfl', optionHeight=40, clearable=False, style={'margin-bottom': 20}),
                                    ]
                                    )
                                ],
                                value='2'
                                ),

                            # dmc.AccordionItem([
                            #     # html.H5('Optional (2) Customise Reductions Layer', style={'font-weight': 'bold', 'margin-top': 20}),
                            #     dmc.AccordionControl('(2 - Optional) Customise Improvements Layer', style={'font-size': 18}),
                            #     dmc.AccordionPanel([
                            #         html.Label('(2a) Download improvements polygons as GPKG:'),
                            #         dcc.Loading(
                            #         id="loading-2",
                            #         type="default",
                            #         children=[dmc.Anchor(dmc.Button('Download land cover'), href='', id='dl_poly', style={'margin-top': 10})],
                            #         ),
                            #         html.Label('NOTE: Only modify existing values. Do not add additional columns; they will be ignored.', style={
                            #             'margin-top': 10
                            #         }
                            #             ),
                            #         html.Label('(2b) Upload modified improvements polygons as GPKG:', style={
                            #             'margin-top': 20
                            #         }
                            #             ),
                            #         dcc.Loading(
                            #             children=[
                            #                 dcc.Upload(
                            #                     id='upload_data_rivers',
                            #                     children=dmc.Button('Upload reductions',
                            #                                          # className="me-1"
                            #                                           # style={
                            #                                           #     'width': '50%',
                            #                                           #     }
                            #                     ),
                            #                     style={
                            #                         'margin-top': 10
                            #                     },
                            #                     multiple=False
                            #                 ),
                            #                 ]
                            #             ),
                            #         dcc.Markdown('', style={
                            #             'margin-top': 10,
                            #             'textAlign': 'left',
                            #                         }, id='upload_error_text'),
                            #         html.Label('(2c) Process the improvements layer and route the improvements downstream:', style={
                            #             'margin-top': 20
                            #         }
                            #             ),
                            #         dcc.Loading(
                            #         id="loading-1",
                            #         type="default",
                            #         children=html.Div([dmc.Button('Process reductions', id='process_reductions_rivers',
                            #                                       # className="me-1",
                            #                                       n_clicks=0),
                            #                             html.Div(id='process_text', style={'margin-top': 10})],
                            #                           style={'margin-top': 10, 'margin-bottom': 10}
                            #                           )
                            #         ),
                            #         ]
                            #         )
                            #     ],
                            #     value='2'
                            #     ),

                            # dmc.AccordionItem([
                            #     dmc.AccordionControl('(3) Query Options', style={'font-size': 18}),
                            #     dmc.AccordionPanel([
                            #         dmc.Text('(3a) Select Indicator:'),
                            #         dcc.Dropdown(options=[{'label': param.rivers_indicator_dict[d], 'value': d} for d in indicators], id='indicator_rivers', optionHeight=40, clearable=False),
                            #         dmc.Text('(3b) Select sampling length (years):', style={'margin-top': 20}),
                            #         dmc.SegmentedControl(data=[{'label': d, 'value': str(d)} for d in param.rivers_time_periods],
                            #                              id='time_period',
                            #                              value='5',
                            #                              fullWidth=True,
                            #                              color=1,
                            #                              ),
                            #         dmc.Text('(3c) Select sampling frequency:', style={'margin-top': 20}),
                            #         dmc.SegmentedControl(data=[{'label': v, 'value': str(k)} for k, v in param.rivers_freq_mapping.items()],
                            #                              id='freq',
                            #                              value='12',
                            #                              fullWidth=True,
                            #                              color=1
                            #                              ),
                            #         html.Label('(3d) Change the percent of the improvements applied (100% is the max realistic improvement):', style={'margin-top': 20}),
                            #         dmc.Slider(id='Reductions_slider',
                            #                    value=100,
                            #                    mb=35,
                            #                    step=10,
                            #                    # min=10,
                            #                    showLabelOnHover=True,
                            #                    disabled=False,
                            #                    marks=param.marks
                            #                    ),
                            #         # dcc.Dropdown(options=[{'label': d, 'value': d} for d in time_periods], id='time_period', clearable=False, value=5),
                            #         # html.Label('Select sampling frequency:'),
                            #         # dcc.Dropdown(options=[{'label': v, 'value': k} for k, v in freq_mapping.items()], id='freq', clearable=False, value=12),

                            #         ],
                            #         )
                            #     ],
                            #     value='3'
                            #     ),

                            # dmc.AccordionItem([
                            #     dmc.AccordionControl('(4) Download Results', style={'font-size': 18}),
                            #     dmc.AccordionPanel([
                            #         dmc.Text('(4a) Download power results given the prior query options (csv):'),
                            #         dcc.Loading(
                            #         type="default",
                            #         children=[html.Div(dmc.Button("Download power results", id='dl_btn_power_rivers'), style={'margin-bottom': 20, 'margin-top': 10}),
                            # dcc.Download(id="dl_power_rivers")],
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
                            )
                        ),
                    dmc.Col(
                        span=4,
                        # style={
                        #     'margin-top': 20
                        #     },
                        children=html.Div([
                            dl.Map(center=param.center, zoom=param.zoom, children=[
                                dl.LayersControl([
                                    dl.BaseLayer(dl.TileLayer(attribution=param.attribution, opacity=0.7), checked=True, name='OpenStreetMap'),
                                    dl.BaseLayer(dl.TileLayer(url='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', attribution='Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap (CC-BY-SA)', opacity=0.6), checked=False, name='OpenTopoMap'),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(url=str(param.rivers_catch_4th_pbf_path), format="geobuf", id='catch_map_hfl', zoomToBoundsOnClick=True, zoomToBounds=False, options=dict(style=catch_style_handle))), name='Catchments', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='marae_map_hfl', zoomToBoundsOnClick=False, zoomToBounds=False, options=dict(pointToLayer=draw_marae))), name='Marae', checked=False),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='reach_map_hfl', options=dict(style=base_reach_style_handle), hideout={}, hoverStyle=reach_hover_style)), name='Rivers', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='sites_points_hfl', options=dict(pointToLayer=sites_points_handle), hideout=param.rivers_points_hideout)), name='Monitoring sites', checked=True),
                                    ],
                                    id='layers_rivers_hfl'
                                    ),
                                gc.colorbar_hfl,
                                # html.Div(id='colorbar', children=colorbar_base),
                                # dmc.Group(id='colorbar', children=colorbar_base),
                                dcc.Markdown(id="info_hfl", className="info", style={"position": "absolute", "top": "10px", "right": "160px", "z-index": "1000"})
                                ],
                                style={'width': '100%', 'height': param.map_height, 'margin': "auto", "display": "block"},
                                id="map2",
                                ),

                            ],
                            # className='five columns', style={'margin': 10}
                            ),
                        ),
                    ]
                    ),
            dcc.Store(id='catch_id_hfl', data=''),
            dcc.Store(id='nzsegments_hfl', data=''),
            dcc.Store(id='nzsegments_hfl_sites', data=''),
            dcc.Store(id='perc_loads_hfl_sites', data=''),
            ]
        )

    return layout


###############################################
### Callbacks

@callback(
    Output('catch_id_hfl', 'data'),
    [Input('catch_map_hfl', 'click_feature')]
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
    Output('catch_name_hfl', 'children'),
    [Input('catch_id_hfl', 'data')]
    )
def update_catch_name(catch_id):
    """

    """
    # print(ds_id)
    if catch_id != '':
        with booklet.open(param.rivers_catch_name_path) as f:
            catch_name = f[int(catch_id)]

        return catch_name


@callback(
        Output('reach_map_hfl', 'data'),
        Input('catch_id_hfl', 'data'),
        )
def update_reaches(catch_id):
    if catch_id != '':
        with booklet.open(param.rivers_reach_gbuf_path, 'r') as f:
            data = base64.b64encode(f[int(catch_id)]).decode()

    else:
        data = ''

    return data


@callback(
        Output('sites_points_hfl', 'data'),
        Output('nzsegments_hfl_sites', 'data'),
        Input('catch_id_hfl', 'data'),
        prevent_initial_call=True
        )
def update_monitor_sites(catch_id):
    if catch_id != '':
        with booklet.open(param.rivers_sites_3rd_path, 'r') as f:
            sites = f[int(catch_id)]

        points_data = base64.b64encode(sites).decode()

        gsites = geobuf.decode(sites)
        nzsegments = [s['properties']['nzsegment'] for s in gsites['features']]

    else:
        points_data = ''
        nzsegments = ''

    return points_data, utils.encode_obj(nzsegments)


@callback(
        Output('marae_map_hfl', 'data'),
        Input('catch_id_hfl', 'data'),
        )
def update_marae(catch_id):
    if catch_id != '':
        with booklet.open(param.rivers_marae_path, 'r') as f:
            data = base64.b64encode(f[int(catch_id)]).decode()

    else:
        data = ''

    return data


# @callback(
#         Output('reach_map_hfl', 'options'),
#         Input('reach_map_hfl', 'hideout'),
#         Input('catch_id_hfl', 'data')
#         )
# def update_reaches_option(hideout, catch_id):
#     trig = ctx.triggered_id

#     if (len(hideout) == 0) or (trig == 'catch_id'):
#         options = dict(style=base_reach_style_handle)
#     else:
#         options = dict(style=reach_style_handle)

#     return options


@callback(
    Output('nzsegments_hfl', 'data'),
    Input('catch_id_hfl', 'data'),
    prevent_initial_call=True)
def update_reach_reductions(catch_id):
    """

    """
    data = ''

    if catch_id != '':
        with booklet.open(param.rivers_reach_mapping_path) as f:
            nzsegments = f[int(catch_id)][int(catch_id)]

        data = utils.encode_obj(nzsegments)

    return data


@callback(
    Output('reach_map_hfl', 'hideout'),
    Output('reach_map_hfl', 'options'),
    Input('indicator_hfl', 'value'),
    Input('nzsegments_hfl', 'data'),
    prevent_initial_call=True
    )
def update_reaches_hideout(indicator, nzsegments_obj):
    """

    """
    hideout_model = {}
    options = dict(style=base_reach_style_handle)

    if (nzsegments_obj != '') and isinstance(indicator, str):
        nzsegments = utils.decode_obj(nzsegments_obj)

        f = xr.open_dataset(param.rivers_high_loads_reaches_path, engine='h5netcdf')

        sites_bool = np.isin(nzsegments, f.nzsegment)

        if sites_bool.any():
            exist_sites = nzsegments[sites_bool]
            other_sites = nzsegments[~sites_bool]

            data = f[indicator].sel(nzsegment=exist_sites, drop=True)
            values = data.values

            # print(np.isnan(values).sum())

            color_arr2 = pd.cut(values, param.bins, labels=param.hfl_colorscale, right=False).tolist()
            color_arr2 = [color if isinstance(color, str) else '#808080' for color in color_arr2]

            combo_sites = np.concatenate((exist_sites, other_sites))

            # print(len(combo_sites))

            color_arr3 = color_arr2 + ['#808080'] * len(other_sites)

            hideout_model = {'colorscale': color_arr3, 'classes': combo_sites, 'style': param.style_power, 'colorProp': 'nzsegment'}
            options = dict(style=reach_style_handle)

    return hideout_model, options


@callback(
    Output('sites_points_hfl', 'hideout'),
    Output('perc_loads_hfl_sites', 'data'),
    Input('nzsegments_hfl_sites', 'data'),
    Input('indicator_hfl', 'value'),
    State('catch_id_hfl', 'data'),
    prevent_initial_call=True
    )
def update_sites_hideout(nzsegments_obj, indicator, catch_id):
    """

    """
    sleep(0.3) # Slow down the callback to ensure the sites get rendered on top
    perc_loads_obj = ''
    if (catch_id != '') and (nzsegments_obj != '') and isinstance(indicator, str):
        nzsegments = utils.decode_obj(nzsegments_obj)

        nzsegments = np.array(nzsegments)

        f = xr.open_dataset(param.rivers_perc_load_above_90_flow_moni_path, engine='h5netcdf')
        sites_bool = np.isin(nzsegments, f.nzsegment)

        ## Monitored
        if sites_bool.any():
            new_sites = nzsegments[sites_bool]
            other_sites = nzsegments[~sites_bool]
            data = f.sel(indicator=indicator, nzsegment=new_sites, drop=True)

            values = data.perc_load_above_90_flow.values

            # print(props_moni)
            color_arr2 = pd.cut(values, param.bins, labels=param.hfl_colorscale, right=False).tolist()

            combo_sites = np.concatenate((new_sites, other_sites))

            color_arr3 = color_arr2 + ['#252525'] * len(other_sites)

            hideout_moni = {'classes': combo_sites, 'colorscale': color_arr3, 'circleOptions': dict(fillOpacity=1, stroke=True, color='black', weight=1, radius=param.site_point_radius), 'colorProp': 'nzsegment'}

            perc_loads_obj = utils.encode_obj(data.to_dataframe())

        else:
            hideout_moni = param.rivers_points_hideout
    else:
        hideout_moni = param.rivers_points_hideout

    return hideout_moni, perc_loads_obj


@callback(
    Output("info_hfl", "children"),
    Input("reach_map_hfl", "click_feature"),
    Input('perc_loads_hfl_sites', 'data'),
    Input('sites_points_hfl', 'click_feature'),
    Input('catch_id_hfl', 'data'),
    prevent_initial_call=True
    )
def update_map_info(reach_feature, perc_loads_obj, sites_feature, catch_id):
    """

    """
    info = """##### Percent load above 90th percentile flow"""

    trig = ctx.triggered_id

    if trig == 'catch_id_hfl':
        pass

    elif trig == 'reach_map_hfl':
        # feature_id = int(reach_feature['id'])
        # info = """\n\n**nzsegment**: {seg}\n\n**Percent load above flow 90th percentile**: {perc_load}""".format(perc_load, seg=feature_id)
        pass

    elif (sites_feature is not None):
        feature_id = int(sites_feature['properties']['nzsegment'])
        site_id = sites_feature['id']
        # print(sites_feature)
        perc_load = 'No continuous flow'

        if (perc_loads_obj != '') and (perc_loads_obj is not None):
            data = utils.decode_obj(perc_loads_obj)

            # print(sites_feature)
            # print(props.nzsegment)
            # print(feature_id)

            if feature_id in data.index:
                try:
                    perc_load = str(int(data.perc_load_above_90_flow.loc[feature_id])) + '%'
                except ValueError:
                    pass

        info = """##### Monitoring Site:

            \n\n**Site name**: {site}\n\n**Percent load above 90th percentile flow**: {perc_load}""".format(perc_load=perc_load, site=site_id)

    return info




# @callback(
#     Output("info_hfl", "children"),
#     [Input('powers_obj', 'data'),
#       # Input('reductions_obj', 'data'),
#       # Input('map_checkboxes_rivers', 'value'),
#       Input("reach_map", "click_feature"),
#       Input('sites_points', 'click_feature')],
#     State("info", "children")
#     )
# def update_map_info(powers_obj, reach_feature, sites_feature, old_info):
#     """

#     """
#     # info = """###### Likelihood of observing a reduction (%)"""
#     info = """"""

#     # if (reductions_obj != '') and (reductions_obj is not None) and ('reductions_poly' in map_checkboxes):
#     #     info = info + """\n\nHover over the polygons to see reduction %"""

#     trig = ctx.triggered_id
#     # print(trig)

#     if (powers_obj != '') and (powers_obj is not None):

#         props = utils.decode_obj(powers_obj)
#         # print(reach_feature)
#         # print(sites_feature)

#         if (trig == 'reach_map'):
#             # print(reach_feature)
#             feature_id = int(reach_feature['id'])

#             reach_data = props.sel(nzsegment=feature_id)

#             info_str = """\n\n**nzsegment**: {seg}\n\n**Improvement**: {red}%\n\n**Likelihood of observing an improvement (power)**: {t_stat}%""".format(red=int(reach_data.reduction), t_stat=int(reach_data.power_modelled), seg=feature_id)

#             info += info_str
#         elif (trig == 'sites_points'):
#             feature_id = int(sites_feature['properties']['nzsegment'])
#             # print(sites_feature)

#             reach_data = props.sel(nzsegment=feature_id)

#             info_str = """\n\n**nzsegment**: {seg}\n\n**Site name**: {site}\n\n**Improvement**: {red}%\n\n**Likelihood of observing an improvement (power)**: {t_stat}%""".format(red=int(reach_data.reduction), t_stat=int(reach_data.power_monitored), seg=feature_id, site=sites_feature['id'])

#             info += info_str
#         else:
#             if 'Site name' in old_info:
#                 feature_id = int(sites_feature['properties']['nzsegment'])
#                 reach_data = props.sel(nzsegment=feature_id)

#                 info_str = """\n\n**nzsegment**: {seg}\n\n**Site name**: {site}\n\n**Improvement**: {red}%\n\n**Likelihood of observing an improvement (power)**: {t_stat}%""".format(red=int(reach_data.reduction), t_stat=int(reach_data.power_monitored), seg=feature_id, site=sites_feature['id'])

#                 info += info_str
#             elif 'nzsegment' in old_info:
#                 feature_id = int(reach_feature['id'])
#                 reach_data = props.sel(nzsegment=feature_id)

#                 info_str = """\n\n**nzsegment**: {seg}\n\n**Improvement**: {red}%\n\n**Likelihood of observing an improvement (power)**: {t_stat}%""".format(red=int(reach_data.reduction), t_stat=int(reach_data.power_modelled), seg=feature_id)

#                 info += info_str
#             else:
#                 info += """\n\nClick on a reach to see info"""


#         # else:
#         #     info = old_info

#     return info


# @callback(
#     Output("dl_power_rivers", "data"),
#     Input("dl_btn_power_rivers", "n_clicks"),
#     State('catch_id', 'data'),
#     State('powers_obj', 'data'),
#     State('indicator_rivers', 'value'),
#     State('time_period', 'value'),
#     State('freq', 'value'),
#     prevent_initial_call=True,
#     )
# def download_power(n_clicks, catch_id, powers_obj, indicator, n_years, n_samples_year):

#     if (catch_id != '') and (powers_obj != '') and (powers_obj is not None):
#         power_data = utils.decode_obj(powers_obj)

#         df1 = power_data.to_dataframe().reset_index()
#         df1['indicator'] = param.rivers_indicator_dict[indicator]
#         df1['n_years'] = n_years
#         df1['n_samples_per_year'] = n_samples_year

#         df2 = df1.set_index(['indicator', 'n_years', 'n_samples_per_year', 'nzsegment']).sort_index()

#         return dcc.send_data_frame(df2.to_csv, f"river_power_{catch_id}.csv")
