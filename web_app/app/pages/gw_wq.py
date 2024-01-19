#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 13:37:46 2022

@author: mike
"""
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
import base64
import geobuf
import booklet
from dash_iconify import DashIconify
import hdf5plugin

# from .app import app

# from app import app
# import utils
# from . import utils

from utils import parameters as param
from utils import components as gc
from utils import utils

##########################################
### Parameters

dash.register_page(
    __name__,
    path='/gw-wq',
    title='Water Quality Sites',
    name='gw_wq',
    description='Groundwater Quality Sites'
)

### Handles
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


###############################################
### Initial processing

with booklet.open(param.gw_points_rc_blt, 'r') as f:
    rcs = list(f.keys())

rcs.sort()

indicators = [{'value': k, 'label': v} for k, v in param.gw_indicator_dict.items()]

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
        children=[
            dmc.Grid(
                columns=7,
                children=[
                    dmc.Col(
                        span=3,
                        children=[dmc.Accordion(
                            value="1",
                            chevronPosition='left',
                            children=[
                            dmc.AccordionItem([
                                dmc.AccordionControl('(1) Select a Regional Council', style={'font-size': 18}),
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
                                dmc.AccordionControl('(2) Select Indicator and an improvement', style={'font-size': 18}),
                                dmc.AccordionPanel([
                                    dmc.Text('(2a) Select Indicator:'),
                                    dcc.Dropdown(options=indicators, id='indicator_gw', optionHeight=40, clearable=False, style={'margin-bottom': 20}),
                                    html.Label('(2b) Select an improvement:'),
                                    dmc.Slider(id='reductions_slider_gw',
                                               value=25,
                                               mb=35,
                                               step=5,
                                               min=5,
                                               max=50,
                                               showLabelOnHover=True,
                                               disabled=False,
                                               marks=param.gw_reductions_options
                                               ),
                                    # dcc.Dropdown(options=gw_reductions_options, id='reductions_gw', optionHeight=40, clearable=False,
                                    #               style={'margin-top': 10}
                                    #               ),
                                    ]
                                    )
                                ],
                                value='2'
                                ),

                            dmc.AccordionItem([
                                dmc.AccordionControl('(3) Query Options', style={'font-size': 18}),
                                dmc.AccordionPanel([
                                    dmc.HoverCard(
                                        withArrow=True,
                                        width=param.hovercard_width,
                                        shadow="md",
                                        openDelay=param.hovercard_open_delay,
                                        children=[
                                            dmc.HoverCardTarget(html.Label('(3a) Select sampling duration (years) (❓):', style={'margin-top': 20})),
                                            dmc.HoverCardDropdown(
                                                dmc.Text(
                                                    """
                                                    The power results for groundwater do not include the impacts of groundwater travel processes (e.g. lag between the source and receptor). Any improvements performed upgradient of the wells will take time to reach the wells. The map therefore shows the maximum possible detection power, which will often be an overestimate.
                                                    """,
                                                    size="sm",
                                                )
                                            ),
                                        ],
                                    ),
                                    # dmc.Group(
                                    #     [dmc.Text('(3a) Select sampling duration (years):', color="black"),
                                    #     dmc.HoverCard(
                                    #         withArrow=True,
                                    #         width=param.hovercard_width,
                                    #         shadow="md",
                                    #         children=[
                                    #             dmc.HoverCardTarget(DashIconify(icon="material-symbols:help", width=30)),
                                    #             dmc.HoverCardDropdown(
                                    #                 dmc.Text(
                                    #                     """
                                    #                     The power results for groundwater only apply after the groundwater lag times of the upgradient improvements. Any improvements performed upgradient of the wells will take time to reach the wells. Click on a well to see the estimated mean residence time.
                                    #                     """,
                                    #                     size="sm",
                                    #                 )
                                    #             ),
                                    #         ],
                                    #     ),
                                    #     ],
                                    #     style={'margin-top': 20}
                                    # ),
                                    dmc.SegmentedControl(data=[{'label': d, 'value': str(d)} for d in param.gw_time_periods],
                                                         id='time_period_gw',
                                                         value='5',
                                                         fullWidth=True,
                                                         color=1,
                                                         ),
                                    dmc.Text('(3b) Select sampling frequency:', style={'margin-top': 20}),
                                    dmc.SegmentedControl(data=[{'label': v, 'value': str(k)} for k, v in param.gw_freq_mapping.items()],
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
                                dmc.AccordionControl('(4) Download Results', style={'font-size': 18}),
                                dmc.AccordionPanel([
                                    dmc.Text('(4a) Download power results given the prior query options (CSV):'),
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
                            ),
                            dcc.Markdown("""The interactive map provides a screening assessment of nitrate concentration change detection power potential and can be used to identify:

- Sites with low detection power potential (<80% = unlikely to be useful for nitrate loss mitigation effectiveness determination); and
- Sites with high detection power potential (≥80% = may be useful for mitigation effectiveness determination, subject to further assessment).

If the detection power is shown to be ≥80%, further assessment of detection power should be undertaken as per the *Mitigation Effectiveness Monitoring Design: Water quality monitoring for management of diffuse nitrate pollution* document. """, style={'font-size': 14, 'margin-top': 10})
                            ]
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
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(url=str(param.rc_bounds_gbuf), format="geobuf", id='rc_map', zoomToBoundsOnClick=True, options=dict(style=rc_style_handle),  hideout={})), name='Regional Councils', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='gw_points', zoomToBounds=True, zoomToBoundsOnClick=True, cluster=False, options=dict(pointToLayer=gw_points_style_handle), hideout=param.gw_points_hideout)), name='GW wells', checked=True),
                                    ],
                                    id='layers_gw'
                                    ),
                                gc.colorbar_power,
                                # html.Div(id='colorbar', children=colorbar_base),
                                # dmc.Group(id='colorbar', children=colorbar_base),
                                dcc.Markdown(id="info_gw", className="info", style={"position": "absolute", "top": "10px", "right": "160px", "z-index": "1000"})
                                ],
                                style={'width': '100%', 'height': param.map_height, 'margin': "auto", "display": "block"},
                                id="map2",
                                ),

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
        with booklet.open(param.gw_points_rc_blt, 'r') as f:
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

        power_data = xr.open_dataset(param.gw_error_path, engine='h5netcdf')
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

        color_arr = pd.cut(props.power.values, param.bins, labels=param.colorscale_power, right=False).tolist()
        # print(color_arr)
        # print(props['gw_id'])

        hideout = {'classes': props['ref'].values, 'colorscale': color_arr, 'circleOptions': dict(fillOpacity=1, stroke=False, radius=param.site_point_radius), 'colorProp': 'tooltip'}
    elif (gw_points_encode is not None):
        # print('trigger')
        gw_refs = utils.decode_obj(gw_points_encode)

        hideout = {'classes': gw_refs, 'colorscale': ['#808080'] * len(gw_refs), 'circleOptions': dict(fillOpacity=1, stroke=False, radius=param.site_point_radius), 'colorProp': 'tooltip'}
    else:
        hideout = param.gw_points_hideout

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

    if isinstance(reductions, int) and (powers_obj != '') and (powers_obj is not None):
        if feature is not None:
            # print(feature)
            gw_refs = utils.decode_obj(gw_points_encode)
            if feature['id'] in gw_refs:
                props = utils.decode_obj(powers_obj)

                # print(feature['properties']['lag_at_site'])

                if feature['properties']['lag_at_site'] is None:
                    info_str = """\n\n**User-defined improvement**: {red}%\n\n**Likelihood of observing the improvement (power)**: {t_stat}%\n\n**Well Depth (m)**: {depth:.1f}\n\n**Mean residence time (MRT) at well (years)**: NA\n\n**MRT within {lag_dist:.1f} km at {depth_min} - {depth_max} m depth**:\n\n&nbsp;&nbsp;&nbsp;&nbsp; **Min - Median - Max**: {lag_min} - {lag_median} - {lag_max} years"""
                    info2 = info_str.format(red=int(reductions), t_stat=int(props[props.ref==feature['id']].iloc[0]['power']), depth=feature['properties']['depth'], lag_median=feature['properties']['lag_median'], lag_dist=feature['properties']['lag_dist']*0.001, depth_min=feature['properties']['depth_min'], depth_max=feature['properties']['depth_max'], lag_min=feature['properties']['lag_min'], lag_max=feature['properties']['lag_max'])
                else:
                    site_lag = str(int(feature['properties']['lag_at_site']))
                    info_str = """\n\n**User-defined improvement**: {red}%\n\n**Likelihood of observing the improvement (power)**: {t_stat}%\n\n**Well Depth (m)**: {depth:.1f}\n\n**Mean residence time (MRT) at well (years)**: {site_lag}"""
                    info2 = info_str.format(red=int(reductions), t_stat=int(props[props.ref==feature['id']].iloc[0]['power']), depth=feature['properties']['depth'], site_lag=site_lag)

                info = info2

        else:
            info = """\n\nClick on a well to see info"""

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

        df1['indicator'] = param.gw_indicator_dict[indicator]
        df1['n_years'] = n_years
        df1['n_samples_per_year'] = n_samples_year
        df1['improvement'] = reductions

        df2 = df1.rename(columns={'ref': 'site_id'}).set_index(['indicator', 'n_years', 'n_samples_per_year', 'improvement', 'site_id']).sort_index()

        return dcc.send_data_frame(df2.to_csv, f"gw_power_{rc_id}.csv")





# with shelflet.open(gw_lc_path, 'r') as f:
#     plan_file = f[str(gw_id)]
