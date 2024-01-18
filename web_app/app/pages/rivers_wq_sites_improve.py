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
# import tethysts
import base64
import geobuf
import booklet

# from .app import app
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
    path='/rivers-wq-sites-improve',
    title='Water Quality Sites Improvements',
    name='rivers_wq_sites_improve',
    description='River Water Quality Sites Improvements'
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
}""", name='rivers_catch_style_handle_sites_change')

base_reach_style_handle = assign("""function style3(feature) {
    return {
        weight: 2,
        opacity: 0.75,
        color: 'grey',
    };
}""", name='rivers_base_reach_style_handle_sites_change')

sites_points_handle = assign("""function rivers_sites_points_handle(feature, latlng, context){
    const {classes, colorscale, circleOptions, colorProp} = context.props.hideout;  // get props from hideout
    const value = feature.properties[colorProp];  // get value the determines the fillColor
    for (let i = 0; i < classes.length; ++i) {
        if (value == classes[i]) {
            circleOptions.fillColor = colorscale[i];  // set the color according to the class
        }
    }

    return L.circleMarker(latlng, circleOptions);
}""", name='rivers_sites_points_handle_sites_change')

draw_marae = assign("""function(feature, latlng){
const flag = L.icon({iconUrl: '/assets/nzta-marae.svg', iconSize: [20, 30]});
return L.marker(latlng, {icon: flag});
}""", name='rivers_sites_change_marae_handle')


# catch_id = 3076139

###############################################
### Initial processing

# with booklet.open(eco_sites_path, 'r') as f:
#     catches = [int(c) for c in f]

# catches.sort()
indicators = list(param.rivers_indicator_dict.keys())
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
                        children=[dmc.Accordion(
                            value="1",
                            chevronPosition='left',
                            children=[
                            dmc.AccordionItem([
                                # html.H5('(1) Catchment selection', style={'font-weight': 'bold'}),
                                dmc.AccordionControl('(1) Catchment Selection', style={'font-size': 18}),
                                dmc.AccordionPanel([

                                    html.Label('(1a) Select a catchment on the map:'),
                                    dmc.Text(id='catch_name_sites_change', weight=700, style={'margin-top': 10}),
                                    ]
                                    )
                                ],
                                value='1'
                                ),

                            dmc.AccordionItem([
                                dmc.AccordionControl('(2) Select Indicator and minimum power', style={'font-size': 18}),
                                dmc.AccordionPanel([
                                    dmc.Text('(2a) Select Indicator:'),
                                    dcc.Dropdown(options=[{'label': param.rivers_indicator_dict[d], 'value': d} for d in indicators], id='indicator_sites_change', optionHeight=40, clearable=False, style={'margin-bottom': 0}),
                                    dmc.HoverCard(
                                        withArrow=True,
                                        width=param.hovercard_width,
                                        shadow="md",
                                        openDelay=param.hovercard_open_delay,
                                        children=[
                                            dmc.HoverCardTarget(html.Label('(2b) Select the minimum detection power (❓):', style={'margin-top': 20})),
                                            dmc.HoverCardDropdown(
                                                dmc.Text(
                                                    """
                                                    This is the minimum power required to detect water quality improvements across the catchment.
                                                    """,
                                                    size="sm",
                                                )
                                            ),
                                        ],
                                    ),
                                    dmc.Slider(id='power_slider_rivers_change',
                                                value=80,
                                                mb=35,
                                                step=5,
                                                min=60,
                                                max=100,
                                                showLabelOnHover=True,
                                                disabled=False,
                                                marks=param.marks_power,
                                                ),
    #                                 html.Label('(2b) Assign a percent improvement by site under the "improvement %" column then press enter to confirm:'),
    #                                 dash_table.DataTable(data=[], style_data={
    #     'whiteSpace': 'normal',
    #     'height': 'auto',
    # }, columns=[{'name': n, 'id': n, 'editable': (n == 'improvement %')} for n in ['site name', 'improvement %']], id='sites_tbl_change', style_cell={'font-size': 11}, style_header_conditional=[{
    #     'if': {'column_id': 'improvement %'},
    #     'font-weight': 'bold'
    # }]),
                                    ]
                                    )
                                ],
                                value='2'
                                ),

                            dmc.AccordionItem([
                                dmc.AccordionControl('(3) Query Options', style={'font-size': 18}),
                                dmc.AccordionPanel([
                                    dmc.Text('(3a) Select sampling duration (years):', style={'margin-top': 20}),
                                    dmc.SegmentedControl(data=[{'label': d, 'value': str(d)} for d in param.rivers_time_periods],
                                                         id='time_period_sites_change',
                                                         value='5',
                                                         fullWidth=True,
                                                         color=1,
                                                         ),
                                    dmc.Text('(3b) Select sampling frequency:', style={'margin-top': 20}),
                                    dmc.SegmentedControl(data=[{'label': v, 'value': str(k)} for k, v in param.rivers_freq_mapping.items()],
                                                          id='freq_sites_change',
                                                          value='12',
                                                          fullWidth=True,
                                                          color=1
                                                          ),
                                    # html.Label('(3d) Change the percent of the reductions applied (100% is the max realistic reduction):', style={'margin-top': 20}),
                                    # dmc.Slider(id='Reductions_slider',
                                    #            value=100,
                                    #            mb=35,
                                    #            step=10,
                                    #            # min=10,
                                    #            showLabelOnHover=True,
                                    #            disabled=False,
                                    #            marks=marks
                                    #            ),
                                    # dcc.Dropdown(options=[{'label': d, 'value': d} for d in time_periods], id='time_period', clearable=False, value=5),

                                    ],
                                    )
                                ],
                                value='3'
                                ),

                            dmc.AccordionItem([
                                dmc.AccordionControl('(4) Download Results', style={'font-size': 18}),
                                dmc.AccordionPanel([
                                    dmc.Text('(4a) Download improvement results given the prior query options (csv):'),
                                    dcc.Loading(
                                    type="default",
                                    children=[html.Div(dmc.Button("Download improvement results", id='dl_btn_power_sites_change'), style={'margin-bottom': 20, 'margin-top': 10}),
                            dcc.Download(id="dl_power_sites_change")],
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
                        # dcc.Markdown("""* The rivers colored with *Low*, *Moderate*, and *High* are the qualitative monitoring priorities as there  is too much uncertainty in estimating the powers per reach.""")
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
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(url=str(param.rivers_catch_pbf_path), format="geobuf", id='catch_map_sites_change', zoomToBoundsOnClick=True, zoomToBounds=False, options=dict(style=catch_style_handle))), name='Catchments', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='marae_map_sites_change', zoomToBoundsOnClick=False, zoomToBounds=False, options=dict(pointToLayer=draw_marae))), name='Marae', checked=False),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='reach_map_sites_change', options=dict(style=base_reach_style_handle), hideout={})), name='Rivers', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='sites_points_sites_change', options=dict(pointToLayer=sites_points_handle), hideout=param.rivers_points_hideout)), name='Monitoring sites', checked=True),
                                    ],
                                    id='layers_sites_change'
                                    ),
                                gc.colorbar_reductions,
                                # html.Div(id='colorbar', children=colorbar_base),
                                # dmc.Group(id='colorbar', children=colorbar_base),
                                dcc.Markdown(id="info_sites_change", className="info", style={"position": "absolute", "top": "10px", "right": "160px", "z-index": "1000"})
                                ],
                                style={'width': '100%', 'height': param.map_height, 'margin': "auto", "display": "block"}
                                ),

                            ],
                            # className='five columns', style={'margin': 10}
                            ),
                        ),
                    ]
                    ),
            dcc.Store(id='catch_id_sites_change', data=''),
            dcc.Store(id='rivers_sites_change', data=''),
            dcc.Store(id='sites_powers_obj_sites_change', data=''),
            ]
        )

    return layout


###############################################
### Callbacks

@callback(
    Output('catch_id_sites_change', 'data'),
    [Input('catch_map_sites_change', 'click_feature')]
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
    Output('catch_name_sites_change', 'children'),
    [Input('catch_id_sites_change', 'data')]
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
        Output('reach_map_sites_change', 'data'),
        Input('catch_id_sites_change', 'data'),
        )
# @cache.memoize()
def update_reaches(catch_id):
    if catch_id != '':
        with booklet.open(param.rivers_reach_gbuf_path, 'r') as f:
            data = base64.b64encode(f[int(catch_id)]).decode()

    else:
        data = ''

    return data


@callback(
        Output('marae_map_sites_change', 'data'),
        Input('catch_id_sites_change', 'data'),
        )
def update_marae(catch_id):
    if catch_id != '':
        with booklet.open(param.rivers_marae_path, 'r') as f:
            data = base64.b64encode(f[int(catch_id)]).decode()

    else:
        data = ''

    return data


@callback(
        Output('sites_points_sites_change', 'data'),
        Output('rivers_sites_change', 'data'),
        Input('catch_id_sites_change', 'data'),
        )
def update_monitor_sites(catch_id):
    if catch_id != '':
        with booklet.open(param.rivers_sites_path, 'r') as f:
            sites = f[int(catch_id)]

        features = geobuf.decode(sites)['features']
        if features:
            tbl_data = {}
            for f in features:
                name = f['id']
                nzsegment = f['properties']['nzsegment']
                tbl_data[nzsegment] = name
        else:
            tbl_data = {}

        points_data = base64.b64encode(sites).decode()

    else:
        points_data = ''
        tbl_data = {}

    tbl_data_encoded = utils.encode_obj(tbl_data)

    return points_data, tbl_data_encoded


@callback(
    Output('sites_powers_obj_sites_change', 'data'),
    [Input('indicator_sites_change', 'value'),
     Input('time_period_sites_change', 'value'),
     Input('freq_sites_change', 'value'),
     Input('power_slider_rivers_change', 'value'),
     Input('rivers_sites_change', 'data'),
     ],
    prevent_initial_call=True
    )
def update_sites_powers_obj(indicator, n_years, n_samples_year, min_power, tbl_data_encoded):
    """

    """
    if isinstance(n_years, str) and isinstance(indicator, str) and isinstance(n_samples_year, str):
        tbl_data = utils.decode_obj(tbl_data_encoded)
        # print(tbl_data)

        if len(tbl_data) > 0:
            n_samples = int(n_samples_year)*int(n_years)

            sites_nzsegments = list(tbl_data.keys())

            power_data = xr.open_dataset(param.rivers_power_moni_path, engine='h5netcdf')
            power_data1 = power_data.sel(indicator=indicator, n_samples=n_samples, drop=True).dropna('nzsegment').copy().load()

            power_segs = power_data1.nzsegment.values

            # conc_perc = power_data1.conc_perc.values
            power_data2 = []
            for seg in sites_nzsegments:
                if seg in power_segs:
                    powers = power_data1.sel(nzsegment=seg, drop=True).power
                    conc_perc = int(powers[powers >= int(min_power)].idxmin())
                    power = int(powers[powers >= int(min_power)].min())
                else:
                    conc_perc = 101
                    power = -1

                site_name = tbl_data[seg]
                power_data2.append({'reduction': 100 - conc_perc, 'nzsegment': seg, 'power': power, 'site_name': site_name})

            # print(power_data1)
            power_data.close()
            del power_data
            power_data1.close()
            del power_data1

            sites_powers_obj = utils.encode_obj(power_data2)
            return sites_powers_obj
        else:
            raise dash.exceptions.PreventUpdate
    else:
        raise dash.exceptions.PreventUpdate


@callback(
    Output('sites_points_sites_change', 'hideout'),
    [Input('sites_powers_obj_sites_change', 'data')],
    prevent_initial_call=True
    )
def update_sites_hideout(sites_powers_obj):
    """

    """
    if (sites_powers_obj != '') and (sites_powers_obj is not None):
        props = utils.decode_obj(sites_powers_obj)

        ## Monitored
        if props:
            # print(props_moni)
            color_arr2 = pd.cut([p['reduction'] for p in props], param.bins_reductions, labels=param.colorscale_reductions, right=False).tolist()
            color_arr2 = [color if isinstance(color, str) else '#252525' for color in color_arr2]

            hideout_moni = {'classes': [p['nzsegment'] for p in props], 'colorscale': color_arr2, 'circleOptions': dict(fillOpacity=1, stroke=True, color='black', weight=1, radius=param.site_point_radius), 'colorProp': 'nzsegment'}

        else:
            hideout_moni = param.rivers_points_hideout
    else:
        hideout_moni = param.rivers_points_hideout

    return hideout_moni


@callback(
    Output("info_sites_change", "children"),
    [Input('sites_powers_obj_sites_change', 'data'),
      Input('sites_points_sites_change', 'click_feature')],
    [State("info_sites_change", "children"),
     ]
    )
def update_map_info(sites_powers_obj, sites_feature, old_info):
    """

    """
    info = """"""

    if (sites_powers_obj != '') and (sites_powers_obj is not None) and (sites_feature is not None):
        props = utils.decode_obj(sites_powers_obj)

        feature_id = int(sites_feature['properties']['nzsegment'])
        # print(sites_feature)
        # print(props)
        # print(feature_id)

        reach_data = [p for p in props if p['nzsegment'] == feature_id]
        if reach_data:
            power = reach_data[0]['power']
            if power == -1:
                # power = 'NA'
                red = 'NA'
            else:
                # power = str(power) + '%'
                red = str(reach_data[0]['reduction']) + '%'

            # info += """##### Monitoring Site:

            #     \n\n**nzsegment**: {seg}\n\n**Site name**: {site}\n\n**Estimated minimum improvement %**: {conc}\n\n**Likelihood of detecting the improvement (power)**: {t_stat}""".format(t_stat=power, conc=red, seg=feature_id, site=sites_feature['id'])
            info += """##### Monitoring Site:

                \n\n**nzsegment**: {seg}\n\n**Site name**: {site}\n\n**Estimated minimum improvement**: {conc}""".format(conc=red, seg=feature_id, site=sites_feature['id'])

    return info


@callback(
    Output("dl_power_sites_change", "data"),
    Input("dl_btn_power_sites_change", "n_clicks"),
    State('catch_id_sites_change', 'data'),
    State('sites_powers_obj_sites_change', 'data'),
    State('indicator_sites_change', 'value'),
    State('time_period_sites_change', 'value'),
    State('freq_sites_change', 'value'),
    prevent_initial_call=True,
    )
def download_power(n_clicks, catch_id, sites_powers_obj, indicator, n_years, n_samples_year):

    if (catch_id != '') and (sites_powers_obj != '') and (sites_powers_obj is not None) and isinstance(n_samples_year, str):
        power_data = utils.decode_obj(sites_powers_obj)

        df1 = pd.DataFrame.from_dict(power_data)

        df1.rename(columns={'reduction': 'improvement'}, inplace=True)
        df1['indicator'] = param.rivers_indicator_dict[indicator]
        df1['n_years'] = n_years
        df1['n_samples_per_year'] = n_samples_year

        df1.loc[df1.power < 0, 'power'] = 'NA'
        df1.loc[df1.improvement < 0, 'improvement'] = 'NA'

        df2 = df1.set_index(['nzsegment', 'improvement', 'indicator', 'n_years', 'n_samples_per_year']).sort_index()

        return dcc.send_data_frame(df2.to_csv, f"river_water_quality_sites_improvements_{catch_id}.csv")
