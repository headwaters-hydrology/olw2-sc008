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
    path='/rivers-eco-sites',
    title='Ecology Sites',
    name='rivers_eco_sites',
    description='River Ecology Sites'
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
}""", name='eco_catch_style_handle')

base_reach_style_handle = assign("""function style3(feature) {
    return {
        weight: 2,
        opacity: 0.75,
        color: 'grey',
    };
}""", name='eco_base_reach_style_handle')

sites_points_handle = assign("""function rivers_sites_points_handle(feature, latlng, context){
    const {classes, colorscale, circleOptions, colorProp} = context.props.hideout;  // get props from hideout
    const value = feature.properties[colorProp];  // get value the determines the fillColor
    for (let i = 0; i < classes.length; ++i) {
        if (value == classes[i]) {
            circleOptions.fillColor = colorscale[i];  // set the color according to the class
        }
    }

    return L.circleMarker(latlng, circleOptions);
}""", name='eco_sites_points_handle')

draw_marae = assign("""function(feature, latlng){
const flag = L.icon({iconUrl: '/assets/nzta-marae.svg', iconSize: [20, 30]});
return L.marker(latlng, {icon: flag});
}""", name='eco_sites_marae_handle')


# catch_id = 3076139

###############################################
### Initial processing

# with booklet.open(eco_sites_path, 'r') as f:
#     catches = [int(c) for c in f]

# catches.sort()
indicators = list(param.eco_indicator_dict.keys())
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
                                dmc.AccordionControl('(1) Catchment Selection', style={'font-size': 18}),
                                dmc.AccordionPanel([

                                    html.Label('(1a) Select a catchment on the map:'),
                                    dmc.Text(id='catch_name_eco_sites', weight=700, style={'margin-top': 10}),
                                    ]
                                    )
                                ],
                                value='1'
                                ),

                            dmc.AccordionItem([
                                dmc.AccordionControl('(2) Define Indicator and improvements by site', style={'font-size': 18}),
                                dmc.AccordionPanel([
                                    dmc.Text('(2a) Select Indicator:'),
                                    dcc.Dropdown(options=[{'label': param.eco_indicator_dict[d], 'value': d} for d in indicators], id='indicator_eco_sites', optionHeight=40, clearable=False, style={'margin-bottom': 20}),
                                    html.Label('(2b) Type in a percent improvement by site under the "improvement %" column then press enter to confirm:'),
                                    dash_table.DataTable(data=[], columns=[{'name': n, 'id': n, 'editable': (n == 'improvement %')} for n in ['site name', 'improvement %']], id='tbl_eco_sites', style_cell={'font-size': 11}, style_header_conditional=[{
        'if': {'column_id': 'improvement %'},
        'font-weight': 'bold'
    }]),
                                    # dmc.Slider(id='reductions_slider_sites',
                                    #            value=25,
                                    #            mb=35,
                                    #            step=5,
                                    #            min=5,
                                    #            max=90,
                                    #            showLabelOnHover=True,
                                    #            disabled=False,
                                    #            marks=reductions_options
                                    #            ),
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
                                    dmc.Text('(3a) Select sampling duration (years):', style={'margin-top': 20}),
                                    dmc.SegmentedControl(data=[{'label': d, 'value': str(d)} for d in param.eco_time_periods],
                                                         id='time_period_eco_sites',
                                                         value='5',
                                                         fullWidth=True,
                                                         color=1,
                                                         ),
                                    dmc.Text('(3b) Select sampling frequency:', style={'margin-top': 20}),
                                    dmc.SegmentedControl(data=param.eco_freq_data_dict['peri'],
                                                        id='freq_eco_sites',
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
                                    dmc.Text('(4a) Download power results given the prior query options (csv):'),
                                    dcc.Loading(
                                    type="default",
                                    children=[html.Div(dmc.Button("Download power results", id='dl_btn_power_eco_sites'), style={'margin-bottom': 20, 'margin-top': 10}),
                            dcc.Download(id="dl_power_eco_sites")],
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
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(url=str(param.rivers_catch_pbf_path), format="geobuf", id='catch_map_eco_sites', zoomToBoundsOnClick=True, zoomToBounds=False, options=dict(style=catch_style_handle))), name='Catchments', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='marae_map_eco_sites', zoomToBoundsOnClick=False, zoomToBounds=False, options=dict(pointToLayer=draw_marae))), name='Marae', checked=False),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='reach_map_eco_sites', options=dict(style=base_reach_style_handle), hideout={})), name='Rivers', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='sites_points_eco_sites', options=dict(pointToLayer=sites_points_handle), hideout=param.rivers_points_hideout)), name='Monitoring sites', checked=True),
                                    ],
                                    id='layers_eco'
                                    ),
                                gc.colorbar_power,
                                dcc.Markdown(id="info_eco_sites", className="info", style={"position": "absolute", "top": "10px", "right": "160px", "z-index": "1000"})
                                ],
                                style={'width': '100%', 'height': param.map_height, 'margin': "auto", "display": "block"}
                                ),

                            ],
                            # className='five columns', style={'margin': 10}
                            ),
                        ),
                    ]
                    ),
            dcc.Store(id='catch_id_eco_sites', data=''),
            dcc.Store(id='sites_powers_obj_eco_sites', data=''),
            # dcc.Store(id='reaches_obj_eco_sites', data=''),
            # dcc.Store(id='catch_power_obj_eco_sites', data=''),
            ]
        )

    return layout


###############################################
### Callbacks

@callback(
    Output('catch_id_eco_sites', 'data'),
    [Input('catch_map_eco_sites', 'click_feature')]
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
    Output('catch_name_eco_sites', 'children'),
    [Input('catch_id_eco_sites', 'data')]
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
        Output('reach_map_eco_sites', 'data'),
        Input('catch_id_eco_sites', 'data'),
        )
# @cache.memoize()
def update_reaches(catch_id):
    if catch_id != '':
        with booklet.open(param.rivers_reach_gbuf_path, 'r') as f:
            data = base64.b64encode(f[int(catch_id)]).decode()

    else:
        data = ''

    return data


# @callback(
#         Output('sites_points_eco_sites', 'data'),
#         Input('catch_id_eco_sites', 'data'),
#         )
# # @cache.memoize()
# def update_monitor_sites(catch_id):
#     if catch_id != '':
#         with booklet.open(eco_sites_path, 'r') as f:
#             data = base64.b64encode(f[int(catch_id)]).decode()

#     else:
#         data = ''

#     return data


@callback(
        Output('marae_map_eco_sites', 'data'),
        Input('catch_id_eco_sites', 'data'),
        )
def update_marae(catch_id):
    if catch_id != '':
        with booklet.open(param.rivers_marae_path, 'r') as f:
            data = base64.b64encode(f[int(catch_id)]).decode()

    else:
        data = ''

    return data


@callback(
    Output("freq_eco_sites", "data"),
    Input('indicator_eco_sites', 'value'),
    prevent_initial_call=True,
    )
def update_freq_options(indicator):

    if isinstance(indicator, str):
        freq_data = param.eco_freq_data_dict[indicator]

        return freq_data


@callback(
    Output("freq_eco_sites", "value"),
    Input("freq_eco_sites", "data"),
    State('indicator_eco_sites', 'value'),
    prevent_initial_call=True,
    )
def update_freq_value(freq_data, indicator):

    if isinstance(indicator, str):
        n_samples_year = param.eco_freq_value_dict[indicator]

        return str(n_samples_year)


@callback(
        Output('sites_points_eco_sites', 'data'),
        Output('tbl_eco_sites', 'data'),
        Input('catch_id_eco_sites', 'data'),
        )
def update_monitor_sites(catch_id):
    if catch_id != '':
        with booklet.open(param.eco_sites_path, 'r') as f:
            sites = f[int(catch_id)]

        points_data = base64.b64encode(sites).decode()

        features = geobuf.decode(sites)['features']
        if features:
            tbl_data = [{'site name': f['id'], 'nzsegment': f['properties']['nzsegment'], 'improvement %': 25} for f in features]
        else:
            tbl_data = []

    else:
        points_data = ''
        tbl_data = []

    return points_data, tbl_data



@callback(
    Output('sites_powers_obj_eco_sites', 'data'),
    Input('tbl_eco_sites', 'data'),
    Input('indicator_eco_sites', 'value'),
    Input('time_period_eco_sites', 'value'),
    Input('freq_eco_sites', 'value'),
    prevent_initial_call=True
    )
def update_sites_powers_obj(tbl_data, indicator, n_years, n_samples_year):
    """

    """
    if (len(tbl_data) > 0) and isinstance(n_years, str) and isinstance(indicator, str) and isinstance(n_samples_year, str):

        red1 = {}
        for r in tbl_data:
            try:
                red_int = int(r['improvement %'])
            except:
                red_int = 0
            red1[int(r['nzsegment'])] = 100 - red_int

        n_samples = int(n_samples_year)*int(n_years)

        power_data = xr.open_dataset(param.eco_power_moni_path, engine='h5netcdf')
        power_data1 = power_data.sel(indicator=indicator, n_samples=n_samples, drop=True).dropna('nzsegment').copy().load()
        power_data2 = []
        for seg, conc_perc in red1.items():
            if seg in power_data1.nzsegment:
                power = int(power_data1.sel(nzsegment=seg, conc_perc=conc_perc).power.values)
                power_data2.append({'conc_perc': conc_perc, 'nzsegment': seg, 'power': power})
            else:
                power_data2.append({'conc_perc': conc_perc, 'nzsegment': seg, 'power': -1})

        # print(power_data1)
        power_data.close()
        del power_data
        power_data1.close()
        del power_data1

        data = utils.encode_obj(power_data2)
        return data
    else:
        raise dash.exceptions.PreventUpdate


@callback(
    Output('sites_points_eco_sites', 'hideout'),
    [Input('sites_powers_obj_eco_sites', 'data')],
    prevent_initial_call=True
    )
def update_sites_hideout(powers_obj):
    """

    """
    if (powers_obj != '') and (powers_obj is not None):
        props = utils.decode_obj(powers_obj)

        ## Monitored
        if props:
            # print(props_moni)
            color_arr2 = pd.cut([p['power'] for p in props], param.bins, labels=param.colorscale_power, right=False).tolist()
            color_arr2 = [color if isinstance(color, str) else '#252525' for color in color_arr2]

            hideout_moni = {'classes': [p['nzsegment'] for p in props], 'colorscale': color_arr2, 'circleOptions': dict(fillOpacity=1, stroke=True, color='black', weight=1, radius=param.site_point_radius), 'colorProp': 'nzsegment'}

        else:
            hideout_moni = param.rivers_points_hideout
    else:
        hideout_moni = param.rivers_points_hideout

    return hideout_moni


@callback(
    Output("info_eco_sites", "children"),
    [Input('sites_powers_obj_eco_sites', 'data'),
      Input('sites_points_eco_sites', 'click_feature')],
    [State("info_eco_sites", "children"),
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
        # print(props.nzsegment)
        # print(feature_id)

        reach_data = [p for p in props if p['nzsegment'] == feature_id]
        if reach_data:
            power = reach_data[0]['power']
            if power == -1:
                power = 'NA'
            else:
                power = str(power) + '%'
            red = 100 - reach_data[0]['conc_perc']

            info += """##### Monitoring Site:

                \n\n**nzsegment**: {seg}\n\n**Site name**: {site}\n\n**Improvement %**: {conc}\n\n**Likelihood of observing an improvement (power)**: {t_stat}""".format(t_stat=power, conc=red, seg=feature_id, site=sites_feature['id'])

    # if info == """""":
    #     info = old_info

    return info


@callback(
    Output("dl_power_eco_sites", "data"),
    Input("dl_btn_power_eco_sites", "n_clicks"),
    State('catch_id_eco_sites', 'data'),
    State('sites_powers_obj_eco_sites', 'data'),
    State('indicator_eco_sites', 'value'),
    State('time_period_eco_sites', 'value'),
    State('freq_eco_sites', 'value'),
    State('tbl_eco_sites', 'data'),
    prevent_initial_call=True,
    )
def download_power(n_clicks, catch_id, powers_obj, indicator, n_years, n_samples_year, tbl_data):

    if (catch_id != '') and (powers_obj != '') and (powers_obj is not None) and isinstance(n_samples_year, str):
        power_data = utils.decode_obj(powers_obj)

        df1 = pd.DataFrame.from_dict(power_data)

        if tbl_data:
            sites_tbl_df = pd.DataFrame(tbl_data).drop('improvement %', axis=1)
            df1 = pd.merge(sites_tbl_df, df1, on='nzsegment')

        df1['improvement'] = 100 - df1['conc_perc']
        df1['indicator'] = param.eco_indicator_dict[indicator]
        df1['n_years'] = n_years
        df1['n_samples_per_year'] = n_samples_year

        df1.loc[df1.power < 0, 'power'] = 'NA'

        df2 = df1.drop('conc_perc', axis=1).set_index(['nzsegment', 'improvement', 'indicator', 'n_years', 'n_samples_per_year']).sort_index()

        return dcc.send_data_frame(df2.to_csv, f"river_ecoology_sites_power_{catch_id}.csv")
