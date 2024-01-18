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
    path='/lakes-wq-sites-improve',
    title='Water Quality Sites Changes Improvements',
    name='lakes_wq_sites_improve',
    description='Lakes Water Quality Sites Improvements'
)

### Handles
lake_style_handle = assign("""function style4(feature, context){
    const {classes, colorscale, style, colorProp} = context.props.hideout;  // get props from hideout
    const value = feature.properties[colorProp];  // get value the determines the color
    for (let i = 0; i < classes.length; ++i) {
        if (value == classes[i]) {
            style.color = colorscale[i];  // set the color according to the class
        }
    }

    return style;
}""", name='lakes_lake_style_handle_sites')

sites_points_handle = assign("""function lakes_sites_points_handle(feature, latlng, context){
    const {classes, colorscale, circleOptions, colorProp} = context.props.hideout;  // get props from hideout
    const value = feature.properties[colorProp];  // get value the determines the fillColor
    for (let i = 0; i < classes.length; ++i) {
        if (value == classes[i]) {
            circleOptions.fillColor = colorscale[i];  // set the color according to the class
        }
    }

    return L.circleMarker(latlng, circleOptions);
}""", name='lakes_sites_points_handle_sites')


# lake_id = 11133

###############################################
### Initial processing

with open(param.assets_path.joinpath('lakes_points.pbf'), 'rb') as f:
    geodict = geobuf.decode(f.read())

lakes_names = {}
for f in geodict['features']:
    label0 = ' '.join(f['properties']['name'].split())
    # label = str(f['id']) + ' - ' + label0
    # lakes_names[int(f['id'])] = label
    lakes_names[int(f['id'])] = label0

lakes_data = {int(f['id']): f['properties'] for f in geodict['features']}

indicators = [{'value': k, 'label': v} for k, v in param.lakes_indicator_dict.items()]

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
                        children=dmc.Accordion(
                            value="1",
                            chevronPosition='left',
                            children=[
                            dmc.AccordionItem([
                                dmc.AccordionControl('(1) Lake Selection', style={'font-size': 18}),
                                dmc.AccordionPanel([

                                    html.Label('(1a) Select a lake/lagoon on the map:'),
                                    # dcc.Dropdown(options=[v for v in lakes_options], id='lake_id', optionHeight=40, clearable=False,
                                    #               style={'margin-top': 10}
                                    #               ),
                                    dmc.Text(id='lake_name_sites_change', weight=700, style={'margin-top': 10}),
                                    ]
                                    )
                                ],
                                value='1'
                                ),

                            dmc.AccordionItem([
                                dmc.AccordionControl('(2) Select Indicator and minimum power', style={'font-size': 18}),
                                dmc.AccordionPanel([
                                    dmc.Text('(2a) Select Indicator:'),
                                    dcc.Dropdown(options=indicators, id='indicator_lakes_sites_change', optionHeight=40, clearable=False, style={'margin-bottom': 20}),
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
                                                    This is the minimum power required to detect water quality improvements across the lake.
                                                    """,
                                                    size="sm",
                                                )
                                            ),
                                        ],
                                    ),
                                    dmc.Slider(id='power_slider_lakes_change',
                                                value=80,
                                                mb=35,
                                                step=5,
                                                min=10,
                                                max=100,
                                                showLabelOnHover=True,
                                                disabled=False,
                                                marks=param.marks_power,
                                                ),
                                    ]
                                    )
                                ],
                                value='2'
                                ),

                            dmc.AccordionItem([
                                dmc.AccordionControl('(3) Query Options', style={'font-size': 18}),
                                dmc.AccordionPanel([
                                    dmc.Text('(3a) Select sampling duration (years):', style={'margin-top': 20}),
                                    dmc.SegmentedControl(data=[{'label': d, 'value': str(d)} for d in param.lakes_time_periods],
                                                         id='time_period_lakes_sites_change',
                                                         value='5',
                                                         fullWidth=True,
                                                         color=1,
                                                         ),
                                    dmc.Text('(3b) Select sampling frequency:', style={'margin-top': 20}),
                                    dmc.SegmentedControl(data=[{'label': v, 'value': str(k)} for k, v in param.lakes_freq_mapping.items()],
                                                         id='freq_lakes_sites_change',
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
                                    children=[html.Div(dmc.Button("Download power results", id='dl_btn_power_lakes_sites_change'), style={'margin-bottom': 20, 'margin-top': 10}),
                            dcc.Download(id="dl_power_lakes_sites_change")],
                                    ),
                                    ],
                                    )
                                ],
                                value='4'
                                ),

                            ],
                            )
                        ),
                    dmc.Col(
                        span=4,
                        children=html.Div([
                            dl.Map(center=param.center, zoom=param.zoom, children=[
                                dl.LayersControl([
                                    dl.BaseLayer(dl.TileLayer(attribution=param.attribution, opacity=0.7), checked=True, name='OpenStreetMap'),
                                    dl.BaseLayer(dl.TileLayer(url='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', attribution='Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap (CC-BY-SA)', opacity=0.6), checked=False, name='OpenTopoMap'),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(url=str(param.lakes_pbf_path), format="geobuf", id='lake_points_sites_change', zoomToBoundsOnClick=True, cluster=True)), name='Lake points', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='lake_poly_sites_change', zoomToBounds=True, options=dict(style=lake_style_handle), hideout={'classes': [''], 'colorscale': ['#808080'], 'style': param.lake_style, 'colorProp': 'tooltip'})), name='Lakes', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='sites_points_lakes_sites_change', options=dict(pointToLayer=sites_points_handle), hideout=param.lakes_points_hideout)), name='Monitoring sites', checked=True),
                                    ],
                                    ),
                                gc.colorbar_reductions,
                                # html.Div(id='colorbar', children=colorbar_base),
                                # dmc.Group(id='colorbar', children=colorbar_base),
                                dcc.Markdown(id="info_lakes_sites_change", className="info", style={"position": "absolute", "top": "10px", "right": "160px", "z-index": "1000"})
                                ],
                                style={'width': '100%', 'height': param.map_height, 'margin': "auto", "display": "block"},
                                ),

                            ],
                            ),
                        ),
                    ]
                    ),
            dcc.Store(id='lake_id_sites_change', data=''),
            dcc.Store(id='powers_obj_lakes_sites_change', data=''),
            dcc.Store(id='lakes_sites_change', data=''),
            dcc.Store(id='sites_powers_obj_lakes_sites_change', data=''),
            ]
        )

    return layout


###############################################
### Callbacks

@callback(
    Output('lake_id_sites_change', 'data'),
    [Input('lake_points_sites_change', 'click_feature')]
    )
def update_lake_id(feature):
    """

    """
    # print(ds_id)
    lake_id = ''
    if feature is not None:
        # print(feature)
        if not feature['properties']['cluster']:
            lake_id = str(feature['id'])

    return lake_id


@callback(
    Output('lake_name_sites_change', 'children'),
    [Input('lake_id_sites_change', 'data')]
    )
def update_lake_name(lake_id):
    """

    """
    # print(ds_id)
    if lake_id != '':
        lake_name = lakes_names[int(lake_id)]

        return lake_name


@callback(
        Output('sites_points_lakes_sites_change', 'data'),
        Output('lakes_sites_change', 'data'),
        Input('lake_id_sites_change', 'data'),
        )
def update_monitor_sites(lake_id):
    if lake_id != '':
        # lake_name = lakes_names[int(lake_id)]

        with booklet.open(param.lakes_moni_sites_gbuf_path, 'r') as f:
            sites = f[int(lake_id)]

        points_data = base64.b64encode(sites).decode()

        features = geobuf.decode(sites)['features']
        if features:
            # tbl_data = []
            # for f in features:
            #     name = f['id']
            #     # site_name = f['properties']['site_name']
            #     # if len(name) > 40:
            #     #     name = name[:40] + '...'
            #     tbl_data.append({'site name': name, 'lake_id': lake_id, 'improvement %': 25})
            tbl_data = []
            for f in features:
                name = f['id']
                tbl_data.append(name)
                # nzsegment = f['properties']['nzsegment']
                # tbl_data[nzsegment] = name
        else:
            tbl_data = []

    else:
        points_data = ''
        tbl_data = []

    tbl_data_encoded = utils.encode_obj(tbl_data)

    return points_data, tbl_data_encoded


@callback(
        Output('lake_poly_sites_change', 'data'),
        Input('lake_id_sites_change', 'data'),
        )
# @cache.memoize()
def update_lake(lake_id):
    if lake_id != '':
        with booklet.open(param.lakes_poly_gbuf_path, 'r') as f:
            data = base64.b64encode(f[int(lake_id)]).decode()
    else:
        data = ''

    return data


@callback(
    Output('powers_obj_lakes_sites_change', 'data'),
    Output('sites_powers_obj_lakes_sites_change', 'data'),
    Input('indicator_lakes_sites_change', 'value'),
    Input('time_period_lakes_sites_change', 'value'),
    Input('freq_lakes_sites_change', 'value'),
    Input('power_slider_lakes_change', 'value'),
    Input('lakes_sites_change', 'data'),
    State('lake_id_sites_change', 'data'),
    prevent_initial_call=True
    )
def update_powers_lakes_sites(indicator, n_years, n_samples_year, min_power, tbl_data_encoded, lake_id):
    """

    """
    power_lake_encoded = ''
    power_sites_encoded = ''

    if isinstance(n_years, str) and isinstance(n_samples_year, str) and isinstance(indicator, str) and (lake_id != ''):
        tbl_data = utils.decode_obj(tbl_data_encoded)

        lake_id_int = int(lake_id)
        # ind_name = param.lakes_indicator_dict[indicator]
        # lake_data = lakes_data[lake_id_int]
        n_samples = int(n_samples_year)*int(n_years)

        ## Overall lake
        # Conc adjustments
        # lake_conc_perc = utils.lakes_conc_adjustment(indicator, 100 - int(lake_improvement), lake_data)
        # lake_conc_perc = 100 - int(lake_improvement)

        # Power
        power_data = xr.open_dataset(param.lakes_power_model_path, engine='h5netcdf')
        power_data1 = power_data.sel(indicator=indicator, LFENZID=lake_id_int, n_samples=n_samples).power_modelled

        lake_conc_perc = int(power_data1[power_data1 >= int(min_power)].idxmin())
        lake_power = int(power_data1[power_data1 >= int(min_power)].min())

        # Encode output
        power_lake_encoded = utils.encode_obj({'reduction': 100 - lake_conc_perc, 'power': lake_power, 'lake_id': lake_id})

        power_data.close()
        del power_data

        # print(power_data1)

        ## Sites
        if len(tbl_data) > 0:

            power_data = xr.open_dataset(param.lakes_power_moni_path, engine='h5netcdf')

            power_data1 = power_data.sel(indicator=indicator, n_samples=n_samples, drop=True).copy().load()
            power_site_ids = power_data1.site_id.values

            power_data2 = []
            for site_id in tbl_data:
                # conc_perc1 = int(conc_perc.sel(site_id=site_id))
                if site_id in power_site_ids:
                    try:
                        powers = power_data1.sel(site_id=site_id).power_monitored
                        conc_perc = int(powers[powers >= int(min_power)].idxmin())
                        power = int(powers[powers >= int(min_power)].min())
                    except ValueError:
                        power = -1
                        conc_perc = 101
                else:
                    power = -1
                    conc_perc = 101

                power_data2.append({'reduction': 100 - conc_perc, 'site_id': site_id, 'power': power, 'site_name': site_id, 'lake_id': lake_id})

            # Encode output
            power_sites_encoded = utils.encode_obj(power_data2)

            power_data.close()
            del power_data

    return power_lake_encoded, power_sites_encoded


@callback(
    Output('lake_poly_sites_change', 'hideout'),
    Output('sites_points_lakes_sites_change', 'hideout'),
    Input('powers_obj_lakes_sites_change', 'data'),
    Input('sites_powers_obj_lakes_sites_change', 'data'),
    Input('lake_id_sites_change', 'data'),
    prevent_initial_call=True
    )
def update_hideout_lakes(powers_obj, sites_powers_obj, lake_id):
    """

    """
    hideout_model = {'classes': [lake_id], 'colorscale': ['#808080'], 'style': param.lake_style, 'colorProp': 'LFENZID'}
    hideout_moni = param.rivers_points_hideout

    if powers_obj != '':
        # print('trigger')
        props = utils.decode_obj(powers_obj)
        # print(props)
        # print(type(lake_id))

        if props['lake_id'] == lake_id:

            color_arr = pd.cut([props['reduction']], param.bins_reductions, labels=param.colorscale_reductions, right=False).tolist()
            # print(color_arr)
            # print(props['lake_id'])

            hideout_model = {'classes': [props['lake_id']], 'colorscale': color_arr, 'style': param.lake_style, 'colorProp': 'LFENZID'}

        ## Monitored
        if sites_powers_obj != '':
            sites_props = utils.decode_obj(sites_powers_obj)
            # print(sites_props)
            color_arr2 = pd.cut([p['reduction'] for p in sites_props], param.bins_reductions, labels=param.colorscale_reductions, right=False).tolist()
            color_arr2 = [color if isinstance(color, str) else '#252525' for color in color_arr2]
            # print(color_arr2)

            # print(sites_props)

            hideout_moni = {'classes': [p['site_id'] for p in sites_props], 'colorscale': color_arr2, 'circleOptions': dict(fillOpacity=1, stroke=True, color='black', weight=1, radius=param.site_point_radius), 'colorProp': 'tooltip'}

    return hideout_model, hideout_moni


@callback(
    Output("info_lakes_sites_change", "children"),
    Input('powers_obj_lakes_sites_change', 'data'),
    Input('sites_powers_obj_lakes_sites_change', 'data'),
    Input("lake_poly_sites_change", "click_feature"),
    Input('sites_points_lakes_sites_change', 'click_feature'),
    Input('lake_id_sites_change', 'data'),
    State("info_lakes_sites_change", "children"),
    )
def update_map_info_lakes(powers_obj, sites_powers_obj, feature, sites_feature, lake_id, old_info):
    """

    """
    info = """"""

    trig = ctx.triggered_id

    # print(ctx.triggered_prop_ids)

    # if (reductions_obj != '') and (reductions_obj is not None) and ('reductions_poly' in map_checkboxes):
    #     info = info + """\n\nHover over the polygons to see reduction %"""

    if trig == 'lake_id_sites_change':
        pass

    elif (powers_obj != '') and (trig == 'lake_poly_sites_change'):
        if feature is not None:
            props = utils.decode_obj(powers_obj)
            # print(props)

            lake_name = lakes_names[int(lake_id)]

            info_str = """**Lake name**: {lake}\n\n**Estimated improvement**: {red}%""".format(red=int(props['reduction']), lake=lake_name)

            if (sites_powers_obj != ''):
                info_str += """\n\n*Estimates from monitoring site(s)*"""
            else:
                info_str += """\n\n*Estimates from numerical model*"""

            info = info_str

    elif (trig == 'sites_points_lakes_sites_change') or ((sites_powers_obj != '') and (sites_feature is not None) and ('Site name' in old_info)):
        if (sites_powers_obj != ''):
            sites_props = utils.decode_obj(sites_powers_obj)
            # print(sites_feature)
            feature_id = sites_feature['id']
            # print(sites_props)

            reach_data = [p for p in sites_props if p['site_id'] == feature_id]
            if reach_data:
                reach_data0 = reach_data[0]
                power = reach_data0['power']
                if power == -1:
                    # power = 'NA'
                    reduction = 'NA'
                else:
                    # power = str(power) + '%'
                    reduction = str(reach_data0['reduction']) + '%'

                site_name = reach_data0['site_name']

                info_str = """**Site name**: {site}\n\n**Estimated improvement**: {red}""".format(red=reduction, site=site_name)

                info = info_str

    elif (trig == 'powers_obj_lakes_sites_change') and (powers_obj != '') and ('Lake name' in old_info):
        # print(reach_feature)
        props = utils.decode_obj(powers_obj)

        lake_name = lakes_names[int(lake_id)]

        info_str = """**Lake name**: {lake}\n\n**Estimated improvement**: {red}%""".format(red=int(props['reduction']), lake=lake_name)

        if (sites_powers_obj != ''):
            info_str += """\n\n*Estimates from monitoring site(s)*"""
        else:
            info_str += """\n\n*Estimates from numerical model*"""

        info = info_str

    return info


@callback(
    Output("dl_power_lakes_sites_change", "data"),
    Input("dl_btn_power_lakes_sites_change", "n_clicks"),
    State('lake_id_sites_change', 'data'),
    State('powers_obj_lakes_sites_change', 'data'),
    State('sites_powers_obj_lakes_sites_change', 'data'),
    State('indicator_lakes_sites_change', 'value'),
    State('time_period_lakes_sites_change', 'value'),
    State('freq_lakes_sites_change', 'value'),
    prevent_initial_call=True,
    )
def download_power(n_clicks, lake_id, powers_obj, sites_powers_obj, indicator, n_years, n_samples_year):

    if (lake_id != '') and (powers_obj != '') and (powers_obj is not None):
        power_data = utils.decode_obj(powers_obj)

        # print(power_data)

        lake_name = lakes_names[int(lake_id)]

        df1 = pd.DataFrame.from_records([power_data]).rename(columns={'power': 'lake_power_modelled', 'lake_id': 'LFENZID', 'reduction': 'lake_improvement'})
        df1['indicator'] = param.lakes_indicator_dict[indicator]
        df1['n_years'] = n_years
        df1['n_samples_per_year'] = n_samples_year
        df1['lake_name'] = lake_name
        # print(df1)

        if sites_powers_obj != '':
            sites_power_data = utils.decode_obj(sites_powers_obj)
            # print(sites_power_data)
            sites_df = pd.DataFrame.from_records(sites_power_data).rename(columns={'power': 'site_power', 'lake_id': 'LFENZID', 'reduction': 'site_improvement'})
            # print(sites_df)
            sites_df.loc[sites_df.site_power < 0, 'site_power'] = 'NA'
            sites_df.loc[sites_df.site_improvement < 0, 'site_improvement'] = 'NA'
            df1 = pd.merge(df1, sites_df.drop(['site_id'], axis=1), on=['LFENZID'], how='left')

        df2 = df1.set_index(['indicator', 'n_years', 'n_samples_per_year', 'LFENZID']).sort_index()

        return dcc.send_data_frame(df2.to_csv, f"lake_sites_improvements_{lake_id}.csv")
