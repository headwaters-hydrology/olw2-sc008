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
    path='/lakes-wq-sites',
    title='Water Quality Sites',
    name='lakes_wq_sites',
    description='Lakes Water Quality Sites'
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
                                    dmc.Text(id='lake_name_sites', weight=700, style={'margin-top': 10}),
                                    ]
                                    )
                                ],
                                value='1'
                                ),

                            dmc.AccordionItem([
                                dmc.AccordionControl('(2) Define Indicator and improvements by site/lake', style={'font-size': 18}),
                                dmc.AccordionPanel([
                                    dmc.Text('(2a) Select Indicator:'),
                                    dcc.Dropdown(options=indicators, id='indicator_lakes_sites', optionHeight=40, clearable=False, style={'margin-bottom': 20}),
                                    html.Label('(2b) Assign a percent improvement for the overall lake:'),
                                    dmc.NumberInput(
                                        id='lake_improvement',
                                        # label="Must be between",
                                        value=25,
                                        precision=0,
                                        min=0,
                                        step=5,
                                        max=100,
                                        style={'margin-bottom': 20},
                                    ),
                                    html.Label('(2c) Assign a percent improvement by site under the "improvement %" column then press enter to confirm:'),
                                    dash_table.DataTable(data=[], style_data={
        'whiteSpace': 'normal',
        'height': 'auto',
    }, columns=[{'name': n, 'id': n, 'editable': (n == 'improvement %')} for n in ['site name', 'improvement %']], id='sites_tbl_lakes_sites', style_cell={'font-size': 11}, style_header_conditional=[{
        'if': {'column_id': 'improvement %'},
        'font-weight': 'bold'
    }]),
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
                                                         id='time_period_lakes_sites',
                                                         value='5',
                                                         fullWidth=True,
                                                         color=1,
                                                         ),
                                    dmc.Text('(3b) Select sampling frequency:', style={'margin-top': 20}),
                                    dmc.SegmentedControl(data=[{'label': v, 'value': str(k)} for k, v in param.lakes_freq_mapping.items()],
                                                         id='freq_lakes_sites',
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
                                    children=[html.Div(dmc.Button("Download power results", id='dl_btn_power_lakes_sites'), style={'margin-bottom': 20, 'margin-top': 10}),
                            dcc.Download(id="dl_power_lakes_sites")],
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
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(url=str(param.lakes_pbf_path), format="geobuf", id='lake_points_sites', zoomToBoundsOnClick=True, cluster=True)), name='Lake points', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='lake_poly_sites', zoomToBounds=True, options=dict(style=lake_style_handle), hideout={'classes': [''], 'colorscale': ['#808080'], 'style': param.lake_style, 'colorProp': 'tooltip'})), name='Lakes', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='sites_points_lakes_sites', options=dict(pointToLayer=sites_points_handle), hideout=param.lakes_points_hideout)), name='Monitoring sites', checked=True),
                                    ],
                                    ),
                                gc.colorbar_power,
                                # html.Div(id='colorbar', children=colorbar_base),
                                # dmc.Group(id='colorbar', children=colorbar_base),
                                dcc.Markdown(id="info_lakes_sites", className="info", style={"position": "absolute", "top": "10px", "right": "160px", "z-index": "1000"})
                                ],
                                style={'width': '100%', 'height': param.map_height, 'margin': "auto", "display": "block"},
                                ),

                            ],
                            ),
                        ),
                    ]
                    ),
            dcc.Store(id='lake_id_sites', data=''),
            dcc.Store(id='powers_obj_lakes_sites', data=''),
            dcc.Store(id='sites_powers_obj_lakes_sites', data=''),
            ]
        )

    return layout


###############################################
### Callbacks

@callback(
    Output('lake_id_sites', 'data'),
    [Input('lake_points_sites', 'click_feature')]
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
    Output('lake_name_sites', 'children'),
    [Input('lake_id_sites', 'data')]
    )
def update_lake_name(lake_id):
    """

    """
    # print(ds_id)
    if lake_id != '':
        lake_name = lakes_names[int(lake_id)]

        return lake_name


@callback(
        Output('sites_points_lakes_sites', 'data'),
        Output('sites_tbl_lakes_sites', 'data'),
        Input('lake_id_sites', 'data'),
        )
def update_monitor_sites(lake_id):
    if lake_id != '':
        # lake_name = lakes_names[int(lake_id)]

        with booklet.open(param.lakes_moni_sites_gbuf_path, 'r') as f:
            sites = f[int(lake_id)]

        points_data = base64.b64encode(sites).decode()

        features = geobuf.decode(sites)['features']
        if features:
            tbl_data = []
            for f in features:
                name = f['id']
                # site_name = f['properties']['site_name']
                # if len(name) > 40:
                #     name = name[:40] + '...'
                tbl_data.append({'site name': name, 'lake_id': lake_id, 'improvement %': 25})
        else:
            tbl_data = []

    else:
        points_data = ''
        tbl_data = []

    return points_data, tbl_data


@callback(
        Output('lake_poly_sites', 'data'),
        Input('lake_id_sites', 'data'),
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
    Output('powers_obj_lakes_sites', 'data'),
    Output('sites_powers_obj_lakes_sites', 'data'),
    Input('indicator_lakes_sites', 'value'),
    Input('time_period_lakes_sites', 'value'),
    Input('freq_lakes_sites', 'value'),
    Input('lake_improvement', 'value'),
    Input('sites_tbl_lakes_sites', 'data'),
    State('lake_id_sites', 'data'),
    prevent_initial_call=True
    )
def update_powers_lakes_sites(indicator, n_years, n_samples_year, lake_improvement, tbl_data, lake_id):
    """

    """
    power_lake_encoded = ''
    power_sites_encoded = ''

    if isinstance(n_years, str) and isinstance(n_samples_year, str) and isinstance(indicator, str) and (lake_id != ''):
        lake_id_int = int(lake_id)
        # ind_name = param.lakes_indicator_dict[indicator]
        lake_data = lakes_data[lake_id_int]
        n_samples = int(n_samples_year)*int(n_years)

        ## Overall lake
        # Conc adjustments
        # lake_conc_perc = utils.lakes_conc_adjustment(indicator, 100 - int(lake_improvement), lake_data)
        lake_conc_perc = 100 - int(lake_improvement)

        # Power
        power_data = xr.open_dataset(param.lakes_power_model_path, engine='h5netcdf')
        power_data1 = int(power_data.sel(indicator=indicator, LFENZID=lake_id_int, n_samples=n_samples, conc_perc=lake_conc_perc).power_modelled.values)

        # Encode output
        power_lake_encoded = utils.encode_obj({'reduction': 100 - lake_conc_perc, 'power': power_data1, 'lake_id': lake_id})

        # print(power_data1)

        ## Sites
        if len(tbl_data) > 0:
            # Conc adjustments
            red1 = {}
            for r in tbl_data:
                try:
                    # new_conc = utils.lakes_conc_adjustment(indicator, 100 - int(r['improvement %']), lake_data)
                    new_conc = 100 - int(r['improvement %'])
                except:
                    new_conc = 0
                red1[r['site name']] = new_conc

            # Iter through sites for power
            with booklet.open(param.lakes_moni_sites_gbuf_path, 'r') as f:
                sites = f[lake_id_int]

            features = geobuf.decode(sites)['features']

            if len(features) > 0:
                power_data = xr.open_dataset(param.lakes_power_moni_path, engine='h5netcdf')

                # print(power_data)
                sites_data = {f1['id']: f1['id'] for f1 in features}

                power_data1 = power_data.sel(indicator=indicator, n_samples=n_samples, drop=True).copy().load()
                power_site_ids = power_data1.site_id.values

                power_data2 = []
                for site_id, site_name in sites_data.items():
                    # conc_perc1 = int(conc_perc.sel(site_id=site_id))
                    if site_id in power_site_ids:
                        conc_perc1 = red1[site_id]
                        try:
                            power = int(power_data1.sel(conc_perc=conc_perc1, site_id=site_id).power_monitored.values)
                        except ValueError:
                            power = -1
                    else:
                        power = -1
                    power_data2.append({'reduction': 100 - conc_perc1, 'site_id': site_id, 'power': power, 'site_name': site_name, 'lake_id': lake_id})

            # Encode output
            power_sites_encoded = utils.encode_obj(power_data2)

    return power_lake_encoded, power_sites_encoded


@callback(
    Output('lake_poly_sites', 'hideout'),
    Output('sites_points_lakes_sites', 'hideout'),
    Input('powers_obj_lakes_sites', 'data'),
    Input('sites_powers_obj_lakes_sites', 'data'),
    Input('lake_id_sites', 'data'),
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

            color_arr = pd.cut([props['power']], param.bins, labels=param.colorscale_power, right=False).tolist()
            # print(color_arr)
            # print(props['lake_id'])

            hideout_model = {'classes': [props['lake_id']], 'colorscale': color_arr, 'style': param.lake_style, 'colorProp': 'LFENZID'}

        ## Monitored
        if sites_powers_obj != '':
            sites_props = utils.decode_obj(sites_powers_obj)
            # print(sites_props)
            color_arr2 = pd.cut([p['power'] for p in sites_props], param.bins, labels=param.colorscale_power, right=False).tolist()
            color_arr2 = [color if isinstance(color, str) else '#252525' for color in color_arr2]
            # print(color_arr2)

            # print(sites_props)

            hideout_moni = {'classes': [p['site_id'] for p in sites_props], 'colorscale': color_arr2, 'circleOptions': dict(fillOpacity=1, stroke=True, color='black', weight=1, radius=param.site_point_radius), 'colorProp': 'tooltip'}

    return hideout_model, hideout_moni


@callback(
    Output("info_lakes_sites", "children"),
    Input('powers_obj_lakes_sites', 'data'),
    Input('sites_powers_obj_lakes_sites', 'data'),
    Input("lake_poly_sites", "click_feature"),
    Input('sites_points_lakes_sites', 'click_feature'),
    Input('lake_id_sites', 'data'),
    State("info_lakes_sites", "children"),
    )
def update_map_info_lakes(powers_obj, sites_powers_obj, feature, sites_feature, lake_id, old_info):
    """

    """
    info = """"""

    trig = ctx.triggered_id

    # print(ctx.triggered_prop_ids)

    # if (reductions_obj != '') and (reductions_obj is not None) and ('reductions_poly' in map_checkboxes):
    #     info = info + """\n\nHover over the polygons to see reduction %"""

    if trig == 'lake_id_sites':
        pass

    elif (powers_obj != '') and (trig == 'lake_poly_sites'):
        if feature is not None:
            props = utils.decode_obj(powers_obj)
            # print(props)

            lake_name = lakes_names[int(lake_id)]

            info_str = """**Lake name**: {lake}\n\n**Predicted improvement**: {red}%\n\n**Likelihood of detecting the improvement (power)**: {power}%""".format(red=int(props['reduction']), power=int(props['power']), lake=lake_name)

            info = info_str

    elif (trig == 'sites_points_lakes_sites') or ((sites_powers_obj != '') and (sites_feature is not None) and ('Site name' in old_info)):
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
                    power = 'NA'
                else:
                    power = str(power) + '%'

                reduction = reach_data0['reduction']
                site_name = reach_data0['site_name']

                info_str = """**Site name**: {site}\n\n**Predicted improvement**: {red}%\n\n**Likelihood of detecting the improvement (power)**: {power}""".format(red=reduction, power=power, site=site_name)

                info = info_str

    elif (trig == 'powers_obj_lakes_sites') and (powers_obj != '') and ('Lake name' in old_info):
        # print(reach_feature)
        props = utils.decode_obj(powers_obj)

        lake_name = lakes_names[int(lake_id)]

        info_str = """**Lake name**: {lake}\n\n**Predicted improvement**: {red}%\n\n**Likelihood of detecting the improvement (power)**: {power}%""".format(red=int(props['reduction']), power=int(props['power']), lake=lake_name)

        info = info_str

    return info


@callback(
    Output("dl_power_lakes_sites", "data"),
    Input("dl_btn_power_lakes_sites", "n_clicks"),
    State('lake_id_sites', 'data'),
    State('powers_obj_lakes_sites', 'data'),
    State('sites_powers_obj_lakes_sites', 'data'),
    State('indicator_lakes_sites', 'value'),
    State('time_period_lakes_sites', 'value'),
    State('freq_lakes_sites', 'value'),
    State('sites_tbl_lakes_sites', 'data'),
    prevent_initial_call=True,
    )
def download_power(n_clicks, lake_id, powers_obj, sites_powers_obj, indicator, n_years, n_samples_year, tbl_data):

    if (lake_id != '') and (powers_obj != '') and (powers_obj is not None):
        power_data = utils.decode_obj(powers_obj)

        # print(power_data)

        lake_name = lakes_names[int(lake_id)]

        df1 = pd.DataFrame.from_records([power_data]).rename(columns={'power': 'overall_lake_power_modelled', 'lake_id': 'LFENZID', 'reduction': 'improvement'})
        df1['indicator'] = param.lakes_indicator_dict[indicator]
        df1['n_years'] = n_years
        df1['n_samples_per_year'] = n_samples_year
        df1['lake_name'] = lake_name
        # print(df1)

        if sites_powers_obj != '':
            sites_power_data = utils.decode_obj(sites_powers_obj)
            # print(sites_power_data)
            sites_df = pd.DataFrame.from_records(sites_power_data).rename(columns={'power': 'monitoring_site_power', 'lake_id': 'LFENZID'})
            # print(sites_df)
            sites_df.loc[sites_df.monitoring_site_power < 0, 'monitoring_site_power'] = np.nan
            df1 = pd.merge(df1, sites_df.drop(['reduction', 'site_id'], axis=1), on=['LFENZID'], how='left')

        df2 = df1.set_index(['indicator', 'n_years', 'n_samples_per_year', 'improvement', 'LFENZID']).sort_index()

        return dcc.send_data_frame(df2.to_csv, f"lake_power_{lake_id}.csv")
