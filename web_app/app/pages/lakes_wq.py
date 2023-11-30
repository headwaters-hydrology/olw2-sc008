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
    path='/lakes-wq',
    title='Water Quality Routing',
    name='lakes_wq',
    description='Lakes Water Quality Routing'
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
}""", name='lakes_lake_style_handle')

catch_style_handle = assign("""function style(feature) {
    return {
        fillColor: 'grey',
        weight: 2,
        opacity: 1,
        color: 'black',
        fillOpacity: 0.1
    };
}""", name='lakes_catch_style_handle')

sites_points_handle = assign("""function lakes_sites_points_handle(feature, latlng, context){
    const {classes, colorscale, circleOptions, colorProp} = context.props.hideout;  // get props from hideout
    const value = feature.properties[colorProp];  // get value the determines the fillColor
    for (let i = 0; i < classes.length; ++i) {
        if (value == classes[i]) {
            circleOptions.fillColor = colorscale[i];  // set the color according to the class
        }
    }

    return L.circleMarker(latlng, circleOptions);
}""", name='lakes_sites_points_handle')

draw_marae = assign("""function(feature, latlng){
const flag = L.icon({iconUrl: '/assets/nzta-marae.svg', iconSize: [20, 30]});
return L.marker(latlng, {icon: flag});
}""", name='lakes_marae_handle')


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
                                    dmc.Text(id='lake_name', weight=700, style={'margin-top': 10}),
                                    ]
                                    )
                                ],
                                value='1'
                                ),

                            dmc.AccordionItem([
                                dmc.AccordionControl('(2 - Optional) Customise the Land Mitigation Layer', style={'font-size': 18}),
                                dmc.AccordionPanel([
                                    html.Label('(2a) Download default Land Mitigation Layer as GPKG:'),
                                    dcc.Loading(
                                    type="default",
                                    children=[dmc.Anchor(dmc.Button('Download default layer'), href='', id='dl_poly_lakes', style={'margin-top': 10})],
                                    ),
                                    html.Label('NOTE: Only modify existing values. Do not add additional columns; they will be ignored.', style={
                                        'margin-top': 10
                                    }
                                        ),
                                    html.Label('(2b) Upload modified Land Mitigation Layer as GPKG:', style={
                                        'margin-top': 20
                                    }
                                        ),
                                    dcc.Loading(
                                        children=[
                                            dcc.Upload(
                                                id='upload_data_lakes',
                                                children=dmc.Button('Upload modified layer',
                                                ),
                                                style={
                                                    'margin-top': 10
                                                },
                                                multiple=False
                                            ),
                                            dcc.Markdown('', style={
                                                'textAlign': 'left',
                                                            }, id='upload_error_text_lakes'),
                                            ]
                                        ),

                                    html.Label('(2c) Process the modified Land Mitigation Layer and route the improvements downstream:', style={
                                        'margin-top': 20
                                    }
                                        ),
                                    dcc.Loading(
                                    type="default",
                                    children=html.Div([dmc.Button('Process improvements', id='process_reductions_lakes',
                                                                  # className="me-1",
                                                                  n_clicks=0),
                                                        html.Div(id='process_text_lakes')],
                                                      style={'margin-top': 10, 'margin-bottom': 10}
                                                      )
                                    ),
                                    ]
                                    )
                                ],
                                value='2'
                                ),

                            dmc.AccordionItem([
                                dmc.AccordionControl('(3) Query Options', style={'font-size': 18}),
                                dmc.AccordionPanel([
                                    dmc.Text('(3a) Select Indicator:'),
                                    dcc.Dropdown(options=indicators, id='indicator_lakes', optionHeight=40, clearable=False),
                                    dmc.Text('(3b) Select sampling duration (years):', style={'margin-top': 20}),
                                    dmc.SegmentedControl(data=[{'label': d, 'value': str(d)} for d in param.lakes_time_periods],
                                                         id='time_period_lakes',
                                                         value='5',
                                                         fullWidth=True,
                                                         color=1,
                                                         ),
                                    dmc.Text('(3c) Select sampling frequency:', style={'margin-top': 20}),
                                    dmc.SegmentedControl(data=[{'label': v, 'value': str(k)} for k, v in param.lakes_freq_mapping.items()],
                                                         id='freq_lakes',
                                                         value='12',
                                                         fullWidth=True,
                                                         color=1
                                                         ),
                                    html.Label('(3d) Change the percent of the improvements applied (100% is the max realistic improvement):', style={'margin-top': 20}),
                                    dmc.Slider(id='reductions_slider_lakes',
                                               value=100,
                                               mb=35,
                                               step=10,
                                               # min=10,
                                               showLabelOnHover=True,
                                               disabled=False,
                                               marks=param.marks
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
                                    children=[html.Div(dmc.Button("Download power results", id='dl_btn_power_lakes'), style={'margin-bottom': 20, 'margin-top': 10}),
                            dcc.Download(id="dl_power_lakes")],
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
                            dl.Map(center=param.center, zoom=param.zoom, children=[
                                dl.LayersControl([
                                    dl.BaseLayer(dl.TileLayer(attribution=param.attribution, opacity=0.7), checked=True, name='OpenStreetMap'),
                                    dl.BaseLayer(dl.TileLayer(url='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', attribution='Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap (CC-BY-SA)', opacity=0.6), checked=False, name='OpenTopoMap'),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(url=str(param.lakes_3rd_pbf_path), format="geobuf", id='lake_points', zoomToBoundsOnClick=True, cluster=True)), name='Lake points', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='catch_map_lakes', zoomToBoundsOnClick=True, zoomToBounds=True, options=dict(style=catch_style_handle))), name='Catchments', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='marae_map_lakes', zoomToBoundsOnClick=False, zoomToBounds=False, options=dict(pointToLayer=draw_marae))), name='Marae', checked=False),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='reach_map_lakes', options=dict(style=param.reach_style))), name='Rivers', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='lake_poly', options=dict(style=lake_style_handle), hideout={'classes': [''], 'colorscale': ['#808080'], 'style': param.lake_style, 'colorProp': 'tooltip'})), name='Lakes', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='sites_points_lakes', options=dict(pointToLayer=sites_points_handle), hideout=param.lakes_points_hideout)), name='Monitoring sites', checked=True),
                                    # dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='sites_points_lakes', options=dict(pointToLayer=sites_points_handle), hideout=rivers_points_hideout)), name='Monitoring sites', checked=True),
                                    ],
                                    ),
                                gc.colorbar_power,
                                # html.Div(id='colorbar', children=colorbar_base),
                                # dmc.Group(id='colorbar', children=colorbar_base),
                                dcc.Markdown(id="info_lakes", className="info", style={"position": "absolute", "top": "10px", "right": "160px", "z-index": "1000"})
                                ],
                                style={'width': '100%', 'height': param.map_height, 'margin': "auto", "display": "block"},
                                id="map2",
                                ),

                            ],
                            ),
                        ),
                    ]
                    ),
            dcc.Store(id='lake_id', data=''),
            dcc.Store(id='powers_obj_lakes', data=''),
            dcc.Store(id='sites_powers_obj_lakes', data=''),
            dcc.Store(id='reaches_obj_lakes', data=''),
            dcc.Store(id='custom_reductions_obj_lakes', data=''),
            dcc.Store(id='base_reductions_obj_lakes', data=''),
            ]
        )

    return layout


###############################################
### Callbacks

@callback(
    Output('lake_id', 'data'),
    [Input('lake_points', 'click_feature')]
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
    Output('lake_name', 'children'),
    [Input('lake_id', 'data')]
    )
def update_lake_name(lake_id):
    """

    """
    # print(ds_id)
    if lake_id != '':
        lake_name = lakes_names[int(lake_id)]

        return lake_name


@callback(
        Output('reach_map_lakes', 'data'),
        Input('lake_id', 'data'),
        )
# @cache.memoize()
def update_reaches_lakes(lake_id):
    if lake_id != '':
        with booklet.open(param.lakes_reach_gbuf_path, 'r') as f:
            data = base64.b64encode(f[int(lake_id)]).decode()

    else:
        data = ''

    return data


@callback(
        Output('marae_map_lakes', 'data'),
        Input('lake_id', 'data'),
        )
def update_marae(lake_id):
    if lake_id != '':
        with booklet.open(param.lakes_marae_path, 'r') as f:
            data = base64.b64encode(f[int(lake_id)]).decode()

    else:
        data = ''

    return data


@callback(
        Output('sites_points_lakes', 'data'),
        Input('lake_id', 'data'),
        )
# @cache.memoize()
def update_monitor_sites(lake_id):
    if lake_id != '':
        with booklet.open(param.lakes_moni_sites_gbuf_path, 'r') as f:
            data = base64.b64encode(f[int(lake_id)]).decode()

    else:
        data = ''

    return data


@callback(
        Output('catch_map_lakes', 'data'),
        Input('lake_id', 'data'),
        )
# @cache.memoize()
def update_catch_lakes(lake_id):
    if lake_id != '':
        with booklet.open(param.lakes_catches_major_path, 'r') as f:
            data = base64.b64encode(f[int(lake_id)]).decode()
    else:
        data = ''

    return data


@callback(
        Output('lake_poly', 'data'),
        Input('lake_id', 'data'),
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
        Output('base_reductions_obj_lakes', 'data'),
        Input('lake_id', 'data'),
        prevent_initial_call=True
        )
# @cache.memoize()
def update_base_reductions_obj(lake_id):
    data = ''

    if lake_id != '':
        with booklet.open(param.lakes_lc_path, 'r') as f:
            data = utils.encode_obj(f[int(lake_id)])

    return data


@callback(
    Output("dl_poly_lakes", "href"),
    # Input('indicator_lc', 'value'),
    Input('lake_id', 'data'),
    prevent_initial_call=True,
    )
def download_catch_lc(lake_id):

    if lake_id != '':
        url = param.lakes_catch_lc_gpkg_str.format(base_url=param.base_data_url, lake_id=lake_id)

        return url


@callback(
        Output('custom_reductions_obj_lakes', 'data'), Output('upload_error_text_lakes', 'children'),
        Input('upload_data_lakes', 'contents'),
        State('upload_data_lakes', 'filename'),
        State('lake_id', 'data'),
        prevent_initial_call=True
        )
# @cache.memoize()
def update_land_reductions(contents, filename, lake_id):
    data = None
    error_text = ''

    if lake_id != '':
        if contents is not None:
            data = utils.parse_gis_file(contents, filename)

            if isinstance(data, list):
                error_text = data[0]
                data = None
            else:
                error_text = 'Upload successful'
    else:
        error_text = 'You need to select a lake before uploading a file. Please refresh the page and start from step (1).'

    return data, error_text


@callback(
    Output('reaches_obj_lakes', 'data'), Output('process_text_lakes', 'children'),
    Input('process_reductions_lakes', 'n_clicks'),
    Input('base_reductions_obj_lakes', 'data'),
    [
      State('lake_id', 'data'),
      State('custom_reductions_obj_lakes', 'data'),
      ],
    prevent_initial_call=True)
def update_reach_reductions(click, base_reductions_obj, lake_id, reductions_obj):
    """

    """
    trig = ctx.triggered_id

    if (trig == 'process_reductions_lakes'):
        if (lake_id != '') and (reductions_obj != '') and (reductions_obj is not None):
            red1 = xr.open_dataset(param.lakes_reductions_model_path, engine='h5netcdf')

            base_props = red1.sel(LFENZID=int(lake_id), drop=True)

            new_reductions = utils.decode_obj(reductions_obj)
            base_reductions = utils.decode_obj(base_reductions_obj)

            diff_cols = utils.diff_reductions(new_reductions, base_reductions, param.lakes_lc_params)

            if diff_cols:
                new_props = utils.calc_lake_reach_reductions(lake_id, new_reductions, base_reductions, diff_cols)
                new_props1 = new_props.combine_first(base_props).copy().load()
                red1.close()
                del red1
                base_props.close()
                del base_props

                data = utils.encode_obj(new_props1)
                text_out = 'Routing complete'
            else:
                data = utils.set_default_lakes_reach_reductions(lake_id)
                text_out = 'The improvements values are identical to the originals. Either skip this step, or modify the improvements values.'
        elif lake_id != '':
            data = utils.set_default_lakes_reach_reductions(lake_id)
            text_out = 'Please upload a Land Mitigation file in step (2b)'
        else:
            data = ''
            text_out = 'Please select a lake before proceeding'
    else:
        if lake_id != '':
            # print('trigger')
            data = utils.set_default_lakes_reach_reductions(lake_id)
            text_out = ''
        else:
            data = ''
            text_out = 'Please select a lake before proceeding'

    return data, text_out


# @callback(
#         Output('red_disclaimer_lakes', 'children'),
#         Input('upload_data_lakes', 'contents'),
#         Input('demo_data_lakes', 'n_clicks'),
#         Input('map_checkboxes_lakes', 'value'),
#         prevent_initial_call=True
#         )
# def update_reductions_diclaimer(contents, n_clicks, map_checkboxes):
#     if (n_clicks is None) or (contents is not None):
#         return ''
#     elif 'reductions_poly' in map_checkboxes:
#         return '''* Areas on the map without polygon reductions are considered to have 0% reductions.'''
#     else:
#         return ''


# @callback(
#         Output('reductions_poly_lakes', 'data'),
#         Input('reductions_obj_lakes', 'data'),
#         Input('map_checkboxes_lakes', 'value'),
#         Input('col_name_lakes', 'value'),
#         )
# def update_reductions_poly_lakes(reductions_obj, map_checkboxes, col_name):
#     # print(reductions_obj)
#     # print(col_name)
#     if (reductions_obj != '') and (reductions_obj is not None) and ('reductions_poly' in map_checkboxes):

#         data = decode_obj(reductions_obj).to_crs(4326)

#         if isinstance(col_name, str):
#             data[col_name] = data[col_name].astype(str).str[:] + '% reduction'
#             data.rename(columns={col_name: 'tooltip'}, inplace=True)

#         gbuf = dlx.geojson_to_geobuf(data.__geo_interface__)

#         return gbuf
#     else:
#         return ''


@callback(
    Output('powers_obj_lakes', 'data'),
    Output('sites_powers_obj_lakes', 'data'),
    [Input('reaches_obj_lakes', 'data'), Input('indicator_lakes', 'value'), Input('time_period_lakes', 'value'), Input('freq_lakes', 'value'), Input('reductions_slider_lakes', 'value')],
    [State('lake_id', 'data')]
    )
def update_powers_data_lakes(reaches_obj, indicator, n_years, n_samples_year, prop_red, lake_id):
    """

    """
    power_model_encoded = ''
    power_moni_encoded = ''

    if (reaches_obj != '') and (reaches_obj is not None) and isinstance(n_years, str) and isinstance(n_samples_year, str) and isinstance(indicator, str):
        ind_name = param.lakes_indicator_dict[indicator]
        n_samples = int(n_samples_year)*int(n_years)
        lake_data = lakes_data[int(lake_id)]

        if prop_red == 0:
            props = 0
        else:
            props = int(utils.decode_obj(reaches_obj)[ind_name].sel(reduction_perc=prop_red, drop=True))
        # print(props)

        conc_perc = utils.lakes_conc_adjustment(indicator, 100 - props, lake_data)

        ## Modelled
        power_data = xr.open_dataset(param.lakes_power_model_path, engine='h5netcdf')
        try:
            power_data1 = power_data.sel(indicator=indicator, LFENZID=int(lake_id), n_samples=n_samples, conc_perc=conc_perc).copy().load()

            power_data2 = int(power_data1.power_modelled.values)
        except:
            power_data2 = 0
        power_data.close()
        del power_data

        power_model_encoded = utils.encode_obj({'reduction': props, 'power': power_data2, 'lake_id': lake_id})

        ## Monitored
        with booklet.open(param.lakes_moni_sites_gbuf_path, 'r') as f:
            sites = f[int(lake_id)]

        features = geobuf.decode(sites)['features']

        if len(features) > 0:
            power_data = xr.open_dataset(param.lakes_power_moni_path, engine='h5netcdf')

            # print(power_data)
            sites_data = {f1['id']: f1['id'] for f1 in features}

            power_data1 = power_data.sel(indicator=indicator, n_samples=n_samples, drop=True).copy().load()
            power_site_ids = power_data1.site_id.values

            power_data2 = []
            for site_id, site_name in sites_data.items():
                if site_id in power_site_ids:
                    try:
                        power = int(power_data1.sel(conc_perc=conc_perc, site_id=site_id).power_monitored.values)
                    except ValueError:
                        power = -1
                else:
                    power = -1
                power_data2.append({'reduction': 100 - conc_perc, 'site_id': site_id, 'power': power, 'site_name': site_name, 'lake_id': lake_id})

            power_data.close()
            del power_data

            # print(power_data2)

            power_moni_encoded = utils.encode_obj(power_data2)

    return power_model_encoded, power_moni_encoded


# @callback(
#         Output('lake_poly', 'options'),
#         Input('lake_poly', 'hideout'),
#         # State('lake_id', 'value')
#         )
# # @cache.memoize()
# def update_lakes_option(hideout):
#     trig = ctx.triggered_id
#     # print(trig)
#     # print(len(hideout))

#     if (len(hideout) == 1) or (trig == 'lake_id'):
#         options = dict(style=lake_style)
#     else:
#         options = dict(style=lake_style_handle)

#     return options


@callback(
    Output('lake_poly', 'hideout'),
    Output('sites_points_lakes', 'hideout'),
    Input('powers_obj_lakes', 'data'),
    Input('sites_powers_obj_lakes', 'data'),
    Input('lake_id', 'data'),
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
    Output("info_lakes", "children"),
    Input('powers_obj_lakes', 'data'),
    Input('sites_powers_obj_lakes', 'data'),
    Input("lake_poly", "click_feature"),
    Input('sites_points_lakes', 'click_feature'),
    Input('lake_id', 'data'),
    State("info_lakes", "children"),
    )
def update_map_info_lakes(powers_obj, sites_powers_obj, feature, sites_feature, lake_id, old_info):
    """

    """
    info = """"""

    trig = ctx.triggered_id

    # print(ctx.triggered_prop_ids)

    # if (reductions_obj != '') and (reductions_obj is not None) and ('reductions_poly' in map_checkboxes):
    #     info = info + """\n\nHover over the polygons to see reduction %"""

    if trig == 'lake_id':
        pass

    elif (powers_obj != '') and (trig == 'lake_poly'):
        if feature is not None:
            props = utils.decode_obj(powers_obj)
            # print(props)

            lake_name = lakes_names[int(lake_id)]

            info_str = """**Lake name**: {lake}\n\n**Predicted improvement**: {red}%\n\n**Likelihood of detecting the improvement (power)**: {power}%""".format(red=int(props['reduction']), power=int(props['power']), lake=lake_name)

            if (sites_powers_obj != ''):
                info_str += """\n\n*Power estimate from monitoring site(s)*"""
            else:
                info_str += """\n\n*Power estimate from numerical model*"""

            info = info_str

    elif (trig == 'sites_points_lakes') or ((sites_powers_obj != '') and (sites_feature is not None) and ('Site name' in old_info)):
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

    elif (trig == 'powers_obj_lakes') and (powers_obj != '') and ('Lake name' in old_info):
        # print(reach_feature)
        props = utils.decode_obj(powers_obj)

        lake_name = lakes_names[int(lake_id)]

        info_str = """**Lake name**: {lake}\n\n**Predicted improvement**: {red}%\n\n**Likelihood of detecting the improvement (power)**: {power}%""".format(red=int(props['reduction']), power=int(props['power']), lake=lake_name)

        if (sites_powers_obj != ''):
            info_str += """\n\n*Power estimate from monitoring site(s)*"""
        else:
            info_str += """\n\n*Power estimate from numerical model*"""

        info = info_str

    return info


@callback(
    Output("dl_power_lakes", "data"),
    Input("dl_btn_power_lakes", "n_clicks"),
    State('lake_id', 'data'),
    State('powers_obj_lakes', 'data'),
    State('sites_powers_obj_lakes', 'data'),
    State('indicator_lakes', 'value'),
    State('time_period_lakes', 'value'),
    State('freq_lakes', 'value'),
    prevent_initial_call=True,
    )
def download_power(n_clicks, lake_id, powers_obj, sites_powers_obj, indicator, n_years, n_samples_year):

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
