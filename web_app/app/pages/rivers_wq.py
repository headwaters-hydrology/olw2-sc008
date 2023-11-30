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
import hdf5plugin
import geobuf
from dash_iconify import DashIconify

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
    path='/rivers-wq',
    title='Water Quality Reaches',
    name='rivers_wq',
    description='River Water Quality Reaches'
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

draw_marae = assign("""function(feature, latlng){
const flag = L.icon({iconUrl: '/assets/nzta-marae.svg', iconSize: [20, 30]});
return L.marker(latlng, {icon: flag});
}""", name='rivers_marae_handle')


# catch_id = 3076139
# catch_id = 10007979

###############################################
### Initial processing

with booklet.open(param.rivers_reach_gbuf_path, 'r') as f:
    catches = [int(c) for c in f]

catches.sort()
indicators = list(param.rivers_indicator_dict.keys())
indicators.remove('NH')
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
                                    dmc.Text(id='catch_name', weight=700, style={'margin-top': 10}),
                                    # dcc.Dropdown(options=[{'label': d, 'value': d} for d in catches], id='catch_id', optionHeight=40, clearable=False,
                                    #               style={'margin-top': 10}
                                    #               ),
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
                                    id="loading-2",
                                    type="default",
                                    children=[dmc.Anchor(dmc.Button('Download default layer'), href='', id='dl_poly', style={'margin-top': 10})],
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
                                                id='upload_data_rivers',
                                                children=dmc.Button('Upload modified layer',
                                                ),
                                                style={
                                                    'margin-top': 10
                                                },
                                                multiple=False
                                            ),
                                            dcc.Markdown('', style={
                                                'margin-top': 10,
                                                'textAlign': 'left',
                                                            }, id='upload_error_text'),
                                            ]
                                        ),
                                    html.Label('(2c) Process the modified Land Mitigation Layer and route the improvements downstream:', style={
                                        'margin-top': 20
                                    }
                                        ),
                                    dcc.Loading(
                                    id="loading-1",
                                    type="default",
                                    children=html.Div([dmc.Button('Process improvements', id='process_reductions_rivers',
                                                                  n_clicks=0),
                                                        html.Div(id='process_text', style={'margin-top': 10})],
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
                                    dcc.Dropdown(options=[{'label': param.rivers_indicator_dict[d], 'value': d} for d in indicators], id='indicator_rivers', optionHeight=40, clearable=False),
                                    dmc.Text('(3b) Select sampling duration (years):', style={'margin-top': 20}),
                                    dmc.SegmentedControl(data=[{'label': d, 'value': str(d)} for d in param.rivers_time_periods],
                                                         id='time_period',
                                                         value='5',
                                                         fullWidth=True,
                                                         color=1,
                                                         ),
                                    dmc.Text('(3c) Select sampling frequency:', style={'margin-top': 20}),
                                    dmc.SegmentedControl(data=[{'label': v, 'value': str(k)} for k, v in param.rivers_freq_mapping.items()],
                                                         id='freq',
                                                         value='12',
                                                         fullWidth=True,
                                                         color=1
                                                         ),
                                    dmc.HoverCard(
                                        withArrow=True,
                                        width=param.hovercard_width,
                                        shadow="md",
                                        openDelay=param.hovercard_open_delay,
                                        children=[
                                            dmc.HoverCardTarget(html.Label('(3d) Select the percentage of the maximum contaminant loss reductions applied: ❓', style={'margin-top': 20})),
                                            dmc.HoverCardDropdown(
                                                dmc.Text(
                                                    """
                                                    The percentage selected will be applied to the contaminant loss reductions specified in the (default or customised) Land Mitigation Layer across the entire catchment. E.g. if 50% is selected, the reductions in the Land Mitigation Layer will be halved; if 100% is selected the reductions will not be altered.
                                                    """,
                                                    size="sm",
                                                )
                                            ),
                                        ],
                                    ),
                                    # dmc.Group([
                                    #     html.Label('(3d) Select the percentage of the maximum contaminant loss reductions applied:'),
                                    #     dmc.HoverCard(
                                    #         withArrow=True,
                                    #         width=param.hovercard_width,
                                    #         shadow="md",
                                    #         children=[
                                    #             dmc.HoverCardTarget(DashIconify(icon="material-symbols:help", width=30)),
                                    #             dmc.HoverCardDropdown(
                                    #                 dmc.Text(
                                    #                     """
                                    #                     The percentage selected will be applied to the contaminant loss reductions specified in the (default or customised) Land Mitigation Layer across the entire catchment. E.g. if 50% is selected, the reductions in the Land Mitigation Layer will be halved; if 100% is selected the reductions will not be altered. Therefore, 0% represents no mitigation and 100% represents the maximum “mitigatable” contaminant loss reduction.
                                    #                     """,
                                    #                     size="sm",
                                    #                 )
                                    #             ),
                                    #         ],
                                    #     ),
                                    #     ],
                                    #     style={'margin-top': 20},
                                    #     noWrap=True
                                    # ),
                                    dmc.Slider(id='Reductions_slider',
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
                            dl.Map(center=param.center, zoom=param.zoom, children=[
                                dl.LayersControl([
                                    dl.BaseLayer(dl.TileLayer(attribution=param.attribution, opacity=0.7), checked=True, name='OpenStreetMap'),
                                    dl.BaseLayer(dl.TileLayer(url='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', attribution='Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap (CC-BY-SA)', opacity=0.6), checked=False, name='OpenTopoMap'),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(url=str(param.rivers_catch_pbf_path), format="geobuf", id='catch_map', zoomToBoundsOnClick=True, zoomToBounds=False, options=dict(style=catch_style_handle))), name='Catchments', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='marae_map', zoomToBoundsOnClick=False, zoomToBounds=False, options=dict(pointToLayer=draw_marae))), name='Marae', checked=False),
                                    # dl.GeoJSON(url='', format="geobuf", id='base_reach_map', options=dict(style=base_reaches_style_handle)),

                                    # dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='sites_points', options=dict(pointToLayer=sites_points_handle), hideout={'circleOptions': dict(fillOpacity=1, stroke=False, radius=5, color='black')})), name='Monitoring sites', checked=True),
                                    # dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='reductions_poly')), name='Land use reductions', checked=False),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='reach_map', options=dict(style=base_reach_style_handle), hideout={}, hoverStyle=arrow_function(dict(weight=10, color='black', dashArray='')))), name='Rivers', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='sites_points', options=dict(pointToLayer=sites_points_handle), hideout=param.rivers_points_hideout)), name='Monitoring sites', checked=True),
                                    ],
                                    id='layers_rivers'
                                    ),
                                gc.colorbar_power,
                                # html.Div(id='colorbar', children=colorbar_base),
                                # dmc.Group(id='colorbar', children=colorbar_base),
                                dcc.Markdown(id="info", className="info", style={"position": "absolute", "top": "10px", "right": "160px", "z-index": "1000"})
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
            dcc.Store(id='catch_id', data=''),
            dcc.Store(id='powers_obj', data=''),
            dcc.Store(id='sites_powers_obj', data=''),
            dcc.Store(id='reaches_obj', data=''),
            dcc.Store(id='custom_reductions_obj', data=''),
            dcc.Store(id='base_reductions_obj', data=''),
            ]
        )

    return layout


###############################################
### Callbacks

@callback(
    Output('catch_id', 'data'),
    [Input('catch_map', 'click_feature')]
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
    Output('catch_name', 'children'),
    [Input('catch_id', 'data')]
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
        Output('reach_map', 'data'),
        Input('catch_id', 'data'),
        )
def update_reaches(catch_id):
    if catch_id != '':
        with booklet.open(param.rivers_reach_gbuf_path, 'r') as f:
            data = base64.b64encode(f[int(catch_id)]).decode()

    else:
        data = ''

    return data


@callback(
        Output('sites_points', 'data'),
        Input('catch_id', 'data'),
        )
def update_monitor_sites(catch_id):
    if catch_id != '':
        with booklet.open(param.rivers_sites_3rd_path, 'r') as f:
            data = base64.b64encode(f[int(catch_id)]).decode()

    else:
        data = ''

    return data


@callback(
        Output('marae_map', 'data'),
        Input('catch_id', 'data'),
        )
def update_marae(catch_id):
    if catch_id != '':
        with booklet.open(param.rivers_marae_path, 'r') as f:
            data = base64.b64encode(f[int(catch_id)]).decode()

    else:
        data = ''

    return data


@callback(
        Output('base_reductions_obj', 'data'),
        Input('catch_id', 'data'),
        prevent_initial_call=True
        )
def update_base_reductions_obj(catch_id):
    data = ''

    if catch_id != '':
        with booklet.open(param.rivers_lc_clean_path, 'r') as f:
            data = utils.encode_obj(f[int(catch_id)])

    return data


@callback(
    Output("dl_poly", "href"),
    # Input('indicator_lc', 'value'),
    Input('catch_id', 'data'),
    prevent_initial_call=True,
    )
def download_catch_lc(catch_id):

    if catch_id != '':
        url = param.rivers_catch_lc_gpkg_str.format(base_url=param.base_data_url, catch_id=catch_id)

        return url


@callback(
        Output('custom_reductions_obj', 'data'), Output('upload_error_text', 'children'),
        Input('upload_data_rivers', 'contents'),
        State('upload_data_rivers', 'filename'),
        State('catch_id', 'data'),
        prevent_initial_call=True
        )
def update_land_reductions(contents, filename, catch_id):
    data = None
    error_text = ''

    if catch_id != '':
        if contents is not None:
            data = utils.parse_gis_file(contents, filename)

            if isinstance(data, list):
                error_text = data[0]
                data = None
            else:
                error_text = 'Upload successful'
    else:
        error_text = 'You need to select a catchment before uploading a file. Please refresh the page and start from step (1).'

    return data, error_text


@callback(
    Output('reaches_obj', 'data'), Output('process_text', 'children'),
    Input('process_reductions_rivers', 'n_clicks'),
    Input('base_reductions_obj', 'data'),
    State('catch_id', 'data'),
    State('custom_reductions_obj', 'data'),
    prevent_initial_call=True)
def update_reach_reductions(click, base_reductions_obj, catch_id, reductions_obj):
    """

    """
    trig = ctx.triggered_id

    if (trig == 'process_reductions_rivers'):
        if (catch_id != '') and (reductions_obj != ''):
            red1 = xr.open_dataset(param.rivers_reductions_model_path, engine='h5netcdf')

            with booklet.open(param.rivers_reach_mapping_path) as f:
                branches = f[int(catch_id)][int(catch_id)]

            base_props = red1.sel(nzsegment=branches)

            new_reductions = utils.decode_obj(reductions_obj)
            base_reductions = utils.decode_obj(base_reductions_obj)

            diff_cols = utils.diff_reductions(new_reductions, base_reductions, param.rivers_lc_params)

            if diff_cols:
                new_props = utils.calc_river_reach_reductions(catch_id, new_reductions, base_reductions, diff_cols)
                new_props1 = new_props.combine_first(base_props).sortby('nzsegment').copy().load()
                red1.close()
                del red1
                base_props.close()
                del base_props

                data = utils.encode_obj(new_props1)
                text_out = 'Routing complete'
            else:
                data = utils.set_default_rivers_reach_reductions(catch_id)
                text_out = 'The improvements values are identical to the originals. Either skip this step, or modify the improvements values.'
        elif catch_id != '':
            data = utils.set_default_rivers_reach_reductions(catch_id)
            text_out = 'Please upload a Land Mitigation file in step (2b)'
        else:
            data = ''
            text_out = 'Please select a catchment before proceeding'
    else:
        if catch_id != '':
            # print('trigger')
            data = utils.set_default_rivers_reach_reductions(catch_id)
            text_out = ''
        else:
            data = ''
            text_out = 'Please select a catchment before proceeding'

    return data, text_out


@callback(
    Output('powers_obj', 'data'),
    Output('sites_powers_obj', 'data'),
    [Input('reaches_obj', 'data'), Input('indicator_rivers', 'value'), Input('time_period', 'value'), Input('freq', 'value'), Input('Reductions_slider', 'value')],
    [State('catch_id', 'data')]
    )
def update_powers_data(reaches_obj, indicator, n_years, n_samples_year, prop_red, catch_id):
    """

    """
    if (reaches_obj != '') and (reaches_obj is not None) and isinstance(indicator, str):
        # print('triggered')

        ind_name = param.rivers_indicator_dict[indicator]

        ## Modelled
        props = utils.decode_obj(reaches_obj)[[ind_name]].sel(reduction_perc=prop_red, drop=True).rename({ind_name: 'reduction'})

        n_samples = int(n_samples_year)*int(n_years)

        power_data = xr.open_dataset(param.rivers_power_model_path, engine='h5netcdf')

        with booklet.open(param.rivers_reach_mapping_path) as f:
            branches = f[int(catch_id)][int(catch_id)]

        power_data1 = power_data.sel(indicator=indicator, nzsegment=branches, n_samples=n_samples, drop=True).copy().load().sortby('nzsegment')
        power_data.close()
        del power_data

        conc_perc = (100 - props.reduction).round().astype('int8')

        # if indicator in ['BD']:
        #     conc_perc = (((conc_perc*0.01)**0.76) * 100).astype('int8')

        new_powers = props.assign(power_modelled=(('nzsegment'), power_data1.sel(conc_perc=conc_perc).power.values.astype('int8')))
        new_powers['nzsegment'] = new_powers['nzsegment'].astype('int32')
        new_powers['reduction'] = new_powers['reduction'].astype('int8')

        power_model_encoded = utils.encode_obj(new_powers)

        ## Monitored
        power_data = xr.open_dataset(param.rivers_power_moni_path, engine='h5netcdf')

        with booklet.open(param.rivers_sites_3rd_path, 'r') as f:
            sites = f[int(catch_id)]

        features = geobuf.decode(sites)['features']

        if len(features) > 0:
            sites_data = {f1['properties']['nzsegment']: f1['id'] for f1 in features}

            power_data1 = power_data.sel(indicator=indicator, n_samples=n_samples, drop=True).copy().load()
            power_nzsegments = power_data1.nzsegment.values

            segs = np.array(list(sites_data.keys()), dtype='int32')

            # other_segs = segs[~np.isin(segs, conc_perc.nzsegment.values)]
            # print(other_segs)

            conc_perc1 = conc_perc.sel(nzsegment=segs)

            power_data2 = []
            for nzsegment, site_name in sites_data.items():
                conc_perc1 = int(conc_perc.sel(nzsegment=nzsegment))
                if nzsegment in power_nzsegments:
                    try:
                        power = int(power_data1.sel(conc_perc=conc_perc1, nzsegment=nzsegment).power.values)
                    except ValueError:
                        power = -1
                else:
                    power = -1
                power_data2.append({'reduction': 100 - conc_perc1, 'nzsegment': nzsegment, 'power_monitored': power, 'site_name': site_name})

            power_data.close()
            del power_data

            # print(power_data2)

            power_moni_encoded = utils.encode_obj(power_data2)

        else:
            power_moni_encoded = ''

        # power_moni_encoded = ''

    else:
        power_model_encoded = ''
        power_moni_encoded = ''

    return power_model_encoded, power_moni_encoded


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
    Output('reach_map', 'options'),
    Output('sites_points', 'hideout'),
    Input('powers_obj', 'data'),
    Input('sites_powers_obj', 'data'),
    prevent_initial_call=True
    )
def update_hideout(powers_obj, sites_powers_obj):
    """

    """
    hideout_model = {}
    options = dict(style=base_reach_style_handle)
    hideout_moni = param.rivers_points_hideout

    if (powers_obj != '') and (powers_obj is not None):
        props = utils.decode_obj(powers_obj)

        ## Modelled
        color_arr = pd.cut(props.power_modelled.values, param.bins, labels=param.colorscale_power, right=False).tolist()

        hideout_model = {'colorscale': color_arr, 'classes': props.nzsegment.values, 'style': param.style_power, 'colorProp': 'nzsegment'}
        options = dict(style=reach_style_handle)

        del props

        ## Monitored
        if sites_powers_obj != '':
            sites_props = utils.decode_obj(sites_powers_obj)
            # print(props_moni)
            color_arr2 = pd.cut([p['power_monitored'] for p in sites_props], param.bins, labels=param.colorscale_power, right=False).tolist()
            color_arr2 = [color if isinstance(color, str) else '#252525' for color in color_arr2]
            # print(color_arr2)

            hideout_moni = {'classes': [p['nzsegment'] for p in sites_props], 'colorscale': color_arr2, 'circleOptions': dict(fillOpacity=1, stroke=True, color='black', weight=1, radius=param.site_point_radius), 'colorProp': 'nzsegment'}

    return hideout_model, options, hideout_moni


@callback(
    Output("info", "children"),
    Input('powers_obj', 'data'),
    Input('sites_powers_obj', 'data'),
    Input("reach_map", "click_feature"),
    Input('sites_points', 'click_feature'),
    Input('catch_id', 'data'),
    State("info", "children"),
    prevent_initial_call=True
    )
def update_map_info(powers_obj, sites_powers_obj, reach_feature, sites_feature, catch_id, old_info):
    """

    """
    info = """"""

    trig = ctx.triggered_id
    # print(trig)

    if trig == 'catch_id':
        pass

    elif (trig == 'reach_map'):

        props = utils.decode_obj(powers_obj)
        # print(reach_feature)
        # print(sites_feature)

        feature_id = int(reach_feature['id'])

        reach_data = props.sel(nzsegment=feature_id)

        info_str = """\n\n**nzsegment**: {seg}\n\n**Predicted improvement**: {red}%\n\n**Likelihood of observing the improvement (power)**: {t_stat}%""".format(red=int(reach_data.reduction), t_stat=int(reach_data.power_modelled), seg=feature_id)

        info = info_str

    elif (trig == 'sites_points') or ((sites_powers_obj != '') and (sites_feature is not None) and ('Site name' in old_info)):
        if (sites_powers_obj != ''):
            sites_props = utils.decode_obj(sites_powers_obj)
            feature_id = int(sites_feature['properties']['nzsegment'])
            # print(sites_feature)

            reach_data = [p for p in sites_props if p['nzsegment'] == feature_id]
            if reach_data:
                reach_data0 = reach_data[0]
                power = reach_data0['power_monitored']
                if power == -1:
                    power = 'NA'
                else:
                    power = str(power) + '%'

                reduction = reach_data0['reduction']
                site_name = reach_data0['site_name']

                info_str = """\n\n**nzsegment**: {seg}\n\n**Site name**: {site}\n\n**Predicted improvement**: {red}%\n\n**Likelihood of detecting the improvement (power)**: {power}""".format(red=reduction, power=power, seg=feature_id, site=site_name)

                info = info_str

    elif (trig == 'powers_obj') and (powers_obj != '') and (reach_feature is not None) and ('nzsegment' in old_info):
        # print(reach_feature)
        props = utils.decode_obj(powers_obj)
        feature_id = int(reach_feature['id'])
        reach_data = props.sel(nzsegment=feature_id)

        info_str = """\n\n**nzsegment**: {seg}\n\n**Predicted improvement**: {red}%\n\n**Likelihood of detecting the improvement (power)**: {t_stat}%""".format(red=int(reach_data.reduction), t_stat=int(reach_data.power_modelled), seg=feature_id)

        info = info_str

    return info


@callback(
    Output("dl_power_rivers", "data"),
    Input("dl_btn_power_rivers", "n_clicks"),
    State('catch_id', 'data'),
    State('powers_obj', 'data'),
    State('sites_powers_obj', 'data'),
    State('indicator_rivers', 'value'),
    State('time_period', 'value'),
    State('freq', 'value'),
    prevent_initial_call=True,
    )
def download_power(n_clicks, catch_id, powers_obj, sites_powers_obj, indicator, n_years, n_samples_year):

    if (catch_id != '') and (powers_obj != '') and (powers_obj is not None):
        power_data = utils.decode_obj(powers_obj)

        df1 = power_data.to_dataframe().reset_index()
        df1['indicator'] = param.rivers_indicator_dict[indicator]
        df1['n_years'] = n_years
        df1['n_samples_per_year'] = n_samples_year

        if sites_powers_obj != '':
            sites_power_data = utils.decode_obj(sites_powers_obj)
            sites_df = pd.DataFrame.from_dict(sites_power_data)
            sites_df.loc[sites_df.power_monitored < 0, 'power_monitored'] = np.nan
            # sites_df['indicator'] = param.rivers_indicator_dict[indicator]
            df1 = pd.merge(df1, sites_df.drop('reduction', axis=1), on=['nzsegment'], how='left')

        df2 = df1.set_index(['indicator', 'n_years', 'n_samples_per_year', 'nzsegment']).sort_index()

        return dcc.send_data_frame(df2.to_csv, f"river_power_{catch_id}.csv")






























