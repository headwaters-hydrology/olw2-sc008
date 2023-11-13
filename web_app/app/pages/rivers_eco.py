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
    path='/rivers-eco',
    title='Ecology Reaches',
    name='rivers_eco',
    description='River Ecology Reaches'
)

## Handles
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

reach_style_handle = assign("""function style2(feature, context){
    const {classes, colorscale, style, colorProp} = context.props.hideout;  // get props from hideout
    const value = feature.properties[colorProp];  // get value the determines the color
    for (let i = 0; i < classes.length; ++i) {
        if (value == classes[i]) {
            style.color = colorscale[i];  // set the fill color according to the class
        }
    }
    return style;
}""", name='eco_reach_style_handle')

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
}""", name='eco_marae_handle')


# catch_id = 3076139

###############################################
### Initial processing

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
                                # html.H5('(1) Catchment selection', style={'font-weight': 'bold'}),
                                dmc.AccordionControl('(1) Catchment Selection', style={'font-size': 18}),
                                dmc.AccordionPanel([

                                    html.Label('(1a) Select a catchment on the map:'),
                                    dmc.Text(id='catch_name_eco', weight=700, style={'margin-top': 10}),
                                    ]
                                    )
                                ],
                                value='1'
                                ),

                            dmc.AccordionItem([
                                dmc.AccordionControl('(2 - Optional) Customise Improvements Layer', style={'font-size': 18}),
                                dmc.AccordionPanel([
                                    html.Label('(2a) Download improvements polygons as GPKG:'),
                                    dcc.Loading(
                                    id="loading-2",
                                    type="default",
                                    children=[dmc.Anchor(dmc.Button('Download land cover'), href='', id='dl_poly_eco', style={'margin-top': 10})],
                                    ),
                                    html.Label('NOTE: Only modify existing values. Do not add additional columns; they will be ignored.', style={
                                        'margin-top': 10
                                    }
                                        ),
                                    html.Label('(2b) Upload modified improvements polygons as GPKG:', style={
                                        'margin-top': 20
                                    }
                                        ),
                                    dcc.Loading(
                                        children=[
                                            dcc.Upload(
                                                id='upload_data_rivers_eco',
                                                children=dmc.Button('Upload improvements',
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
                                        'margin-top': 10,
                                        'textAlign': 'left',
                                                    }, id='upload_error_text_eco'),
                                    html.Label('(2c) Process the improvements layer and route the improvements downstream:', style={
                                        'margin-top': 20
                                    }
                                        ),
                                    dcc.Loading(
                                    id="loading-1",
                                    type="default",
                                    children=html.Div([dmc.Button('Process improvements', id='process_reductions_rivers_eco',
                                                                  # className="me-1",
                                                                  n_clicks=0),
                                                        html.Div(id='process_text_eco', style={'margin-top': 10})],
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
                                    dcc.Dropdown(options=[{'label': param.eco_indicator_dict[d], 'value': d} for d in indicators], id='indicator_eco', optionHeight=40, clearable=False, style={'margin-bottom': 20}),
                                    html.Label('(3b) Select a percent improvement for the overall catchment:'),
                                    dmc.Slider(id='reductions_slider_eco',
                                               value=30,
                                               mb=35,
                                               step=10,
                                               min=10,
                                               max=100,
                                               showLabelOnHover=True,
                                               disabled=False,
                                               marks=param.eco_reductions_options
                                               ),
                                    dmc.Text('(3c) Select sampling duration (years):', style={'margin-top': 20}),
                                    dmc.SegmentedControl(data=[{'label': d, 'value': str(d)} for d in param.eco_time_periods],
                                                         id='time_period_eco',
                                                         value='5',
                                                         fullWidth=True,
                                                         color=1,
                                                         ),
                                    # dmc.Text('(3c) Select sampling frequency (monitoring site power):', style={'margin-top': 20}),
                                    # dmc.SegmentedControl(data=[{'label': v, 'value': str(k)} for k, v in freq_mapping.items()],
                                    #                       id='freq_eco',
                                    #                       value='12',
                                    #                       fullWidth=True,
                                    #                       color=1
                                    #                       ),
                                    dmc.Text('(3d) Select the number of sites per catchment:', style={'margin-top': 20}),
                                    dmc.SegmentedControl(data=[{'label': k, 'value': str(k)} for k in param.eco_n_sites],
                                                          id='n_sites_eco',
                                                          value='10',
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
                                    # html.Label('Select sampling frequency:'),
                                    # dcc.Dropdown(options=[{'label': v, 'value': k} for k, v in freq_mapping.items()], id='freq', clearable=False, value=12),

                                    ],
                                    )
                                ],
                                value='3'
                                ),

                            # dmc.AccordionItem([
                            #     dmc.AccordionControl('(4) Download Results', style={'font-size': 18}),
                            #     dmc.AccordionPanel([
                            #         dmc.Text('(4a) Download power results given the prior query options (csv):'),
                            #         dcc.Loading(
                            #         type="default",
                            #         children=[html.Div(dmc.Button("Download power results", id='dl_btn_power_eco'), style={'margin-bottom': 20, 'margin-top': 10}),
                            # dcc.Download(id="dl_power_eco")],
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
                            ),
                        dcc.Markdown("""* The rivers colored with *Low*, *Moderate*, and *High* are the qualitative monitoring priorities as there  is too much uncertainty in estimating the powers per reach.""")]
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
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(url=str(param.rivers_catch_pbf_path), format="geobuf", id='catch_map_eco', zoomToBoundsOnClick=True, zoomToBounds=False, options=dict(style=catch_style_handle))), name='Catchments', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='marae_map_eco', zoomToBoundsOnClick=False, zoomToBounds=False, options=dict(pointToLayer=draw_marae))), name='Marae', checked=False),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='reach_map_eco', options=dict(style=base_reach_style_handle), hideout={})), name='Rivers', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='sites_points_eco', options=dict(pointToLayer=sites_points_handle), hideout=param.rivers_points_hideout)), name='Monitoring sites', checked=True),
                                    ],
                                    id='layers_eco'
                                    ),
                                gc.colorbar_weights,
                                dcc.Markdown(id="info_eco", className="info", style={"position": "absolute", "top": "10px", "right": "160px", "z-index": "1000"})
                                ],
                                style={'width': '100%', 'height': param.map_height, 'margin': "auto", "display": "block"}
                                ),

                            ],
                            # className='five columns', style={'margin': 10}
                            ),
                        ),
                    ]
                    ),
            dcc.Store(id='catch_id_eco', data=''),
            dcc.Store(id='reaches_obj_eco', data=''),
            dcc.Store(id='reaches_weights_obj_eco', data=''),
            dcc.Store(id='catch_power_obj_eco', data=''),
            dcc.Store(id='custom_reductions_obj_eco', data=''),
            dcc.Store(id='base_reductions_obj_eco', data=''),
            ]
        )

    return layout


###############################################
### Callbacks

@callback(
    Output('catch_id_eco', 'data'),
    [Input('catch_map_eco', 'click_feature')]
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
    Output('catch_name_eco', 'children'),
    [Input('catch_id_eco', 'data')]
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
        Output('reach_map_eco', 'data'),
        Input('catch_id_eco', 'data'),
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
        Output('sites_points_eco', 'data'),
        Input('catch_id_eco', 'data'),
        )
# @cache.memoize()
def update_monitor_sites(catch_id):
    if catch_id != '':
        with booklet.open(param.eco_sites_path, 'r') as f:
            data = base64.b64encode(f[int(catch_id)]).decode()

    else:
        data = ''

    return data


@callback(
        Output('marae_map_eco', 'data'),
        Input('catch_id_eco', 'data'),
        )
def update_marae(catch_id):
    if catch_id != '':
        with booklet.open(param.rivers_marae_path, 'r') as f:
            data = base64.b64encode(f[int(catch_id)]).decode()

    else:
        data = ''

    return data


@callback(
        Output('base_reductions_obj_eco', 'data'),
        Input('catch_id_eco', 'data'),
        prevent_initial_call=True
        )
def update_base_reductions_obj(catch_id):
    data = ''

    if catch_id != '':
        with booklet.open(param.rivers_lc_clean_path, 'r') as f:
            data = utils.encode_obj(f[int(catch_id)])

    return data


@callback(
    Output("dl_poly_eco", "href"),
    # Input('indicator_lc', 'value'),
    Input('catch_id_eco', 'data'),
    prevent_initial_call=True,
    )
def download_catch_lc(catch_id):

    if catch_id != '':
        url = param.rivers_catch_lc_gpkg_str.format(base_url=param.base_data_url, catch_id=catch_id)

        return url


@callback(
        Output('custom_reductions_obj_eco', 'data'), Output('upload_error_text_eco', 'children'),
        Input('upload_data_rivers_eco', 'contents'),
        State('upload_data_rivers_eco', 'filename'),
        State('catch_id_eco', 'data'),
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
                error_text = 'Upload sucessful'
    else:
        error_text = 'You need to select a catchment before uploading a file. Please refresh the page and start from step (1).'

    return data, error_text


@callback(
    Output('reaches_obj_eco', 'data'), Output('process_text_eco', 'children'),
    Input('process_reductions_rivers_eco', 'n_clicks'),
    Input('base_reductions_obj_eco', 'data'),
    [
      State('catch_id_eco', 'data'),
      State('custom_reductions_obj_eco', 'data'),
      ],
    prevent_initial_call=True)
def update_reach_reductions(click, base_reductions_obj, catch_id, reductions_obj):
    """

    """
    trig = ctx.triggered_id

    if (trig == 'process_reductions_rivers_eco'):
        if (catch_id != '') and (reductions_obj != '') and (reductions_obj is not None):
            new_reductions = utils.decode_obj(reductions_obj)
            base_reductions = utils.decode_obj(base_reductions_obj)

            new_props1 = utils.calc_river_reach_eco_weights(catch_id, new_reductions, base_reductions)

            data = utils.encode_obj(new_props1)
            text_out = 'Routing complete'
        elif catch_id != '':
            data = utils.set_default_eco_reach_weights(catch_id)
            text_out = 'Please upload a polygon improvements file in step (2b)'
        else:
            data = ''
            text_out = 'Please select a catchment before proceding'
    else:
        if catch_id != '':
            # print('trigger')
            data = utils.set_default_eco_reach_weights(catch_id)
            text_out = ''
        else:
            data = ''
            text_out = 'Please select a catchment before proceding'

    return data, text_out


@callback(
    Output('reaches_weights_obj_eco', 'data'),
    Input('reaches_obj_eco', 'data'),
    Input('reductions_slider_eco', 'value'),
    Input('indicator_eco', 'value'),
    State('catch_id_eco', 'data'),
    prevent_initial_call=True
    )
def update_reaches_obj(reaches_obj, reductions, indicator, catch_id):
    """

    """
    if (reaches_obj != '') and (reaches_obj is not None) and isinstance(reductions, (str, int)) and isinstance(indicator, str) and (catch_id != ''):
        reach_weights0 = utils.decode_obj(reaches_obj)
        reach_weights1 = reach_weights0.sel(reduction_perc=int(reductions), drop=True).rename({indicator: 'weights'}).load().copy()
        reach_weights0.close()
        del reach_weights0

        data = utils.encode_obj(reach_weights1)
        return data
    else:
        raise dash.exceptions.PreventUpdate


@callback(
    Output('catch_power_obj_eco', 'data'),
    [Input('reductions_slider_eco', 'value'), Input('indicator_eco', 'value'), Input('time_period_eco', 'value'), Input('n_sites_eco', 'value')],
    [State('catch_id_eco', 'data')],
    prevent_initial_call=True
    )
def update_catch_power_obj(reductions, indicator, n_years, n_sites, catch_id):
    """

    """
    if isinstance(reductions, (str, int)) and isinstance(n_years, str) and isinstance(indicator, str) and isinstance(n_sites, str) and (catch_id != ''):
        n_samples = int(n_sites)*int(n_years)

        power_data = xr.open_dataset(param.eco_power_catch_path, engine='h5netcdf')
        power_data1 = int(power_data.sel(nzsegment=int(catch_id), indicator=indicator, n_samples=n_samples, conc_perc=100-int(reductions), drop=True).power.values)
        power_data.close()
        del power_data

        data = str(power_data1)
        return data
    else:
        raise dash.exceptions.PreventUpdate


@callback(
    Output('reach_map_eco', 'hideout'),
    Output('reach_map_eco', 'options'),
    Input('reaches_weights_obj_eco', 'data'),
    prevent_initial_call=True
    )
def update_reach_hideout(reaches_obj):
    """

    """
    if (reaches_obj != '') and (reaches_obj is not None):
        props = utils.decode_obj(reaches_obj)

        ## Modelled
        values = props.weights.values
        bins_weights = np.quantile(values, [0, 0.50, 0.75, 1])
        bins_weights[-1] += 0.01
        color_arr = pd.cut(values, bins_weights, labels=param.colorscale_weights, right=False).tolist()

        hideout_model = {'colorscale': color_arr, 'classes': props.nzsegment.values.astype(int), 'style': param.style_power, 'colorProp': 'nzsegment'}
        options = dict(style=reach_style_handle)

    else:
        hideout_model = {}
        options = dict(style=base_reach_style_handle)

    return hideout_model, options


@callback(
    Output("info_eco", "children"),
    [
      Input('catch_power_obj_eco', 'data'),
      ],
    [State("info_eco", "children"),
     State('n_sites_eco', 'value')
     ]
    )
def update_map_info(catch_power_obj, old_info, n_sites):
    """

    """
    info = """"""

    if (catch_power_obj != '') and (catch_power_obj is not None):

        catch_power = int(catch_power_obj)

        info += """##### Catchment:

            \n\n**Likelihood of observing an improvement (power)**: {power}%\n\n**Number of sites**: {n_sites}\n\n""".format(power = catch_power, n_sites=n_sites)

    if info == """""":
        info = old_info

    return info


# @callback(
#     Output("dl_power_eco", "data"),
#     Input("dl_btn_power_eco", "n_clicks"),
#     State('catch_id_eco', 'data'),
#     State('indicator_eco', 'value'),
#     State('time_period_eco', 'value'),
#     prevent_initial_call=True,
#     )
# def download_power(n_clicks, catch_id, powers_obj, indicator, n_years):

#     if (catch_id != '') and (powers_obj != '') and (powers_obj is not None) and isinstance(n_samples_year, str):
#         power_data = decode_obj(powers_obj)

#         df1 = power_data.to_dataframe().reset_index()
#         df1['indicator'] = eco_indicator_dict[indicator]
#         df1['n_years'] = n_years
#         df1['n_samples_per_year'] = n_samples_year

#         df2 = df1.set_index(['indicator', 'n_years', 'n_samples_per_year', 'nzsegment']).sort_index()

#         return dcc.send_data_frame(df2.to_csv, f"ecoology_sites_power_{catch_id}.csv")
