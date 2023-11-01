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
    title='Water Quality',
    name='lakes_wq',
    description='Lakes and Lagoons Water Quality'
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
}""", name='rivers_catch_style_handle')

# lake_id = 48177

###############################################
### Initial processing

with open(param.assets_path.joinpath('lakes_points.pbf'), 'rb') as f:
    geodict = geobuf.decode(f.read())

lakes_names = {}
for f in geodict['features']:
    label0 = ' '.join(f['properties']['name'].split())
    label = str(f['id']) + ' - ' + label0
    lakes_names[int(f['id'])] = label

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
                                dmc.AccordionControl('(2 - Optional) Customise Improvements Layer', style={'font-size': 18}),
                                dmc.AccordionPanel([
                                    html.Label('(2a) Download improvements polygons as GPKG:'),
                                    dcc.Loading(
                                    type="default",
                                    children=[dmc.Anchor(dmc.Button('Download land cover'), href='', id='dl_poly_lakes', style={'margin-top': 10})],
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
                                                id='upload_data_lakes',
                                                children=dmc.Button('Upload improvements',
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
                                                    }, id='upload_error_text_lakes'),
                                    html.Label('(2c) Process the improvements layer and route the improvements downstream:', style={
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
                                    dmc.Text('(3b) Select sampling length (years):', style={'margin-top': 20}),
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
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(url=str(param.lakes_pbf_path), format="geobuf", id='lake_points', zoomToBoundsOnClick=True, cluster=True)), name='Lake points', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='catch_map_lakes', zoomToBoundsOnClick=True, zoomToBounds=True, options=dict(style=catch_style_handle))), name='Catchments', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='reach_map_lakes', options=dict(style=param.reach_style))), name='Rivers', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='lake_poly', options=dict(style=lake_style_handle), hideout={'classes': [''], 'colorscale': ['#808080'], 'style': param.lake_style, 'colorProp': 'name'})), name='Lakes', checked=True),
                                    # dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='sites_points_lakes', options=dict(pointToLayer=sites_points_handle), hideout=rivers_points_hideout)), name='Monitoring sites', checked=True),
                                    ],
                                    id='layers_gw',
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
def update_catch_name(lake_id):
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


# @callback(
#         Output('sites_points', 'data'),
#         Input('lake_id', 'value'),
#         )
# # @cache.memoize()
# def update_monitor_sites(lake_id):
#     if (lake_id is not None):
#         with booklet.open(lakes_sites_path, 'r') as f:
#             data = base64.b64encode(f[int(lake_id)]).decode()

#     else:
#         data = ''

#     return data


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


# @callback(
#     Output("dl_poly_lakes", "data"),
#     Input("dl_btn_lakes", "n_clicks"),
#     State('lake_id', 'value'),
#     prevent_initial_call=True,
#     )
# def download_lc(n_clicks, lake_id):
#     if isinstance(lake_id, str):
#         path = lakes_catch_lc_dir.joinpath(lakes_catch_lc_gpkg_str.format(lake_id))

#         return dcc.send_file(path)


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
                error_text = 'Upload sucessful'
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
            text_out = 'Please upload a polygon improvements file in step (2b)'
        else:
            data = ''
            text_out = 'Please select a lake before proceding'
    else:
        if lake_id != '':
            # print('trigger')
            data = utils.set_default_lakes_reach_reductions(lake_id)
            text_out = ''
        else:
            data = ''
            text_out = 'Please select a lake before proceding'

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
    [Input('reaches_obj_lakes', 'data'), Input('indicator_lakes', 'value'), Input('time_period_lakes', 'value'), Input('freq_lakes', 'value'), Input('reductions_slider_lakes', 'value')],
    [State('lake_id', 'data')]
    )
def update_powers_data_lakes(reaches_obj, indicator, n_years, n_samples_year, prop_red, lake_id):
    """

    """
    if (reaches_obj != '') and (reaches_obj is not None) and isinstance(n_years, str) and isinstance(n_samples_year, str) and isinstance(indicator, str):
        ind_name = param.lakes_indicator_dict[indicator]

        props = int(utils.decode_obj(reaches_obj)[ind_name].sel(reduction_perc=prop_red, drop=True))
        # print(props)

        conc_perc = 100 - props

        lake_data = lakes_data[int(lake_id)]

        ## Lake residence time calcs
        if indicator in ['TP', 'CHLA', 'Secchi']:
            if lake_data['max_depth'] > 7.5:
                b = 1 + (0.44*(lake_data['residence_time']**0.13))
                conc_perc = int(round(((conc_perc*0.01)**(1/b)) * 100))
        elif indicator == 'TN':
            conc_perc = int(round(((conc_perc*0.01)**0.54) * 100))
        if indicator in ['CHLA', 'Secchi']:
            conc_perc = int(round(((conc_perc*0.01)**1.25) * 100))
        if indicator == 'Secchi':
            if lake_data['max_depth'] > 20:
                conc_perc = int(round(((conc_perc*0.01)**(0.9)) * 100))
            else:
                conc_perc = int(round(((conc_perc*0.01)**(0.9)) * 100))

        if conc_perc < 1:
            conc_perc = 1
        elif conc_perc > 100:
            conc_perc = 100

        ## Lookup power
        n_samples = int(n_samples_year)*int(n_years)

        power_data = xr.open_dataset(param.lakes_power_combo_path, engine='h5netcdf')
        try:
            power_data1 = power_data.sel(indicator=indicator, LFENZID=int(lake_id), n_samples=n_samples, conc_perc=conc_perc).copy().load()

            power_data2 = [int(power_data1.power_modelled.values), float(power_data1.power_monitored.values)]
        except:
            power_data2 = [0, np.nan]
        power_data.close()
        del power_data

        data = utils.encode_obj({'reduction': props, 'power': power_data2, 'lake_id': lake_id})
    else:
        data = ''

    return data


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
    [Input('powers_obj_lakes', 'data')],
    Input('lake_id', 'data'),
    prevent_initial_call=True
    )
def update_hideout_lakes(powers_obj, lake_id):
    """

    """
    if (powers_obj != '') and (powers_obj is not None):
        # print('trigger')
        props = utils.decode_obj(powers_obj)
        # print(props)
        # print(type(lake_id))

        if props['lake_id'] == lake_id:

            color_arr = pd.cut([props['power'][0]], param.bins, labels=param.colorscale_power, right=False).tolist()
            # print(color_arr)
            # print(props['lake_id'])

            hideout = {'classes': [props['lake_id']], 'colorscale': color_arr, 'style': param.lake_style, 'colorProp': 'LFENZID'}
        else:
            hideout = {'classes': [lake_id], 'colorscale': ['#808080'], 'style': param.lake_style, 'colorProp': 'LFENZID'}
    else:
        hideout = {'classes': [lake_id], 'colorscale': ['#808080'], 'style': param.lake_style, 'colorProp': 'LFENZID'}

    return hideout


@callback(
    Output("info_lakes", "children"),
    [Input('powers_obj_lakes', 'data'),
      Input("lake_poly", "click_feature")],
    )
def update_map_info_lakes(powers_obj, feature):
    """

    """
    info = """"""

    # if (reductions_obj != '') and (reductions_obj is not None) and ('reductions_poly' in map_checkboxes):
    #     info = info + """\n\nHover over the polygons to see reduction %"""

    if (powers_obj != '') and (powers_obj is not None):
        if feature is not None:
            props = utils.decode_obj(powers_obj)

            if np.isnan(props['power'][1]):
                moni1 = 'NA'
            else:
                moni1 = str(int(props['power'][1])) + '%'

            info_str = """\n\n**Improvement**: {red}%\n\n**Likelihood of observing an improvement (power)**:\n\n&nbsp;&nbsp;&nbsp;&nbsp;**Modelled**: {t_stat1}%\n\n&nbsp;&nbsp;&nbsp;&nbsp;**Monitored**: {t_stat2}""".format(red=int(props['reduction']), t_stat1=int(props['power'][0]), t_stat2=moni1)

            info = info + info_str

        else:
            info = info + """\n\nClick on a lake to see info"""

    return info


@callback(
    Output("dl_power_lakes", "data"),
    Input("dl_btn_power_lakes", "n_clicks"),
    State('lake_id', 'data'),
    State('powers_obj_lakes', 'data'),
    State('indicator_lakes', 'value'),
    State('time_period_lakes', 'value'),
    State('freq_lakes', 'value'),
    prevent_initial_call=True,
    )
def download_power(n_clicks, lake_id, powers_obj, indicator, n_years, n_samples_year):

    if (lake_id != '') and (powers_obj != '') and (powers_obj is not None):
        power_data = utils.decode_obj(powers_obj)

        df1 = pd.DataFrame([power_data['power']], columns=['modelled', 'monitored'])
        df1['indicator'] = param.lakes_indicator_dict[indicator]
        df1['n_years'] = n_years
        df1['n_samples_per_year'] = n_samples_year
        df1['LFENZID'] = int(lake_id)
        df1['improvement'] = power_data['reduction']

        df2 = df1.set_index(['indicator', 'n_years', 'n_samples_per_year', 'improvement', 'LFENZID']).sort_index()

        return dcc.send_data_frame(df2.to_csv, f"lake_power_{lake_id}.csv")
