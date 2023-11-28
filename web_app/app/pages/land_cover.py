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
import hdf5plugin

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
    path='/land-cover',
    title='Land Cover',
    name='land_cover',
    description='Land Cover'
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

lc_style_handle = assign("""function style2(feature, context){
    const {classes, colorscale, style, colorProp} = context.props.hideout;  // get props from hideout
    const value = feature.properties[colorProp];  // get value the determines the color
    for (let i = 0; i < classes.length; ++i) {
        if (value >= classes[i]) {
            style.fillColor = colorscale[i];  // set the fill color according to the class
        }
    }
    return style;
}""", name='rivers_lc_style_handle')

sites_points_handle = assign("""function rivers_sites_points_handle(feature, latlng, context){
    const {classes, colorscale, circleOptions, colorProp} = context.props.hideout;  // get props from hideout
    const value = feature.properties[colorProp];  // get value the determines the fillColor
    for (let i = 0; i < classes.length; ++i) {
        if (value == classes[i]) {
            circleOptions.fillColor = colorscale[i];  // set the color according to the class
        }
    }

    return L.circleMarker(latlng, circleOptions);
}""", name='rivers_lc_points_handle')

draw_marae = assign("""function(feature, latlng){
const flag = L.icon({iconUrl: '/assets/nzta-marae.svg', iconSize: [20, 30]});
return L.marker(latlng, {icon: flag});
}""", name='rivers_lc_marae_handle')

# catch_id = 3076139

###############################################
### Initial processing

with booklet.open(param.rivers_reach_gbuf_path, 'r') as f:
    catches = [int(c) for c in f]

catches.sort()
indicators = list(set(param.rivers_lc_param_mapping.values()))
indicators.sort()

lc_mapping_inverse = {value: key for key, value in param.rivers_lc_param_mapping.items()}

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
                                    dmc.Text(id='catch_name_lc', weight=700, style={'margin-top': 10}),
                                    ]
                                    )
                                ],
                                value='1'
                                ),

                            dmc.AccordionItem([
                                dmc.AccordionControl('(2) Query Options', style={'font-size': 18}),
                                dmc.AccordionPanel([
                                    dmc.Text('(2a) Select Indicator:'),
                                    dcc.Dropdown(options=[{'label': d.capitalize(), 'value': d} for d in indicators], id='indicator_lc', optionHeight=40, clearable=False),

                                    dmc.Text(dcc.Markdown('(2b) Change the percent of the improvements applied. 100% is the max realistic improvement (**This option only applies to the river reaches**):'), style={'margin-top': 20}),
                                    dmc.Slider(id='Reductions_slider_lc',
                                                value=100,
                                                mb=35,
                                                step=10,
                                                # min=10,
                                                showLabelOnHover=True,
                                                disabled=False,
                                                marks=param.marks
                                                ),
                                    dmc.Text('NOTE', weight=700, underline=True, style={'margin-top': 20}),
                                    dmc.Text('The river reaches can be added to the map via the layer button on the top right corner of the map, and the land cover can also be removed.')
                                    ],
                                    )
                                ],
                                value='2'
                                ),

                            dmc.AccordionItem([
                                dmc.AccordionControl('(3) Download Results', style={'font-size': 18}),
                                dmc.AccordionPanel([
                                    dmc.Text('(3a) Download land cover improvements for the selected catchment (gpkg):'),
                                    dcc.Loading(
                                    type="default",
                                    children=[
                            dmc.Anchor(dmc.Button('Download land cover'), href='', id='lc_dl1')],
                                    ),
                                    ],
                                    ),
                                dmc.AccordionPanel([
                                    dmc.Text('(3b) Download river reach improvements for the selected catchment (csv):'),
                                    dcc.Loading(
                                    type="default",
                                    children=[
                            dmc.Button('Download reaches', id='reach_dl_btn'),
                            dcc.Download(id='reach_dl1')],
                                    ),
                                    ],
                                    ),
                            #     dmc.AccordionPanel([
                            #         dmc.Text('(3c) Download land cover reductions for all NZ (gpkg):'),
                            #         dcc.Loading(
                            #         type="default",
                            #         children=[
                            # dmc.Anchor(dmc.Button('Download land cover NZ-wide'), href=lc_url, id='lc_dl2')],
                            #         ),
                            #         ],
                            #         ),
                            #     dmc.AccordionPanel([
                            #         dmc.Text('(3d) Download river reach reductions for all NZ (csv):'),
                            #         dcc.Loading(
                            #         type="default",
                            #         children=[
                            # dmc.Anchor(dmc.Button('Download reaches NZ-wide'), href=rivers_red_url, id='reach_dl2')],
                            #         ),
                            #         ],
                            #         )
                                ],
                                value='3'
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
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(url=str(param.rivers_catch_pbf_path), format="geobuf", id='catch_map_lc', zoomToBoundsOnClick=True, zoomToBounds=False, options=dict(style=catch_style_handle))), name='Catchments', checked=True),
                                    # dl.GeoJSON(url='', format="geobuf", id='base_reach_map', options=dict(style=base_reaches_style_handle)),

                                    # dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='sites_points', options=dict(pointToLayer=sites_points_handle), hideout={'circleOptions': dict(fillOpacity=1, stroke=False, radius=5, color='black')})), name='Monitoring sites', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='reductions_poly_lc', hoverStyle=arrow_function(dict(weight=5, color='#666', dashArray='')), options=dict(style=lc_style_handle), hideout={})), name='Land cover', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='marae_map_lc', zoomToBoundsOnClick=False, zoomToBounds=False, options=dict(pointToLayer=draw_marae))), name='Marae', checked=False),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='reach_map_lc', options={}, hideout={}, hoverStyle=arrow_function(dict(weight=8, color='black', dashArray='')))), name='Rivers', checked=False),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='sites_points_lc', options=dict(pointToLayer=sites_points_handle), hideout=param.rivers_points_hideout)), name='Monitoring sites', checked=False),
                                    ], id='layers_lc'),
                                gc.colorbar_power,
                                # html.Div(id='colorbar', children=colorbar_base),
                                # dmc.Group(id='colorbar', children=colorbar_base),
                                dcc.Markdown(id="info_lc", className="info", style={"position": "absolute", "top": "10px", "right": "160px", "z-index": "1000"})
                                                ], style={'width': '100%', 'height': param.map_height, 'margin': "auto", "display": "block"}, id="map2_lc"),

                            ],
                            # className='five columns', style={'margin': 10}
                            ),
                        ),
                    ]
                    ),
            dcc.Store(id='catch_id_lc', data=''),
            dcc.Store(id='powers_obj_lc', data=''),
            dcc.Store(id='reaches_obj_lc', data=''),
            dcc.Store(id='base_reductions_obj_lc', data=''),
            ]
        )

    return layout


###############################################
### Callbacks

@callback(
    Output('catch_id_lc', 'data'),
    [Input('catch_map_lc', 'click_feature')]
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
    Output('catch_name_lc', 'children'),
    [Input('catch_id_lc', 'data')]
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
        Output('reach_map_lc', 'data'),
        Input('catch_id_lc', 'data'),
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
        Output('marae_map_lc', 'data'),
        Input('catch_id_lc', 'data'),
        )
def update_marae(catch_id):
    if catch_id != '':
        with booklet.open(param.rivers_marae_path, 'r') as f:
            data = base64.b64encode(f[int(catch_id)]).decode()

    else:
        data = ''

    return data


@callback(
        Output('reductions_poly_lc', 'data'),
        Input('catch_id_lc', 'data'),
        )
# @cache.memoize()
def update_lc_map(catch_id):
    if catch_id != '':
        with booklet.open(param.lc_catch_pbf_path, 'r') as f:
            data = base64.b64encode(f[int(catch_id)]).decode()

    else:
        data = ''

    return data


@callback(
        Output('sites_points_lc', 'data'),
        Input('catch_id_lc', 'data'),
        )
# @cache.memoize()
def update_monitor_sites(catch_id):
    if catch_id != '':
        with booklet.open(param.rivers_sites_path, 'r') as f:
            data = base64.b64encode(f[int(catch_id)]).decode()

    else:
        data = ''

    return data


@callback(
        Output('reach_map_lc', 'options'),
        Input('reach_map_lc', 'hideout'),
        Input('catch_id_lc', 'data')
        )
# @cache.memoize()
def update_reaches_option(hideout, catch_id):
    trig = ctx.triggered_id

    if (len(hideout) == 0) or (trig == 'catch_id_lc'):
        options = dict(style=base_reach_style_handle)
    else:
        options = dict(style=reach_style_handle)

    return options


@callback(
        Output('base_reductions_obj_lc', 'data'),
        Input('catch_id_lc', 'data'),
        prevent_initial_call=True
        )
# @cache.memoize()
def update_base_reductions_obj(catch_id):
    data = ''

    if catch_id != '':
        with booklet.open(param.rivers_lc_clean_path, 'r') as f:
            data = utils.encode_obj(f[int(catch_id)])

    return data


@callback(
    Output('reaches_obj_lc', 'data'),
    Input('base_reductions_obj_lc', 'data'),
    [
      State('catch_id_lc', 'data'),
      ],
    prevent_initial_call=True)
def update_reach_reductions(base_reductions_obj, catch_id):
    """

    """
    if catch_id != '':
        # print('trigger')
        red1 = xr.open_dataset(param.rivers_reductions_model_path, engine='h5netcdf')

        with booklet.open(param.rivers_reach_mapping_path) as f:
            branches = f[int(catch_id)][int(catch_id)]

        base_props = red1.sel(nzsegment=branches).sortby('nzsegment').copy().load()
        red1.close()
        del red1
        # print(base_props)

        data = utils.encode_obj(base_props)
    else:
        data = ''

    return data


@callback(
    Output('reach_map_lc', 'hideout'),
    [Input('reaches_obj_lc', 'data'),
     Input('indicator_lc', 'value'),
     Input('Reductions_slider_lc', 'value')],
    prevent_initial_call=True
    )
def update_reach_hideout(reaches_obj, indicator, prop_red):
    """

    """
    if (reaches_obj != '') and (reaches_obj is not None) and isinstance(indicator, str):
        ind_name = lc_mapping_inverse[indicator]

        props = utils.decode_obj(reaches_obj)[[ind_name]].sel(reduction_perc=prop_red, drop=True).rename({ind_name: 'reduction'})

        ## Modelled
        color_arr = pd.cut(props.reduction.values, param.bins, labels=param.colorscale_power, right=False).tolist()

        hideout = {'colorscale': color_arr, 'classes': props.nzsegment.values, 'style': param.style_power, 'colorProp': 'nzsegment'}

    else:
        hideout = {}

    return hideout


@callback(
    Output('reductions_poly_lc', 'hideout'),
    [
     Input('indicator_lc', 'value'),
     ],
    prevent_initial_call=True
    )
def update_lc_hideout(indicator):
    """

    """
    if isinstance(indicator, str):
        hideout = {'colorscale': param.colorscale_power, 'classes': param.classes, 'style': param.lc_style, 'colorProp': indicator}

    else:
        hideout = {}

    return hideout


@callback(
    Output("info_lc", "children"),
    [Input('reaches_obj_lc', 'data'),
      Input("reach_map_lc", "click_feature"),
      Input('reductions_poly_lc', 'click_feature'),
      Input('indicator_lc', 'value'),
      Input('Reductions_slider_lc', 'value')],
    )
def update_map_info(reaches_obj, reach_feature, lc_feature, indicator, prop_red):
    """

    """
    info = """"""

    trig = ctx.triggered_id
    # print(trig)

    if isinstance(indicator, str):
        ind_name = lc_mapping_inverse[indicator]

        if trig == 'reach_map_lc':
            props = utils.decode_obj(reaches_obj)[[ind_name]].sel(reduction_perc=prop_red, drop=True).rename({ind_name: 'reduction'})
            # print(reach_feature)
            feature_id = int(reach_feature['id'])

            if feature_id in props.nzsegment:

                reach_data = props.sel(nzsegment=feature_id)

                info_str = """**nzsegment**: {seg}\n\n**Improvement**: {red}%""".format(red=int(reach_data.reduction), seg=feature_id)

                info = info + info_str

            else:
                info = info + """Click on a reach to see info"""

        elif trig == 'reductions_poly_lc':
            feature = lc_feature['properties']
            # print(feature)

            info_str = """**Typology**: {typo}\n\n**Land Cover**: {lc}\n\n**Improvement**: {red}%""".format(red=int(feature[indicator]), typo=feature['typology'], lc=feature['land_cover'])

            info = info + info_str

        else:
            info = info + """Click on a reach/polygon to see info"""

    return info


@callback(
    Output("lc_dl1", "href"),
    # Input('indicator_lc', 'value'),
    Input('catch_id_lc', 'data'),
    prevent_initial_call=True,
    )
def download_catch_lc(catch_id):

    if catch_id != '':
        url = param.rivers_catch_lc_gpkg_str.format(base_url=param.base_data_url, catch_id=catch_id)

        return url


@callback(
    Output("reach_dl1", "data"),
    Input("reach_dl_btn", "n_clicks"),
    State('catch_id_lc', 'data'),
    State('reaches_obj_lc', 'data'),
    prevent_initial_call=True,
    )
def download_power(n_clicks, catch_id, reaches_obj):

    if catch_id != '':
        props = utils.decode_obj(reaches_obj)

        df1 = props.to_dataframe().reset_index()
        # print(df1)
        for col in df1.columns:
            df1[col] = df1[col].astype(int)

        df2 = df1.rename(columns={'reduction_perc': 'improvement_perc'}).set_index(['improvement_perc', 'nzsegment'])

        return dcc.send_data_frame(df2.to_csv, f"rivers_improvement_{catch_id}.csv")

