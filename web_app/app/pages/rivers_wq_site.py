#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 13:37:46 2022

@author: mike
"""
import io
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
# import requests
import zstandard as zstd
import codecs
import pickle
import os
# import tethysts
import base64
import geobuf
import pathlib
import hdf5plugin
import booklet
import hdf5tools

# from .app import app
# from . import utils

# from app import app
# import utils

##########################################
### Parameters

# assets_path = pathlib.Path(os.path.split(os.path.realpath(os.path.dirname(__file__)))[0]).joinpath('assets')

dash.register_page(
    __name__,
    path='/rivers-wq-site',
    title='Water Quality Sites Only',
    name='rivers_wq_site',
    description='Rivers Water Quality Sites Only'
)

### Paths
assets_path = pathlib.Path(os.path.realpath(os.path.dirname(__file__))).parent.joinpath('assets')

app_base_path = pathlib.Path('/assets')

# base_data_url = 'https://b2.tethys-ts.xyz/file/'

# lc_url = '{}olw-data/olw-sc008/olw_land_cover_reductions.gpkg'.format(base_data_url)
# rivers_red_url = '{}olw-data/olw-sc008/olw_sites_reductions.csv.zip'.format(base_data_url)

rivers_power_moni_path = assets_path.joinpath('rivers_reaches_power_monitored.h5')

rivers_catch_pbf_path = app_base_path.joinpath('rivers_catchments.pbf')

rivers_reach_gbuf_path = assets_path.joinpath('rivers_reaches.blt')
river_catch_name_path = assets_path.joinpath('rivers_catchments_names.blt')

rivers_sites_path = assets_path.joinpath('rivers_sites_catchments.blt')
river_marae_path = assets_path.joinpath('rivers_catchments_marae.blt')
# eco_loads_rec_path = assets_path.joinpath('rivers_loads_rec.blt')

# rivers_catch_lc_dir = assets_path.joinpath('rivers_land_cover_gpkg')
# rivers_catch_lc_gpkg_str = '{}_sites_land_cover_reductions.gpkg'
# rivers_catch_lc_gpkg_str = '{base_url}olw-data/olw-sc008/rivers_land_cover_gpkg/{catch_id}_sites_land_cover_reductions.gpkg'

### Layout
map_height = 700
center = [-41.1157, 172.4759]

attribution = 'Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'

reductions_values = range(5, 95, 10)

reductions_options = [{'value': v, 'label': str(v)+'%'} for v in reductions_values]

style = dict(weight=4, opacity=1, color='white')

site_point_radius = 6

# reduction_ratios = range(10, 101, 10)
# red_ratios = np.array(list(reduction_ratios), dtype='int8')

freq_mapping = {4: 'quarterly', 12: 'monthly', 26: 'fortnightly', 52: 'weekly', 104: 'biweekly', 364: 'daily'}
time_periods = [5, 10, 20, 30]

style = dict(weight=4, opacity=1, color='white')
classes = [0, 20, 40, 60, 80]
bins = classes.copy()
bins.append(101)
# colorscale = ['#808080', '#FED976', '#FEB24C', '#FC4E2A', '#BD0026', '#800026']
colorscale = ['#808080', '#FED976', '#FD8D3C', '#E31A1C', '#800026']
# reductions_colorscale = ['#edf8fb','#b2e2e2','#66c2a4','#2ca25f','#006d2c']
# ctg = ["{}%+".format(cls, classes[i + 1]) for i, cls in enumerate(classes[1:-1])] + ["{}%+".format(classes[-1])]
# ctg.insert(0, 'NA')
ctg = ["{}%+".format(cls, classes[i + 1]) for i, cls in enumerate(classes[:-1])] + ["{}%+".format(classes[-1])]
# ctg.insert(0, 'NA')

site_point_radius = 6

reduction_ratios = range(10, 101, 10)
red_ratios = np.array(list(reduction_ratios), dtype='int8')

rivers_points_hideout = {'classes': [], 'colorscale': ['#232323'], 'circleOptions': dict(fillOpacity=1, stroke=True, weight=1, color='black', radius=site_point_radius), 'colorProp': 'nzsegment'}

rivers_indicator_dict = {'BD': 'Visual Clarity', 'EC': 'E.coli', 'DRP': 'Dissolved reactive phosporus', 'NH': 'Ammoniacal nitrogen', 'NO': 'Nitrate', 'TN': 'Total nitrogen', 'TP': 'Total phosphorus'}

rivers_reduction_cols = list(rivers_indicator_dict.values())

catch_style_handle = assign("""function style(feature) {
    return {
        fillColor: 'grey',
        weight: 2,
        opacity: 1,
        color: 'black',
        fillOpacity: 0.1
    };
}""", name='rivers_catch_style_handle_sites')

base_reach_style_handle = assign("""function style3(feature) {
    return {
        weight: 2,
        opacity: 0.75,
        color: 'grey',
    };
}""", name='rivers_base_reach_style_handle_sites')

# reach_style_handle = assign("""function style2(feature, context){
#     const {classes, colorscale, style, colorProp} = context.props.hideout;  // get props from hideout
#     const value = feature.properties[colorProp];  // get value the determines the color
#     for (let i = 0; i < classes.length; ++i) {
#         if (value == classes[i]) {
#             style.color = colorscale[i];  // set the fill color according to the class
#         }
#     }
#     return style;
# }""", name='rivers_reach_style_handle')

sites_points_handle = assign("""function rivers_sites_points_handle(feature, latlng, context){
    const {classes, colorscale, circleOptions, colorProp} = context.props.hideout;  // get props from hideout
    const value = feature.properties[colorProp];  // get value the determines the fillColor
    for (let i = 0; i < classes.length; ++i) {
        if (value == classes[i]) {
            circleOptions.fillColor = colorscale[i];  // set the color according to the class
        }
    }

    return L.circleMarker(latlng, circleOptions);
}""", name='rivers_sites_points_handle_sites')

draw_marae = assign("""function(feature, latlng){
const flag = L.icon({iconUrl: '/assets/nzta-marae.svg', iconSize: [20, 30]});
return L.marker(latlng, {icon: flag});
}""", name='rivers_sites_marae_handle')

### Colorbar
colorbar_base = dl.Colorbar(style={'opacity': 0})
base_reach_style = dict(weight=4, opacity=1, color='white')

indices = list(range(len(ctg) + 1))
colorbar_power = dl.Colorbar(min=0, max=len(ctg), classes=indices, colorscale=colorscale, tooltip=True, tickValues=[item + 0.5 for item in indices[:-1]], tickText=ctg, width=300, height=30, position="bottomright")

marks = []
for i in range(0, 101, 10):
    if (i % 20) == 0:
        marks.append({'label': str(i) + '%', 'value': i})
    else:
        marks.append({'value': i})

# catch_id = 3076139

###############################################
### Helper Functions


# def read_pkl_zstd(obj, unpickle=False):
#     """
#     Deserializer from a pickled object compressed with zstandard.

#     Parameters
#     ----------
#     obj : bytes or str
#         Either a bytes object that has been pickled and compressed or a str path to the file object.
#     unpickle : bool
#         Should the bytes object be unpickled or left as bytes?

#     Returns
#     -------
#     Python object
#     """
#     if isinstance(obj, (str, pathlib.Path)):
#         with open(obj, 'rb') as p:
#             dctx = zstd.ZstdDecompressor()
#             with dctx.stream_reader(p) as reader:
#                 obj1 = reader.read()

#     elif isinstance(obj, bytes):
#         dctx = zstd.ZstdDecompressor()
#         obj1 = dctx.decompress(obj)
#     else:
#         raise TypeError('obj must either be a str path or a bytes object')

#     if unpickle:
#         obj1 = pickle.loads(obj1)

#     return obj1


# def encode_xr(obj: xr.Dataset):
#     """

#     """
#     i1 = io.BytesIO()
#     hdf5tools.xr_to_hdf5(obj, i1)
#     str_obj = codecs.encode(i1.read(), encoding="base64").decode()

#     return str_obj


# def decode_xr(str_obj):
#     """

#     """
#     i1 = io.BytesIO(codecs.decode(str_obj.encode(), encoding="base64"))
#     x1 = xr.load_dataset(i1)

#     return x1


def encode_obj(obj):
    """

    """
    cctx = zstd.ZstdCompressor(level=1)
    c_obj = codecs.encode(cctx.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)), encoding="base64").decode()

    return c_obj


def decode_obj(str_obj):
    """

    """
    dctx = zstd.ZstdDecompressor()
    obj1 = dctx.decompress(codecs.decode(str_obj.encode(), encoding="base64"))
    d1 = pickle.loads(obj1)

    return d1



###############################################
### Initial processing

# with booklet.open(eco_sites_path, 'r') as f:
#     catches = [int(c) for c in f]

# catches.sort()
indicators = list(rivers_indicator_dict.keys())
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
                                    dmc.Text(id='catch_name_sites', weight=700, style={'margin-top': 10}),
                                    ]
                                    )
                                ],
                                value='1'
                                ),

                            dmc.AccordionItem([
                                dmc.AccordionControl('(2) Define an improvement by site', style={'font-size': 18}),
                                dmc.AccordionPanel([
                                    html.Label('(2a) Type in a percent improvement by site under the "improvement %" column then press enter to confirm:'),
                                    dash_table.DataTable(data=[], columns=[{'name': n, 'id': n, 'editable': (n == 'improvement %')} for n in ['site name', 'improvement %']], id='sites_tbl', style_cell={'font-size': 11}, style_header_conditional=[{
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
                                    dmc.Text('(3a) Select Indicator:'),
                                    dcc.Dropdown(options=[{'label': rivers_indicator_dict[d], 'value': d} for d in indicators], id='indicator_sites', optionHeight=40, clearable=False),
                                    dmc.Text('(3b) Select sampling length (years):', style={'margin-top': 20}),
                                    dmc.SegmentedControl(data=[{'label': d, 'value': str(d)} for d in time_periods],
                                                         id='time_period_sites',
                                                         value='5',
                                                         fullWidth=True,
                                                         color=1,
                                                         ),
                                    dmc.Text('(3c) Select sampling frequency (monitoring site power):', style={'margin-top': 20}),
                                    dmc.SegmentedControl(data=[{'label': v, 'value': str(k)} for k, v in freq_mapping.items()],
                                                          id='freq_sites',
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
                                    dmc.Text('(4a) Download power results given the prior query options (csv):'),
                                    dcc.Loading(
                                    type="default",
                                    children=[html.Div(dmc.Button("Download power results", id='dl_btn_power_sites'), style={'margin-bottom': 20, 'margin-top': 10}),
                            dcc.Download(id="dl_power_sites")],
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
                            dl.Map(center=center, zoom=6, children=[
                                dl.LayersControl([
                                    dl.BaseLayer(dl.TileLayer(attribution=attribution, opacity=0.7), checked=True, name='OpenStreetMap'),
                                    dl.BaseLayer(dl.TileLayer(url='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', attribution='Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap (CC-BY-SA)', opacity=0.6), checked=False, name='OpenTopoMap'),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(url=str(rivers_catch_pbf_path), format="geobuf", id='catch_map_sites', zoomToBoundsOnClick=True, zoomToBounds=False, options=dict(style=catch_style_handle))), name='Catchments', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='marae_map_sites', zoomToBoundsOnClick=False, zoomToBounds=False, options=dict(pointToLayer=draw_marae))), name='Marae', checked=False),
                                    # dl.GeoJSON(url='', format="geobuf", id='base_reach_map', options=dict(style=base_reaches_style_handle)),

                                    # dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='sites_points', options=dict(pointToLayer=sites_points_handle), hideout={'circleOptions': dict(fillOpacity=1, stroke=False, radius=5, color='black')})), name='Monitoring sites', checked=True),
                                    # dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='reductions_poly')), name='Land use reductions', checked=False),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='reach_map_sites', options=dict(style=base_reach_style_handle), hideout={})), name='Rivers', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='sites_points_sites', options=dict(pointToLayer=sites_points_handle), hideout=rivers_points_hideout)), name='Monitoring sites', checked=True),
                                    ], id='layers_sites'),
                                colorbar_power,
                                # html.Div(id='colorbar', children=colorbar_base),
                                # dmc.Group(id='colorbar', children=colorbar_base),
                                dcc.Markdown(id="info_sites", className="info", style={"position": "absolute", "top": "10px", "right": "160px", "z-index": "1000"})
                                                ], style={'width': '100%', 'height': '100vh', 'margin': "auto", "display": "block"}),

                            ],
                            # className='five columns', style={'margin': 10}
                            ),
                        ),
                    ]
                    ),
            dcc.Store(id='catch_id_sites', data=''),
            dcc.Store(id='sites_powers_obj_sites', data=''),
            ]
        )

    return layout


###############################################
### Callbacks

@callback(
    Output('catch_id_sites', 'data'),
    [Input('catch_map_sites', 'click_feature')]
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
    Output('catch_name_sites', 'children'),
    [Input('catch_id_sites', 'data')]
    )
def update_catch_name(catch_id):
    """

    """
    # print(ds_id)
    if catch_id != '':
        with booklet.open(river_catch_name_path) as f:
            catch_name = f[int(catch_id)]

        return catch_name


@callback(
        Output('reach_map_sites', 'data'),
        Input('catch_id_sites', 'data'),
        )
# @cache.memoize()
def update_reaches(catch_id):
    if catch_id != '':
        with booklet.open(rivers_reach_gbuf_path, 'r') as f:
            data = base64.b64encode(f[int(catch_id)]).decode()

    else:
        data = ''

    return data


@callback(
        Output('marae_map_sites', 'data'),
        Input('catch_id_sites', 'data'),
        )
def update_marae(catch_id):
    if catch_id != '':
        with booklet.open(river_marae_path, 'r') as f:
            data = base64.b64encode(f[int(catch_id)]).decode()

    else:
        data = ''

    return data


@callback(
        Output('sites_points_sites', 'data'),
        Output('sites_tbl', 'data'),
        Input('catch_id_sites', 'data'),
        )
# @cache.memoize()
def update_monitor_sites(catch_id):
    if catch_id != '':
        with booklet.open(rivers_sites_path, 'r') as f:
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
    Output('sites_powers_obj_sites', 'data'),
    [Input('indicator_sites', 'value'), Input('time_period_sites', 'value'), Input('freq_sites', 'value'),
    Input('sites_tbl', 'data')],
    prevent_initial_call=True
    )
def update_sites_powers_obj(indicator, n_years, n_samples_year, tbl_data):
    """

    """
    if isinstance(n_years, str) and isinstance(indicator, str) and isinstance(n_samples_year, str) and (len(tbl_data) > 0):
        n_samples = int(n_samples_year)*int(n_years)

        red1 = {}
        for r in tbl_data:
            try:
                red_int = int(r['improvement %'])
            except:
                red_int = 0
            red1[int(r['nzsegment'])] = 100 - red_int

        power_data = xr.open_dataset(rivers_power_moni_path, engine='h5netcdf')
        power_data1 = power_data.sel(indicator=indicator, n_samples=n_samples, drop=True).dropna('nzsegment').load().copy()
        power_data2 = []
        for seg, conc_perc in red1.items():
            if seg in power_data1.nzsegment:
                power = int(power_data1.sel(nzsegment=seg, conc_perc=conc_perc).power.values)
                power_data2.append({'conc_perc': conc_perc, 'nzsegment': seg, 'power': power})

        # print(power_data1)
        power_data.close()
        del power_data

        data = encode_obj(power_data2)
        return data
    else:
        raise dash.exceptions.PreventUpdate


@callback(
    Output('sites_points_sites', 'hideout'),
    [Input('sites_powers_obj_sites', 'data')],
    prevent_initial_call=True
    )
def update_sites_hideout(powers_obj):
    """

    """
    if (powers_obj != '') and (powers_obj is not None):
        props = decode_obj(powers_obj)

        ## Monitored
        if props:
            # print(props_moni)
            color_arr2 = pd.cut([p['power'] for p in props], bins, labels=colorscale, right=False).tolist()

            hideout_moni = {'classes': [p['nzsegment'] for p in props], 'colorscale': color_arr2, 'circleOptions': dict(fillOpacity=1, stroke=True, color='black', weight=1, radius=site_point_radius), 'colorProp': 'nzsegment'}

        else:
            hideout_moni = rivers_points_hideout
    else:
        hideout_moni = rivers_points_hideout

    return hideout_moni


@callback(
    Output("info_sites", "children"),
    [Input('sites_powers_obj_sites', 'data'),
      Input('sites_points_sites', 'click_feature')],
    [State("info_sites", "children"),
     ]
    )
def update_map_info(sites_powers_obj, sites_feature, old_info):
    """

    """
    info = """"""

    if (sites_powers_obj != '') and (sites_powers_obj is not None) and (sites_feature is not None):
        props = decode_obj(sites_powers_obj)

        feature_id = int(sites_feature['properties']['nzsegment'])
        # print(sites_feature)
        # print(props.nzsegment)
        # print(feature_id)

        reach_data = [p for p in props if p['nzsegment'] == feature_id]
        if reach_data:
            power = reach_data[0]['power']
            red = 100 - reach_data[0]['conc_perc']

            info += """##### Monitoring Site:

                \n\n**nzsegment**: {seg}\n\n**Site name**: {site}\n\n**Improvement %**: {conc}\n\n**Likelihood of observing an improvement (power)**: {t_stat}%""".format(t_stat=power, conc=red, seg=feature_id, site=sites_feature['id'])

    # if info == """""":
    #     info = old_info

    return info


@callback(
    Output("dl_power_sites", "data"),
    Input("dl_btn_power_sites", "n_clicks"),
    State('catch_id_sites', 'data'),
    State('sites_powers_obj_sites', 'data'),
    State('indicator_sites', 'value'),
    State('time_period_sites', 'value'),
    State('freq_sites', 'value'),
    prevent_initial_call=True,
    )
def download_power(n_clicks, catch_id, powers_obj, indicator, n_years, n_samples_year):

    if (catch_id != '') and (powers_obj != '') and (powers_obj is not None) and isinstance(n_samples_year, str):
        power_data = decode_obj(powers_obj)

        df1 = pd.DataFrame.from_dict(power_data)
        df1['improvement'] = 100 - df1['conc_perc']
        df1['indicator'] = rivers_indicator_dict[indicator]
        df1['n_years'] = n_years
        df1['n_samples_per_year'] = n_samples_year

        df2 = df1.drop('conc_perc', axis=1).set_index(['nzsegment', 'improvement', 'indicator', 'n_years', 'n_samples_per_year']).sort_index()

        return dcc.send_data_frame(df2.to_csv, f"sites_power_{catch_id}.csv")
