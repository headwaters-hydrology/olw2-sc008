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
import geopandas as gpd
from gistools import vector
import os
# import tethysts
import base64
import geobuf
import pathlib
import hdf5plugin
import booklet
from dash_iconify import DashIconify

# from .app import app

# from app import app
# import utils
# from . import utils

##########################################
### Parameters

dash.register_page(
    __name__,
    path='/gw-wq',
    title='Water Quality',
    name='gw_wq',
    description='Groundwater Quality'
)

### Paths
assets_path = pathlib.Path(os.path.realpath(os.path.dirname(__file__))).parent.joinpath('assets')

app_base_path = pathlib.Path('/assets')

gw_error_path = assets_path.joinpath('gw_points_error.h5')

# gw_pbf_path = app_base_path.joinpath('gw_points.pbf')
gw_points_rc_blt = assets_path.joinpath('gw_points_rc.blt')
rc_bounds_gbuf = app_base_path.joinpath('rc_bounds.pbf')

### Layout
map_height = 700
center = [-41.1157, 172.4759]

attribution = 'Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'

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

gw_points_hideout = {'classes': [], 'colorscale': ['#808080'], 'circleOptions': dict(fillOpacity=1, stroke=False, radius=site_point_radius), 'colorProp': 'tooltip'}

gw_freq_mapping = {1: 'Yearly', 4: 'Quarterly', 12: 'monthly', 26: 'fortnightly', 52: 'weekly'}

gw_indicator_dict = {'Nitrate': 'Nitrate'}

gw_reductions_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

gw_reductions_options = [{'value': v, 'label': str(v)+'%'} for v in gw_reductions_values]

rc_style_handle = assign("""function style(feature) {
    return {
        fillColor: 'grey',
        weight: 2,
        opacity: 1,
        color: 'black',
        fillOpacity: 0.1
    };
}""", name='gw_rc_style_handle')

gw_points_style_handle = assign("""function gw_points_style_handle(feature, latlng, context){
    const {classes, colorscale, circleOptions, colorProp} = context.props.hideout;  // get props from hideout
    const value = feature.properties[colorProp];  // get value the determines the fillColor
    for (let i = 0; i < classes.length; ++i) {
        if (value == classes[i]) {
            circleOptions.fillColor = colorscale[i];  // set the color according to the class
        }
    }

    return L.circleMarker(latlng, circleOptions);
}""", name='gw_points_style_handle')

### Colorbar
colorbar_base = dl.Colorbar(style={'opacity': 0})
base_reach_style = dict(weight=4, opacity=1, color='white')

indices = list(range(len(ctg) + 1))
colorbar_power = dl.Colorbar(min=0, max=len(ctg), classes=indices, colorscale=colorscale, tooltip=True, tickValues=[item + 0.5 for item in indices[:-1]], tickText=ctg, width=300, height=30, position="bottomright")


###############################################
### Helper Functions


def read_pkl_zstd(obj, unpickle=False):
    """
    Deserializer from a pickled object compressed with zstandard.

    Parameters
    ----------
    obj : bytes or str
        Either a bytes object that has been pickled and compressed or a str path to the file object.
    unpickle : bool
        Should the bytes object be unpickled or left as bytes?

    Returns
    -------
    Python object
    """
    if isinstance(obj, (str, pathlib.Path)):
        with open(obj, 'rb') as p:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(p) as reader:
                obj1 = reader.read()

    elif isinstance(obj, bytes):
        dctx = zstd.ZstdDecompressor()
        obj1 = dctx.decompress(obj)
    else:
        raise TypeError('obj must either be a str path or a bytes object')

    if unpickle:
        obj1 = pickle.loads(obj1)

    return obj1


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


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
            [1, 4, 7],
            [1, 5, 6],
            [1, 5, 7],
            [2, 4, 6],
            [2, 4, 7],
            [2, 5, 6],
            [2, 5, 7],
            [3, 4, 6],
            [3, 4, 7],
            [3, 5, 6],
            [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]

    return out


def parse_gis_file(contents, filename):
    """

    """
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        plan1 = gpd.read_file(io.BytesIO(decoded))

        output = encode_obj(plan1)
    except:
        output = ['Wrong file type. It must be a GeoPackage (gpkg).']

    return output


def check_reductions_input(new_reductions, base_reductions):
    """

    """
    base_typos = base_reductions.typology.unique()
    try:
        missing_typos = np.in1d(new_reductions.typology.unique(), base_typos).all()
    except:
        missing_typos = False

    return missing_typos


def diff_reductions(new_reductions, base_reductions, reduction_cols):
    """

    """
    new_reductions1 = new_reductions.set_index('typology').sort_index()[reduction_cols]
    base_reductions1 = base_reductions.set_index('typology').sort_index()[reduction_cols]
    temp1 = new_reductions1.compare(base_reductions1, align_axis=0)

    return list(temp1.columns)


def xr_concat(datasets):
    """
    A much more efficient concat/combine of xarray datasets. It's also much safer on memory.
    """
    # Get variables for the creation of blank dataset
    coords_list = []
    chunk_dict = {}

    for chunk in datasets:
        coords_list.append(chunk.coords.to_dataset())
        for var in chunk.data_vars:
            if var not in chunk_dict:
                dims = tuple(chunk[var].dims)
                enc = chunk[var].encoding.copy()
                dtype = chunk[var].dtype
                _ = [enc.pop(d) for d in ['original_shape', 'source'] if d in enc]
                var_dict = {'dims': dims, 'enc': enc, 'dtype': dtype, 'attrs': chunk[var].attrs}
                chunk_dict[var] = var_dict

    try:
        xr3 = xr.combine_by_coords(coords_list, compat='override', data_vars='minimal', coords='all', combine_attrs='override')
    except:
        xr3 = xr.merge(coords_list, compat='override', combine_attrs='override')

    # Create the blank dataset
    for var, var_dict in chunk_dict.items():
        dims = var_dict['dims']
        shape = tuple(xr3[c].shape[0] for c in dims)
        xr3[var] = (dims, np.full(shape, np.nan, var_dict['dtype']))
        xr3[var].attrs = var_dict['attrs']
        xr3[var].encoding = var_dict['enc']

    # Update the attributes in the coords from the first ds
    for coord in xr3.coords:
        xr3[coord].encoding = datasets[0][coord].encoding
        xr3[coord].attrs = datasets[0][coord].attrs

    # Fill the dataset with data
    for chunk in datasets:
        for var in chunk.data_vars:
            if isinstance(chunk[var].variable._data, np.ndarray):
                xr3[var].loc[chunk[var].transpose(*chunk_dict[var]['dims']).coords.indexes] = chunk[var].transpose(*chunk_dict[var]['dims']).values
            elif isinstance(chunk[var].variable._data, xr.core.indexing.MemoryCachedArray):
                c1 = chunk[var].copy().load().transpose(*chunk_dict[var]['dims'])
                xr3[var].loc[c1.coords.indexes] = c1.values
                c1.close()
                del c1
            else:
                raise TypeError('Dataset data should be either an ndarray or a MemoryCachedArray.')

    return xr3


###############################################
### Initial processing

# sel1 = xr.open_dataset(base_path.joinpath(sel_data_h5), engine='h5netcdf')

# with booklet.open(gw_catches_major_path, 'r') as f:
#     lakes = list(f.keys())

# lakes.sort()

with booklet.open(gw_points_rc_blt, 'r') as f:
    rcs = list(f.keys())

rcs.sort()

indicators = [{'value': k, 'label': v} for k, v in gw_indicator_dict.items()]

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
                                dmc.AccordionControl('(1) Select a Regional Council', style={'font-size': 18}),
                                dmc.AccordionPanel([

                                    html.Label('(1a) Select a Regional Council on the map:'),
                                    dcc.Dropdown(options=[{'label': d, 'value': d} for d in rcs], id='rc_id', optionHeight=40, clearable=False,
                                                  style={'margin-top': 10}
                                                  ),
                                    ]
                                    )
                                ],
                                value='1'
                                ),

                            dmc.AccordionItem([
                                dmc.AccordionControl('(2) Select a reduction', style={'font-size': 18}),
                                dmc.AccordionPanel([
                                    html.Label('(2a) Select a reduction:'),
                                    dmc.Slider(id='reductions_slider_gw',
                                               value=25,
                                               mb=35,
                                               step=5,
                                               min=5,
                                               max=50,
                                               showLabelOnHover=True,
                                               disabled=False,
                                               marks=gw_reductions_options
                                               ),
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
                                    dcc.Dropdown(options=indicators, id='indicator_gw', optionHeight=40, clearable=False),
                                    dmc.Group(
                                        [dmc.Text('(3b) Select sampling length (years):', color="black"),
                                        dmc.HoverCard(
                                            withArrow=True,
                                            width=300,
                                            shadow="md",
                                            children=[
                                                dmc.HoverCardTarget(DashIconify(icon="material-symbols:help", width=30)),
                                                dmc.HoverCardDropdown(
                                                    dmc.Text(
                                                        """
                                                        The power results for groundwater only apply after the groundwater lag times of the upgradient mitigation actions. Any mitigation actions performed upgradient of the wells will take time to reach the wells. Click on a well to see the estimated mean residence time.
                                                        """,
                                                        size="sm",
                                                    )
                                                ),
                                            ],
                                        ),
                                        ],
                                        style={'margin-top': 20}
                                    ),
                                    # dmc.Text('(3b) Select sampling length (years):', style={'margin-top': 20}),
                                    dmc.SegmentedControl(data=[{'label': d, 'value': str(d)} for d in time_periods],
                                                         id='time_period_gw',
                                                         value='5',
                                                         fullWidth=True,
                                                         color=1,
                                                         ),
                                    dmc.Text('(3c) Select sampling frequency:', style={'margin-top': 20}),
                                    dmc.SegmentedControl(data=[{'label': v, 'value': str(k)} for k, v in gw_freq_mapping.items()],
                                                         id='freq_gw',
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
                                    children=[html.Div(dmc.Button("Download power results", id='dl_btn_power_gw'), style={'margin-bottom': 20, 'margin-top': 10}),
                            dcc.Download(id="dl_power_gw")],
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
                            dl.Map(center=center, zoom=6, children=[
                                dl.LayersControl([
                                    dl.BaseLayer(dl.TileLayer(attribution=attribution, opacity=0.7), checked=True, name='OpenStreetMap'),
                                    dl.BaseLayer(dl.TileLayer(url='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', attribution='Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap (CC-BY-SA)', opacity=0.6), checked=False, name='OpenTopoMap'),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(url=str(rc_bounds_gbuf), format="geobuf", id='rc_map', zoomToBoundsOnClick=True, options=dict(style=rc_style_handle),  hideout={})), name='Regional Councils', checked=True),
                                    dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='gw_points', zoomToBounds=True, zoomToBoundsOnClick=True, cluster=False, options=dict(pointToLayer=gw_points_style_handle), hideout=gw_points_hideout)), name='GW wells', checked=True),
                                    # dl.Overlay(dl.LayerGroup(dl.GeoJSON(data='', format="geobuf", id='sites_points_gw', options=dict(pointToLayer=sites_points_handle), hideout=rivers_points_hideout)), name='Monitoring sites', checked=True),
                                    ], id='layers_gw'),
                                colorbar_power,
                                # html.Div(id='colorbar', children=colorbar_base),
                                # dmc.Group(id='colorbar', children=colorbar_base),
                                dcc.Markdown(id="info_gw", className="info", style={"position": "absolute", "top": "10px", "right": "160px", "z-index": "1000"})
                                                ], style={'width': '100%', 'height': '100vh', 'margin': "auto", "display": "block"}, id="map2"),

                            ],
                            ),
                        ),
                    ]
                    ),
            dcc.Store(id='powers_obj_gw', data=None),
            dcc.Store(id='gw_points_ids', data=None),
            ]
        )

    return layout


###############################################
### Callbacks


@callback(
    Output('rc_id', 'value'),
    [Input('rc_map', 'click_feature')]
    )
def update_rc_id(feature):
    """

    """
    # print(ds_id)

    if feature is not None:
        if not feature['properties']['cluster']:
            rc_id = feature['id']
    else:
        rc_id = None

    return rc_id


@callback(
        Output('gw_points', 'data'),
        Output('gw_points_ids', 'data'),
        Input('rc_id', 'value'),
        )
# @cache.memoize()
def update_gw_points(rc_id):
    if (rc_id is not None):
        with booklet.open(gw_points_rc_blt, 'r') as f:
            data0 = f[rc_id]
            geo1 = geobuf.decode(data0)
            gw_points = [s['id'] for s in geo1['features']]
            gw_points_encode = encode_obj(gw_points)
            data = base64.b64encode(data0).decode()
    else:
        data = None
        gw_points_encode = None

    return data, gw_points_encode


@callback(
    Output('powers_obj_gw', 'data'),
    [Input('reductions_slider_gw', 'value'), Input('indicator_gw', 'value'), Input('time_period_gw', 'value'), Input('freq_gw', 'value')],
    # [State('gw_id', 'value')]
    )
def update_props_data_gw(reductions, indicator, n_years, n_samples_year):
    """

    """
    if isinstance(reductions, (str, int)) and isinstance(n_years, str) and isinstance(n_samples_year, str) and isinstance(indicator, str):
        n_samples = int(n_samples_year)*int(n_years)

        power_data = xr.open_dataset(gw_error_path, engine='h5netcdf')
        power_data1 = power_data.sel(indicator=indicator, n_samples=n_samples, conc_perc=100-int(reductions), drop=True).to_dataframe().reset_index()
        power_data.close()
        del power_data

        data = encode_obj(power_data1)
        return data
    else:
        raise dash.exceptions.PreventUpdate


@callback(
    Output('gw_points', 'hideout'),
    Input('powers_obj_gw', 'data'),
    Input('gw_points_ids', 'data'),
    prevent_initial_call=True
    )
def update_hideout_gw_points(powers_obj, gw_points_encode):
    """

    """
    if (powers_obj != '') and (powers_obj is not None):

        # print('trigger')
        props = decode_obj(powers_obj)
        # print(props)
        # print(type(gw_id))

        color_arr = pd.cut(props.power.values, bins, labels=colorscale, right=False).tolist()
        # print(color_arr)
        # print(props['gw_id'])

        hideout = {'classes': props['ref'].values, 'colorscale': color_arr, 'circleOptions': dict(fillOpacity=1, stroke=False, radius=site_point_radius), 'colorProp': 'tooltip'}
    elif (gw_points_encode is not None):
        # print('trigger')
        gw_refs = decode_obj(gw_points_encode)

        hideout = {'classes': gw_refs, 'colorscale': ['#808080'] * len(gw_refs), 'circleOptions': dict(fillOpacity=1, stroke=False, radius=site_point_radius), 'colorProp': 'tooltip'}
    else:
        hideout = gw_points_hideout

    return hideout


@callback(
    Output("info_gw", "children"),
    [Input('powers_obj_gw', 'data'),
      Input('reductions_slider_gw', 'value'),
      Input("gw_points", "click_feature")],
    State('gw_points_ids', 'data')
    )
def update_map_info_gw(powers_obj, reductions, feature, gw_points_encode):
    """

    """
    info = """"""
    # info_str = """\n\n**Reduction**: {red}%\n\n**Likelihood of observing a reduction (power)**: {t_stat}%\n\n**Well Depth (m)**: {depth:.1f}\n\n**mean residence time (years)**: {lag}"""

    # if (reductions_obj != '') and (reductions_obj is not None) and ('reductions_poly' in map_checkboxes):
    #     info = info + """\n\nHover over the polygons to see reduction %"""

    if isinstance(reductions, int) and (powers_obj != '') and (powers_obj is not None):
        if feature is not None:
            # print(feature)
            gw_refs = decode_obj(gw_points_encode)
            if feature['id'] in gw_refs:
                props = decode_obj(powers_obj)

                # print(feature['properties']['lag_at_site'])

                if feature['properties']['lag_at_site'] is None:
                    info_str = """\n\n**Reduction**: {red}%\n\n**Likelihood of observing a reduction (power)**: {t_stat}%\n\n**Well Depth (m)**: {depth:.1f}\n\n**Mean residence time (MRT) at well (years)**: NA\n\n**MRT at nearest well**: {lag_median} years within a distance of {lag_dist:,} m"""
                    info2 = info_str.format(red=int(reductions), t_stat=int(props[props.ref==feature['id']].iloc[0]['power']), depth=feature['properties']['depth'], lag_median=feature['properties']['lag_median'], lag_dist=feature['properties']['lag_dist'])
                else:
                    site_lag = str(int(feature['properties']['lag_at_site']))
                    info_str = """\n\n**Reduction**: {red}%\n\n**Likelihood of observing a reduction (power)**: {t_stat}%\n\n**Well Depth (m)**: {depth:.1f}\n\n**Mean residence time (MRT) at well (years)**: {site_lag}"""
                    info2 = info_str.format(red=int(reductions), t_stat=int(props[props.ref==feature['id']].iloc[0]['power']), depth=feature['properties']['depth'], site_lag=site_lag)

                info = info2

        else:
            info = """\n\nClick on a well to see info"""

    return info


@callback(
    Output("dl_power_gw", "data"),
    Input("dl_btn_power_gw", "n_clicks"),
    State('powers_obj_gw', 'data'),
    State('rc_id', 'value'),
    State('indicator_gw', 'value'),
    State('time_period_gw', 'value'),
    State('freq_gw', 'value'),
    State('reductions_slider_gw', 'value'),
    prevent_initial_call=True,
    )
def download_power(n_clicks, powers_obj, rc_id, indicator, n_years, n_samples_year, reductions):

    if (powers_obj != '') and (powers_obj is not None):
        df1 = decode_obj(powers_obj)

        df1['power'] = df1['power'].astype(int)

        df1['indicator'] = gw_indicator_dict[indicator]
        df1['n_years'] = n_years
        df1['n_samples_per_year'] = n_samples_year
        df1['reduction'] = reductions

        df2 = df1.rename(columns={'ref': 'site_id'}).set_index(['indicator', 'n_years', 'n_samples_per_year', 'reduction', 'site_id']).sort_index()

        return dcc.send_data_frame(df2.to_csv, f"gw_power_{rc_id}.csv")





# with shelflet.open(gw_lc_path, 'r') as f:
#     plan_file = f[str(gw_id)]
