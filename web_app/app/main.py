#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 09:55:36 2021

@author: mike
"""
import dash
from dash import Dash, html, dcc, callback
from dash.dependencies import Input, Output, State
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc
# from dash_iconify import DashIconify
import pathlib
from time import sleep

##############################################
### The app
# sleep(10)
app = dash.Dash(__name__,
                use_pages=True,
                external_stylesheets=[dbc.themes.BOOTSTRAP]
                )
server = app.server

app_base_path = pathlib.Path('/assets')

# def sidebar():
#     return html.Div(
#         dbc.Nav(
#             [
#                 dbc.NavLink(
#                     [
#                         html.Div('- ' + page["name"], className="ms-2"),
#                     ],
#                     href=page["path"],
#                     active="exact",
#                 )
#                 for page in dash.page_registry.values()
#             ],
#             vertical=True,
#             pills=True,
#             className="bg-light",
#         )
#     )


# app.layout = html.Div(children=[
#     html.Div([dcc.Link(html.Img(src=str(app_base_path.joinpath('our-land-and-water-logo.svg'))), href='https://ourlandandwater.nz/'),
#               html.H3('Contents'),
#               sidebar(),
#               ], className='one column', style={'margin': 0, 'margin-top': 15}),
#     html.Div([dash.page_container
#               ], className='eleven columns', style={'margin': 0})
#     ])

page_path_names = {v['path']: v['description'] for k, v in dash.page_registry.items()}

def create_nav_link(label, href):
    return dcc.Link(
        dmc.Group(
            [
                # dmc.ThemeIcon(
                #     DashIconify(icon=icon, width=18),
                #     size=30,
                #     radius=30,
                #     variant="light",
                # ),
                dmc.Text(label, size="sm", color="grey"),
            ]
        ),
        href=href,
        style={"textDecoration": "none"},
    )


def create_sidebar_children(pages):
    """

    """
    list1 = [
        dmc.Group(
        # direction="column",
        children=[
            create_nav_link(
                label="Home",
                href="/",
                ),
            ],
            ),
        dmc.Divider(
            label="User Guides", style={"marginBottom": 20, "marginTop": 20}
            ),
        dmc.Divider(
            label="Rivers", style={"marginBottom": 20, "marginTop": 20}
            ),
        dmc.Group(
            children=[
                create_nav_link(
                    label=pages["pages.rivers_wq"]['title'], href=pages["pages.rivers_wq"]["path"]
                    )
                ], style={"marginBottom": 10}
            ),
        dmc.Group(
            children=[
                create_nav_link(
                    label=pages["pages.rivers_eco"]['title'], href=pages["pages.rivers_eco"]["path"]
                    )
                ], style={"marginBottom": 10}
            ),
        dmc.Group(
            children=[
                create_nav_link(
                    label=pages["pages.land_cover"]['title'], href=pages["pages.land_cover"]["path"]
                    )
                ], style={"marginBottom": 10}
            ),
        dmc.Divider(
            label="Lakes", style={"marginBottom": 20, "marginTop": 20}
            ),
        dmc.Group(
            children=[
                create_nav_link(
                    label=pages["pages.lakes_wq"]['title'], href=pages["pages.lakes_wq"]["path"]
                    )
                ], style={"marginBottom": 10}
            ),
        dmc.Divider(
            label="Groundwater", style={"marginBottom": 20, "marginTop": 20}
            ),
        dmc.Group(
            children=[
                create_nav_link(
                    label=pages["pages.gw_wq"]['title'], href=pages["pages.gw_wq"]["path"]
                    )
                ], style={"marginBottom": 10}
            ),
        ]

    # dmc.Group(
    #     children=[
    #         create_nav_link(
    #             label=page["name"], href=page["path"]
    #         )
    #     ], style={"marginBottom": 10}
    # )
    # for page in pages if 'Home' not in page["name"]]

    return list1


sidebar = dmc.Navbar(
    fixed=True,
    width={"base": 180},
    position={"top": 80},
    px=10,
    # height=300,
    children=[
        dmc.ScrollArea(
            offsetScrollbars=True,
            type="scroll",
            children=create_sidebar_children(dash.page_registry)
            )
        ]
    )

app.layout = html.Div(
    [
     dmc.Header(
        height=60,
        fixed=True,
        px=25,
        children=[
            dmc.Grid(
                # gutter='xl',
                style={"height": 60},
                children=[
                    dmc.Col(
                        dmc.Anchor(
                            'Mitigation Effectiveness Monitoring Design',
                            size="xl",
                            href="/",
                            underline=False,
                            # align='center',
                            # style={'vertical-align': 'middle'}

                        ),
                        span=6,
                        style={'padding': '20px 0'}
                        # style={'vertical-align': 'middle'}
                        ),
                    dmc.Col(
                        dmc.Text(
                            '',
                            id='title',
                            size=28,
                            ),
                        span=3,
                        style={'padding': '20px 0'}
                        ),
                    dmc.Col(
                        dmc.Anchor(
                            dmc.Image(
                                src=str(app_base_path.joinpath('our-land-and-water-logo.svg')),
                                fit='cover',
                                width='90%'
                                ),
                            href='https://ourlandandwater.nz/'
                            ),
                        span=3,
                        offset=0
                        ),
                    ]
                )
            ]
        ),

        #     html.Div(
        #  dcc.Link(html.Img(src=str(app_base_path.joinpath('our-land-and-water-logo.svg')), style={'height': 60}), href='https://ourlandandwater.nz/'),
        #         # style={"backgroundColor": "#228be6"},
        # className='three columns'
        # ),
        # html.Div(
        # html.H2('', id='title'), className='seven columns'
        # ),
        sidebar,
        # html.Div(dash.page_container, className='eleven columns', style={'margin': 0, "marginLeft": 190})
        # html.Div(dash.page_container, style={"margin-top": 80, 'margin-left': 220, 'margin-right': 220})
        dmc.Container(
            dash.page_container,
            size="xl",
            # pt=20,
            style={"margin-top": 80, 'margin-left': 200, 'margin-right': 20},
        ),
    ],
)

@callback(
    Output('title', 'children'),
    Input('_pages_location', 'pathname'),
    prevent_initial_call=True
    )
def updated_title(path):
    # print(path)
    if isinstance(path, str):
        title = page_path_names[path]
        if title == 'Home':
            # title = 'Mitigation Effectiveness Monitoring Design'
            title = ''

        return title
    else:
        raise dash.exceptions.PreventUpdate


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=80)

# if __name__ == '__main__':
#     app.run_server(debug=True, host='0.0.0.0', port=8000)
