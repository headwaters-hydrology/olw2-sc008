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
# from dash_iconify import DashIconify
import pathlib
from time import sleep

##############################################
### The app
# sleep(10)
app = dash.Dash(__name__, use_pages=True)
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



page_mapping = {v['path']: v['title'] for k, v in dash.page_registry.items()}

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
                dmc.Text(label, size="sm", color="gray"),
            ]
        ),
        href=href,
        style={"textDecoration": "none"},
    )


def create_sidebar_children(pages):
    """

    """
    list1 = [dmc.Group(
        direction="column",
        children=[
            create_nav_link(
                label="Home",
                href="/",
            ),
        ],
    ),
    dmc.Divider(
        label="Pages", style={"marginBottom": 20, "marginTop": 20}
    )]

    list2 = [
    dmc.Group(
        direction="column",
        children=[
            create_nav_link(
                label=page["name"], href=page["path"]
            )
        ], style={"marginBottom": 20}
    )
    for page in pages if 'Home' not in page["name"]]

    return list1 + list2


sidebar = dmc.Navbar(
    fixed=True,
    width={"base": 180},
    position={"top": 80},
    height=300,
    children=[
        dmc.ScrollArea(
            offsetScrollbars=True,
            type="scroll",
            children=create_sidebar_children(dash.page_registry.values())
            )
        ]
    )

app.layout = html.Div(
    [html.Div(
         dcc.Link(html.Img(src=str(app_base_path.joinpath('our-land-and-water-logo.svg')), style={'height': 60}), href='https://ourlandandwater.nz/'),
                # style={"backgroundColor": "#228be6"},
        className='three columns'
        ),
        html.Div(
        html.H1('', id='title'), className='seven columns'
        ),
        sidebar,
        html.Div(dash.page_container, className='eleven columns', style={'margin': 0, "marginLeft": 190})
        # dmc.Container(
        #     dash.page_container,
        #     size="lg",
        #     # pt=20,
        #     style={"marginLeft": 190},
        # ),
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
        title = page_mapping[path]
        if title == 'Home':
            title = 'Mitigation Effectiveness Monitoring Design'

        return title
    else:
        raise dash.exceptions.PreventUpdate


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=80)

# if __name__ == '__main__':
#     app.run_server(debug=True, host='0.0.0.0', port=8000)
