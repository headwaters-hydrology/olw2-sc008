#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 14:06:48 2022

@author: mike
"""
import dash
from dash import html, dcc

dash.register_page(__name__, path='/')

intro_text = """
#### Introduction
This webpage contains various dashboards that help provide freshwater stewards and
kaitiaki with new tools to decide on what to measure, where and when, and
how much it will cost to monitor the freshwater outcomes of their actions.

Click on a link on the left to go to a specific dashboard.

##### **Disclaimer**
**This webpage is currently under active development and should not be used for operational purposes.**

"""

attribution_text = """
#### Data attribution
A number of datasets were used in the development of this app.

* Regional and District Council monitoring data licensed under the [CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/)
* [New Zealand River Environment Classification (REC)](https://data.mfe.govt.nz/layer/51845-river-environment-classification-new-zealand-2010/) version 2.5 - Data licensed by the Ministry for the Environment (MfE) under the [CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/)
* [New Zealand Land Cover Database (LCDB)](https://lris.scinfo.org.nz/layer/104400-lcdb-v50-land-cover-database-version-50-mainland-new-zealand/) version 5 - Data licensed by Landcare Research under the [CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/)
* [New Zealand Primary Land Parcels](https://data.linz.govt.nz/layer/50823-nz-primary-land-parcels/) - Data licensed by LINZ under the [CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/)
* [Freshwater Ecosystems New Zealand](https://www.doc.govt.nz/our-work/freshwater-ecosystems-of-new-zealand/) - Data licensed by DOC under the [CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/)

"""

layout = html.Div(children=[
    dcc.Markdown(intro_text),
    dcc.Markdown(attribution_text, id="attribution")
], className='eight columns')
