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
Many datasets were used in the development of this app.

* Regional and District Council monitoring data licensed under the [CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/)
* [New Zealand River Environment Classification (REC)](https://data.mfe.govt.nz/layer/51845-river-environment-classification-new-zealand-2010/) version 2.5 - Data licensed by the Ministry for the Environment (MfE) under the [CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/)
* [New Zealand Land Cover Database (LCDB)](https://lris.scinfo.org.nz/layer/104400-lcdb-v50-land-cover-database-version-50-mainland-new-zealand/) version 5 - Data licensed by Landcare Research under the [CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/)
* [New Zealand Primary Land Parcels](https://data.linz.govt.nz/layer/50823-nz-primary-land-parcels/) - Data licensed by LINZ under the [CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/)
* [Freshwater Ecosystems New Zealand](https://www.doc.govt.nz/our-work/freshwater-ecosystems-of-new-zealand/) - Data licensed by DOC under the [CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/)
* [Regional Council 2023 Clipped](https://datafinder.stats.govt.nz/layer/111181-regional-council-2023-clipped-generalised/) - Data licensed by Stats NZ under the [CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/)
* [River water quality: Nitrogen, modelled, 2016 - 2020](https://data.mfe.govt.nz/layer/109888-river-water-quality-nitrogen-modelled-2016-2020/) - Data licensed by MfE under the [CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/)
* [River water quality: Phosphorus, modelled, 2016 - 2020](https://data.mfe.govt.nz/layer/109934-river-water-quality-phosphorus-modelled-2016-2020/) - Data licensed by MfE under the [CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/)
* [River water quality: Escherichia coli, modelled, 2016 - 2020](https://data.mfe.govt.nz/layer/109886-river-water-quality-escherichia-coli-modelled-2016-2020/) - Data licensed by MfE under the [CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/)
* [Updated suspended sediment yield estimator and estuarine trap efficiency model results 2019](https://data.mfe.govt.nz/layer/103686-updated-suspended-sediment-yield-estimator-and-estuarine-trap-efficiency-model-results-2019/) - Data licensed by MfE under the [CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/)
* [Marae of Aotearoa](https://www.arcgis.com/home/item.html?id=3b9e52a2012a4e4cb434e07ce19b36dd) - Data licensed by Te Puni Kōkiri under the [CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/)

#### Research articles
* [Quantifying contaminant losses to water from pastoral land uses in New Zealand III. What could be achieved by 2035?](https://doi.org/10.1080/00288233.2020.1844763)


"""

layout = html.Div(children=[
    dcc.Markdown(intro_text),
    dcc.Markdown(attribution_text, id="attribution")
], className='eight columns')
