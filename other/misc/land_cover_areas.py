#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 07:46:35 2023

@author: mike
"""

import pandas as pd
import geopandas as gpd
import booklet

pd.options.display.max_columns = 10


#####################################################
### Parameters

lcdb_red_path = '/home/mike/data/OLW/web_app/land_use/lcdb_reductions.feather'

snb_dairy_red_path = '/home/mike/data/OLW/web_app/land_use/snb_dairy_reductions.feather'

catch_lc_path = '/home/mike/data/OLW/web_app/output/assets/rivers_catch_lc.blt'

typo_nz_feather_path = '/home/mike/data/OLW/web_app/land_use/typo_nz_reductions.feather'

typo_area_nz_path = '/home/mike/data/OLW/web_app/land_use/typo_area_nz.csv'

##################################################
### Combine data for all of NZ - takes too long...

# lcdb0 = gpd.read_feather(lcdb_red_path)
# snb_dairy0 = gpd.read_feather(snb_dairy_red_path)

# lcdb1 = lcdb0.overlay(snb_dairy0, how='difference', keep_geom_type=True)
# combo2 = pd.concat([snb_dairy0, lcdb1])


#################################################
### Process by catchment and combine

results_list = []
with booklet.open(catch_lc_path) as f:
    for catch_id, lc0 in f.items():
        if not lc0.empty:
            lc1 = lc0[['typology', 'farm_type', 'land_cover', 'geometry']].copy()
            lc1['area'] = lc1.geometry.area*0.000001
            lc1.drop('geometry', axis=1, inplace=True)
            results_list.append(lc1)


results1 = pd.concat(results_list)

results2 = results1.groupby('typology')['area'].sum().round(3).sort_values()
results2.name = 'area_km2'

results2.to_csv(typo_area_nz_path)

















































