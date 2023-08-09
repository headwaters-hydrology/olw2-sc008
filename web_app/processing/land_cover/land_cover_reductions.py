#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 13:11:07 2022

@author: mike
"""
import os
from gistools import vector, rec
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import intersection
import hdf5tools
import xarray as xr
import dbm
import booklet
import pickle
import zstandard as zstd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.pipeline import make_pipeline

import sys
if '..' not in sys.path:
    sys.path.append('..')

import utils

pd.options.display.max_columns = 10

##########################################
### land use/cover

lcdb_reductions = {'nitrogen': {
                       'Exotic Forest': 0,
                       'Forest - Harvested': 0,
                       'Orchard, Vineyard or Other Perennial Crop': 15,
                       'Short-rotation Cropland': 30,
                       'Built-up Area (settlement)': 0,
                       'High Producing Exotic Grassland': 30,
                       'Low Producing Grassland': 10,
                       'Mixed Exotic Shrubland': 0
                       },
                   'phosphorus': {
                       'Exotic Forest': 30,
                       'Forest - Harvested': 30,
                       'Orchard, Vineyard or Other Perennial Crop': 50,
                       'Short-rotation Cropland': 50,
                       'Built-up Area (settlement)': 0,
                       'High Producing Exotic Grassland': 30,
                       'Low Producing Grassland': 10,
                       'Mixed Exotic Shrubland': 0
                       },
                   'sediment': {
                       'Exotic Forest': 30,
                       'Forest - Harvested': 30,
                       'Orchard, Vineyard or Other Perennial Crop': 50,
                       'Short-rotation Cropland': 50,
                       'Built-up Area (settlement)': 0,
                       'High Producing Exotic Grassland': 30,
                       'Low Producing Grassland': 10,
                       'Mixed Exotic Shrubland': 0
                       },
                   'e.coli': {
                       'Exotic Forest': 0,
                       'Forest - Harvested': 0,
                       'Orchard, Vineyard or Other Perennial Crop': 0,
                       'Short-rotation Cropland': 30,
                       'Built-up Area (settlement)': 0,
                       'High Producing Exotic Grassland': 30,
                       'Low Producing Grassland': 10,
                       'Mixed Exotic Shrubland': 0
                       },
                   }


snb_reductions = {'sediment': 30,
                  'e.coli': 35
                  }
dairy_reductions = {'sediment': 65,
                  'e.coli': 75
                  }

features_cols = ['Climate', 'Slope', 'Drainage', 'Wetness']

param_mapping = {'Visual Clarity': 'sediment',
                 'E.coli': 'e.coli',
                 'Dissolved reactive phosporus': 'phosphorus',
                 'Ammoniacal nitrogen': 'nitrogen',
                 'Nitrate': 'nitrogen',
                 'Total nitrogen': 'nitrogen',
                 'Total phosphorus': 'phosphorus',
                 'Chlorophyll a': 'e.coli',
                 'Total Cyanobacteria': 'e.coli',
                 'Secchi Depth': 'sediment'
                 }


def land_cover_reductions():
    ### Apply the default reductions
    ## SnB
    snb1 = pd.read_csv(utils.snb_typo_path, header=[0, 1])
    phos1 = snb1['Phosphorus'].copy()
    phos2 = ((1 - (phos1['2035 potential load (kg)'] / phos1['2015 current load (kg)'])) * 100).round().astype('int8')
    phos2.name = 'phosphorus'
    nitrate1 = snb1['Nitrogen'].copy()
    nitrate2 = ((1 - (nitrate1['2035 potential load (kg)'] / nitrate1['2015 current load (kg)'])) * 100).round().astype('int8')
    nitrate2.name = 'nitrogen'
    snb2 = pd.concat([snb1['typology']['typology'], phos2, nitrate2], axis=1)
    snb2['typology'] = snb2.typology.str.title()
    snb2['typology'] = snb2['typology'].str.replace('Bop', 'BoP')
    for col, red in snb_reductions.items():
        snb2[col] = red

    # Determine missing typologies for snb
    snb_geo = gpd.read_feather(utils.snb_geo_clean_path)

    miss_snb = snb_geo[['typology']][~snb_geo['typology'].isin(snb2.typology)].drop_duplicates(subset=['typology'])

    if not miss_snb.empty:
        raise ValueError('What the heck!')

        snb2['base'] = snb2['typology'].apply(lambda x: x.split('(')[0])
        extra_snb = snb2.groupby('base')[['phosphorus', 'nitrogen']].mean().round().astype('int8')
        snb2.drop('base', axis=1, inplace=True)
        miss_snb['base'] = miss_snb['typology'].apply(lambda x: x.split('(')[0])
        miss_snb1 = pd.merge(miss_snb, extra_snb.reset_index(), on='base').drop('base', axis=1)

        snb2 = pd.concat([snb2, miss_snb1])

    snb3 = snb_geo.merge(snb2, on='typology')

    ## dairy
    dairy1 = pd.read_csv(utils.dairy_typo_path, header=[0, 1])
    phos1 = dairy1['Phosphorus'].copy()
    phos2 = ((1 - (phos1['2035 potential load (kg)'] / phos1['2015 current load (kg)'])) * 100).round().astype('int8')
    phos2.name = 'phosphorus'
    nitrate1 = dairy1['Nitrogen'].copy()
    nitrate2 = ((1 - (nitrate1['2035 potential load (kg)'] / nitrate1['2015 current load (kg)'])) * 100).round().astype('int8')
    nitrate2.name = 'nitrogen'
    dairy2 = pd.concat([dairy1['typology'], phos2, nitrate2], axis=1)
    dairy2 = dairy2.replace({'Wetness': {'Irrig': 'Irrigated'},
                             'Slope': {'Mod': 'Moderate',
                                       'Flat': 'Low'}})

    for col, red in dairy_reductions.items():
        dairy2[col] = red

    # Determine missing typologies for dairy
    dairy_geo = gpd.read_feather(utils.dairy_geo_clean_path)

    d_typo1 = dairy2['Climate'] + '/' + dairy2['Slope'] + '/' + dairy2['Drainage'] + '/' + dairy2['Wetness']

    g_typo1 = pd.Series(dairy_geo['typology'][~dairy_geo['typology'].isin(d_typo1)].unique()).copy()
    t1 = g_typo1.str.split('/')
    typo2 = pd.DataFrame.from_dict(dict(zip(t1.index, t1.values)), orient='index')
    typo2.columns = features_cols
    typo2 = typo2.drop_duplicates()
    typo3 = typo2.copy()

    t2 = pd.Series(dairy_geo['typology'].unique()).str.split('/')
    typo_all = pd.DataFrame.from_dict(dict(zip(t2.index, t2.values)), orient='index')
    typo_all.columns = features_cols

    # The model
    ordinal_encoder = make_column_transformer(
        (
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
            make_column_selector(dtype_include="category"),
        ),
        remainder="passthrough",
        # Use short feature names to make it easier to specify the categorical
        # variables in the HistGradientBoostingRegressor in the next step
        # of the pipeline.
        verbose_feature_names_out=False,
    )
    ordinal_encoder.fit(typo_all.astype('category'))

    # model = make_pipeline(
    # ordinal_encoder, HistGradientBoostingRegressor(loss='squared_error', max_iter=100, learning_rate=0.05, early_stopping=False, categorical_features=list(range(4))))
    # model = HistGradientBoostingRegressor(loss='squared_error', categorical_features=list(range(4)))
    # model = RandomForestRegressor()

    train_features = dairy2[features_cols].astype('category').copy()

    for param in ['phosphorus', 'nitrogen']:
        model = GradientBoostingRegressor(loss='absolute_error')

        train_targets = dairy2[param].copy()

        model.fit(ordinal_encoder.transform(train_features), train_targets.values)

        typo3[param] = model.predict(ordinal_encoder.transform(typo2.astype('category'))).round().astype('int8')

    # Export the results for checking
    typo3.to_csv(utils.dairy_model_typo_path, index=False)

    for col, red in dairy_reductions.items():
        typo3[col] = red

    # Combine with original data
    dairy3 = pd.concat([dairy2, typo3])
    dairy3['typology'] = dairy3['Climate'] + '/' + dairy3['Slope'] + '/' + dairy3['Drainage'] + '/' + dairy3['Wetness']

    dairy4 = dairy_geo.merge(dairy3.drop(features_cols, axis=1), on='typology')

    combo1 = pd.concat([snb3, dairy4])
    for param, col in param_mapping.items():
        combo1[param] = combo1[col].copy()

    combo2 = combo1.drop(set(param_mapping.values()), axis=1).reset_index(drop=True)

    # combo1['default_reductions'] = combo1[['phosphorus', 'nitrogen']].mean(axis=1).round().astype('int8')

    utils.gpd_to_feather(combo2, utils.snb_dairy_red_path)

    combo3 = combo2.drop('geometry', axis=1).drop_duplicates(subset=['typology'])
    lcdb_extras = combo3.drop(['farm_type', 'typology'], axis=1).groupby('land_cover').mean().round().astype('int8')

    ### LCDB
    lcdb0 = gpd.read_feather(utils.lcdb_clean_path)

    lcdb_red_list = []
    for col, red in lcdb_reductions.items():
        t1 = pd.DataFrame.from_dict(red, orient='index')
        t1.index.name = 'typology'
        t1.columns = [col]
        lcdb_red_list.append(t1)

    lcdb_red = pd.concat(lcdb_red_list, axis=1).reset_index()

    lcdb1 = lcdb0.merge(lcdb_red, on='typology')
    for param, col in param_mapping.items():
        lcdb1[param] = lcdb1[col].copy()

    lcdb2 = lcdb1.drop(set(param_mapping.values()), axis=1).reset_index(drop=True)

    lcdb2.loc[lcdb2.land_cover == 'Low Producing Grassland', lcdb_extras.columns] = lcdb_extras.loc['Sheep and Beef'].values
    lcdb2.loc[lcdb2.land_cover == 'High Producing Exotic Grassland', lcdb_extras.columns] = lcdb_extras.loc['Dairy'].values

    utils.gpd_to_feather(lcdb2, utils.lcdb_red_path)

    ## Save the reductions only
    combo3 = combo2.drop('geometry', axis=1).drop_duplicates(subset=['typology'])

    lcdb3 = lcdb2.drop('geometry', axis=1).drop_duplicates(subset=['typology'])

    lc_red = pd.concat([combo3, lcdb3])

    lc_red.to_csv(utils.lc_red_csv_path, index=False)




    # combo2 = pd.concat([snb2, combo1.drop(features_cols, axis=1), lcdb_red])

    # combo3 = lc0.merge(combo2, on='typology', how='left')
    # combo3['default_reductions'] = combo3[['phosphorus', 'nitrogen']].mean(axis=1).round().astype('int8')

    # combo3.to_file(utils.lc_red_gpkg_path)

    # ## Simplify for app
    # combo4 = combo3[combo3.default_reductions > 0].copy()
    # combo4['geometry'] = combo4['geometry'].buffer(1).simplify(1)
    # combo4 = combo4.dissolve('typology')

    # utils.gpd_to_feather(combo4.reset_index(), utils.lc_red_feather_path)





































































































