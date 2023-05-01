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

import utils

pd.options.display.max_columns = 10

##########################################
### land use/cover

lcdb_reductions = {'Exotic Forest': 0, 'Forest - Harvested': 0, 'Orchard, Vineyard or Other Perennial Crop': 10, 'Short-rotation Cropland': 30}

features_cols = ['Climate', 'Slope', 'Drainage', 'Wetness']


def land_cover_reductions():
    ### land cover
    lc0 = gpd.read_feather(utils.lc_clean_diss_path)

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

    # Combine with original data
    combo1 = pd.concat([dairy2, typo3])
    combo1['typology'] = combo1['Climate'] + '/' + combo1['Slope'] + '/' + combo1['Drainage'] + '/' + combo1['Wetness']

    lcdb_red_list = []
    for param in ['phosphorus', 'nitrogen']:
        t1 = pd.DataFrame.from_dict(lcdb_reductions, orient='index')
        t1.index.name = 'typology'
        t1.columns = [param]
        lcdb_red_list.append(t1)

    lcdb_red = pd.concat(lcdb_red_list, axis=1).reset_index()
    combo2 = pd.concat([snb2, combo1.drop(features_cols, axis=1), lcdb_red])

    combo3 = lc0.merge(combo2, on='typology', how='left')
    combo3['default_reductions'] = combo3[['phosphorus', 'nitrogen']].mean(axis=1).round().astype('int8')

    combo3.to_file(utils.lc_red_gpkg_path)

    ## Simplify for app
    combo4 = combo3[combo3.default_reductions > 0].copy()
    combo4['geometry'] = combo4['geometry'].buffer(1).simplify(1)
    combo4 = combo4.dissolve('typology')

    utils.gpd_to_feather(combo4.reset_index(), utils.lc_red_feather_path)





































































































