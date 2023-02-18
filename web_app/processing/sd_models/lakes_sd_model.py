#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:53:39 2023

@author: mike
"""
import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
# from sklearn.preprocessing import PowerTransformer
from sklearn import linear_model, kernel_ridge
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score, cross_val_predict
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
import scipy
import hdf5tools


if '..' not in sys.path:
    sys.path.append('..')

import utils


pd.options.display.max_columns = 10

################################################
### Parameters

min_area = 500000

cat_cols = ['Current5', 'GeomorphicType']
num_cols = ['MaxDepth', 'LakeArea', 'DecTemp', 'DecSolrad', 'Fetch', 'SumWind', 'CatBeech', 'CatGlacial', 'CatHard', 'CatPeat', 'CatPhos', 'CatSlope', 'CatAnnTemp', 'DirectDistCoast', 'ResidenceTime', 'Urban', 'Pasture', 'LakeElevation', 'MeanWind']
model_cols = cat_cols + num_cols

scoring = "neg_mean_absolute_percentage_error"
n_cv_folds = 5

model_names = ['BoostingRegressor', 'RandomForestRegressor', 'Ridge', 'Lasso', 'BayesianRidge', 'ARDRegression', 'LinearRegression']

encodings = {'stdev': {'scale_factor': 0.001, '_FillValue': -999, 'dtype': 'int16'},
             'data_mean': {'scale_factor': 0.001, '_FillValue': -999, 'dtype': 'int16'},
             'data_stdev': {'scale_factor': 0.001, '_FillValue': -999, 'dtype': 'int16'},
             'data_min': {'scale_factor': 0.001, '_FillValue': -999, 'dtype': 'int16'},
             'data_max': {'scale_factor': 0.001, '_FillValue': -999, 'dtype': 'int16'}
             }

################################################
### Functions


def reg_transform(x, y, slope, intercept):
    """

    """
    y_new = y - (slope*x + intercept)
    return y_new



################################################
### Individual SD from regressions

data0 = pd.read_feather(utils.lakes_data_clean_path)

grp1 = data0.groupby(['indicator'])['value']
for i, v in grp1:
    new_data = np.log(v)
    data0.loc[data0.indicator == i, 'value'] = new_data


grp2 = data0.groupby(['LawaSiteID', 'indicator'])[['date', 'value']]

stdev_list = []
for i, v in grp2:
    x = v.date.values.astype('datetime64[D]').astype(int)
    y = v.value
    slope, intercept, r, p, se = scipy.stats.linregress(x, y)
    if p < 0.05:
        new_y = []
        for x1, y1 in zip(x, y):
            new_y.append(reg_transform(x1, y1, slope, intercept))
        new_y = np.array(new_y)
    else:
        new_y = y - y.mean()

    stdev = new_y.std()
    stdev_list.append([i[0], i[1], stdev])

stdev_df = pd.DataFrame(stdev_list, columns=['LawaSiteID', 'indicator', 'stdev'])

stdev_df1 = pd.merge(data0[['LawaSiteID', 'LFENZID']].drop_duplicates(), stdev_df, on=['LawaSiteID'])

stdev_df2 = stdev_df1.groupby(['LFENZID', 'indicator'])['stdev'].mean().reset_index()

ind_count = stdev_df2.groupby('indicator')['LFENZID'].count()
indicators = ind_count[ind_count > 10].index.values

stdev_df3 = stdev_df2[stdev_df2.indicator.isin(indicators)].copy()

###########################################
### Global model

lakes_class0 = pd.read_csv(utils.lakes_class_csv)
lakes_class0 = lakes_class0.rename(columns={'LID': 'LFENZID'})
lakes_class0 = lakes_class0.dropna(subset=model_cols).copy()

lakes_class0[cat_cols] = lakes_class0[cat_cols].astype("category")

for cat_col in cat_cols:
    lakes_class0[cat_col] = lakes_class0[cat_col].cat.codes

lakes_class1 = lakes_class0[lakes_class0.Name.notnull() & lakes_class0.LFENZID.notnull()].copy()
lakes_class1['LFENZID'] = lakes_class1['LFENZID'].astype('int32')
lakes_class1 = lakes_class1.drop_duplicates(subset=['LFENZID'])

data_lakes0 = pd.merge(stdev_df3, lakes_class0, on='LFENZID')

# Prep container
lids = lakes_class1.LFENZID.unique()
indicators = data_lakes0.indicator.unique()

filler1 = np.empty((len(model_names), len(indicators), len(lids)), dtype='float')
filler2 = np.empty((len(model_names), len(indicators)), dtype='int16')
filler3 = np.empty((len(indicators)), dtype='float')
filler4 = np.empty((len(indicators)), dtype='int16')

results = xr.Dataset(data_vars={'stdev': (('model', 'indicator', 'LFENZID'), filler1.copy()),
                                'stdev_error_perc': (('model', 'indicator'), filler2.copy()),
                                'stdev_error_stdev_perc': (('model', 'indicator'), filler2.copy()),
                                'data_mean': (('indicator'), filler3.copy()),
                                'data_stdev': (('indicator'), filler3.copy()),
                                'data_min': (('indicator'), filler3.copy()),
                                'data_max': (('indicator'), filler3.copy()),
                                'sample_size': (('indicator'), filler4.copy()),

                              },
                   coords={'indicator': indicators,
                           'LFENZID': lids,
                           'model': model_names
                           }
                   )

for f, enc in encodings.items():
    results[f].encoding = enc


for indicator in indicators:
    print(indicator)

    ## Model build
    tn_data = data_lakes0[data_lakes0.indicator == indicator]

    X = tn_data[cat_cols + num_cols].copy()
    # X[cat_cols] = X[cat_cols].astype("category")

    y = tn_data['stdev']

    data_mean = round(y.mean(), 3)
    data_stdev = round(y.std(), 3)
    data_min = round(y.min(), 3)
    data_max = round(y.max(), 3)
    sample_size = len(y)

    results['data_mean'].loc[{'indicator': indicator}] = data_mean
    results['data_stdev'].loc[{'indicator': indicator}] = data_stdev
    results['data_min'].loc[{'indicator': indicator}] = data_min
    results['data_max'].loc[{'indicator': indicator}] = data_max
    results['sample_size'].loc[{'indicator': indicator}] = sample_size

    categorical_columns = X.columns.isin(cat_cols)

    br_model = make_pipeline(
        HistGradientBoostingRegressor(
            min_samples_leaf=2,
            # random_state=42,
            categorical_features=categorical_columns,
        ),
    )

    rf_model = make_pipeline(
        RandomForestRegressor(
            # min_samples_leaf=2,
            # random_state=42,
            # categorical_features=categorical_columns,
        ),
    )

    ridge_model = make_pipeline(
        linear_model.Ridge(
            # random_state=42,
            # categorical_features=categorical_columns,
        ),
    )

    lasso_model = make_pipeline(
        linear_model.Lasso(
            # random_state=42,
            # categorical_features=categorical_columns,
        ),
    )

    b_ridge_model = make_pipeline(
        linear_model.BayesianRidge(
            # random_state=42,
            # categorical_features=categorical_columns,
        ),
    )

    b_ard_model = make_pipeline(
        linear_model.ARDRegression(
            # random_state=42,
            # categorical_features=categorical_columns,
        ),
    )

    kr_model = make_pipeline(
        kernel_ridge.KernelRidge(
            # random_state=42,
            # categorical_features=categorical_columns,
        ),
    )

    lr_model = make_pipeline(
        linear_model.LinearRegression(
            # random_state=42,
            # categorical_features=categorical_columns,
        ),
    )

    models = {'BoostingRegressor': br_model, 'RandomForestRegressor': rf_model, 'Ridge': ridge_model, 'Lasso': lasso_model, 'BayesianRidge': b_ridge_model, 'ARDRegression': b_ard_model, 'LinearRegression': lr_model}

    for model_name, model in models.items():
        native_result = cross_val_score(model, X, y, cv=n_cv_folds, scoring=scoring)

        mean_score = np.abs(native_result).mean()
        sd_score = np.abs(native_result).std()

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        # m1 = hist_native.fit(X_train, y_train)

        m1 = model.fit(X, y)

        ## Prediction
        predict_data = lakes_class1.copy()

        Xn = predict_data[cat_cols + num_cols].copy()
        # Xn[cat_cols] = Xn[cat_cols].astype("category")

        predictions = model.predict(Xn)

        results['stdev_error_perc'].loc[{'model': model_name, 'indicator': indicator}] = int(round(mean_score*100))
        results['stdev_error_stdev_perc'].loc[{'model': model_name, 'indicator': indicator}] = int(round(sd_score*100))
        results['stdev'].loc[{'model': model_name, 'indicator': indicator}] = predictions.round(3)


hdf5tools.xr_to_hdf5(results, utils.lakes_stdev_path)




































































