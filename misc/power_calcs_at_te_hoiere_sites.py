#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 07:39:50 2023

@author: mike
"""
import os
import numpy as np
import pandas as pd
import pathlib
import xarray as xr
import hdf5tools
import hdf5plugin
import nzrec

pd.options.display.max_columns = 10

######################################################
### Parameters

base_path = pathlib.Path('/home/mike/data/olw')

rivers_reach_error_path = base_path.joinpath('web_app/output/assets/rivers_reaches_power_modelled.h5')
# rivers_reach_error_path = base_path.joinpath('web_app/output/assets/rivers_reaches_power_monitored.h5')

sites_csv = 'te_hoiere_sites.csv'

output_csv = 'freq_at_te_hoiere_sites.csv'

reductions = [30]

n_years = [5, 20]
n_samples_year = [12, 26, 52, 104, 364]

n_samples0 = hdf5tools.utils.cartesian([n_years, n_samples_year])
n_samples = np.prod(n_samples0, axis=1)

n_samples_df = pd.DataFrame(n_samples0, columns=['n_years', 'n_samples_year'])
n_samples_df['n_samples'] = n_samples

#####################################################
### Process data

w0 = nzrec.Water('/home/mike/git/nzrec/data', download_files=False)

sites0 = pd.read_csv(base_path.joinpath(sites_csv))

power_data = xr.open_dataset(rivers_reach_error_path, engine='h5netcdf')

# power_data['nzsegment'] = power_data['nzsegment'].astype('int32')
# power_data['n_samples'] = power_data['n_samples'].astype('int16')
# power_data['conc_perc'] = power_data['conc_perc'].astype('int8')

# n_samples = power_data.n_samples.values
freqs = np.array([4, 12, 26, 52, 104, 364])
n_samples = freqs * 10

nzsegments = power_data.nzsegment.values

results_dict = {}
for i, row in sites0.iterrows():
    seg = row.nzsegment
    ind = row.indicator
    if seg not in nzsegments:
        way1 = w0.add_way(seg)
        way2 = way1.downstream()
        segs = way2.ways

        seg_list = []
        seg_index = []
        for seg in segs:
            if seg in nzsegments:
                way3 = w0.add_way(seg)
                down_len = len(way3.ways)
                seg_index.append(down_len)
                seg_list.append(seg)

        if not seg_index:
            continue
        index = np.argmax(seg_index)
        seg = seg_list[index]

    pd0 = power_data.sel(conc_perc=row.conc_perc, nzsegment=seg, n_samples=n_samples, indicator=row.indicator).copy().load()
    pd1 = pd0.where(pd0.power >= 80, drop=True)
    vals = pd1.n_samples.values
    if len(vals) == 0:
        results_dict[(row.site_id, ind)] = 0
    else:
        freq = pd1.n_samples.values[0] * 0.1
        results_dict[(row.site_id, ind)] = int(freq)




