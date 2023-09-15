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

base_path = pathlib.Path('/home/mike/data/OLW')

lakes_sims_h5_path = '/home/mike/data/OLW/web_app/output/assets/lakes_power_combo.h5'

output_csv = 'power_calcs_at_all_lakes_v02.csv'
output2_csv = 'n_samples_at_all_lakes_gte_80_power_v02.csv'

conc_perc = [70]

n_years = [5, 20]
n_samples_year = [6, 12, 26, 52, 104, 364]

n_samples0 = hdf5tools.utils.cartesian([n_years, n_samples_year])
n_samples = np.prod(n_samples0, axis=1)

n_samples_df = pd.DataFrame(n_samples0, columns=['n_years', 'n_samples_year'])
n_samples_df['n_samples'] = n_samples

#####################################################
### Estimate power

lake_sims = xr.open_dataset(lakes_sims_h5_path, engine='h5netcdf')

lake_sims1 = lake_sims.sel(conc_perc=conc_perc, n_samples=n_samples).power_modelled.astype('int8')

power_data0 = lake_sims1.to_dataframe().reset_index()

power_data1 = pd.merge(n_samples_df, power_data0, on='n_samples').drop('n_samples', axis=1)

power_data1['conc_perc'] = 100 - power_data1['conc_perc']
power_data1 = power_data1.rename(columns={'conc_perc': 'reduction'})

power_data1.set_index(['indicator', 'LFENZID', 'reduction', 'n_years', 'n_samples_year']).to_csv(base_path.joinpath(output_csv))


### Estimate n samples
lake_sims2 = lake_sims.sel(conc_perc=conc_perc).power_modelled.astype('int8').to_dataframe().power_modelled
lake_sims3 = lake_sims2[lake_sims2 >= 80].reset_index()

lake_sims4 = pd.merge(n_samples_df, lake_sims3, on='n_samples').drop('n_samples', axis=1)

lake_sims4['conc_perc'] = 100 - lake_sims4['conc_perc']
lake_sims4 = lake_sims4.rename(columns={'conc_perc': 'reduction'})

lake_sims4.set_index(['indicator', 'LFENZID', 'reduction', 'n_years', 'n_samples_year']).to_csv(base_path.joinpath(output2_csv))


































