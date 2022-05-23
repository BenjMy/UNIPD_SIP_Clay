#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 09:45:59 2022
@author: ben
"""

from lib_cc_fit import cc_fit as cc_fit
from lib_cc_fit import utils

#%%
# Dataset is:
# 59 frequencies
files = [ 'rawData/III-IV_sc_2_4_6_8_calc_cleaned2.xlsx',
         'rawData/V_scs2_2_4_6_8_calc_cleaned2.xlsx'
         ]
# V_scs2_2_4_6_8_calc_cleaned2

sc_nb = 2 #[2,4,6,8] # sc_nb, nb of the excel tab ?
sat_idx = 3 # 3 # saturation
max_freq = 1e4
min_freq = 1e-3
nr = 1  #  number of Cole-Cole terms to fit
ini_val = 1


data, sat, theta = utils.load_excel(filename=files[0],
                                    sheet_nb=sc_nb,
                                    min_freq=min_freq,
                                    max_freq=max_freq
                                    )    

# we should build a datframe rather than a dict here, add collums to specify how to filter and how to fit
# pd.DataFrame.from_dict(data)
# path = './'

#%%
len(data['data_asc'])


lab_SIP = cc_fit.cc_fit()
lab_SIP.load_data(data['data_asc'][:,sat_idx],ignore=data['ign_freq'])  # you can ignore frequencies here
id_2_rmv = utils.filter_data(data['data_asc'][:,sat_idx],data['freq'])

if len(id_2_rmv)>1:
    pass

else:
    lab_SIP.data     
    lab_SIP.load_frequencies_array(data['freq_asc'],ignore= data['ign_freq'])  # you can ignore frequencies here
    lab_SIP.set_nr_cc_terms(nr=nr)
    lab_SIP.set_initial_values(ini_val)
    lab_SIP.fit_all_spectra()
    
    lab_SIP.cc_pars
    # rho0_init, m_init, np.log(0.01), 0.6
    
    
    lab_SIP.plot_all_spectra('./', prefix=str(sat[sat_idx]))


