#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 09:45:59 2022

@author: ben
"""
from CC_fit_src_MW.lib.lib_cc_fit import cc_fit as cc_fit

from utils import load_excel, filter_data, plot_data_spectra

# Dataset is:
# 59 frequencies, 21 saturation rate


clay_cont = 2 #[2,4,6,8] # clay content ?
sat_idx = 3 # 3 # saturation
max_freq = 1e3
min_freq = 1e-3
nr = 2 
ini_val = 2
    
data = load_excel(clay_cont,min_freq=min_freq,max_freq=max_freq)    
sat = data['sat']
path = './'

lab_SIP = cc_fit.cc_fit()
lab_SIP.load_data(data['data_asc'][:,sat_idx],ignore=data['ign_freq'])  # you can ignore frequencies here
id_2_rmv = filter_data(data['data_asc'][:,sat_idx],data['freq'])

if len(id_2_rmv)>1:
    pass

else:
    lab_SIP.data     
    lab_SIP.load_frequencies_array(data['freq_asc'].to_numpy(),ignore= data['ign_freq'])  # you can ignore frequencies here
    lab_SIP.set_nr_cc_terms(nr=nr)
    lab_SIP.set_initial_values(ini_val)
    lab_SIP.fit_all_spectra()
    lab_SIP.plot_all_spectra(path+'./', prefix=str(sat[sat_idx]))

