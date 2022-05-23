#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 09:45:59 2022
@author: ben
"""

from lib_cc_fit import utils
import pandas as pd

sc_nb = [2,4,6,8] # varying parameters = excel tabs
sat_idx = [] #np.arange(0,21) # 3 # saturation
max_freq = 1e3 # minimum freq to fit
min_freq = 1e-3 # maximum freq to fit
nr = 2 # number of peaks?
ini_val = 1 # initial guess of CC params

# Theta value not calculated for sc == 2!

#%% note

# !! Tau values are log-transformed during fitting! need to apply exp operator before ploting


#%% excel files
files = [ 'rawData/III-IV_sc_2_4_6_8_calc_cleaned2.xlsx',
          'rawData/V_scs2_2_4_6_8_calc_cleaned2.xlsx'
         ]

#%% varying clay content, fixed salinity

varying_parm_name = '% of clay (a)'
data_mat = utils.crawl_excel(varying_parm= sc_nb,
                              filename=files[0],
                              sat_idx=[],
                              nr = nr,
                              ini_val = 1,
                              max_freq = max_freq,
                              min_freq = min_freq,
                              plot_raw = False,
                              varying_parm_name = varying_parm_name,
                              no_filter = True,
                            )
# add error bars to points based on RMS

savename = 'CC_matrice'+varying_parm_name

# for ext in ['.png','.eps','.svg']:
#     utils.plot_CC_matrice(data_mat,varying_parm_name, savename + ext)

for ext in ['.png','.eps','.svg']:
    utils.plot_CC_matrice(data_mat,varying_parm_name, savename + 'n_peaks' + str(nr) + ext, 
                          minmax_y_tau=[-15,5],
                          minmax_y_m=[0,0.5],
                          minmax_y_c=[0.1,0.45],
                          minmax_y_rho0=[3,6],
                          )
    

#%% varying salinity content, fixed clay content

varying_parm_name = '% of clay (b)'
data_mat = utils.crawl_excel(varying_parm= sc_nb,
                             filename=files[1],
                             sat_idx=[],
                             nr = nr,
                             ini_val = 1,
                             max_freq = max_freq,
                             min_freq = min_freq,
                             plot_raw = False,
                             varying_parm_name = varying_parm_name,
                             no_filter = True,
                            )

savename = 'CC_matrice'+varying_parm_name
for ext in ['.png','.eps','.svg']:
    # utils.plot_CC_matrice(data_mat,varying_parm_name, savename + ext)

    utils.plot_CC_matrice(data_mat,varying_parm_name, savename + 'n_peaks' + str(nr) + ext, 
                          minmax_y_tau=[-15,5],
                          minmax_y_m=[0,0.5],
                          minmax_y_c=[0.1,0.45],
                          minmax_y_rho0=[3,6],
                          )
    
    
    
