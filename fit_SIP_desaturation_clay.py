#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 09:45:59 2022
@author: ben
"""

from lib_cc_fit import utils

clay_cont = [2,4,6,8] # clay content ?
sat_idx = [] #np.arange(0,21) # 3 # saturation
max_freq = 1e3 # minimum freq to fit
min_freq = 1e-3 # maximum freq to fit
nr = 1  # number of peaks?
ini_val = 1 # initial guess of CC params

# Theta value not calculated for sc == 2!

data_mat = utils.crawl_excel(clay_cont= [2,4,6,8],
            sat_idx=[],
            nr = nr,
            ini_val = 1,
            max_freq = max_freq,
            min_freq = min_freq,
            plot_raw = False
            )

utils.plot_CC_matrice(data_mat, 'CC_matrice.png')


