#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 09:45:59 2022

@author: ben
"""
import matplotlib.pyplot as plt
import matplotlib as mpl

import lib_cc_fit.cc_fit as cc_fit
import numpy as np
from lib_cc_fit import utils

# from utils import (load_excel, 
#                    filter_data, 
#                    plot_data_spectra,
#                    map_color)

# Dataset is:
# 59 frequencies, 21 saturation rate


clay_cont = [2] #[2,4,6,8] # clay content ?
sat_idx = [] # 3 # saturation
max_freq = 1e3
min_freq = 1e-3
nr = 1 
ini_val = 1
    


import matplotlib.colors as mcolors
import matplotlib.cm
import matplotlib.colors
import seaborn as sns



clist = [(0, 'powderblue'), (1, 'darkblue')]
colors = mcolors.LinearSegmentedColormap.from_list("", clist)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", clist)
matplotlib.cm.register_cmap("mycolormap",cmap)
#https://stackoverflow.com/questions/43805821/matplotlib-add-colorbar-to-non-mappable-object




for cc in enumerate(clay_cont): # Loop over sc

    fig, axs = plt.subplots(2, 1, figsize=(7, 6), dpi=300)

    data = utils.load_excel(cc[1],min_freq=min_freq,max_freq=max_freq)    
    if len(sat_idx)>0:
        sat = sat_idx
    else:
        sat = data['sat']

    cpal = sns.color_palette("mycolormap", n_colors=len(sat)+1, desat=1)

    for ss in enumerate(sat): # Loop over saturation
        path = './'
        
        lab_SIP = cc_fit.cc_fit()
        lab_SIP.load_data(data['data_asc'][:,ss[0]],ignore=data['ign_freq'])  # you can ignore frequencies here
        id_2_rmv = utils.filter_data(data['data_asc'][:,ss[0]],data['freq'])
        
        if len(id_2_rmv)>1:
            pass
        
        else:
            
            # sat_new = 
            lab_SIP.data     
            lab_SIP.load_frequencies_array(data['freq_asc'].to_numpy(),ignore= data['ign_freq'])  # you can ignore frequencies here
            lab_SIP.set_nr_cc_terms(nr=nr)
            lab_SIP.set_initial_values(ini_val)
            lab_SIP.fit_all_spectra()
            # lab_SIP.plot_all_spectra(path+'./', prefix=str(ss[1]), ax=ax)

            plt = utils.plot_data_spectra(lab_SIP.data, lab_SIP.frequencies, path+'./', 
                              prefix=str(ss[1]), 
                              axes=[axs,fig], 
                              c=cpal[len(sat)-ss[0]],
                              label=str(ss))

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# fig.colorbar(im, cax=cbar_ax)

norm = mpl.colors.Normalize(vmin=min(sat),vmax=max(sat))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ticks=np.linspace(min(sat),max(sat),int(len(sat)/2)), 
             boundaries=np.arange(min(sat),max(sat),2),cax=cbar_ax)
cbar.set_label('# saturation')

plt.show()