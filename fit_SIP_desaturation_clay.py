#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 09:45:59 2022

@author: ben
"""
import pandas as pd
import matplotlib.pyplot as plt
import lib_cc_fit.cc_fit as cc_fit
import numpy as np
import os
import sys 
import seaborn as sns

from utils import load_excel, filter_data

# Dataset is:
# 59 frequencies, 21 saturation rate


clay_cont = [2,4,6] # clay content ?
sat_idx = [] #np.arange(0,21) # 3 # saturation
max_freq = 1e3 # minimum freq to fit
min_freq = 1e-3 # maximum freq to fit
nr = 1  # number of peaks?
ini_val = 1 # initial guess of CC params

# Theta value not calculated for sc == 2!

    
# def scatterCC(key='sat'):
    
CC_mat = [] # store ColeCole parameters per saturation per clay content

for cc in enumerate(clay_cont): # Loop over sc

    data = load_excel(cc[1],min_freq=min_freq,max_freq=max_freq)
    cc_pars_ss = []
    
    if len(sat_idx)>0:
        sat = sat_idx
    else:
        sat = data['sat']

        
    for ss in enumerate(sat): # Loop over saturation
       
        path = './'
        
        lab_SIP = cc_fit.cc_fit()
        lab_SIP.load_data(data['data_asc'][:,ss[0]],ignore=data['ign_freq'])  # you can ignore frequencies here
        id_2_rmv = filter_data(data['data_asc'][:,ss[0]],data['freq'])
        
        if len(id_2_rmv)>1:
            cc_pars_ss.append([np.nan,np.nan,np.nan,np.nan,ss[1]])
            pass
        
        else:
            lab_SIP.data     
            lab_SIP.load_frequencies_array(data['freq_asc'].to_numpy(),ignore= data['ign_freq'])  # you can ignore frequencies here
            lab_SIP.set_nr_cc_terms(nr=nr)
            lab_SIP.set_initial_values(ini_val)
            lab_SIP.fit_all_spectra()
            lab_SIP.plot_all_spectra(path+'./', prefix=str(ss[1]))
            cc_pars_ss.append(np.r_[lab_SIP.cc_pars[0],ss[1]])
    
    
    cc_pars_ss = np.vstack(cc_pars_ss)
    CC_mat.append(cc_pars_ss)
        
CC_mat = np.array(CC_mat)
CC_mat = np.hstack(CC_mat)

#%%

# index = pd.MultiIndex.from_product(iterables, names=['i', "j"])
m,n,r = CC_mat.shape
out_arr = np.column_stack((np.repeat(np.arange(m),n),CC_mat.reshape(m*n,-1)))
CC_df = pd.DataFrame(out_arr)
CC_df.rename(columns={0:'ClayContent', 1:'C0',2:'C1',3:'C2',4:'C3',5:'saturation'},inplace=True)

CC_df['C0'].to_numpy()

# plt.scatter(CC_df['saturation'].to_numpy(), CC_df['C0'].to_numpy())

# CC_df.groupby('ClayContent', dropna=True)['C0'].plot()
# plt.show()

CC_df.reset_index()
# CC_df.pivot(index=['saturation'],,values='C0').plot()

colors = {0:'red', 1:'green', 2:'blue', 3:'yellow'}
# !pip install seaborn
# https://kanoki.org/2020/08/30/matplotlib-scatter-plot-color-by-category-in-python/
# https://seaborn.pydata.org/tutorial/regression.html


fig, axs = plt.subplots(2,2,sharex='all')

# sns.lmplot('saturation', 'C0', data=test, hue='ClayContent', fit_reg=False, ax=axs[0])
axs[0,0].scatter(CC_df['saturation'], CC_df['C0'], c=CC_df['ClayContent'].map(colors)
                 ,label=[''])
axs[0,0].set_xlabel('saturation')

axs[1,0].scatter(CC_df['saturation'], CC_df['C1'], c=CC_df['ClayContent'].map(colors))
axs[0,1].scatter(CC_df['saturation'], CC_df['C2'], c=CC_df['ClayContent'].map(colors))
axs[1,1].scatter(CC_df['saturation'], CC_df['C3'], 
                 c=CC_df['ClayContent'].map(colors))
axs[0,0].set_ylabel('rho0')
axs[1,0].set_ylabel('m')
axs[0,1].set_ylabel('tau')
axs[1,1].set_ylabel('c')
axs[1,0].set_xlabel('saturation')
axs[0,1].set_xlabel('saturation')
axs[1,1].set_xlabel('saturation')
plt.show()

