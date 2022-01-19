#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 10:00:04 2022

@author: ben
"""
import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt


import matplotlib.colors as mcolors
import matplotlib.cm
import matplotlib.colors
import seaborn as sns

from lib_cc_fit import cc_fit as cc_fit

def map_color(n_colors, desat=1):
    
    clist = [(0, 'powderblue'), (1, 'darkblue')]
    colors = mcolors.LinearSegmentedColormap.from_list("", clist)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", clist)
    # matplotlib.cm.register_cmap("mycolormap", cmap)
    cpal = sns.color_palette("mycolormap", n_colors=n_colors, desat=1)
    
    return cpal
    
    

def filter_data(data,freq,minmax_mag=[]):
    
    data_mag = data[0:int(len(freq)/2)]
    data_phase = data[int(len(freq)/2):]

    # positive phase
    # ------------------
    id_2_rmv = list(np.where(data_phase>0)[0])
    # negative amp
    # ------------------
    id_2_rmv.append(list(np.where(data_mag<0)[0]))

    # extreme amp
    # ------------------
    id_2_rmv.append(list(np.where(data_mag>1e3)[0]))

    
    return np.hstack(id_2_rmv)


def load_excel(sheet_nb,**kwargs):
    '''
    

    Parameters
    ----------
    sheet_nb : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    dict_data_excel : TYPE
        DESCRIPTION.

    '''

    
    max_freq = 1e99
    min_freq = 1e-99
    
    if 'max_freq' in kwargs:
        max_freq = kwargs['max_freq']
    if 'min_freq' in kwargs:
        min_freq = kwargs['min_freq']   
        
        
    # Load data from excel file
    # ------------------------------------------------------------------------
    # 'III-IV_sc_2_4_6_8_calc_cleaned.xlsx'    
    # 'III-IV_sc_2_4_6_8_calc_modifiedBM.xlsx'
    
    filename = 'III-IV_sc_2_4_6_8_calc_cleaned.xlsx'
    # filename = 'III-IV_sc_2_4_6_8_calc_modifiedBM.xlsx'
    phase = pd.read_excel(filename,
                          sheet_name=str(sheet_nb) + '_phase',
                          index_col=None, header=4,
                          na_values='NA',
                          nrows= 118)  
    
    phase.rename(columns={"theta": "freq"},inplace=True)
    data_phase = phase.to_numpy()
    data_phase = data_phase[:,1:]


    # df_t = phase.T.reset_index()
    # theta = df_t['index'][1:]
    
    sat_df = pd.read_excel(filename,
                          sheet_name=str(sheet_nb) + '_resistivity',
                          index_col=0, header=2,
                          #converters={'saturation':np.float64}
                          )       
    
    df_t = sat_df.T.reset_index()
    sat = (df_t['saturation'][:].to_numpy())
    # sat = np.flip(df_t['saturation'][:].to_numpy())
    sat = sat[~np.isnan(sat)]


    magnitude = pd.read_excel(filename,
                              sheet_name=str(sheet_nb) + '_resistivity',
                              index_col=None, header=4,
                              na_values='NA',
                              nrows= 118)  
    data_mag = magnitude.to_numpy()
    data_mag = data_mag[:,1:]
    data_mag.shape
    
    print(data_mag)
    np.log(data_mag)
    
    
    # Infer saturation
    # ------------------------------------------------------------------------
    # Sat_idx = range(data_mag.shape[1])
    sat_idx = range(len(sat))
    
    
    # Infer frequencies
    # ------------------------------------------------------------------------
    freq = phase['freq']
    freq_asc = freq[0:int(len(freq)/2)]
    freq_dsc = freq[int(len(freq)/2):]
    
    # split ascending/descending freq
    # ------------------------------------------------------------------------
    data_phase_asc = data_phase[0:int(len(freq)/2)]
    data_phase_dsc = data_phase[int(len(freq)/2):]
    
    data_mag_asc = data_mag[0:int(len(freq)/2),:]
    data_mag_dsc = data_mag[int(len(freq)/2):]
    
   
    # # stack magnitudes and phases 
    # # ------------------------------------------------------------------------
    data_asc =  np.vstack([data_mag_asc,data_phase_asc])

    
    
    index = freq_asc.index.to_numpy()
    condition = ((freq_asc>=max_freq) + (freq_asc<=min_freq))
    ign_freq = index[condition]
    # ign_freq = np.array([])
    
    
    dict_data_excel = { 'sat':sat, 
                        'sat_idx':sat_idx, 
                        'freq':freq, 
                        'freq_asc':freq_asc, 
                        'freq_dsc':freq_dsc, 
                        'data_phase_asc':data_phase_asc, 
                        'data_phase_dsc':data_phase_dsc, 
                        'data_mag_asc':data_mag_asc, 
                        'data_mag_dsc':data_mag_dsc,
                        'data_asc':data_asc,
                        'ign_freq':ign_freq
                        }
    
    
    return dict_data_excel

def crawl_excel(clay_cont= [2,4,6,8], 
                sat_idx=[], 
                nr=1, 
                ini_val=1, 
                **kwargs):

    pathfig = './figs/'
    
    if not os.path.exists(pathfig):
        os.makedirs(pathfig)
        
    
    max_freq = 1e4
    min_freq = 1e-4
    
    plot_raw = False
    if 'plot_raw' in kwargs:
        plot_raw = kwargs['plot_raw']
        
    if 'max_freq' in kwargs:
        max_freq = kwargs['max_freq']
    if 'min_freq' in kwargs:
        min_freq = kwargs['min_freq']
        
    CC_mat = [] # store ColeCole parameters per saturation per clay content

    for cc in enumerate(clay_cont): # Loop over sc
    
        if plot_raw == True:
            fig, axs = plt.subplots(2, 1, figsize=(7, 6))

        data = load_excel(cc[1],
                                min_freq=min_freq,
                                max_freq=max_freq)
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
                cc_pars_ss.append([np.nan,np.nan,np.nan,np.nan,ss[1],cc[1]])
                pass
            
            else:
                lab_SIP.data     
                lab_SIP.load_frequencies_array(data['freq_asc'].to_numpy(),ignore= data['ign_freq'])  # you can ignore frequencies here
                lab_SIP.set_nr_cc_terms(nr=nr)
                lab_SIP.set_initial_values(ini_val)
                lab_SIP.fit_all_spectra()
                cc_pars_ss.append(np.r_[lab_SIP.cc_pars[0],ss[1],cc[1]])
        

                if 'plot_raw' == True:
                
                    cpal = map_color(n_colors=len(sat)+1, desat=1)
                    # cpal = sns.color_palette("mycolormap", n_colors=len(sat)+1, desat=1)
                    plot_data_spectra(lab_SIP.data, lab_SIP.frequencies, path+'./', 
                                      prefix=str(ss[1]), 
                                      axes=[axs,fig], 
                                      c=cpal[len(sat)-ss[0]],
                                      label=str(ss))
                    plt.savefig(pathfig + 'raw_data_clay' + str(cc[1]),dpi=300)
                    plt.title('Data clay' + str(cc[1]))
                
                else:
                    lab_SIP.plot_all_spectra(pathfig, prefix=str(ss[1]))

                
            
        cc_pars_ss = np.vstack(cc_pars_ss)
        CC_mat.append(cc_pars_ss)
        
    CC_sat_param_clay = np.vstack(CC_mat)
    
    return CC_sat_param_clay



def plot_CC_matrice(data_mat, savename='CC_matrice.png'):

    pathfig = './figs/'
    
    if not os.path.exists(pathfig):
        os.makedirs(pathfig)
        
    CC_df = pd.DataFrame(data_mat)
    CC_df.rename(columns={0:'C0',1:'C1',2:'C2',3:'C3',4:'saturation',5:'clay'},inplace=True)
    CC_df.reset_index()
    
    # colors = {0:'red', 1:'green', 2:'blue', 3:'yellow',4:'yellow',5:'black'}
    # !pip install seaborn
    # https://kanoki.org/2020/08/30/matplotlib-scatter-plot-color-by-category-in-python/
    # https://seaborn.pydata.org/tutorial/regression.html
    
    fig, axs = plt.subplots(2,2,sharex='all', figsize=(10,4), dpi=300)
    
    # sns.lmplot('saturation', 'C0', data=test, hue='ClayContent', fit_reg=False, ax=axs[0])
    map1 = axs[0,0].scatter(CC_df['saturation'], CC_df['C0'], c=CC_df['clay']
                     ,label=[''],cmap='viridis')
    axs[0,0].set_xlabel('saturation')
    
    axs[1,0].scatter(CC_df['saturation'], CC_df['C1'], c=CC_df['clay'])
    axs[0,1].scatter(CC_df['saturation'], CC_df['C2'], c=CC_df['clay'])
    axs[1,1].scatter(CC_df['saturation'], CC_df['C3'], 
                      c=CC_df['clay'])
    axs[0,0].set_ylabel(r'$\rho_{0}$')
    axs[1,0].set_ylabel('m')
    axs[0,1].set_ylabel(r'$\tau$')
    axs[1,1].set_ylabel('c')
    axs[1,0].set_xlabel('saturation')
    axs[0,1].set_xlabel('saturation')
    axs[1,1].set_xlabel('saturation')
    # fig.colorbar(map1, ax=axs[0,0])
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cb = fig.colorbar(map1, cax=cbar_ax)
    cb.set_label('Clay content')    
    plt.savefig(pathfig + savename, dpi=300)
    
    pass



def plot_data_spectra(data, frequencies, directory=None, prefix=None, 
                      axes=[], 
                      c=[], 
                      label=[]):
    
    pathfig = './figs/'
    
    if not os.path.exists(pathfig):
        os.makedirs(pathfig)
        
        
    if len(axes)>0:
        axs = axes[0]
        fig = axes[1]

    else:
        fig, axes = plt.subplots(2, 1, figsize=(7, 6))

        
    if prefix is None:
        prefix = '_'
        
    for id in range(0, data.shape[0]):
    
        # filename = directory + os.sep + '{1}spectrum_{0:02}'.format(id + 1,
        #                                                          prefix)  
              
        print('Plotting spectrum {0} of {1}'.format(id + 1,
                                                    data.shape[0]))
        # use more frequencies
        # f_e = np.logspace(np.log10(self.frequencies.min()),
        #                   np.log10(self.frequencies.max()),
        #                   100)
    
    
        # plot magnitude
        ax = axs[0]
        ax.semilogx(frequencies,
                    (np.exp(data[id, 0: int(len(data[id, :]) / 2)])), '.', c=c,
                    linewidth=2.0,
                    label=label)
        ax.set_xlabel('frequency [Hz]')
        ax.set_ylabel(r'$|Z/\rho| [\Omega (m)]$')
    
        # plot phase
        ax = axs[1]
        ax.semilogx(frequencies,
                    -data[id, int(len(data[id, :]) / 2):],
                    '.', linewidth=2.0, c=c,
                    label=label)
    
        # ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0, 0, 1, 1),
        #           bbox_transform=fig.transFigure)
        ax.set_xlabel('frequency [Hz]')
        ax.set_ylabel(r'$-\varphi~[mrad]$')
    
        for ax in axs:
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
        fig.tight_layout()
        fig.subplots_adjust(top=0.8)
        # fig.savefig('{0}.png'.format(filename))
        # plt.show()
        # plt.close(fig)
            
    return plt
    
            