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

def map_color():
    
    clist = [(0, 'powderblue'), (1, 'darkblue')]
    colors = mcolors.LinearSegmentedColormap.from_list("", clist)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", clist)
    matplotlib.cm.register_cmap("mycolormap", cmap)
    cpal = sns.color_palette("mycolormap", n_colors=21, desat=1)
    
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
    phase = pd.read_excel('../III-IV_sc_2_4_6_8_calc_modifiedBM.xlsx',
                          sheet_name=str(sheet_nb) + '_phase',
                          index_col=None, header=4)  
    
    
    phase.rename(columns={"theta": "freq"},inplace=True)
    data_phase = phase.to_numpy()
    data_phase = data_phase[:,1:]
    
    # df_t = phase.T.reset_index()
    # theta = df_t['index'][1:]
    
    sat_df = pd.read_excel('../III-IV_sc_2_4_6_8_calc_modifiedBM.xlsx',
                          sheet_name=str(sheet_nb) + '_resistivity',
                          index_col=0, header=2,
                          #converters={'saturation':np.float64}
                          )       
    
    df_t = sat_df.T.reset_index()
    sat = (df_t['saturation'][:].to_numpy())
    # sat = np.flip(df_t['saturation'][:].to_numpy())
    sat = sat[~np.isnan(sat)]


    magnitude = pd.read_excel('../III-IV_sc_2_4_6_8_calc_modifiedBM.xlsx',
                              sheet_name=str(sheet_nb) + '_resistivity',
                              index_col=None, header=4)  
    data_mag = magnitude.to_numpy()
    data_mag = data_mag[:,1:]
    data_mag.shape
    
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


def plot_data_spectra(data, frequencies, directory=None, prefix=None, 
                      axes=[], 
                      c=[], 
                      label=[]):
    
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
    
            