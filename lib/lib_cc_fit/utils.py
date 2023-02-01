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



import matplotlib.font_manager
import matplotlib.style
import matplotlib as mpl
import matplotlib.dates as mdates

mpl.style.use('default')

mpl.rcParams['grid.color'] = 'k'
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.25
# mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 0.75


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
    if len(np.where(data_phase>0)[0])>0:
        print('Positive phase - data removed')
    # negative amp
    # ------------------
    id_2_rmv.append(list(np.where(data_mag<0)[0]))
    if len(np.where(data_mag<0)[0])>0:
        print('Negative amplitude - data removed')
    # extreme amp
    # ------------------
    id_2_rmv.append(list(np.where(data_mag>1e3)[0]))
    if len(np.where(data_mag<0)[0])>0:
        print('Amplitude>10^3 - data removed')
    
    return np.hstack(id_2_rmv)


def load_excel(sheet_nb,
               filename='III-IV_sc_2_4_6_8_calc_cleaned.xlsx',
               **kwargs):
    '''
    

    Parameters
    ----------
    sheet_nb : tab 
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
      
    phase = pd.read_excel(filename,skiprows=2,
                          sheet_name=str(sheet_nb) + '_phase',
                          index_col=0, header=[0,1,2],
                          na_values='NA',
                          nrows= 118)  
    
    
    magnitude = pd.read_excel(filename,skiprows=2,
                          sheet_name=str(sheet_nb) + '_resistivity',
                          index_col=0, header=[0,1,2],
                          na_values='NA',
                          nrows= 118)  
    
    phase = phase.transpose()
    phase.rename_axis(index=["f_nb", "saturation", "theta"])
    magnitude = magnitude.transpose()
    magnitude.rename_axis(index=["f_nb", "saturation", "theta"])

    data_phase = phase.to_numpy()   
    data_mag = magnitude.to_numpy()   
    sat = phase.index.get_level_values(1).to_numpy() 
    theta = phase.index.get_level_values(2).to_numpy() 

    # Infer saturation
    # ------------------------------------------------------------------------
    # Sat_idx = range(data_mag.shape[1])
    sat_idx = range(len(sat))
    
    
    # Infer frequencies
    # ------------------------------------------------------------------------
    freq = phase.columns.to_numpy()
    freq_asc = freq[0:int(len(freq)/2)]
    freq_dsc = freq[int(len(freq)/2):]
    
    # split ascending/descending freq
    # ------------------------------------------------------------------------
    data_phase_asc = data_phase[:,0:int(len(freq)/2)].T
    data_phase_dsc = data_phase[:,int(len(freq)/2):].T
    
    data_mag_asc = data_mag[:,0:int(len(freq)/2)].T
    data_mag_dsc = data_mag[:,int(len(freq)/2):].T
    
   
    # # stack magnitudes and phases 
    # # ------------------------------------------------------------------------
    data_asc =  np.vstack([data_mag_asc,data_phase_asc])

    
    condition = ((freq_asc>=max_freq) + (freq_asc<=min_freq))
    # ign_freq = freq_asc[condition]
    ign_freq = list(np.where(freq_asc[condition])[0])
    # np.shape(data_mag_asc)
    
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
    
    
    return dict_data_excel, sat, theta

def crawl_excel(filename,
                varying_parm= [2,4,6,8], 
                sat_idx=[], 
                nr=1, 
                ini_val=1, 
                savefig=False,
                fixC = True,
                **kwargs):

    pathfig = './figs/'
    
    if not os.path.exists(pathfig):
        os.makedirs(pathfig)
        
    
    max_freq = 1e4
    min_freq = 1e-4
    plot_raw = False
    no_filter = False
    if 'plot_raw' in kwargs:
        plot_raw = kwargs['plot_raw']
    if 'no_filter' in kwargs:
        no_filter = kwargs['no_filter']
    if 'max_freq' in kwargs:
        max_freq = kwargs['max_freq']
    if 'min_freq' in kwargs:
        min_freq = kwargs['min_freq']
        
        
    varying_parm_name = 'variable_name'
    if 'varying_parm_name' in kwargs:
        varying_parm_name = kwargs['varying_parm_name']
        
    CC_mat = [] # store ColeCole parameters per saturation per clay content

    for cc in enumerate(varying_parm): # Loop over sc
    
        if plot_raw == True:
            fig, axs = plt.subplots(2, 1, figsize=(7, 6))

        data, sat, theta = load_excel(cc[1],filename,
                            min_freq=min_freq,
                            max_freq=max_freq)
        cc_pars_ss = []
        

        for ss in enumerate(sat): # Loop over saturation
            path = './'
            lab_SIP = cc_fit.cc_fit()
            lab_SIP.load_data(data['data_asc'][:,ss[0]],ignore=data['ign_freq'])  # you can ignore frequencies here
            
            if no_filter:
                id_2_rmv = []
            else:
                id_2_rmv = filter_data(data['data_asc'][:,ss[0]],data['freq'])
            
            if len(id_2_rmv)>1:
                if nr==1:
                    cc_pars_ss.append([np.nan,np.nan,np.nan,np.nan,
                                       np.nan,np.nan,
                                       ss[1],
                                       cc[1]])
                elif nr==2:
                    cc_pars_ss.append([np.nan,
                                       np.nan,np.nan,np.nan,
                                       np.nan,np.nan,np.nan,
                                       np.nan,np.nan,
                                       ss[1],
                                       cc[1]])
                    # cc_pars_ss_df = pd.DataFrame(columns=['c1_peak1','c2_peak2','c3_peak1','c4_peak2'])
                pass
            
            else:
                print('no filter applied for data of clay content: ' + str(cc[1]) 
                      + ' with saturation_level: ' + str(ss[1])
                      )
                lab_SIP.data     
                lab_SIP.load_frequencies_array(data['freq_asc'],ignore= data['ign_freq'])  # you can ignore frequencies here
                lab_SIP.set_nr_cc_terms(nr=nr)
                lab_SIP.set_initial_values(ini_val)
                lab_SIP.cc_pars_init
                lab_SIP.fit_all_spectra(fixC)
                cc_pars_ss.append(np.r_[lab_SIP.cc_pars[0],
                                        lab_SIP.magnitude_rms,lab_SIP.phase_rms,
                                        ss[1],
                                        cc[1]])
                # np.shape(cc_pars_ss)
                
                CC_df = pd.DataFrame(cc_pars_ss)



                if 'plot_raw' == True:
                    cpal = map_color(n_colors=len(sat)+1, desat=1)
                    # cpal = sns.color_palette("mycolormap", n_colors=len(sat)+1, desat=1)
                    plot_data_spectra(lab_SIP.data, lab_SIP.frequencies, path+'./', 
                                      prefix=str(ss[1]), 
                                      axes=[axs,fig], 
                                      c=cpal[len(sat)-ss[0]],
                                      label=str(ss))
                    
                    plt.title(varying_parm_name + ', sat_level:' + str(cc[1]))

                    if savefig:
                        plt.savefig(pathfig + varying_parm_name + ', sat_level:' + str(cc[1]),dpi=300)
                
                else:
                    lab_SIP.plot_all_spectra(pathfig, prefix=str(ss[1]))

                
            
        cc_pars_ss = np.vstack(cc_pars_ss)
        CC_mat.append(cc_pars_ss)
        
    CC_sat_param_clay = np.vstack(CC_mat)
    
    return CC_sat_param_clay


def set_fig_attribute(fig,axs,varying_parm_name,map1,**kwargs):
    if 'minmax_y_rho0' in kwargs:
        axs[0,0].set_ylim([kwargs['minmax_y_rho0'][0],kwargs['minmax_y_rho0'][1]])
    if 'minmax_y_tau' in kwargs:
        axs[0,1].set_ylim([kwargs['minmax_y_tau'][0],kwargs['minmax_y_tau'][1]])
    if 'minmax_y_m' in kwargs:
        axs[1,0].set_ylim([kwargs['minmax_y_m'][0],kwargs['minmax_y_m'][1]])
    if 'minmax_y_c' in kwargs:
        axs[1,1].set_ylim([kwargs['minmax_y_c'][0],kwargs['minmax_y_c'][1]])
        
    axs[0,1].grid(True)
    axs[1,1].set_ylabel('c (-)')
    axs[1,0].set_xlabel('saturation (-)')
    axs[0,1].set_xlabel('saturation (-)')
    axs[1,1].set_xlabel('saturation (-)')

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.025, 0.7])
    cb = fig.colorbar(map1,shrink=0.45, cax=cbar_ax)
    cb.set_label(varying_parm_name)  
    pass
        
def plot_CC_matrice(data_mat, 
                    varying_parm_name='clay content', 
                    savename='CC_matrice.png', 
                    samefig=True,
                    **kwargs):

    pathfig = './figs/'
    
    if not os.path.exists(pathfig):
        os.makedirs(pathfig)
        
    CC_df = pd.DataFrame(data_mat)
    if CC_df.shape[1]>10:
        CC_df.rename(columns={0:'C0',
                              1:'C1',2:'C2',3:'C3',
                              4:'C1_p2',5:'C2_p2',6:'C3_p2',
                              7:'rms_mag',8:'rms_phase',
                              9:'saturation',10:varying_parm_name},inplace=True) 
    else:
        CC_df.rename(columns={0:'C0',1:'C1',2:'C2',3:'C3',
                              4:'rms_mag',5:'rms_phase',
                              6:'saturation',7:varying_parm_name},inplace=True)
       
    fig, axs = plt.subplots(2,2,sharex='all', figsize=(10,4), dpi=300)
    # sns.lmplot('saturation', 'C0', data=test, hue='ClayContent', fit_reg=False, ax=axs[0])
    map1 = axs[0,0].scatter(CC_df['saturation'], CC_df['C0'], c=CC_df[varying_parm_name]
                     ,label=[''],cmap='viridis')
    # axs[0,0].errorbar(CC_df['saturation'], CC_df['C0'], yerr=CC_df['C0']*CC_df['rms_mag'], fmt="o")
    axs[0,0].set_xlabel('saturation (-)')
    axs[0,0].grid(True)
    axs[1,0].scatter(CC_df['saturation'], CC_df['C1'], c=CC_df[varying_parm_name])
    axs[1,0].grid(True)
    axs[0,1].scatter(CC_df['saturation'], CC_df['C2'], c=CC_df[varying_parm_name])
    axs[1,1].scatter(CC_df['saturation'], CC_df['C3'], 
                      c=CC_df[varying_parm_name])
    axs[1,1].grid(True)
    axs[0,0].set_ylabel(r'$\rho_{0} (\Omega) $')
    axs[1,0].set_ylabel('m (-)')
    axs[0,1].set_ylabel(r'$log(\tau)$ (s)')
    
    if CC_df.shape[1]>10:
        if samefig:
            if hasattr(CC_df, 'C1_p2'):
                axs[1,0].scatter(CC_df['saturation'], CC_df['C1_p2'], c=CC_df[varying_parm_name],
                                 marker='^')
            if hasattr(CC_df, 'C2_p2'):
                axs[0,1].scatter(CC_df['saturation'], CC_df['C2_p2'], c=CC_df[varying_parm_name],
                                 marker='^')
            if hasattr(CC_df, 'C3_p2'):
                axs[1,1].scatter(CC_df['saturation'], CC_df['C3_p2'], c=CC_df[varying_parm_name],
                                 marker='^')
            set_fig_attribute(fig,axs,varying_parm_name,map1,**kwargs)
            plt.savefig(pathfig + '_peak12_' + savename, dpi=450)     
        else:
            plt.savefig(pathfig + '_peak1_' + savename , dpi=450)     
            fig, axs = plt.subplots(2,2,sharex='all', figsize=(10,4), dpi=300)
            map1 = axs[0,0].scatter(CC_df['saturation'], CC_df['C0'], c=CC_df[varying_parm_name]
                             ,label=[''],cmap='viridis')
            axs[0,0].set_xlabel('saturation (-)')
            axs[0,0].grid(True)
            axs[1,0].scatter(CC_df['saturation'], CC_df['C1_p2'], c=CC_df[varying_parm_name],marker='^')
            axs[1,0].grid(True)
            axs[0,1].scatter(CC_df['saturation'], CC_df['C2_p2'], c=CC_df[varying_parm_name],marker='^')
            axs[1,1].scatter(CC_df['saturation'], CC_df['C3_p2'],marker='^',
                              c=CC_df[varying_parm_name])
            axs[1,1].grid(True)
            axs[0,0].set_ylabel(r'$\rho_{0} (\Omega) $')
            axs[1,0].set_ylabel('m (-)')
            axs[0,1].set_ylabel(r'$log(\tau)$ (s)')
            set_fig_attribute(fig,axs,varying_parm_name,map1,**kwargs)
            plt.savefig(pathfig + '_peak2_' + savename , dpi=450)     
    else:
        set_fig_attribute(fig,axs,varying_parm_name,map1,**kwargs)
        plt.savefig(pathfig + savename, dpi=450)     
        
    
    return CC_df


def export2csv(df,filename):
    
    pathproc = './process/'
    if not os.path.exists(pathproc):
        os.makedirs(pathproc)
        
    df.to_csv(pathproc+filename,index=False)
    
    
    
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
              
        # print('Plotting spectrum {0} of {1}'.format(id + 1,
        #                                             data.shape[0]))
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
    
            