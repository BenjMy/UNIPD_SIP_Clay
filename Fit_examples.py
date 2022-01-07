#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 09:45:59 2022

@author: ben
"""
from CC_fit_src_MW.lib.lib_cc_fit import cc_fit as cc_fit
import os
# import lib_cc_fit.cc_fit as cls_cc_fit
import numpy as np
# !cc_fit.py --h
# test -d results && rm -r results
# !cls_cc_fit -f frequencies.dat -d data.dat -c 1 --plot

# os.chdir('./CCfit/examples/1-term/01/')

def get_filter_ids(filter_string, nr_frequencies=None):
    """
    If nr_frequencies is provided, then range can also have the form "%i-",
    e.g. "4-", and the end index will be set to the largest frequency.
    """
    sections = filter_string.split(';')

    filter_ids = []
    # now look for ranges and expand if necessary
    for section in sections:
        filter_range = section.split('-')
        if(len(filter_range) == 2):
            start = filter_range[0]
            end = filter_range[1]
            # check for an open range, e.g. 4-
            if(end == ''):
                if(nr_frequencies is not None):
                    end = nr_frequencies
                else:
                    continue
            filter_ids += range(int(start) - 1, int(end))
        else:
            filter_ids.append(int(section) - 1)
    return np.array(filter_ids)

filt = np.array([0,1,2,3,4,5,6,7])

filt = np.array([])

# path = './CCfit/examples/1-term/01/'
# lab_SIP = cc_fit.cc_fit()
# lab_SIP.load_data(path+ 'data.dat',ignore=filt)
# lab_SIP.data

# # plt.plot(np.hstack(lab_SIP.data))
# # plt.show()
# lab_SIP.load_frequencies(path+'frequencies.dat',ignore=filt)  # you can ignore frequencies here
# lab_SIP.fin
# lab_SIP.fin

# np.shape(lab_SIP.fin)

# lab_SIP.set_nr_cc_terms(nr=1)

# lab_SIP.set_initial_values(1)

# lab_SIP.fit_all_spectra()
# lab_SIP.plot_all_spectra(path+'./', prefix='')


# lab_SIP.save_non_essential_results(path+'./')
# lab_SIP.save_cc_pars(path+'cc_fits.dat')
# lab_SIP.save_cc_errors(path+'cc_fits.dat.err')



# path = './CCfit/examples/1-term/04_mean/'
# lab_SIP = cc_fit.cc_fit()
# lab_SIP.load_data(path+ 'fpi/rho_model_04_specs_fpi.dat',ignore= [])
# lab_SIP.data

# lab_SIP.load_frequencies(path+'fpi/extract_frequencies.dat',ignore= [])  # you can ignore frequencies here
# lab_SIP.fin
# lab_SIP.set_nr_cc_terms(nr=1)

# lab_SIP.set_initial_values(2)

# lab_SIP.fit_all_spectra()
# lab_SIP.plot_all_spectra(path+'./', prefix='')


# lab_SIP.save_non_essential_results(path+'./')
# lab_SIP.save_cc_pars(path+'cc_fits.dat')
# lab_SIP.save_cc_errors(path+'cc_fits.dat.err')






path = './CCfit/examples/2-term/01/'

lab_SIP = cc_fit.cc_fit()
lab_SIP.load_data(path+'data.dat',ignore= [])
lab_SIP.data

lab_SIP.load_frequencies(path+'frequencies.dat',ignore= [])
lab_SIP.fin
lab_SIP.set_nr_cc_terms(nr=2)

lab_SIP.set_initial_values(2)

lab_SIP.fit_all_spectra()
lab_SIP.plot_all_spectra(path+ './', prefix='')


lab_SIP.save_non_essential_results(path + './')
lab_SIP.save_cc_pars(path+ 'cc_fits.dat')
lab_SIP.save_cc_errors(path+ 'cc_fits.dat.err')


