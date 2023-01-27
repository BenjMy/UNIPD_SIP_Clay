#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 14:13:48 2022

@author: ben
"""

# /home/ben/Documents/GitHub/BenjMy/UNIPD_SIP_Clay/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filename = 'scs2_2_4_4B' # scs2_2_4_4B scs2_2_4_4A
data = pd.read_csv(filename + '.csv', sep=';', nrows=118)
data.columns = ['f','Abs(Zm)','Std(Abs)','Phi(Zm)',' Std(Phi)','Re(Zm)','Im(Zm)','Time [s]']
data['f']

# data = data[list(np.arange(0,len(data)/2))]
id_2_remove = list(np.arange(len(data)/2,len(data)))

data = data.filter(id_2_remove, axis=0)


data['phase (mrad)'] = data['Phi(Zm)']*1e3
data['resistivity (Ohm.m)'] = (data['Abs(Zm)']*0.001134)/0.05
data['conductivity (mS/m)'] = (1/data['resistivity (Ohm.m)'])*1000
data['real c (mS/m)'] = abs(data['conductivity (mS/m)'])*np.cos(data['Phi(Zm)'])
data['im c (uS/m)'] = abs(abs(data['conductivity (mS/m)'])*np.sin(data['Phi(Zm)'])*1e3)

# matplotlib.pyplot.plot()
# plt.plot()

fig, ax = plt.subplots(2,1)
data.plot.scatter(x='f',y='real c (mS/m)',logx=True, ax=ax[0])
data.plot.scatter(x='f',y='im c (uS/m)',logx=True, ax=ax[1])

data.to_csv(filename+'_new.csv',index=False, sep=';')



filename_log = 'log_SIP - Foglio1'
log = pd.read_csv(filename_log + '.csv', sep=',')


