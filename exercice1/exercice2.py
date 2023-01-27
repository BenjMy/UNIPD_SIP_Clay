#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:07:58 2022

@author: ben
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np

filename_log = 'log_SIP - Foglio1'
log = pd.read_csv(filename_log + '.csv', sep=',')

min_clay = 2
max_clay = 7

min_sal = 0
max_sal = 1.1

min_sat = 2
max_sat = 9

log_filtered = log[(log['Satur'] >= min_sat) & (log['Satur'] <= max_sat )]
log_filtered = log_filtered[(log_filtered['Clay'] >= min_clay) & (log_filtered['Clay'] <= max_clay )]
log_filtered = log_filtered[(log_filtered['Salinity'] >= min_sal) & (log_filtered['Salinity'] <= max_sal )]


   
fig, ax = plt.subplots(2,1)
color = cm.viridis(np.linspace(0, 1, len(log_filtered)))

# colormap='viridis'
for i, f in enumerate(log_filtered['Filename']):
    print(f)
    data = pd.read_csv(f + '.csv', sep=';')
    lgd = 'Satur' + str(log_filtered['Satur'].iloc[i]) + '_Clay'
    data.plot.scatter(x='f',y='real c (mS/m)',logx=True, ax=ax[0],label=lgd,color=color[i])
    data.plot.scatter(x='f',y='im c (uS/m)',logx=True, ax=ax[1],color=color[i])#,label=lgd)

plt.legend()

data.columns
fig, ax = plt.subplots(2,1)
color = cm.viridis(np.linspace(0, 1, len(log_filtered)))

for i, f in enumerate(log_filtered['Filename']):
    print(f)
    data = pd.read_csv(f + '.csv', sep=';')
    lgd = 'Satur' + str(log_filtered['Satur'].iloc[i]) + '_Clay'
    data.plot.scatter(x='f',y='resistivity (Ohm.m)',logx=True, ax=ax[0],label=lgd,color=color[i])
    data.plot.scatter(x='f',y='phase (mrad)',logx=True, ax=ax[1],color=color[i])#,label=lgd)

plt.legend()
