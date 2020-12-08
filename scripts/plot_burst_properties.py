#!/usr/bin/env python3
import numpy as np
import json, logging
import argparse
import pandas as pd
from astropy.time import Time, TimeDelta
from astropy import units as u
import datetime
import pylab as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec
from frbpa.utils import get_phase, get_cycle, get_params

# Initial parameters
P = 16.29
REFMJD = 58369.9

# Opening files
csvname = '/home/ines/Documents/projects/R3/arts/arts_r3_properties.csv'
burst_data = pd.read_csv(csvname)
burst_data['phase'] = get_phase(burst_data['bary_mjd'], P, ref_mjd=REFMJD)
burst_data['cycle'] = get_cycle(burst_data['bary_mjd'], P, ref_mjd=REFMJD)
cycmin = min(burst_data['cycle'])
cycmax = max(burst_data['cycle'])
ncycles = cycmax - cycmin

# Plotting
# plt.rcParams.update({
#         'font.size': 12,
#         'font.family': 'serif',
#         'axes.labelsize': 12,
#         'axes.titlesize': 14,
#         'xtick.labelsize': 12,
#         'ytick.labelsize': 12,
#         'xtick.direction': 'in',
#         'ytick.direction': 'in',
#         'xtick.minor.visible': True,
#         'ytick.minor.visible': True,
#         'xtick.top': True,
#         'ytick.right': True,
#         'lines.linewidth': 0.5,
#         'lines.markersize': 5,
#         'legend.fontsize': 12,
#         # 'legend.borderaxespad': 0,
#         # 'legend.frameon': True,
#         'legend.loc': 'upper right'})
plt.style.use('/home/ines/.config/matplotlib/stylelib/paper.mplstyle')



# Properties to plot
properties = ['struct_opt_dm', 'drift_rate', 'fluence_Jyms', 'pa_deg']
errors = ['struct_opt_dm_err', 'drift_rate_err', 'fluence_err', 'pa_err']
ylabel = ['DM (pc cm$^{-3}$)', 'Drift rate (MHz ms$^{-1}$)',
        'Fluence (Jy ms)', 'PA (deg)']
ylim = [(346.5, 351.5), (-120, 0), (-2, 70), (-10, 50)]
plt_cycles = np.unique(burst_data['cycle'])
plt_cycles.sort()
# plt_cycles = [i for i,c in enumerate(plt_cycles)]
# print(plt_cycles)

fig = plt.figure(figsize=(8,13))
gs = gridspec.GridSpec(len(properties),1, hspace=0.01, wspace=0.01)

#colors = ['#577590', '#43aa8b', '#90be6d', '#f9c74f', '#f8961e', '#f3722c', '#f94144']
colors = ['#577590', '#43aa8b', '#90be6d', '#f9c74f', '#f8961e', '#f94144']


for ii,k in enumerate(properties):
    ax = fig.add_subplot(gs[ii, 0])
    # Defining color
    for jj in range(len(burst_data['bary_mjd'])):
        cm = plt.cm.get_cmap('cubehelix')
        col = np.where(plt_cycles == burst_data['cycle'][jj])[0]
        #cycmax - burst_data['cycle'][jj]
        color = colors[int(col)]
        if burst_data['detection_snr'][jj] > 20:
            ecolor = 'k'
            size = 6
        else:
            ecolor = color
            size = 5
        ax.errorbar(burst_data['phase'][jj], burst_data[k][jj],
                yerr=burst_data[errors[ii]][jj], ms=size,
                fmt='o', color=color, mec=ecolor, mew=0.5)
    if k == 'struct_opt_dm':
        ax.hlines(348.75, 0, 1, color='gray', linestyle='--')
        ax.text(0.535, 348.8, '348.75 pc cm$^{-3}$', horizontalalignment='left')
    ax.set_ylabel(ylabel[ii])
    ax.set_xlim(0.36,0.59)
    ax.set_ylim(ylim[ii])
    #ax.set_xlim(0.15,0.4)
    if (ii == len(properties)-1):
        ax.set_xlabel('Phase')
    else:
        ax.set_xticklabels([])
    ax.tick_params(axis='x', which='both', direction='in', bottom=True,
            top=True)
    ax.tick_params(axis='y', which='both', direction='in', left=True,
            right=True)
    ax.label_outer()
    ax.yaxis.set_label_coords(-0.075, 0.5)

# Legend
lines = [plt.plot([], 'o', color=colors[i])[0]
        for i,c in enumerate(plt_cycles)]
labels=["Cycle {}".format(int(c))
        for i,c in enumerate(plt_cycles)]
ax.legend(lines, labels, fontsize=12, loc='lower left')

#plt.scatter(burst_data['phase'], burst_data['fluence_Jyms'])
plt_out = '/home/ines/Documents/projects/R3/arts/burst_properties.pdf'
print('Plot saved:', plt_out)
plt.savefig(plt_out, pad_inches=0, bbox_inches='tight')
plt.show()
