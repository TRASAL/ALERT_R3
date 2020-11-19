from math import *
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
from scipy.optimize import curve_fit

def sort_dict(dictionary, list):
    sorted_dict = {k: dictionary[k] for k in list if k in dictionary.keys()}
    return sorted_dict

def open_json(data_json):
    with open(data_json, 'r') as f:
        data = json.load(f)

    assert 'obs_duration' in data.keys()
    assert 'bursts' in data.keys()
    assert 'obs_startmjds' in data.keys()

    burst_dict = data['bursts']
    snr_dict = data['snr']
    obs_duration_dict = data['obs_duration']
    obs_startmjds_dict = data['obs_startmjds']
    fmin_dict = data['freq_min']
    fmax_dict = data['freq_max']

    assert len(obs_duration_dict.keys()) == len(obs_startmjds_dict.keys())
    assert len(obs_duration_dict.keys()) < 20
    assert len(burst_dict.keys()) < 10
    assert len(fmin_dict.keys()) ==  len(fmax_dict.keys())

    telescopes = list(obs_duration_dict.keys())

    new_obs_startmjds_dict = {}
    new_obs_duration_dict = {}
    fcen_dict = {}
    for k in obs_startmjds_dict.keys():
        start_times = obs_startmjds_dict[k]
        durations = obs_duration_dict[k]
        fmin = fmin_dict[k]
        fmax = fmax_dict[k]
        #new_start_times = []
        new_durations = []
        for i, t in enumerate(start_times):
            new_durations.append(durations[i]/(3600))
        new_obs_duration_dict[k] = new_durations
        fcen_dict[k] = (fmax + fmin)/2
    obs_duration_dict = new_obs_duration_dict

    # Sorting dictionaries by central frequency
    fcen_dict = {k: v for k, v in sorted(fcen_dict.items(),
            key=lambda item: item[1])}
    burst_dict = sort_dict(burst_dict, fcen_dict.keys())
    snr_dict = sort_dict(snr_dict, fcen_dict.keys())
    obs_duration_dict = sort_dict(obs_duration_dict, fcen_dict.keys())
    obs_startmjds_dict = sort_dict(obs_startmjds_dict, fcen_dict.keys())
    fmin_dict = sort_dict(fmin_dict, fcen_dict.keys())
    fmax_dict = sort_dict(fmax_dict, fcen_dict.keys())

    return burst_dict, snr_dict, obs_duration_dict, obs_startmjds_dict, fmin_dict, fmax_dict, fcen_dict

def fluence_to_energy(fluence, d_L=149, BW=300, f_b=1):
    """
    Converting fluence (Jy ms) into energy (erg)

    Parameters
    ----------
        fluence: float or np.array in Jy ms
        d_L: luminosity distance in Mpc
        BW: bandwidth in MHz
        f_b: beaming fraction

    Returns
    -------
        energy in ergs
    """

    fluence = fluence * u.Jy * u.ms
    d_L = d_L * u.Mpc
    BW = BW * u.MHz

    energy = 4*pi * d_L**2 * f_b * fluence * BW
    return energy.to('erg')

def func_powerlaw(x, alpha, c):
    return c * x**(alpha+1)

# Initial parameters
period = 16.29
ref_mjd = 58369.9
d_L = 149
BW = 300

# Opening files
data_json = '/home/ines/Documents/projects/R3/periodicity/r3all_data.json'
burst_dict, snr_dict, obs_duration_dict, obs_startmjds_dict, fmin_dict, fmax_dict, fcen_dict = open_json(data_json)

fluence_fn = '/home/ines/Documents/projects/R3/arts/fluxcal/fluence_int.txt'
fl = np.genfromtxt(fluence_fn, names=True)
arts_fluence, arts_ferr = [], []
for i in range(len(fl)):
        arts_fluence.append(fl['fint_Jyms'][i])
        arts_ferr.append(fl['fint_err'][i])

# Sorting by fluence
arts_width = [x for _,x in sorted(zip(arts_fluence,fl['width_ms']))]
arts_snr = [x for _,x in sorted(zip(arts_fluence,fl['snr']))]
arts_mjd = [x for _,x in sorted(zip(arts_fluence,fl['MJD']))]
arts_ferr = [x for _,x in sorted(zip(arts_fluence,arts_ferr))]
arts_phase = get_phase(fl['MJD'], period, ref_mjd=ref_mjd)
arts_phase = [x for _,x in sorted(zip(arts_fluence,arts_phase))]
arts_fluence.sort()
print("ARTS fluences", arts_fluence)

# Plotting fluence vs. phase
plt.errorbar(arts_phase, arts_fluence, yerr=arts_ferr, fmt='o', color='k',
        zorder=10)
plt.ylabel('Fluence (Jy ms)')
plt.xlabel('Phase')
plt.xlim(0.35,0.6)
plt.ylim(0,1.15*max(arts_fluence))
plt.show()

# Comparing fluence SNR-width and fluence integral
arts_fluence = []
for i in range(len(arts_mjd)):
    j = i+1
    if fl['snr'][i] >= 15:
        plt.errorbar(j, fl['fluence_Jyms'][i], yerr=fl['fluence_err'][i],
                marker='^', color='k', zorder=10)
        plt.errorbar(j, fl['fint_Jyms'][i], yerr=fl['fint_err'][i],
                marker='o', color='c', zorder=10)
        arts_fluence.append(fl['fint_Jyms'][i])
    else:
        plt.errorbar(j, fl['fluence_Jyms'][i], yerr=fl['fluence_err'][i],
                     marker='o', color='k', zorder=10)
        plt.errorbar(j, fl['fint_Jyms'][i], yerr=fl['fint_err'][i], marker='^',
                     color='c', zorder=10)
        arts_fluence.append(fl['fluence_Jyms'][i])


lines = [plt.plot([], 'o', color='k')[0],
        plt.plot([], 'o', color='c')[0]]
labels=['boxcar', 'integral']

plt.legend(lines, labels)


plt.ylabel('Fluence (Jy ms)')
plt.xlabel('ID')
plt.show()

# Cumulative distribution function
csvname = '/home/ines/Documents/projects/R3/arts/arts_r3_properties.csv'
burst_data = np.genfromtxt(csvname, delimiter=',', names=True)

arts_fluence = burst_data['fluence_Jyms']
#arts_width = [x for _,x in sorted(zip(arts_fluence,fl['width_ms']))]
arts_snr = [x for _,x in sorted(zip(arts_fluence,burst_data['snr']))]
arts_mjd = [x for _,x in sorted(zip(arts_fluence,burst_data['bary_mjd']))]
arts_ferr = [x for _,x in sorted(zip(arts_fluence,burst_data['fluence_err']))]
arts_phase = get_phase(burst_data['bary_mjd'], period, ref_mjd=ref_mjd)
arts_phase = [x for _,x in sorted(zip(arts_fluence,arts_phase))]
arts_fluence.sort()

arts_obs_time = np.sum(obs_duration_dict['Apertif'])
cumulative_rate = np.array([(len(arts_fluence)-i)/arts_obs_time
        for i in range(len(arts_fluence))])
cumulative_n = np.array([len(arts_fluence)-i for i in range(len(arts_fluence))])
cumulative_snr = np.array([len(arts_snr)-i for i in range(len(arts_fluence))])

# Converting fluence to energy
arts_energy = fluence_to_energy(arts_fluence)

# Fitting CFD to powerlaw and plotting
cm = plt.cm.get_cmap('twilight')
fig = plt.figure()
gs = gridspec.GridSpec(1,1)

ax1 = fig.add_subplot(gs[0, 0])
# ax1.errorbar(arts_fluence, cumulative_n, yerr=np.sqrt(cumulative_n),
#         errorevery=3, zorder=10, linestyle='-', lw=1, marker='o', color='gray',
#         label="All bursts")
ax1.plot(arts_fluence, cumulative_n, zorder=10, linestyle='-', lw=1,
        marker='o', color='k', label="All bursts")
ax1.set_xlabel('Fluence (Jy ms)')
ax1.set_ylabel('N (>F)')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(3e-1,30)
ax1.set_ylim(8e-1,70)

fl_hi, fl_md, fl_lo = [], [], []
cm_hi, cm_md, cm_lo = [], [], []

for j,f in enumerate(arts_fluence):
    if f < 1.4:
        fl_lo.append(f)
        cm_lo.append(cumulative_n[j])
    elif f > 5.3:
        fl_hi.append(f)
        cm_hi.append(cumulative_n[j])
    else:
        fl_md.append(f)
        cm_md.append(cumulative_n[j])

#print(fl_md, cm_md)

# Fitting
p0=[-2.3, 100]
coeff_hi, var_hi = curve_fit(func_powerlaw, fl_hi, cm_hi, p0=p0,
        sigma=np.sqrt(cm_hi))
print("High fluence pl fit", coeff_hi)
ax1.plot(np.logspace(-1,2), func_powerlaw(np.logspace(-1,2), *coeff_hi),
         color='k', alpha=0.4, linestyle='--', label='pl > 5.3 Jy ms')

coeff_md, var_md = curve_fit(func_powerlaw, fl_md, cm_md, p0=p0,
        sigma=np.sqrt(cm_md))
print("Medium fluence pl fit", coeff_md)
ax1.plot(np.logspace(-1,2), func_powerlaw(np.logspace(-1,2), *coeff_md),
         color='k', alpha=0.4, linestyle='dotted', label='pl 1.4-5.3 Jy ms')

coeff_lo, var_lo = curve_fit(func_powerlaw, fl_lo, cm_lo, p0=p0,
        sigma=np.sqrt(cm_lo))
print("Low fluence pl fit", coeff_lo)
ax1.plot(np.logspace(-1,2), func_powerlaw(np.logspace(-1,2), *coeff_lo),
         color='k', alpha=0.4, linestyle='-.', label='pl < 1.4 Jy ms')

phase_range = [0.36, 0.44, 0.47, 0.49, 0.52, 0.6]
color_test=['#95b88c', '#538d8c', '#2a3857', '#965da6', '#d685a4', 'C6', 'C7']
for i,p in enumerate(phase_range[:-1]):
    c = color_test[i]
    flist = []
    for j,f in enumerate(arts_fluence):
        if arts_phase[j] > p and arts_phase[j] < phase_range[i+1]:
            flist.append(f)
    ax1.plot(flist, [len(flist)-i for i in range(len(flist))], linestyle='-',
            marker='o', color=c, label="phase range "+str(i-2), markersize=5,
            linewidth=0.5)
ax1.legend()

# ax1.axvline(1.4, ymin=0, ymax=1e3, zorder=0, color='k', ls=(0, (5, 1)), alpha=0.3)
# ax1.axvline(5.3, ymin=0, ymax=1e3, zorder=0, color='k', ls=(0, (5, 1)), alpha=0.3)

plt_fl = '/home/ines/Documents/projects/R3/arts/fluxcal/cdf_fluence.pdf'
plt.savefig(plt_fl, pad_inches=0, bbox_inches='tight', dpi=200)
plt.show()
