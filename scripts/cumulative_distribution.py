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
from frbpa.utils import get_phase#, get_cycle, get_params
from scipy.optimize import curve_fit

def pl(x, xmin=None):
   """ Get the maximum likelihood power-law
   index for the distribution x
   """
   if xmin is None:
       xmin = x.min()
   return (len(x)-1)/(float(len(x)))*(1./len(x) * np.sum(np.log(x/xmin)))**-1

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

def brokenpl(x, *p):
    "Broken power law"
    (c1, xb, a1, a2) = p
    c2 = c1 * xb ** (a1 - a2)
    res = np.zeros(x.shape)
    for ii,xx in enumerate(x):
        if xx < xb:
            res[ii] = c1 * xx ** a1
        else:
            res[ii] = c2 * xx ** a2
    return res

def brokenpl2(x, *p):
    "Two times broken power law"
    (c1, xb1, xb2, a1, a2, a3) = p
    c2 = c1 * xb1 ** (a1 - a2)
    c3 = c2 * xb2 ** (a2 - a3)
    res = np.zeros(x.shape)
    for ii,xx in enumerate(x):
        if xx < xb1:
            res[ii] = c1 * xx ** a1
        elif xx < xb2:
            res[ii] = c2 * xx ** a2
        else:
            res[ii] = c3 * xx ** a3
    return res

# ------------------------------------------------------------------------- #
# Initial parameters
period = 16.29
ref_mjd = 58369.9
d_L = 149
BW = 300

# Opening files
data_json = '/home/ines/Documents/projects/R3/periodicity/r3all_data.json'
# Liam edit
#data_json = './r3all_data.json'
burst_dict, snr_dict, obs_duration_dict, obs_startmjds_dict, fmin_dict, fmax_dict, fcen_dict = open_json(data_json)

fluence_fn = '/home/ines/Documents/projects/R3/arts/fluxcal/fluence_int.txt'
# Liam edit
#fluence_fn = './fluence_int.txt'
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

# Liam edit: get observing time in each phase bin
arts_time_phase_bin = get_phase(np.array(obs_startmjds_dict['Apertif']), period, ref_mjd=ref_mjd)
arts_obs_duration = np.array(obs_duration_dict['Apertif'])
print("Fluence boxcar", fl['fluence_Jyms'])
print("ARTS fluences", fl['fint_Jyms'])

# Plotting fluence vs. phase
plt.errorbar(arts_phase, arts_fluence, yerr=arts_ferr, fmt='o', color='k',
        zorder=10)
plt.ylabel('Fluence (Jy ms)')
plt.xlabel('Phase')
plt.xlim(0.35,0.6)
plt.ylim(0,1.15*max(arts_fluence))
#plt.show()

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
#plt.show()

# ------------------------------------------------------------------------- #
# Cumulative distribution function
## ARTS
csvname = '/home/ines/Documents/projects/R3/arts/arts_r3_properties.csv'
#csvname = 'arts_r3_properties.csv'
burst_data = np.genfromtxt(csvname, delimiter=',', names=True)

arts_fluence = burst_data['fluence_Jyms']
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

## LOFAR
csvname = '/home/ines/Documents/projects/R3/lofar/lofar_r3_properties.csv'
burst_data = np.genfromtxt(csvname, delimiter=',', names=True)

Tobs_lofar = 48.3
duty_cycle_lofar = 1.0
lofar_fluence = burst_data['fluence_Jyms']
lofar_snr = burst_data['detection_snr']

lofar_fluence.sort()
# do the same for LOFAR
cumulative_n_lofar = np.array([len(lofar_fluence)-i
        for i in range(len(lofar_fluence))])

print("LOFAR fluence slope %0.2f" % pl(np.array(lofar_fluence)))
print("ARTS fluence slope %0.2f" % pl(np.array(arts_fluence)))

print("LOFAR SNR slope %0.2f" % pl(np.array(lofar_snr)))
print("ARTS SNR slope %0.2f" % pl(np.array(arts_snr)))

# Converting fluence to energy
arts_energy = fluence_to_energy(arts_fluence)

# Fitting CFD to powerlaw and plotting
#cm = plt.cm.get_cmap('twilight')
#cm = ''
fig = plt.figure(figsize=(10,7))
plt.style.use('/home/ines/.config/matplotlib/stylelib/paper.mplstyle')
plt.rcParams.update({
        'lines.linewidth': 1,
        'legend.fontsize': 10,
        'legend.loc': 'lower left'})
gs = gridspec.GridSpec(1,1)

colors = ['#7ECCA5', '#9E0142']

ax1 = fig.add_subplot(gs[0, 0])
# ax1.errorbar(arts_fluence, cumulative_n, yerr=np.sqrt(cumulative_n),
#         errorevery=3, zorder=10, linestyle='-', lw=1, marker='o', color='gray',
#         label="All bursts")

ax1.plot(arts_fluence, cumulative_n/arts_obs_time, zorder=10, linestyle='-',
        lw=1, marker='o', color=colors[0], label="All Apertif bursts")
ax1.plot(lofar_fluence, cumulative_n_lofar/Tobs_lofar*duty_cycle_lofar,
        zorder=10, linestyle='-', lw=1,
        marker='s', color=colors[1], label="All LOFAR bursts")
ax1.set_xlabel('Fluence (Jy ms)')
ax1.set_ylabel(r'Rate (>F)   hr$^{-1}$')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(7e-1,400)
ax1.set_ylim(1e-3, 1)

# FITTING CDF
# Fitting Apertif to 2 times broken power law
c, x1, x2, a1, a2, a3 = 100, 2.7, 3.5, -0.17, -0.58, -1.38
p0 = [c, x1, x2, a1, a2, a3]
coeff, var = curve_fit(brokenpl2, arts_fluence, cumulative_n, p0=p0,
        sigma=np.sqrt(cumulative_n))
ax1.plot(np.logspace(-1,2),
        brokenpl2(np.logspace(-1,2), *coeff)/arts_obs_time,
        color='k', alpha=0.4, linestyle='-.', label='Apertif broken pl')
c, x1, x2 = coeff[0], coeff[1], coeff[2]
a1, a2, a3= coeff[3]-1, coeff[4]-1, coeff[5]-1
(c_err, x1_err, x2_err, a1_err, a2_err, a3_err) = np.sqrt(np.diag(var))
print("Apertif fit\n", coeff, "\n", np.sqrt(np.diag(var)))

# Fitting LOFAR to broken power law
cl, xl, a1l, a2l = 100, 100, -0.15, -1.4
p0 = [cl, xl, a1l, a2l]
coeff, var = curve_fit(brokenpl, lofar_fluence, cumulative_n_lofar, p0=p0,
        sigma=np.sqrt(cumulative_n_lofar))
ax1.plot(np.logspace(1,3),
        brokenpl(np.logspace(1,3), *coeff)/Tobs_lofar*duty_cycle_lofar,
        color='k', alpha=0.4, linestyle='dotted', label='LOFAR broken pl')
xl = coeff[1]
print("LOFAR\n", coeff, "\n", np.sqrt(np.diag(var)))

# Dividing Apertif phase range
phase_range = [0.35, 0.46, 0.51, 0.62]
color_test = ['#98C56D', '#34835A', '#17343A']
for i,p in enumerate(phase_range[:-1]):
    c = color_test[i]
    flist = []
    for j,f in enumerate(arts_fluence):
        if arts_phase[j] > p and arts_phase[j] < phase_range[i+1]:
            # Liam edit: convert y-axis into a rate
            arts_time_phase_bin = get_phase(
                    np.array(obs_startmjds_dict['Apertif']), period,
                    ref_mjd=ref_mjd)
            tobs_j = np.sum(arts_obs_duration[np.where(
                    (arts_time_phase_bin<phase_range[i+1]) & \
                    (arts_time_phase_bin>p))[0]])
            flist.append(f)

    leglabel="phase: %0.2f-%0.2f "%(p,phase_range[i+1])
    ax1.plot(flist, ([len(flist)-i for i in range(len(flist))])/tobs_j,
            linestyle='-', marker='', color=c, label=leglabel, markersize=5,
            linewidth=0.8)

ax1.legend()

ax1.axvline(x1, ymin=0, ymax=1e3, zorder=0, color='k', ls=(0, (5, 1)),
        alpha=0.3)
ax1.axvline(x2, ymin=0, ymax=1e3, zorder=0, color='k', ls=(0, (5, 1)),
        alpha=0.3)
ax1.axvline(xl, ymin=0, ymax=1e3, zorder=0, color='k', ls=(0, (5, 1)),
        alpha=0.3)

plt_fl = '/home/ines/Documents/projects/R3/arts/fluxcal/cdf_fluence.pdf'
#plt_fl = '/home/ines/Documents/PhD/meetings/20210303-Astrolunch_talk/figs/cdf_fluence.png'
#plt_fl = 'cdf_fluence.pdf'
print("Saving figure", plt_fl)
plt.savefig(plt_fl, pad_inches=0, bbox_inches='tight', dpi=200)
plt.show()
