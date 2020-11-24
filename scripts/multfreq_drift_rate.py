import math
import glob
import numpy as np
import uncertainties.unumpy as unp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
import tools
from frbpa.utils import get_phase, get_cycle
from astropy.time import Time

def powlaw(x, *p):
    (c, a) = p
    res = c * x ** (a)
    return res

def linear(x, *p):
    c = p
    res = c * x
    return res

#colors = ['#577590', '#90be6d', '#f9c74f', '#f3722c', '#f94144']
colors = ['#7ecca5', '#feeb9d', '#f67a49', '#9e0142']
burst_data = pd.read_csv('/home/ines/Documents/projects/R3/arts/arts_r3_properties.csv')
print(burst_data.columns)
burst_data[['bary_mjd', 'ncomp', 'drift_rate', 'drift_trust']]

period = 16.29
drift_rate_list, drift_rate_err = [], []
fcen_arts_all = []

for ii,burst in burst_data.iterrows():
    phase = get_phase(burst['detection_mjd'], period, ref_mjd=58369.30)
    if burst['drift_trust']:
        if burst['ncomp'] == 2:
            drift_rate_list.append(burst['drift_rate'])
            drift_rate_err.append(0.0)
        if burst['ncomp'] == 3:
            drift_rate_list.append(burst['drift_rate'])
            drift_rate_err.append(burst['drift_rate_err'])
        if burst['ncomp'] == 5:
            drift_rate_list.append(burst['drift_rate'])
            drift_rate_err.append(burst['drift_rate_err'])

drift_rate_chime = [-3.6660, -8.4000, -6.5665]

dr_chgbt = -4.2
dr_chgbt_err = 0.4
fcen_chgbt = 400

dr_chime = np.mean(drift_rate_chime)
dr_chime_err = np.std(drift_rate_chime)
fcen_chime = [449.98, 530.93, 672.05] #600
print("Drift rate mean {:.1f} +- {:.1f}".format(dr_chime, dr_chime_err))

dr_arts = np.mean(drift_rate_list)
dr_arts_err = np.std(drift_rate_list)
fcen_arts = [1437.52,1459.5,1417.97,1422.25,1452.66,1451.64,1288.48,1428.31]
print(len(fcen_arts), len(drift_rate_list))

# ---------
# Plotting
plt.figure(figsize=(8,5))
plt.style.use('/home/ines/.config/matplotlib/stylelib/paper.mplstyle')

plt.errorbar(fcen_chgbt, dr_chgbt, yerr = dr_chgbt_err, mfc=colors[2],
             marker='^', ls='none', label='CHIME-GBT',
             mec='k', mew=0.5, ecolor='k')
plt.errorbar(fcen_chime, drift_rate_chime, yerr = dr_chime_err, mfc=colors[1],
             marker='s', ls='none', label='CHIME',
             mec='k', mew=0.5, ecolor='k')
plt.errorbar(fcen_arts, drift_rate_list, yerr = drift_rate_err,
             mfc=colors[0], marker='o', ls='none', label='Apertif',
             mec='k', mew=0.5, ecolor='k')

# Fitting drift rate to power law
drifts = [dr_chgbt]
drifts += drift_rate_chime + drift_rate_list
freq = [fcen_chgbt] + fcen_chime + fcen_arts
fval = np.logspace(2,4,num=100)

p0 = [-1,1]
coeff_pow, var_matrix_pow = curve_fit(powlaw, freq, drifts, p0=p0, bounds=((-np.inf,0), (0,np.inf)))
err_pow = np.sqrt(np.diag(var_matrix_pow))
print(coeff_pow, err_pow)

p0=-1
coeff_lin, var_matrix_lin = curve_fit(linear, freq, drifts, p0=p0)
err_lin = np.sqrt(np.diag(var_matrix_lin))
print(coeff_lin, err_lin)


# Fitted curve
sep_pow_fit = powlaw(fval, *coeff_pow)
sep_lin_fit = linear(fval, *coeff_lin)

plt.plot(fval, sep_pow_fit, color='k', linestyle='--', label='power-law fit', lw=1)
plt.plot(fval, sep_lin_fit, color='k', linestyle='dotted', label='linear fit', lw=1)
plt.axvspan(110, 190, color=colors[3], alpha=0.5, label='LOFAR')
print('power-law drift rate 150 MHz:', powlaw(150, *coeff_pow))
print('linear drift rate 150 MHz:', coeff_lin[0]*150)


plt.legend(loc='lower center', fontsize=10)
plt.xlim(100,1600)
plt.ylim(-50, 0)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Drift rate (MHz ms$^{-1}$)')
plt_out = '/home/ines/Documents/projects/R3/arts/drift_rate/freq_drift_rate.pdf'
print("Saving figure", plt_out)
plt.savefig(plt_out, pad_inches=0, bbox_inches='tight', dpi=200)

plt.show()
