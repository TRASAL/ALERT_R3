import math
import glob
import numpy as np
import uncertainties.unumpy as unp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
import scipy as sp
import lmfit
from lmfit.models import PowerLawModel, LinearModel
from lmfit import Model, Minimizer, Parameters
import tools
from frbpa.utils import get_phase, get_cycle
from astropy.time import Time

# ------------------------------------------------------------------------- #

def powlaw(x, amplitude, exponent):
    amp = amplitude #par['amplitude']
    k = exponent #par['exponent']
    res = amp * x ** k
    return res

def linear(x, slope):
    res = slope * x
    return res

def res_linear(params, x, data, sigma):
    slope = params['slope']
    model = slope * x
    res = model - data
    weighted = np.sqrt(res**2/sigma**2)
    return weighted

def res_powlaw(params, x, data, sigma):
    amp = params['amplitude']
    k = params['exponent']
    model = amp * x ** k
    res = model - data
    weighted = np.sqrt(res**2/sigma**2)
    return weighted

# ------------------------------------------------------------------------- #

colors = ['#7ecca5', '#feeb9d', '#f67a49', '#9e0142']
burst_data = pd.read_csv('/home/ines/Documents/projects/R3/arts/arts_r3_properties.csv')
burst_data[['bary_mjd', 'ncomp', 'drift_rate', 'drift_trust']]

# Drift rate Apertif
drift_rate_list, drift_rate_err, fcen_arts = [], [], []
fcen_arts_all = []

for ii,burst in burst_data.iterrows():
    #if burst['drift_trust']:
    if burst['ncomp'] == 2:
        drift_rate_list.append(burst['drift_rate'])
        drift_rate_err.append(np.sqrt(np.abs(burst['drift_rate'])))
        fcen_arts.append(burst['fcen'])
    if burst['ncomp'] > 2:
        drift_rate_list.append(burst['drift_rate'])
        drift_rate_err.append(burst['drift_rate_err'])
        fcen_arts.append(burst['fcen'])


dr_arts = np.mean(drift_rate_list)
dr_arts_err = np.std(drift_rate_list)
print("Apertif {:.4f}+-{:.4f} MHz/ms".format(dr_arts, dr_arts_err))


# Drift rate CHIME-GBT
# from Chawla+2020
dr_chgbt = -4.2
dr_chgbt_err = 0.4
fcen_chgbt = 400
print("CHIME-GBT {:.4f}+-{:.4f} MHz/ms".format(dr_chgbt, dr_chgbt_err))


# Drift rate CHIME
# From Chamma+2020
drift_rate_chime = [
        -28.6001169835474, -9.10167488518107, -11.8715199167045,
        -11.4566937888086, -21.4756530472919, -16.2887489846749,
        -9.31480412059403, -9.6311058232698, -25.5671081443539,
        -15.8686741184703, -19.2043364935631, -8.91264015179834,
        -42.4356309778676, -17.0918083584993, -43.503305977501,
        -44.8424422226505
        ]

drift_rate_chime_err = [np.sqrt(np.abs(dr)) for dr in drift_rate_chime]

dr_chime = np.mean(drift_rate_chime)
dr_chime_err = np.std(drift_rate_chime)
print("CHIME {:.4f}+-{:.4f} MHz/ms".format(dr_chime, dr_chime_err))

fcen_chime = [
        694.359833147273, 513.724573392185, 649.122616735104,
        494.136507812055, 497.038997189982, 622.446260101918,
        534.272499903657, 703.912708008321, 552.54046456535,
        536.144084799638, 619.254769561924, 468.222272721977,
        624.695039955252, 624.212334439836, 563.686814233842,
        712.51972145998
        ]

# Plotting
plt.figure(figsize=(8,5))
plt.style.use('/home/ines/.config/matplotlib/stylelib/paper.mplstyle')

plt.errorbar(fcen_chgbt, dr_chgbt, yerr=dr_chgbt_err, mfc=colors[2],
             marker='^', ls='none', label='CHIME/FRB-GBT',
             mec='k', mew=0.5, ecolor='k')
plt.errorbar(fcen_chime, drift_rate_chime, yerr=drift_rate_chime_err, mfc=colors[1],
             marker='s', ls='none', label='CHIME/FRB',
             mec='k', mew=0.5, ecolor='k')
plt.errorbar(fcen_arts, drift_rate_list, yerr=drift_rate_err,
             mfc=colors[0], marker='o', ls='none', label='Apertif',
             mec='k', mew=0.5, ecolor='k')


# ------------------------------------------------------------------------- #
# FITTING MULTI-FREQUENCY DRIFT RATE
## Fitting drift rate to power law
drifts = np.array([dr_chgbt] + drift_rate_chime + drift_rate_list)
drifts_err = np.array([dr_chgbt_err] + drift_rate_chime_err + drift_rate_err)
freq = np.array([fcen_chgbt] + fcen_chime + fcen_arts)
freqn = freq - np.mean(freq)
fval = np.logspace(2,4,num=100)

## Using lmfit.fit
plmodel = Model(powlaw)
params = Parameters()
params.add('amplitude', value=-0.6, max=0.)
params.add('exponent', value=0.5, min=0.)
plfit = plmodel.fit(drifts, params, x=freq, weights=1/drifts_err)
print("\nPOWER LAW FIT\n", plfit.fit_report())

# Using lmfit.minimize
# params = Parameters()
# params.add('amplitude', value=-0.6, max=0.)
# params.add('exponent', value=0.5, min=0.)
# plminner = Minimizer(res_powlaw, params, fcn_args=(freq, drifts, drifts_err))
# plfit = plminner.minimize()
#
# out = lmfit.fit_report(plfit.params,sort_pars='True')
# chisq = sum((plfit.residual)**2) # Calculating the reduced chisq
# redch = chisq/len(freq) # of the fit
# print("\nPOWER LAW MINIMIZATION\n", out)
# print("    redch:", redch)

amp = plfit.params.get("amplitude").value
amp_err = plfit.params.get("amplitude").stderr
exp = plfit.params.get("exponent").value
exp_err = plfit.params.get("exponent").stderr
coeff_pow = [amp, exp]
err_pow = [amp_err, exp_err]

## Fitting drift rate to linear model

# Using lmfit.fit
lmodel = Model(linear)
params = Parameters()
params.add('slope', value=-0.02, max=0.)
lfit = lmodel.fit(drifts, params, x=freq, weights=1/drifts_err)
slope = lfit.params.get("slope").value
slope_err = lfit.params.get("slope").stderr
print("\nLINEAR FIT\n", lfit.fit_report())

# Using lmfit.Minimize
# params = Parameters()
# params.add('slope', value=-0.02, max=0.)
# lminner = Minimizer(res_linear, params, fcn_args=(freq, drifts, drifts_err))
# lfit = lminner.minimize()
#
# out = lmfit.fit_report(lfit.params,sort_pars='True')
# chisq = sum((lfit.residual)**2) # Calculating the reduced chisq
# redch = chisq/len(freq) # of the fit
# slope = lfit.params.get("slope").value
# slope_err = lfit.params.get("slope").stderr
# print("\nLINEAR MINIMIZATION\n", out)
# print("    redch:", redch)

coeff_lin = [slope]
err_lin = [slope_err]

## Fitted curve
sep_pow_fit = powlaw(fval, *coeff_pow)
sep_lin_fit = linear(fval, *coeff_lin)

plt.plot(fval, sep_pow_fit, color='k', linestyle='--', label='power-law fit', lw=1)
plt.plot(fval, sep_lin_fit, color='k', linestyle='dotted', label='linear fit', lw=1)
plt.axvspan(110, 190, color=colors[3], alpha=0.5, label='LOFAR')
print('power-law drift rate 150 MHz:', powlaw(150, *coeff_pow))
print('linear drift rate 150 MHz:', coeff_lin[0]*150)


plt.legend(loc='lower center', fontsize=10)
plt.xlim(100,1600)
plt.ylim(-100, 0)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Drift rate (MHz ms$^{-1}$)')
plt_out = '/home/ines/Documents/projects/R3/arts/drift_rate/freq_drift_rate.pdf'
print("Saving figure", plt_out)
plt.savefig(plt_out, pad_inches=0, bbox_inches='tight', dpi=200)

plt.show()
