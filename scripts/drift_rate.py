import math
import glob, sys
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

# Defining functions
def rebin(arr, binsz):
    new_shape = [arr.shape[0] //  binsz[0], arr.shape[1] // binsz[1]]
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

# Define model function to be used to fit to the data above:
def quintuple_gaussian(x, *p):
    (c1, mu1, sigma1, c2, mu2, sigma2, c3, mu3, sigma3, c4, mu4, sigma4, c5, mu5, sigma5) = p
    res =    c1 * np.exp( - (x - mu1)**2.0 / (2.0 * sigma1**2.0) ) \
          +  c2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) ) \
          +  c3 * np.exp( - (x - mu3)**2.0 / (2.0 * sigma3**2.0) ) \
          +  c4 * np.exp( - (x - mu4)**2.0 / (2.0 * sigma4**2.0) ) \
          +  c5 * np.exp( - (x - mu5)**2.0 / (2.0 * sigma5**2.0) )
    return res

def triple_gaussian(x, *p):
    (c1, mu1, sigma1, c2, mu2, sigma2, c3, mu3, sigma3) = p
    res =    c1 * np.exp( - (x - mu1)**2.0 / (2.0 * sigma1**2.0) ) \
          +  c2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) ) \
          +  c3 * np.exp( - (x - mu3)**2.0 / (2.0 * sigma3**2.0) )
    return res

def double_gaussian(x, *p):
    (c1, mu1, sigma1, c2, mu2, sigma2) = p
    res =    c1 * np.exp( - (x - mu1)**2.0 / (2.0 * sigma1**2.0) ) \
          +  c2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) )
    return res

def gaussian(x, *p):
    (c, mu, sigma) = p
    res =    c * np.exp( - (x - mu)**2.0 / (2.0 * sigma**2.0) )
    return res

def drift(x, a, b):
    res = b + a * x
    return res

def powlaw(x, *p):
    (c, a) = p
    res = c * x ** (a)
    return res

# Opening data
fname = '/home/ines/Documents/projects/R3/arts/drift_rate/drift_rate.txt'
file_info = np.genfromtxt(fname, names=True,
                         dtype=('<U3', '<U35', 'f8', 'i8', 'f8', 'f8', '<U41', '<U41', '<U41', '<U41'))
print(file_info.dtype.names)

ii = int(sys.argv[1])
ref_dm = 348.8

data_path = '/home/ines/Documents/projects/R3/arts/drift_rate/iquv_npy/'
fn = data_path + file_info['file_'][ii]
idb = file_info['id'][ii]
dm = file_info['dm'][ii]
ncomp = file_info['ncomp'][ii]
t0 = file_info['tdrift'][ii].strip('][').split(',')
p0 = np.array([[.1, float(t), 0.1] for t in t0]).flatten()
# p0[-3] = 0.2
# p0[-6] = 0.3
print("Number of components", ncomp)
print("Initial conditions:", p0)
#p0 = [0.2, -1.7074, 0.1, 0.6, -0.9222, 0.1, 0.9, 0.0649, 0.1, 0.3, 1.0513, 0.1, 0.1, 1.7553, 0.1]
#p0 = [0.1, -1.4, 0.1, 0.3, -0.0264, 0.2, 0.1, 2, 0.2]
#p0 = [0.1, -0.9036, 0.1, 0.3, 0.2168, 0.1, 0.2, 0.9834, 0.1]

# Loading data
data = np.load(fn)
data = data - np.mean(data)
print(dm, dm - ref_dm)
# Dedispersing data
data = tools.dedisperse(data, -(dm - ref_dm))
# Reshaping data
imin = data.shape[1]//2-256
imax = data.shape[1]//2+256
data = data[:,imin:imax]

# Waterfalling
fscrunch, bscrunch = 8, 1
waterfall = rebin(data, [fscrunch,bscrunch])
print(data.shape)

nfreq, nsamp = waterfall.shape
tsamp = 0.08192 * bscrunch # ms
fmin, fmax = 1219.50561523, 1519.50561523
fsamp = (fmax - fmin) / nfreq
tval = np.arange(-nsamp/2, nsamp/2) * tsamp
fval = np.arange(fmin, fmax, fsamp)

pulse = np.mean(waterfall, axis=0)

# Plotting dynamic spectrum
# fig = plt.figure(figsize=(9,9))
# gs = gridspec.GridSpec(2,1, hspace=0.0, wspace=0.0, height_ratios=[1,3])
#
# ax1 = fig.add_subplot(gs[0,0])
# ax2 = fig.add_subplot(gs[1,0], sharex=ax1)
#
# ax1.plot(tval, pulse, color='k')
#
# if ncomp == 2:
#     #pulse0 = double_gaussian(tval, *coeff_p.flatten())
#     pulse0 = double_gaussian(tval, *p0)
#     ax1.plot(tval, pulse0, color='gray')
# elif ncomp == 3:
#     #pulse0 = triple_gaussian(tval, *coeff_p.flatten())
#     pulse0 = triple_gaussian(tval, *p0)
#     ax1.plot(tval, pulse0, color='gray')
# elif ncomp == 5:
#     #pulse0 = triple_gaussian(tval, *coeff_p.flatten())
#     pulse0 = quintuple_gaussian(tval, *p0)
#     ax1.plot(tval, pulse0, color='gray')
#
#
# ax2.imshow(waterfall, interpolation='nearest', aspect='auto', origin='lower', cmap='viridis',
#            extent=[tval[0], tval[-1], fmin, fmax])
#
# ax2.set_xlabel('Time (ms)')
# ax2.set_ylabel('Frequency (MHz)')
#
# ax1.set_xlim(tval[0], tval[-1])
# ax2.set_ylim(fmin, fmax)
#plt.show()

# Fitting pulse to gaussians
# p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
if ncomp == 2:
    #p0 = [9., -0.6, .2, 11., 0.6, .1]

    # Fitting
    coeff_p, var_matrix_p = curve_fit(double_gaussian, tval, pulse, p0=p0)
    err_p = np.reshape(np.sqrt(np.diag(var_matrix_p)), (2,3))

    # Fitted curve
    pulse_fit = double_gaussian(tval, *coeff_p)
    coeff_p = np.reshape(coeff_p, (2,3))

elif ncomp == 3:
    #p0 = [6., -1, .1, 10., -0., .1, 7., 2., .2]
    print("Fitting 3 components")

    # Fitting
    coeff_p, var_matrix_p = curve_fit(triple_gaussian, tval, pulse, p0=p0)
    err_p = np.reshape(np.sqrt(np.diag(var_matrix_p)), (3,3))

    # Fitted curve
    pulse_fit = triple_gaussian(tval, *coeff_p)
    coeff_p = np.reshape(coeff_p, (3,3))

elif ncomp == 5:
    #p0 = [6., -1, .1, 10., -0., .1, 7., 2., .2]
    print("Fitting 5 components")

    # Fitting
    coeff_p, var_matrix_p = curve_fit(quintuple_gaussian, tval, pulse, p0=p0)
    err_p = np.reshape(np.sqrt(np.diag(var_matrix_p)), (5,3))

    # Fitted curve
    pulse_fit = quintuple_gaussian(tval, *coeff_p)
    coeff_p = np.reshape(coeff_p, (5,3))
print(coeff_p, err_p)

# Fitting spectra of subbursts to gaussians
spec = []
for ii in range(ncomp):
    imin = np.abs(tval - (coeff_p[ii,1]-2*coeff_p[ii,2])).argmin()
    imax = np.abs(tval - (coeff_p[ii,1]+2*coeff_p[ii,2])).argmin()
    spec.append(np.mean(waterfall[:,imin:imax], axis=1))

#print(spec)
spec_fit = []
coeff_s = []
err_s = []
for ii,ss in enumerate(spec):
    # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
    s0 = [5., 1400., 100.]

    # Fitting
    coeff, var_matrix = curve_fit(gaussian, fval, ss, p0=s0)

    # Fitted curve
    spec_fit.append(gaussian(fval, *coeff))
    coeff_s.append(coeff)
    err_s.append(np.sqrt(np.diag(var_matrix)))
print(err_s)

# Fitting drift rate
tdrift = [c[1] for c in coeff_p]
terr = [e[1] for e in err_p]
fdrift = [c[1] for c in coeff_s]
ferr = [e[1] for e in err_s]

d0 = [-4.2, 0]

coeff, var_matrix = curve_fit(drift, tdrift, fdrift, p0=d0, sigma=ferr)
f0 = coeff[1]
drift_err = np.sqrt(np.diag(var_matrix))

# Freezing one component
coeff, var_matrix = curve_fit(lambda x, dft: drift(x, dft, f0), tdrift, fdrift, sigma=ferr)
r_drift = coeff[0]

drift_fit = drift(tval, r_drift, f0)
drift_err[0] = np.sqrt(var_matrix[0,0])

print([r_drift, f0], drift_err)
print('Drift rate: {:.4f} +- {:.4f} MHz/ms\n'.format(coeff[0], drift_err[0]))

print(idb, fn.split('/')[-1], dm, ncomp, '{:.4f} {:.4f}'.format(coeff[0], drift_err[0]), end = ' ')
print([round(x, 4) for x in tdrift], [round(x, 4) for x in terr], end=' ')
print([round(x, 2) for x in fdrift], [round(x, 2) for x in ferr])

# PLotting
nrows, ncols = 2, 2
fig = plt.figure(figsize=(9,6.5))
gs = gridspec.GridSpec(nrows,ncols, hspace=0.0, wspace=0.0, height_ratios=[1,3], width_ratios=[3,1])
colors = ['#577590', '#90be6d', '#f9c74f', '#f3722c', '#f94144']
mpl.rcParams.update({'font.size': 18})

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[1,0], sharex=ax1)
ax3 = fig.add_subplot(gs[1,1])

ax1.plot(tval, pulse, color='k')
ax1.plot(tval, pulse_fit, color='gray')

ax2.imshow(waterfall, interpolation='nearest', aspect='auto', cmap='viridis', origin='lower',
           extent=[tval[0], tval[-1], fmin, fmax])

ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Frequency (MHz)')

for i in range(ncomp):
    ax1.axvspan(coeff_p[i,1]-2*coeff_p[i,2], coeff_p[i,1]+2*coeff_p[i,2],
                alpha=0.3, color=colors[i])
    rspec = spec[i].reshape(len(spec[i])//2, 2).mean(-1)
    rfval = fval.reshape(len(fval)//2,2).mean(-1)
    ax3.plot(rspec, rfval, ls='-', lw=0.7, color=colors[i])
    ax3.plot(spec_fit[i], fval, ls='--', lw=1.5, color=colors[i], alpha=0.8)

ax2.errorbar(tdrift, fdrift, xerr=terr, yerr=ferr,
             zorder=10, color='w', marker='x')
ax2.plot(tval, drift_fit, zorder=9, color='w')

ax1.text(0.05, 0.95, idb, horizontalalignment='left',
        verticalalignment='top', transform=ax1.transAxes)
#ax3.set_xticks([])
#ax3.set_yticks([])
ax1.set_xlim(-10, 10)
ax1.set_yticklabels([])
ax2.set_ylim(fmin, fmax)
ax3.set_ylim(fmin, fmax)
ax3.set_yticklabels([])
ax3.set_xticklabels([])

for ax in (ax1, ax2, ax3):
    ax.tick_params(axis='x', which='both', direction='in', bottom=True,
            top=True)
    ax.tick_params(axis='y', which='both', direction='in', left=True,
            right=True)
    ax.label_outer()

plt_out = fn.replace('.npy', '.pdf').replace('iquv_npy', 'plot_drift')
plt.savefig(plt_out, pad_inches=0, bbox_inches='tight', dpi=200)
print("Saving", plt_out)
plt.show()
