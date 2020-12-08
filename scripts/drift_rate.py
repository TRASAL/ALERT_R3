import math
import glob, sys
import numpy as np
import uncertainties.unumpy as unp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.stats import norm
from lmfit import Model, Parameters
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
def duodecuple_gaussian(x, *p):
    (c1, mu1, sigma1, c2, mu2, sigma2, c3, mu3, sigma3, c4, mu4, sigma4, c5, mu5, sigma5, c6, mu6, sigma6, c7, mu7, sigma7, c8, mu8, sigma8, c9, mu9, sigma9, c10, mu10, sigma10, c11, mu11, sigma11, c12, mu12, sigma12) = p
    res =    c1 * np.exp( - (x - mu1)**2.0 / (2.0 * sigma1**2.0) ) \
        +  c2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) ) \
        +  c3 * np.exp( - (x - mu3)**2.0 / (2.0 * sigma3**2.0) ) \
        +  c4 * np.exp( - (x - mu4)**2.0 / (2.0 * sigma4**2.0) ) \
        +  c5 * np.exp( - (x - mu5)**2.0 / (2.0 * sigma5**2.0) ) \
        +  c6 * np.exp( - (x - mu6)**2.0 / (2.0 * sigma6**2.0) ) \
        +  c7 * np.exp( - (x - mu7)**2.0 / (2.0 * sigma7**2.0) ) \
        +  c8 * np.exp( - (x - mu8)**2.0 / (2.0 * sigma8**2.0) ) \
        +  c9 * np.exp( - (x - mu9)**2.0 / (2.0 * sigma9**2.0) ) \
        +  c12 * np.exp( - (x - mu10)**2.0 / (2.0 * sigma10**2.0) ) \
        +  c11 * np.exp( - (x - mu11)**2.0 / (2.0 * sigma11**2.0) ) \
        +  c12 * np.exp( - (x - mu12)**2.0 / (2.0 * sigma12**2.0) )
    return res

def quintuple_gaussian(x, *p):
    (c1, mu1, sigma1, c2, mu2, sigma2, c3, mu3, sigma3, c4, mu4, sigma4, c5, mu5, sigma5) = p
    res =    c1 * np.exp( - (x - mu1)**2.0 / (2.0 * sigma1**2.0) ) \
          +  c2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) ) \
          +  c3 * np.exp( - (x - mu3)**2.0 / (2.0 * sigma3**2.0) ) \
          +  c4 * np.exp( - (x - mu4)**2.0 / (2.0 * sigma4**2.0) ) \
          +  c5 * np.exp( - (x - mu5)**2.0 / (2.0 * sigma5**2.0) )
    return res

def quadruple_gaussian(x, *p):
    (c1, mu1, sigma1, c2, mu2, sigma2, c3, mu3, sigma3, c4, mu4, sigma4) = p
    res =    c1 * np.exp( - (x - mu1)**2.0 / (2.0 * sigma1**2.0) ) \
          +  c2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) ) \
          +  c3 * np.exp( - (x - mu3)**2.0 / (2.0 * sigma3**2.0) ) \
          +  c4 * np.exp( - (x - mu4)**2.0 / (2.0 * sigma4**2.0) )
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

def drift_error(x, y, xerr, yerr, model, num=10):
    "Getting drift rate errors through simulations"
    drift_sim = np.zeros(num)
    freq_sim = np.zeros(num)
    xsim = np.array([norm.rvs(loc=x[i], scale=xerr[i], size=num)
            for i in range(len(x))])
    ysim = np.array([norm.rvs(loc=y[i], scale=yerr[i], size=num)
            for i in range(len(y))])
    for i in range(num):
        coeff, _ = curve_fit(model, xsim[:,i], ysim[:,i])
        drift_sim[i] = coeff[0]
        freq_sim[i] = coeff[1]
    return [np.std(drift_sim), np.std(freq_sim)]

# Opening data
burst_data = pd.read_csv(
        '/home/ines/Documents/projects/R3/arts/arts_r3_properties.csv')
fname = '/home/ines/Documents/projects/R3/arts/drift_rate/drift_rate.txt'
file_info = np.genfromtxt(fname, names=True,
        dtype=('<U3', '<U35', 'f8', 'i8', 'f8', 'f8', '<U41',
        '<U41', '<U41', '<U41'))
print(file_info.dtype.names)

id = sys.argv[1]
ref_dm = 348.75
burst = burst_data.loc[burst_data['paper_name'] == id]

# Defining burst properties
data_path = '/home/ines/Documents/projects/R3/arts/drift_rate/iquv_npy/'
mjd = burst['detection_mjd'].values[0]
print(data_path + "*{:.6f}*".format(mjd))
fn = glob.glob(data_path + "*{:.6f}*".format(mjd))[0] #file_info['file_'][ii]
print(fn)
dm = 348.75 #file_info['dm'][ii]
ncomp = int(burst['ncomp'].values[0]) #file_info['ncomp'][ii]

pmax=99.5
pmin=1.0
if ncomp == 1:
    t0 = 0.
    p0 = [.1, t0, .1]
elif ncomp <=5 :
    ii = np.where(id == file_info['id'])[0][0] #int(id.replace('A', ''))-1
    print("Line", ii)
    t0 = file_info['tdrift'][ii].strip('][').split(',')
    p0 = np.array([[.1, float(t), .1] for t in t0]).flatten()
    # t0 = 0.
    # p0 = [.1, 0, .1, .1, .8, .1]
if id == 'A17':
    p0 = [0.2, -.8, 0.1,
          2., -0.1, 0.15,
          1.3, 1., 0.3,
          0.3, 2., 0.15,
          0.15, 2.75, 0.2]
    pmax=99.9
    pmin=1.0
if id == 'A53':
    p0 = [.2, -11, 0.1,
      .4, -8.4, 0.1,
      .2, -7.5, 0.2,
      .25, -3, .2,
      .2, -1.5, .4,
      .1, 0.5, .2,
      .3, 1.5, .15,
      .07, 7, .5,
      .2, 10.4, .05,
      .4, 11.5, .15,
      .2, 13, .15,
      .1, 14.5, .15]
    pmax = 99.9
    pmin = 1.0

print("Number of components", ncomp)
print("Initial conditions:", p0)

# Loading data
data = np.load(fn)
#data = data - np.mean(data)
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
if id == 'A17':
    pulse += 0.05

# Plotting dynamic spectrum
fig = plt.figure(figsize=(9,9))
gs = gridspec.GridSpec(2,1, hspace=0.0, wspace=0.0, height_ratios=[1,3])

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[1,0], sharex=ax1)

ax1.plot(tval, pulse, color='k')

if ncomp == 1:
    pulse0 = gaussian(tval, *p0)
    ax1.plot(tval, pulse0, color='gray')
elif ncomp == 2:
    print("Initial parameters", p0)
    pulse0 = double_gaussian(tval, *p0)
    ax1.plot(tval, pulse0, color='gray')
elif ncomp == 3:
    pulse0 = triple_gaussian(tval, *p0)
    ax1.plot(tval, pulse0, color='gray')
elif ncomp == 4:
    pulse0 = quadruple_gaussian(tval, *p0)
    ax1.plot(tval, pulse0, color='gray')
elif ncomp == 5:
    pulse0 = quintuple_gaussian(tval, *p0)
    ax1.plot(tval, pulse0, color='gray')
elif ncomp == 12:
    pulse0 = duodecuple_gaussian(tval, *p0)
    ax1.plot(tval, pulse0, color='gray')

ax2.imshow(waterfall, interpolation='nearest', aspect='auto', origin='lower',
        cmap='viridis', extent=[tval[0], tval[-1], fmin, fmax])

ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Frequency (MHz)')

ax1.set_xlim(tval[0], tval[-1])
ax2.set_ylim(fmin, fmax)
plt.show()

# Fitting pulse to gaussians
# p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
if ncomp == 2:
    print("Fitting 2 components")

    # Fitting
    coeff_p, var_matrix_p = curve_fit(double_gaussian, tval, pulse, p0=p0)
    err_p = np.reshape(np.sqrt(np.diag(var_matrix_p)), (2,3))

    # Fitted curve
    pulse_fit = double_gaussian(tval, *coeff_p)
    coeff_p = np.reshape(coeff_p, (2,3))

elif ncomp == 3:
    print("Fitting 3 components")

    # Fitting
    coeff_p, var_matrix_p = curve_fit(triple_gaussian, tval, pulse, p0=p0)
    err_p = np.reshape(np.sqrt(np.diag(var_matrix_p)), (3,3))

    # Fitted curve
    pulse_fit = triple_gaussian(tval, *coeff_p)
    coeff_p = np.reshape(coeff_p, (3,3))

elif ncomp == 4:
    print("Fitting 4 components")

    # Fitting
    coeff_p, var_matrix_p = curve_fit(quadruple_gaussian, tval, pulse, p0=p0)
    err_p = np.reshape(np.sqrt(np.diag(var_matrix_p)), (4,3))

    # Fitted curve
    pulse_fit = quadruple_gaussian(tval, *coeff_p)
    coeff_p = np.reshape(coeff_p, (4,3))

elif ncomp == 5:
    print("Fitting 5 components")

    # Fitting
    coeff_p, var_matrix_p = curve_fit(quintuple_gaussian, tval, pulse, p0=p0)
    err_p = np.reshape(np.sqrt(np.diag(var_matrix_p)), (5,3))

    # Fitted curve
    pulse_fit = quintuple_gaussian(tval, *coeff_p)
    coeff_p = np.reshape(coeff_p, (5,3))

elif ncomp == 12:
    print("Fitting 12 components")

    # Fitting
    coeff_p, var_matrix_p = curve_fit(duodecuple_gaussian, tval, pulse, p0=p0)
    err_p = np.reshape(np.sqrt(np.diag(var_matrix_p)), (12,3))

    # Fitted curve
    pulse_fit = duodecuple_gaussian(tval, *coeff_p)
    coeff_p = np.reshape(coeff_p, (12,3))

else:
    print("Fitting 1 component")
    # Fitting
    coeff_p, var_matrix_p = curve_fit(gaussian, tval, pulse, p0=p0)
    err_p = np.reshape(np.sqrt(np.diag(var_matrix_p)), (1,3))

    # Fitted curve
    pulse_fit = gaussian(tval, *coeff_p)
    coeff_p = np.reshape(coeff_p, (1,3))

print(coeff_p, err_p)

# Fitting spectra of subbursts to gaussians
spec = []
for ii in range(ncomp):
    imin = np.abs(tval - (coeff_p[ii,1] \
            - 2 * coeff_p[ii,2])).argmin()
    imax = np.abs(tval - (coeff_p[ii,1] \
            + 2 * coeff_p[ii,2])).argmin()
    spec_comp = np.mean(waterfall[:,imin:imax], axis=1)
    spec_comp = np.nan_to_num(spec_comp)
    spec.append(spec_comp)
    print(imin, imax)
    #spec.append(np.mean(waterfall[:,imin:imax], axis=1))

#print(spec)
spec_fit = []
coeff_s = []
err_s = []
for ii,ss in enumerate(spec):
    # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
    f0 = fval[np.argmax(ss)]
    s0 = [.5, f0, 100.]
    print("Initial spec params", s0)
    # plt.plot(fval, ss)
    # plt.plot(fval, gaussian(fval, *s0))
    # plt.show()

    # Fitting
    coeff, var_matrix = curve_fit(gaussian, fval, ss, p0=s0)

    # Fitted curve
    spec_fit.append(gaussian(fval, *coeff))
    coeff_s.append(coeff)
    err_s.append(np.sqrt(np.diag(var_matrix)))

coeff_s = np.array(coeff_s)
print("Frequencies", coeff_s[:,1])
print("\nCentral frequency", np.mean(coeff_s[:,1]))
#print(err_s)

# Fitting drift rate
if ncomp > 1:
    tdrift = np.array([c[1] for c in coeff_p])
    terr = np.array([e[1] for e in err_p])
    fdrift = np.array([c[1] for c in coeff_s])
    ferr = np.array([e[1] for e in err_s])

    d0 = np.array([-4.2, 0])

    coeff, var_matrix = curve_fit(drift, tdrift, fdrift, p0=d0, sigma=ferr)
    drift_fit = drift(tval, *coeff)
    if ncomp == 2:
        drift_err = drift_error(tdrift, fdrift, terr, ferr, drift, num=10000)
    else:
        drift_err = np.sqrt(np.diag(var_matrix))

    print(coeff, drift_err)
    print('Drift rate: {:.4f} +- {:.4f} MHz/ms\n'.format(coeff[0], drift_err[0]))

    print(id, fn.split('/')[-1], dm, ncomp, '{:.4f} {:.4f}'.format(coeff[0],
            drift_err[0]), end = ' ')
    print([round(x, 4) for x in tdrift], [round(x, 4) for x in terr], end=' ')
    print([round(x, 2) for x in fdrift], [round(x, 2) for x in ferr])

# ------------------------------------------------------------------------- #
# Plotting
nrows, ncols = 2, 2
fig = plt.figure(figsize=(9,6.5))
gs = gridspec.GridSpec(nrows,ncols, hspace=0.0, wspace=0.0,
        height_ratios=[1,3], width_ratios=[6,1])
colors = ['#577590', '#90be6d', '#f9c74f', '#f3722c', '#f94144']
plt.style.use('/home/ines/.config/matplotlib/stylelib/paper.mplstyle')

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[1,0], sharex=ax1)
ax3 = fig.add_subplot(gs[1,1])

ax1.plot(tval, pulse, color='k')
ax1.plot(tval, pulse_fit, color='gray')
vmin = np.percentile(waterfall, pmin)
vmax = np.percentile(waterfall, pmax)
ax2.imshow(waterfall, interpolation='nearest', aspect='auto', cmap='viridis',
        origin='lower', extent=[tval[0], tval[-1], fmin, fmax],
        vmin=vmin, vmax=vmax)

ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Frequency (MHz)')

if ncomp <= 5:
    for i in range(ncomp):
        ax1.axvspan(coeff_p[i,1]-2*coeff_p[i,2],
                coeff_p[i,1]+2*coeff_p[i,2],
                alpha=0.3, color=colors[i])
        rspec = spec[i].reshape(len(spec[i])//2, 2).mean(-1)
        rfval = fval.reshape(len(fval)//2,2).mean(-1)
        ax3.plot(rspec, rfval, ls='-', lw=0.7, color=colors[i])
        ax3.plot(spec_fit[i], fval, ls='--', lw=1.5, color=colors[i], alpha=0.8)
else:
    cm = plt.cm.get_cmap('Spectral_r')
    for i in range(ncomp):
        c = i/ncomp
        col = cm(c)
        ax1.axvspan(coeff_p[i,1]-2*coeff_p[i,2], coeff_p[i,1]+2*coeff_p[i,2],
                    alpha=0.3, color=col)
        rspec = spec[i].reshape(len(spec[i])//2, 2).mean(-1)
        rfval = fval.reshape(len(fval)//2,2).mean(-1)
        ax3.plot(rspec, rfval, ls='-', lw=0.7, color=col)
        ax3.plot(spec_fit[i], fval, ls='--', lw=1.5, color=col, alpha=0.8)

if ncomp > 1:
    ax2.errorbar(tdrift, fdrift, xerr=terr, yerr=ferr,
                 zorder=10, color='w', marker='x')
    ax2.plot(tval, drift_fit, zorder=9, color='w')

ax1.text(0.05, 0.95, id, horizontalalignment='left',
        verticalalignment='top', transform=ax1.transAxes)
#ax3.set_xticks([])
#ax3.set_yticks([])
if id != 'A53':
    ax1.set_xlim(-10, 10)
else:
    ax1.set_xlim(-20, 20)
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
