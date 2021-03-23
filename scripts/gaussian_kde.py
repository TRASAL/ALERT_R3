from math import *
import numpy as np
from numpy import inf
import json
import matplotlib.pylab as plt
import matplotlib.mlab as mlab
from scipy import stats
from scipy.optimize import curve_fit
from astropy.time import Time
from frbpa.utils import get_params
import matplotlib.gridspec as gridspec
from lmfit.models import GaussianModel

def gaussian(x, *p):
    (c, mu, sigma) = p
    res =    c * np.exp( - (x - mu)**2.0 / (2.0 * sigma**2.0) )
    return res

def gaussian_fit(bins, y):
	# centers = (0.5*(bins[1:]+bins[:-1]))
	# pars, cov = curve_fit(lambda x, mu, sig : stats.norm.pdf(
	# 		x, loc=mu, scale=sig), centers, y, p0=[0,1])
	# mu, sigma = pars[0], pars[1]
	# muerr, serr = np.sqrt(cov[0,0]), np.sqrt(cov[1,1])

	x = (0.5*(bins[1:]+bins[:-1]))
	pars, cov = curve_fit(gaussian, x, y, p0=[1,0.5,1])
	c, mu, sigma = pars[0], pars[1], pars[2]
	err = np.sqrt(np.diag(cov))
	cerr, muerr, serr = err[0], err[1], err[2] #np.sqrt(cov[0,0]), np.sqrt(cov[1,1])
	return c, cerr, mu, muerr, sigma, serr

P = 16.29
fndir = '/home/ines/Documents/projects/R3/periodicity/burst_phases/'

fnarts = fndir + 'phase_Apertif_p{:.2f}_f1370.0.npy'.format(P)
fnchime = fndir + 'phase_CHIME_p{:.2f}_f600.0.npy'.format(P)
fnlofar = fndir + 'phase_LOFAR_p{:.2f}_f150.0.npy'.format(P)

data_r3 = np.loadtxt(fndir+'counts_per_phase_p{:.2f}.txt'.format(P))
data_arts=np.load(fnarts)
data_chime=np.load(fnchime)
data_lofar=np.load(fnlofar)
chime_mjd=Time(data_chime[0], format='mjd')
year_arr=[]
period = fnarts.split('/')[-1].split('_')[2].replace('p','')

plt_out = "/home/ines/Documents/projects/R3/periodicity/gaussian_kde.pdf"
#plt_out = "/home/ines/Documents/PhD/meetings/20210303-Astrolunch_talk/figs/gaussian_kde.png"

for dd in chime_mjd.datetime:
	year_arr.append(dd.year)
year_arr=np.array(year_arr)
obs_t_ch = data_r3[:, -3]
obs_t_arts = data_r3[:, -2]
obs_t_lofar = data_r3[:, -1]
obs_phase = data_r3[:, 0]
weights_arts, weights_ch, weights_lofar = [],[],[]

# create weights array for each detection
# based on the reciprical of the amount of
# observing time spent in near that phase bin
for pb_arts in data_arts[1]:
	ind=np.argmin(np.abs(pb_arts-obs_phase))
	weights_arts.append(obs_t_arts[ind]**-1)
weights_arts = np.array(weights_arts)
weights_arts[weights_arts == inf] = 0

for pb_chime in data_chime[1]:
	ind=np.argmin(np.abs(pb_chime-obs_phase))
	weights_ch.append(obs_t_ch[ind]**-1)
weights_ch = np.array(weights_ch)
weights_ch[weights_ch == inf] = 0

for pb_lofar in data_lofar[1]:
	ind=np.argmin(np.abs(pb_lofar-obs_phase))
	weights_lofar.append(obs_t_lofar[ind]**-1)
weights_lofar = np.array(weights_lofar)
weights_lofar[weights_lofar == inf] = 0

# Average rate per instrument
rate_chime = data_chime.shape[1] / np.sum(obs_t_ch)
rate_arts = data_arts.shape[1] / np.sum(obs_t_arts)
rate_lofar = data_lofar.shape[1] / np.sum(obs_t_lofar)

# Printing average rate:
print("Rate ARTS:", rate_arts)
print("Rate CHIME:", rate_chime)
print("Rate LOFAR:", rate_lofar)

# Get the Gaussian Kernel Density Estimator for
# CHIME and ARTS
func_arts = stats.gaussian_kde(data_arts[1], weights=weights_arts)
func_lofar = stats.gaussian_kde(data_lofar[1], weights=weights_lofar)
func_chime = stats.gaussian_kde(data_chime[1], weights=weights_ch)
func_chime_2018 = stats.gaussian_kde(data_chime[1][year_arr<2020])
func_chime_2020 = stats.gaussian_kde(data_chime[1][year_arr==2020])
phase_bin = np.linspace(0,1,1000)

# Defining colors
cm = plt.cm.get_cmap('Spectral_r')
max_freq=2500
min_freq=200
fcen_dict = {'Apertif': 1370.0, 'CHIME': 600.0, 'LOFAR': 150.0}
obs_colors = {'Apertif': 'C2', 'CHIME': 'C1', 'LOFAR': 'C3'}

# Printing phase max:
print("Max phase ARTS:", phase_bin[np.argmax(func_arts(phase_bin))])
print("Max phase CHIME:", phase_bin[np.argmax(func_chime(phase_bin))])
print("Max phase LOFAR:", phase_bin[np.argmax(func_lofar(phase_bin))])


# Plotting
fig = plt.figure(figsize=(8,5))
gs = gridspec.GridSpec(1,1, hspace=0.0)
params = get_params()
#plt.rcParams.update(params)
plt.style.use('/home/ines/.config/matplotlib/stylelib/paper.mplstyle')

axl = fig.add_subplot(gs[0, 0])
axr = axl.twinx()

# Plotting gaussian KDEs
axr.plot(phase_bin, func_arts(phase_bin)*rate_arts,
		color=obs_colors['Apertif'], lw=3,
		label='Apertif {} MHz'.format(int(fcen_dict['Apertif'])))
axr.plot(phase_bin, func_chime(phase_bin)*rate_chime,
		color=obs_colors['CHIME'], lw=3,
		label='CHIME/FRB {} MHz'.format(int(fcen_dict['CHIME'])))
axr.plot(phase_bin, func_chime_2018(phase_bin)*rate_chime,
		color=obs_colors['CHIME'], lw=1, linestyle='dotted',
		label='CHIME/FRB <2020')
axr.plot(phase_bin, func_chime_2020(phase_bin)*rate_chime,
		color=obs_colors['CHIME'], lw=1, linestyle='dashed',
		label='CHIME/FRB 2020')
axr.plot(phase_bin, func_lofar(phase_bin)*rate_lofar,
		color=obs_colors['LOFAR'], lw=3,
		label='LOFAR {} MHz'.format(int(fcen_dict['LOFAR'])))

# Plotting histograms
ych,xch,_ = axl.hist(data_chime[1], bins=50, range=(0,1),
		alpha=0.4, color=obs_colors['CHIME'])
yarts,xarts,_ = axl.hist(data_arts[1], bins=50, range=(0,1),
		alpha=0.4, color=obs_colors['Apertif'])
ylofar,xlofar,_ = axl.hist(data_lofar[1], bins=50, range=(0,1),
		alpha=0.4, color=obs_colors['LOFAR'])


axl.set_ylabel("Detections")
axr.set_ylabel("Rate (h$^{-1}$)")
axl.set_xlim(0,1)
axl.set_xlim(0,1)
axl.set_ylim(0,20)
axr.set_ylim(0,2)
axr.legend()
#axl.set_title('Gaussian KDE', fontsize=14)
#ax.set_xticklabels([])
axl.text(0.05, 0.95, "P = {} days".format(period), verticalalignment='top',
		transform=axl.transAxes, fontsize=14)
axl.set_xlabel("Phase", fontsize=12)


# Plotting observation time
# ax1 = fig.add_subplot(gs[1, 0])
#
# ax1.step(obs_phase-obs_phase[0], obs_t_lofar, color=obs_colors['LOFAR'],
# 		label='LOFAR')
# ax1.set_ylabel("Obs. duration (h)", fontsize=12)
# ax1.set_xlim(0,1)
# ax1.set_ylim(0,9)
# ax1.legend(fontsize=12, loc=0)
# print(np.sum(obs_t_lofar))

# Fitting to gaussian
model = GaussianModel()

# create parameters with initial guesses:
params = model.make_params(center=0.5, amplitude=10, sigma=1)

print("Apertif")
x = (0.5*(xarts[1:]+xarts[:-1]))
result = model.fit(yarts, params, x=x)
#print(result.fit_report())
mu = result.params.get("center").value
muerr = result.params.get("center").stderr
fwhm = result.params.get("fwhm").value
fwhmerr = result.params.get("fwhm").stderr
print("\tmu {:.3f}+-{:.3f} (phase) \n" \
		"\t{:.2f}+-{:.2f} (days) \n" \
		"\tfwhm {:.2f}+-{:.2f} (days)".format(
		mu, muerr, mu*P, muerr*P, fwhm*P, fwhmerr*P))
# axl.plot(x, gaussian(x, result.params.get("height").value,
# 		mu, result.params.get("sigma").value), color=obs_colors['Apertif'])

print("CHIME")
x = (0.5*(xch[1:]+xch[:-1]))
result = model.fit(ych, params, x=x)
#print(result.fit_report())
mu = result.params.get("center").value
muerr = result.params.get("center").stderr
fwhm = result.params.get("fwhm").value
fwhmerr = result.params.get("fwhm").stderr
print("\tmu {:.3f}+-{:.3f} (phase) \n" \
		"\t{:.2f}+-{:.2f} (days) \n" \
		"\tfwhm {:.2f}+-{:.2f} (days)".format(
		mu, muerr, mu*P, muerr*P, fwhm*P, fwhmerr*P))
# axl.plot(x, gaussian(x, result.params.get("height").value,
# 		mu, result.params.get("sigma").value), color=obs_colors['CHIME'])

print("LOFAR")
x = (0.5*(xlofar[1:]+xlofar[:-1]))
result = model.fit(ylofar, params, x=x)
#print(result.fit_report())
mu = result.params.get("center").value
muerr = result.params.get("center").stderr
fwhm = result.params.get("fwhm").value
fwhmerr = result.params.get("fwhm").stderr
print("\tmu {:.3f}+-{:.3f} (phase) \n" \
		"\t{:.2f}+-{:.2f} (days) \n" \
		"\tfwhm {:.2f}+-{:.2f} (days)".format(
		mu, muerr, mu*P, muerr*P, fwhm*P, fwhmerr*P))
# axl.plot(x, gaussian(x, result.params.get("height").value,
# 		mu, result.params.get("sigma").value), color=obs_colors['LOFAR'])

# Plotting gaussians
# axl.plot(xch, ynch, color=obs_colors['CHIME'])
# axl.plot(xarts, ynarts, color=obs_colors['Apertif'])
# axl.plot(xlofar, ynlofar, color=obs_colors['LOFAR'])

print('Saving figure to ', plt_out)
plt.savefig(plt_out, pad_inches=0, bbox_inches='tight')
plt.show()
