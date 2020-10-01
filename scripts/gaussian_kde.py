import numpy as np
from numpy import inf
import json
import matplotlib.pylab as plt
from scipy import stats
from astropy.time import Time
from frbpa.utils import get_params
import matplotlib.gridspec as gridspec

P = 16.29
fndir = '/home/ines/Documents/projects/R3/periodicity/burst_phases/'
# fnchime = fndir + 'phase_CHIME_p16.28_f600.0.npy'
# fnarts = fndir + 'phase_ARTS_p16.28_f1370.0.npy'
# fnlofar = fndir + 'phase_LOFAR_p16.28_f150.0.npy'

fnarts = fndir + 'phase_ARTS_p{:.2f}_f1370.0.npy'.format(P)
fnchime = fndir + 'phase_CHIME_p{:.2f}_f600.0.npy'.format(P)
fnlofar = fndir + 'phase_LOFAR_p{:.2f}_f150.0.npy'.format(P)

data_r3 = np.loadtxt(fndir+'counts_per_phase_p{:.2f}.txt'.format(P))
data_arts=np.load(fnarts)
data_chime=np.load(fnchime)
data_lofar=np.load(fnlofar)
chime_mjd=Time(data_chime[0], format='mjd')
year_arr=[]
period = fnarts.split('/')[-1].split('_')[2].replace('p','')
plt_out = "/home/ines/Documents/projects/R3/periodicity/gaussian_kde.png"
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
fcen_dict = {'ARTS': 1370.0, 'CHIME': 600.0, 'LOFAR': 150.0}
obs_colors = {'ARTS': 'C2', 'CHIME': 'C1', 'LOFAR': 'C3'}

# Printing phase max:
print("Max phase ARTS:", phase_bin[np.argmax(func_arts(phase_bin))])
print("Max phase CHIME:", phase_bin[np.argmax(func_chime(phase_bin))])
print("Max phase LOFAR:", phase_bin[np.argmax(func_lofar(phase_bin))])


# Plotting
fig = plt.figure(figsize=(8,5))
gs = gridspec.GridSpec(1,1, hspace=0.0)
params = get_params()
#plt.rcParams.update(params)
plt.rcParams.update({
        'font.size': 14,
		'font.family': 'serif',
        'axes.labelsize': 14,
        'axes.titlesize': 18,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        # 'xtick.minor.visible': True,
        # 'ytick.minor.visible': True,
        'xtick.top': True,
        'ytick.right': True,
        'lines.linewidth': 0.5,
        'lines.markersize': 5,
        'legend.fontsize': 12,
        # 'legend.borderaxespad': 0,
        # 'legend.frameon': True,
        'legend.loc': 'upper right'})

axl = fig.add_subplot(gs[0, 0])
axr = axl.twinx()

# Plotting gaussian KDEs
axr.plot(phase_bin, func_arts(phase_bin)*rate_arts, color=obs_colors['ARTS'], lw=3, label='Apertif {} MHz'.format(int(fcen_dict['ARTS'])))
axr.plot(phase_bin, func_chime(phase_bin)*rate_chime, color=obs_colors['CHIME'], lw=3, label='CHIME {} MHz'.format(int(fcen_dict['CHIME'])))
axr.plot(phase_bin, func_chime_2018(phase_bin)*rate_chime,
	color=obs_colors['CHIME'], lw=1, linestyle='dotted', label='CHIME <2020')
axr.plot(phase_bin, func_chime_2020(phase_bin)*rate_chime,
	color=obs_colors['CHIME'], lw=1, linestyle='dashed', label='CHIME 2020')
axr.plot(phase_bin, func_lofar(phase_bin)*rate_lofar, color=obs_colors['LOFAR'], lw=3, label='LOFAR {} MHz'.format(int(fcen_dict['LOFAR'])))

# Plotting histograms
ych,xch,_ = axl.hist(data_chime[1], bins=50, range=(0,1),
		alpha=0.4, color=obs_colors['CHIME'])
yarts,xarts,_ = axl.hist(data_arts[1], bins=50, range=(0,1),
		alpha=0.4, color=obs_colors['ARTS'])
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

print('Saving figure to ', plt_out)
plt.savefig(plt_out, pad_inches=0, bbox_inches='tight')
plt.show()
