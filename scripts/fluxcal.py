from __future__ import print_function
from __future__ import division
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import astropy.units as u
from astropy.time import Time
import datetime
import argparse
import tools
import pol
#import read_IQUV_dada
from calibration_tools import run_fluxcal, CalibrationTools, Plotter
SNRtools = tools.SNR_Tools()

# Initial parameters
t_res = 0.8192*1e-4 # s
f_res = 0.1953125 * 1e6 # Hz
t_cal = t_res*1e4 # s
Ndish = 10.0
IAB = True
NPOL = 2
BW = 3e8 # Hz
# width = 1 # ms
# SNR = 25

# Functions
def closest_mjd_fct(list, mjd):
    diff = np.array([abs(l-mjd) for l in list])
    i = np.argmin(diff)
    return i

def peak_flux(snr, width, sefd_rms, BW=3e8, NPOL=2):
    flux_freq = sefd_rms / np.sqrt(NPOL * BW * width*1e-3) * snr
    flux = np.mean(flux_freq)
    return flux, 0.1*flux

def fluence_boxcar(snr, width, sefd_rms, BW=3e8, NPOL=2):
    fluence_freq = sefd_rms / np.sqrt(NPOL * BW * width*1e-3) * snr * width
    fluence = np.median(fluence_freq)
    return fluence, 0.1*fluence, fluence_freq

def fluence_integral(data, sefd_rms,
        t_res=0.8192*1e-4, f_res=0.1953125*1e6, NPOL=2):
    # sefd_rms_all = np.transpose(np.tile(sefd_rms,
    #         (data.shape[1],1)))
    # # with t_res*1e3 I get ~10 times the boxcar fluence value
    # fluence_tf = sefd_rms_all / np.sqrt(NPOL * f_res * t_res) * data * t_res*1e3
    # fluence_freq = np.sum(fluence_tf, axis=1)
    # fluence = np.median(fluence_freq)
    pulse = np.nanmean(data, axis=0)
    fluence_t = np.median(sefd_rms)/np.sqrt(NPOL*f_res*t_res) * pulse*t_res*1e3
    fluence = np.sum(fluence_t)
    #print(np.median(sefd_rms), np.sum(pulse), 1/np.sqrt(NPOL*f_res*t_res))
    return fluence, 0.1*fluence #, fluence_freq

# File paths
data_path = '/home/arts/pastor/R3/fluxcal/i_npy/'
drift_path = '/tank/data/FRBs/R3/driftscan/'
driftnames = [
        #'2020-03-23-23:35:00.3C286drift00/CB00_00_downsamp10000_dt0.819.npy',
        # '2020-03-18-18:45:00.3C147drift00/CB00_00_downsamp10000_dt0.819.npy',
        # '2020-05-07-19:40:00.3C286drift00/CB00_00_downsamp10000_dt0.819.npy',
        # #'2020-05-15-18:30:00.3C286drift00/CB00_00_downsamp10000_dt0.819.npy',
        # '2020-05-21-19:55:00.3C286drift00/CB00_00_downsamp10000_dt0.819.npy',
        # '2020-05-26-19:00:00.3C286drift00/CB00_00_downsamp10000_dt0.819.npy',
        '2020-08-27-04:45:00.3C147drift00/CB00_00_downsamp10000_dt0.819.npy']
sefdnames = [
    '20200827_3C147'
    ]
driftmjds = []
for i,d in enumerate(driftnames):
    date = d.split('.')[0]
    t = 'T'.join(date.rsplit('-', 1))
    driftmjds.append(Time(t, format='isot', scale='utc').mjd)

snr_fname = '/home/arts/pastor/R3/fluxcal/snr_width_all.txt'
snr_width = np.genfromtxt(snr_fname, dtype=('<U30', 'float', 'int'), names=True)
fluence_fname = '/home/arts/pastor/R3/fluxcal/fluence_int.txt'

# Computing fluence
fout = open(fluence_fname, 'w')
fout.write('# MJD width_ms snr fluence_Jyms fluence_err fint_Jyms fint_err ' \
        'flux_Jy flux_err\n')
#print('# MJD \t\t width_ms \tSNR \tfluence_Jyms\n')

# Plotting
nrows,ncols = 7,8
fig = plt.figure(figsize=(21,29))
gs = gridspec.GridSpec(nrows,ncols, hspace=0.05, wspace=0.05)

for i,b in enumerate(snr_width):
    burst_id = 'A{:02}'.format(i+1)
    snr = b['SNR_boxcar']
    bwidth = b['boxcar_width_samples']
    width = bwidth * t_res *1e3 # s
    filename = b['filename'].replace('.npy:', '.npy')
    mjd = float(filename.split('_')[1].replace('mjd', ''))

    closest_driftscan = driftnames[closest_mjd_fct(driftmjds, mjd)]
    driftname = drift_path + closest_driftscan
    src = closest_driftscan.split('/')[0].split('.')[1].replace('drift00', '')

    # Calibration
    driftscan = np.load(driftname)

    driftscan[driftscan!=driftscan] = 0.
    nt = driftscan.shape[-1]
    data = driftscan.reshape(-1, 2, nt).mean(1)
    nfreq = driftscan.shape[0]

    # Calibrating
    CalTools = CalibrationTools(t_res=t_cal, Ndish=Ndish, IAB=IAB, nfreq=nfreq)
    tsys_rms = CalTools.tsys_rms_allfreq(driftscan, off_samp=(0, 200), src=src)
    sefd_rms = CalTools.tsys_to_sefd(tsys_rms)
    #sefd_rms[np.where(sefd_rms > 3*np.std(sefd_rms))] = 0.0

    # Saving SEFD
    sefdname = sefdnames[closest_mjd_fct(driftmjds, mjd)]
    #print("Saving SEFD", data_path + sefdname)
    np.save(data_path + sefdname, sefd_rms)

    # Computing fluence
    ## boxcar SNR-width
    fluence, fluence_err, fluence_freq = fluence_boxcar(snr, width,
            sefd_rms, BW, NPOL)

    ## Integral
    np_file = glob.glob(data_path + filename + '*')[0]
    np_data = np.load(np_file)
    np_data = np_data - np.mean(np_data)
    peak = np.argmax(np_data.mean(0))
    #waterfall = np_data[:,peak-128:peak+128]
    imin = np_data.shape[1]//2-128
    imax = np_data.shape[1]//2+128
    waterfall = np_data[:,imin:imax]
    pulse= np.mean(waterfall, axis=0)

    # fint, fint_err, fint_freq = fluence_integral(waterfall, sefd_rms,
    #         t_res=t_res, f_res=f_res, NPOL=NPOL)
    fint, fint_err = fluence_integral(waterfall, sefd_rms,
            t_res=t_res, f_res=f_res, NPOL=NPOL)

    # Computing flux
    flux, flux_err = peak_flux(snr, width, sefd_rms, BW, NPOL)

    # Writing data to file
    print("{} {:.6f} \t{:.2f} \t{:.2f} \t{:.2f} \t{:.2f} \t{:.2f} \t{:.2f}".format(
            burst_id, mjd, fluence, fluence_err, fint, fint_err, flux, flux_err))
    fout.write("{:.6f} \t{:.2f} \t{:.2f} \t{:.2f} \t{:.2f} " \
            "\t{:.2f} \t{:.2f} \t{:.2f} \t{:.2f}\n".format(mjd,
            width, snr, fluence, fluence_err, fint, fint_err, flux, flux_err))

    # Plotting
    # gss = gridspec.GridSpecFromSubplotSpec(2, 1,
    #         subplot_spec=gs[i//ncols,i%ncols], hspace=0, wspace=0,
    #         height_ratios=[1,1])
    # ax1 = fig.add_subplot(gss[0,0])
    # ax1.plot(pulse)
    # #ax1.axvspan(peak-128,peak+128, alpha=0.3)
    # ax1.text(0.05, 0.95, burst_id, horizontalalignment='left',
    #         verticalalignment='top', transform=ax1.transAxes)
    # ax2 = fig.add_subplot(gss[1,0])
    # ax2.plot(fint_freq, color='C1')
    #ax2.hlines(3*np.std(sefd_rms), 0, 1536)
    #ax2.set_yscale('log')
    # plt.show()

fout.close()
print('Fluence written to ', fluence_fname)

#plt.show()
