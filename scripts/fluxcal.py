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
src='3C286'
NPOL = 2
BW = 3e8 # Hz
# width = 1 # ms
# SNR = 25

# Functions
def closest_mjd_fct(list, mjd):
    diff = np.array([abs(l-mjd) for l in list])
    i = np.argmin(diff)
    return i

def fluence_boxcar(snr, width, sefd_rms, BW=3e8, NPOL=2):
    fluence_freq = sefd_rms / np.sqrt(NPOL * BW * width*1e-3) * snr * width
    fluence = np.median(fluence_freq)
    return fluence

def fluence_integral(data, sefd_rms,
        t_res=0.8192*1e-4, f_res=0.1953125*1e6, NPOL=2):
    # sefd_rms_all = np.transpose(np.tile(sefd_rms,
    #         (data.shape[1],1))/sefd_rms.shape)
    sefd_rms_all = np.transpose(np.tile(sefd_rms, (data.shape[1],1)))
    fluence_tf = sefd_rms_all / np.sqrt(NPOL * f_res * t_res) * data
    # fluence_freq = np.sum(fluence_tf, axis=1)
    #fluence_freq = np.mean(fluence_tf, axis=1)
    fluence = np.mean(fluence_tf)
    return fluence

# File paths
#data_path = '/tank/data/FRBs/R3/'
data_path = '/home/arts/pastor/scripts/arts-analysis/iquv_npy/'
drift_path = '/tank/data/FRBs/R3/driftscan/'
driftnames = [
        '2020-03-23-23:35:00.3C286drift00/CB00_00_downsamp10000_dt0.819.npy',
        '2020-05-07-19:40:00.3C286drift00/CB00_00_downsamp10000_dt0.819.npy',
        #'2020-05-15-18:30:00.3C286drift00/CB00_00_downsamp10000_dt0.819.npy',
        '2020-05-21-19:55:00.3C286drift00/CB00_00_downsamp10000_dt0.819.npy',
        '2020-05-26-19:00:00.3C286drift00/CB00_00_downsamp10000_dt0.819.npy']
driftmjds = []
for i,d in enumerate(driftnames):
    date = d.split('.')[0]
    t = 'T'.join(date.rsplit('-', 1))
    driftmjds.append(Time(t, format='isot', scale='utc').mjd)

snr_fname = '/home/arts/pastor/R3/fluxcal/snr_width_all.txt'
snr_width = np.genfromtxt(snr_fname, dtype=('<U30', 'float', 'int'), names=True)
# With boxcar width and SNR
#fluence_fname = '/home/arts/pastor/R3/fluxcal/fluence.txt'
# With integral if SNR >= 15
fluence_fname = '/home/arts/pastor/R3/fluxcal/fluence_int.txt'

# Computing fluence
fout = open(fluence_fname, 'w')
#fout.write('# MJD \t\t width_ms \t fluence_Jyms\n')
print('# MJD \t\t width_ms \tSNR \tfluence_Jyms\n')
for i,b in enumerate(snr_width):
    snr = b['SNR_boxcar']
    bwidth = b['boxcar_width_samples']
    width = bwidth * t_res *1e3 # s
    filename = b['filename'].replace('.npy:', '.npy')
    mjd = float(filename.split('_')[1].replace('mjd', ''))

    closest_driftscan = driftnames[closest_mjd_fct(driftmjds, mjd)]
    driftname = drift_path + closest_driftscan

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

    fluence = fluence_boxcar(snr, width, sefd_rms, BW, NPOL)
    #fout.write("{:.6f} \t {:.2f} \t\t {:.2f}\n".format(mjd, width, fluence))

    if snr >= 15:
        np_file = glob.glob(data_path + filename + '*')[0]
        np_data = np.load(np_file)
        peak = np.argmax(np_data.mean(0))
        waterfall = np_data[:,peak-2*bwidth:peak+2*bwidth]
        fint = fluence_integral(np_data, sefd_rms,
                t_res=t_res, f_res=f_res, NPOL=NPOL)
        print("{:.6f} \t{:.2f} \t{:.2f} \t{:.2f} \t{:.2f}".format(mjd,
                width, snr, fluence, fint))
        # fig = plt.figure()
        # ax = plt.subplot(1,1,1)
        # ax.plot(np_data.mean(0))
        # ax.axvspan(peak-bwidth,peak+bwidth, alpha=0.3)
        # plt.show()

        # (snr, width) = SNRtools.calc_snr_matchedfilter(np_data.mean(0), widths=range(500))
        # print("{}: {:.2f} {}".format(filename + '.npy', snr, width))
    else:
        print("{:.6f} \t{:.2f} \t{:.2f} \t{:.2f}".format(mjd,
                width, snr, fluence))

fout.close()
print('Fluence written to ', fluence_fname)
