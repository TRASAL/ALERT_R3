from __future__ import print_function
from __future__ import division
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import astropy.units as u
from astropy.time import Time
from pypulsar.formats import filterbank
import datetime
import argparse
import tools
import pol
import triggers
from plot_r3_iquv import get_i
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
def fil_snr_width(ff, sb_generator, t0=5.0):
    rawdatafile = filterbank.filterbank(ff)
    header = rawdatafile.header
    nchans = header['nchans']
    ntime_plot = 2048
    dm=348.75
    CB='00'

    f = ff.replace('{}.fil'.format(CB), '')

    # Getting pulse peak
    full_dm_arr, full_freq_arr, time_res, params = triggers.proc_trigger(
            f, dm, t0, -1,
            ndm=32, mk_plot=False, downsamp=1,
            beamno=CB, fn_mask=None, nfreq_plot=nchans,
            ntime_plot=ntime_plot,
            cmap='viridis', cand_no=1, multiproc=False,
            rficlean=True, snr_comparison=-1,
            outdir='./data', sig_thresh_local=0.0,
            subtract_zerodm=False,
            threshold_time=3.25, threshold_frequency=2.75,
            bin_size=32, n_iter_time=3,
            n_iter_frequency=5,
            clean_type='perchannel', freq=1370,
            sb_generator=sb_generator, sb=35)

    # SNR width
    D = full_freq_arr
    #D[np.isnan(D)] = np.nanmean(D)
    pulse = np.nanmean(D, axis=0)

    (snr, width) = SNRtools.calc_snr_matchedfilter(pulse, widths=range(500))
    return snr, width

def closest_mjd_fct(list, mjd):
    diff = np.array([abs(l-mjd) for l in list])
    i = np.argmin(diff)
    return i

def peak_flux(snr, width, sefd_rms, BW=3e8, NPOL=2):
    flux_freq = sefd_rms / np.sqrt(NPOL * BW * width*1e-3) * snr
    flux = np.mean(flux_freq)
    return flux, 0.2*flux

def fluence_boxcar(snr, width, sefd_rms, BW=3e8, NPOL=2):
    fluence_freq = sefd_rms / np.sqrt(NPOL * BW * width*1e-3) * snr * width
    fluence = np.median(fluence_freq)
    return fluence, 0.2*fluence, fluence_freq

def fluence_integral(data, sefd_rms,
        t_res=0.8192*1e-4, f_res=0.1953125*1e6, NPOL=2):
    pulse = np.nanmean(data, axis=0)
    #pulse -= np.median(pulse)
    #print(np.sum(pulse), t_res*1e3*pulse.shape[0])
    fluence_t = np.median(sefd_rms)/np.sqrt(NPOL*f_res*t_res) * pulse*t_res*1e3
    fluence = np.sum(fluence_t)
    return fluence, 0.2*fluence #, fluence_freq

if __name__=='__main__':
    # File paths
    data_path = '/home/arts/pastor/R3/fluxcal/i_npy/'
    drift_path = '/tank/data/FRBs/R3/driftscan/'
    driftnames = [
            # '2020-03-23-23:35:00.3C286drift00/CB00_00_downsamp10000_dt0.819.npy',
            # '2020-03-18-18:45:00.3C147drift00/CB00_00_downsamp10000_dt0.819.npy',
            # '2020-05-07-19:40:00.3C286drift00/CB00_00_downsamp10000_dt0.819.npy',
            # #'2020-05-15-18:30:00.3C286drift00/CB00_00_downsamp10000_dt0.819.npy',
            # '2020-05-21-19:55:00.3C286drift00/CB00_00_downsamp10000_dt0.819.npy',
            # '2020-05-26-19:00:00.3C286drift00/CB00_00_downsamp10000_dt0.819.npy',
            '2020-08-27-04:45:00.3C147drift00/CB00_00_downsamp10000_dt0.819.npy']
    sefdnames = [
        # '20200323_3C286',
        # '20200318_3C147',
        # '20200507_3C286',
        # '20200521_3C286',
        # '20200526_3C286',
        '20200827_3C147'
        ]
    driftmjds = []
    for i,d in enumerate(driftnames):
        date = d.split('.')[0]
        t = 'T'.join(date.rsplit('-', 1))
        driftmjds.append(Time(t, format='isot', scale='utc').mjd)

    # Defining filenames
    #snr_fname = '/home/arts/pastor/R3/fluxcal/snr_width_all.txt'
    #snr_width = np.genfromtxt(snr_fname, dtype=('<U30', 'float', 'int'), names=True)
    fname = '/home/arts/pastor/R3/arts_r3_properties.csv'
    burst_data = pd.read_csv(fname)
    fluence_fname = '/home/arts/pastor/R3/fluxcal/fluence_int.txt'

    # Computing fluence
    fout = open(fluence_fname, 'w')
    fout.write('# MJD width_ms snr fluence_Jyms fluence_err fint_Jyms fint_err ' \
            'flux_Jy flux_err\n')
    #print('# MJD \t\t width_ms \tSNR \tfluence_Jyms\n')

    sb_generator = triggers.SBGenerator.from_science_case(science_case=4)
    sb_generator.reversed = True

    # Plotting
    nrows,ncols = 7,8
    fig = plt.figure(0, figsize=(21,29))
    gs = gridspec.GridSpec(nrows,ncols, hspace=0.05, wspace=0.05)

    #for i,b in enumerate(snr_width):
    for i,burst in burst_data.iterrows():
        burst_id = burst['paper_name'] #'A{:02}'.format(i+1)
        # snr = b['SNR_boxcar']
        # bwidth = b['boxcar_width_samples']
        filename = burst['file_location']
        t0 = float(burst['t_peak'])
        # if burst_id == "A26":
        #     print(t0)

        #b['filename'].replace('.npy:', '.npy')
        snr, bwidth = fil_snr_width(filename, sb_generator, t0=t0)
        width = bwidth * t_res *1e3 # ms
        mjd = burst['detection_mjd']
        #float(filename.split('_')[1].replace('mjd', ''))


        closest_driftscan = driftnames[closest_mjd_fct(driftmjds, mjd)]
        driftname = drift_path + closest_driftscan
        src = closest_driftscan.split('/')[0].split('.')[1].replace(
                'drift00', '')

        # Calibration
        driftscan = np.load(driftname)

        driftscan[driftscan!=driftscan] = 0.
        nt = driftscan.shape[-1]
        data = driftscan.reshape(-1, 2, nt).mean(1)
        nfreq = driftscan.shape[0]

        # Calibrating
        CalTools = CalibrationTools(t_res=t_cal, Ndish=Ndish, IAB=IAB,
                nfreq=nfreq)
        tsys_rms = CalTools.tsys_rms_allfreq(driftscan, off_samp=(0, 200),
                src=src)
        sefd_rms = CalTools.tsys_to_sefd(tsys_rms)
        #sefd_rms[np.where(sefd_rms > 3*np.std(sefd_rms))] = 0.0

        # Saving SEFD
        sefdname = sefdnames[closest_mjd_fct(driftmjds, mjd)]
        #print("Saving SEFD", data_path + sefdname)
        #np.save(data_path + sefdname, sefd_rms)

        # Computing fluence
        ## boxcar SNR-width
        fluence, fluence_err, fluence_freq = fluence_boxcar(snr, width,
                sefd_rms, BW, NPOL)

        ## Integral
        # np_file = glob.glob(data_path + filename + '*')[0]
        # np_data = np.load(np_file)
        # np_data = np_data - np.mean(np_data)
        # peak = np.argmax(np_data.mean(0))
        # #waterfall = np_data[:,peak-128:peak+128]
        # imin = np_data.shape[1]//2-128
        # imax = np_data.shape[1]//2+128
        # waterfall = np_data[:,imin:imax]
        # pulse= np.mean(waterfall, axis=0)

        if i>35:
            rficlean = True
        else:
            rficlean = False

        waterfall, tval, t0 = get_i(filename, t0, sb_generator, nsamp=512,
                bscrunch=1, fscrunch=1, dm=348.75, rficlean=rficlean, snr=False)
        pulse= np.nanmean(waterfall, axis=0)
        # baseline = np.median(pulse[:128])
        # waterfall -= baseline
        # pulse -= baseline

        # fig2 = plt.figure(1, figsize=(13,9))
        # gsub = gridspec.GridSpec(2,1, hspace=0., wspace=0.)
        #
        # ax1 = fig2.add_subplot(gsub[0,0])
        # ax1.plot(pulse, 'k')
        #
        # ax2 = fig2.add_subplot(gsub[1,0], sharex=ax1)
        # ax2.imshow(waterfall, interpolation='nearest', aspect='auto')
        # ax2.set_xlabel('Time (ms)')
        # ax2.set_ylabel('Frequency (MHz)')

        waterfall = waterfall[:, 128:384]
        pulse = pulse[128:384]

        fint, fint_err = fluence_integral(waterfall, sefd_rms,
                t_res=t_res, f_res=f_res, NPOL=NPOL)

        # Computing flux
        flux, flux_err = peak_flux(snr, width, sefd_rms, BW, NPOL)

        # Writing data to file
        print("{} {:.6f} {:.1f} \t{:.1f} \t{:.2f} \t{:.2f} \t{:.2f} \t{:.2f} \t{:.2f} \t{:.2f}".format(
                burst_id, mjd, snr, width, fluence, fluence_err, fint, fint_err, flux, flux_err))
        fout.write("{:.6f} \t{:.2f} \t{:.2f} \t{:.2f} \t{:.2f} " \
                "\t{:.2f} \t{:.2f} \t{:.2f} \t{:.2f}\n".format(mjd,
                width, snr, fluence, fluence_err, fint, fint_err, flux, flux_err))

        # Plotting
        gss = gridspec.GridSpecFromSubplotSpec(1, 1,
                subplot_spec=gs[i//ncols,i%ncols], hspace=0, wspace=0)
        ax1 = fig.add_subplot(gss[0,0])
        ax1.plot(pulse)
        #ax1.axvspan(peak-128,peak+128, alpha=0.3)
        ax1.text(0.05, 0.95, burst_id, horizontalalignment='left',
                verticalalignment='top', transform=ax1.transAxes)
        # ax2 = fig.add_subplot(gss[1,0])
        # ax2.plot(fint_freq, color='C1')
        # ax2.hlines(3*np.std(sefd_rms), 0, 1536)
        # ax2.set_yscale('log')
        #plt.show()

    fout.close()
    print('Fluence written to ', fluence_fname)

    plt.show()
