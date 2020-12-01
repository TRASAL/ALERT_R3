from __future__ import print_function
from __future__ import division
import numpy as np
import pandas as pd
import glob, sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator, MaxNLocator)
from matplotlib.lines import Line2D
from astropy.time import Time
import astropy.units as u
import datetime
import argparse
from pypulsar.formats import filterbank

# Local imports
import tools
import pol
import read_IQUV_dada
import reader
import triggers
from calibration_tools import run_fluxcal, CalibrationTools, Plotter
from frbpa.utils import get_phase
SNRtools = tools.SNR_Tools()

# Example usage
# python /home/arts/pastor/scripts/arts-analysis/plot_r3_iquv.py -f 1 -b 1 -w
# -n 512 -s --nfig 3 --save_npy --cmap viridis

# --------------------------------------------------------------------- #
# User defined functions
# --------------------------------------------------------------------- #

def rebin(arr, binsz):
    new_shape = [int(arr.shape[0] //  int(binsz[0])),
            int(arr.shape[1] // int(binsz[1]))]
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

def get_cycle(bursts, period, ref_mjd=58369.30):
    return ((np.array(bursts) - ref_mjd) // period)

def get_pa_iquv(d, fn_xyphase, nfreq=1536, ntime=1e4, bw=300., nsamp=512,
                bscrunch=2, fscrunch=2, id=None,
                fmin=1219.50561523, fmax=1519.50561523, fn_bandpass=None):
    """
    Getting PA values from IQUV array after callibration and bandpass correction
    """

    XYduncal,XYdcal = 0,0
    SS=0
    Dcal_spectrum=0

    freq = np.arange(fmin, fmax, bw/nfreq)
    xyphase = np.load(fn_xyphase)
    if fn_bandpass != None:
        print("Loading bandpass %s" % fn_bandpass)
        bandpass_arr = np.load(fn_bandpass)
        bandpass_arr[np.isnan(bandpass_arr)] = np.nanmean(bandpass_arr)
        bandpass_arr[np.where(bandpass_arr == 0)] = np.mean(bandpass_arr)
    else:
        print("not using bandpass")
        bandpass_arr = np.ones(nfreq)

    # Frequency mask
    f_mask = range(1532,1536)
    if id == "A09":
        f_mask.extend(range(175, 200))
    elif id == "A01":
        f_mask.extend(range(305, 330))
    snr_max, width_max = SNRtools.calc_snr_matchedfilter(d[0].mean(0),
            widths=range(250))
    #get channel weights
    nsubband_snr=192
    Isubband = d[0].reshape(nfreq//nsubband_snr, nsubband_snr, ntime).mean(1)
    width_max = max(4,width_max)
    weights_f = []

    for kk in range(Isubband.shape[0]):
        Isub_ = Isubband[kk, :ntime//width_max*width_max]
        Isub_ = Isub_.reshape(-1, width_max).mean(-1)
        Isub_ -= np.median(Isub_)
        snr_subband = Isub_[int(ntime/2./width_max)]/np.std(Isub_[:int(ntime/2./width_max)-5])
        # Create a S/N^2 weights array for full 1536 channel spectrum
        weight_kk = (max(snr_subband,0)*np.ones([nsubband_snr]))**2
        weights_f.append(weight_kk)


#    for kk in range(Isubband.shape[0]):
#        Isub_ = Isubband[kk, :ntime//width_max*width_max]
#        Isub_ = Isub_.reshape(-1, width_max).mean(-1)
#        Isub_ -= np.median(Isub_)
#        snr_subband = Isub_[int(ntime/2./width_max)] \
#                / np.std(Isub_[:int(ntime/2./width_max)-5])
#        weight_kk = (max(snr_subband,0)*np.ones([nsubband_snr]))**2
#        weights_f.append((max(snr_subband,0)*np.ones([96]))**2)

    weights_f = np.array(np.concatenate(weights_f)).flatten()
    weights_f = weights_f[:, None]
    on = d[0, :, 5000-width_max//2:5000+width_max//2].mean(1)
    off = d[0, :, :4500].mean(1)
    frb_spectrum = (on - off)/bandpass_arr[:,None]
    frb_spectrum[f_mask] = 0.0
    frb_spectrum = frb_spectrum.reshape(-1, 16).mean(-1)

    for jj in range(4):
        # d[jj] = d[jj].astype(float)
        # off = d[jj, :, :4500].mean(1)
        # d[jj] = (d[jj] - off[:, None]) #/ off[:, None]
        # d[jj] -= np.median(d[jj].mean(0)[:4000])
        # d[jj] /= np.std(d[jj].mean(0)[:4000])
        #d[jj] = np.divide(d[jj].T, bandpass_arr).T

        d[jj] = d[jj].astype(float)
        #d[jj] -= np.median(d[jj], axis=-1)[:, None]
        off = np.mean(d[jj, :, :4500], axis=1) #d[jj, :, :4500].mean(1)
        d[jj] = (d[jj] - off[:, None]) #/ off[:, None]
        d[jj] /= np.std(d[jj])
        bp_arr = np.std(d[jj][...,
                int((d[jj].shape[0]-nsamp)/2):int(d[jj].shape[0]/2-nsamp/6)],
                axis=1)
        bp_arr[np.where(bp_arr == 0)] = np.mean(bp_arr)
        d[jj] = np.divide(d[jj].T, bp_arr).T

        # d[jj] -= np.median(d[jj], axis=-1)[:, None]
        # #d[jj] = rebin(d[jj], [fscrunch,bscrunch])
        # d[jj] /= np.std(d[jj])
        # bandpass_arr = np.std(d[jj][...,:int(d[jj].shape[0]/3)], axis=1)
        # d[jj] = np.divide(d[jj].T, bandpass_arr).T
        # tmed = np.median(d[jj], axis=-1, keepdims=True)
        # d[jj] -= tmed

    # Mask
    d[np.isnan(d)] = 0.0
    d[:, f_mask, :] = 0.0

    # Burst selection
    wm2 = width_max//2
    sampmin = int(5000-nsamp/2)
    sampmax = int(5000+nsamp/2)
    D = d[:, :, sampmin:sampmax] #-d[:,:,:4500, None].mean(-2)
    SS = (d[:, :, 5000-width_max//2:5000+width_max//2].mean(-1))
    #np.save('test%d.npy' % ii, D)

    # Phase correction
    XYdata = D[2]+1j*D[3]
    XYduncal += XYdata
    XYdata *= np.exp(-1j*(xyphase))[:, None]
    XYdcal += XYdata

    D[2], D[3] = XYdata.real, XYdata.imag
    P = D[1]+1j*D[2]

    frb_spectrum_stokes = (D[:, :,
            D.shape[-1]//2-width_max//2:D.shape[-1]//2+width_max//2].mean(-1))
    freqArr_Hz = pol.freq_arr*1e6

    # TODO: replace derot_phase by single RM value
    #derot_phase = np.exp(2j*114.6*pol.lam_arr**2)[:, None] #np.load('R3_RM_phase.npy')
    #P *= derot_phase[:, None]
    P *= np.exp(2j*114.6*pol.lam_arr**2)[:, None]
    D[1],D[2] = P.real, P.imag

    Dcal_spectrum += (D[:, :,
            D.shape[-1]//2-width_max//2:D.shape[-1]//2+width_max//2].mean(-1))

#    weights_f = 1
    print(P.shape, weights_f.shape)
    PA = np.angle(np.mean(P*weights_f,0),deg=True)

    freqArr_Hz = pol.freq_arr*1e6

    # Temporal mask. index in mask allowed for PA calculation
    std_I = np.std(d[0, ..., 3900:4900].mean(0))
    t_mask = np.where(D[0].mean(0) > 4.5*std_I)[0]
    t_mask = t_mask[np.where(t_mask>50)]
    #t_mask = np.where(D[0].mean(0).reshape(-1,1).mean(1) > 2.0*std_I)[0]
    return D, PA, weights_f, t_mask

def get_i(ff, t0, sb_generator, nsamp=512, bscrunch=2, fscrunch=2, dm=348.75,
          tburst=None, rficlean=True, snr=False):

    CB = '00'

    # Reading filterbank file
    rawdatafile = filterbank.filterbank(ff)
    header = rawdatafile.header

    tstart = header['tstart']  # MJD
    if tburst is None:
        tburst = float(ff.split('/')[-1].split('_')[3].replace('t', '')) * u.s # s
    mjd = tstart + tburst.to("d").value

    nchans = header['nchans']
    nsamp0 = max(4096, nsamp*4)
    fmax = header['fch1'] #1519.50561523
    fmin = fmax + nchans * header['foff'] #1219.50561523
    nfreq_plot = int(nchans/fscrunch)
    ntime_plot = int(nsamp0/bscrunch)

    f = ff.replace('00.fil', '')

    # Getting pulse peak
    full_dm_arr, full_freq_arr, time_res, params = triggers.proc_trigger(
            f, dm, t0, -1,
            ndm=32, mk_plot=False, downsamp=1,
            beamno=CB, fn_mask=None, nfreq_plot=nchans,
            ntime_plot=ntime_plot,
            cmap='viridis', cand_no=1, multiproc=False,
            rficlean=rficlean, snr_comparison=-1,
            outdir='./data', sig_thresh_local=0.0,
            subtract_zerodm=False,
            # threshold_time=3.5, threshold_frequency=2.75,
            # bin_size=32, n_iter_time=2, n_iter_frequency=3,
            threshold_time=3.75, threshold_frequency=3.5,
            bin_size=32, n_iter_time=3, n_iter_frequency=3,
            clean_type='perchannel', freq=1370,
            sb_generator=sb_generator, sb=35, dumb_mask=False)

    full_freq_arr[np.isnan(full_freq_arr)] = 0.0
    full_freq_arr = rebin(full_freq_arr, [fscrunch,bscrunch])
    ntime_plot = int(nsamp/bscrunch)
    p = np.argmax(np.mean(full_freq_arr, axis=0))
    D = full_freq_arr[:, int(p-ntime_plot/2):int(p+ntime_plot/2)]
    if D.shape[-1]==0:
        print("Time axis is length zero")
        return D, None, None

    pulse = np.nanmean(D, axis=0)
    tsamp = time_res * 1000 # ms
    tval = np.arange(-ntime_plot/2, ntime_plot/2) * tsamp * bscrunch
    tcen = tval[np.argmax(pulse)] * u.ms # ms
    t0 += tcen.to("s").value
    D = np.flip(D, axis=0)

    # Baseline subtraction
    baseline = np.median(pulse[:nsamp//4])
    D -= baseline
    pulse -= baseline
    off = np.mean(D[:, :nsamp//4], axis=1) #d[jj, :, :4500].mean(1)
    D = (D - off[:, None])

    # # Bandpass correction
    # bp_arr = np.std(D[..., :int(ntime_plot/3)], axis=1)
    # bp_arr[np.where(bp_arr == 0)] = np.mean(bp_arr)
    # D = np.divide(D.T, bp_arr).T

    # SNR units
    if snr:
        sig = np.std(D)
        dmax = (D.copy()).max()
        dmed = np.median(D)
        N = len(D)
        D = (D - dmed)/(1.048*sig)

    return D, tval, t0

def iquv_plotter(D, PA, tval, fig, gss, ii, weights_f, t_mask, k_fluence=None,
        nsamp=512, tsamp=8.192e-5, fmin=1220, fmax=1520,
        bscrunch=2, fscrunch=2, tot=34, nsub=34, nfig=1,
        ncols=4, nrows=6, cmap='viridis', waterfall=False,
        stokes='iquv', id=None, cycle=False, cycle_color=None):
    """
    Plotting IQUV data
    """

    #colors = ['#577590', '#90be6d', '#f8961e', '#f94144']
    #colors = ['#080808', '#525252', '#858585', '#C2C2C2']

    ax1 = fig.add_subplot(gss[1, 0])
    nfreq, ntime = D.shape[1]//fscrunch, D.shape[2]//bscrunch
    iquv = [rebin(d, [1,bscrunch]).mean(0) for d in D]
    if k_fluence is not None:
        for kk in range(len(iquv)):
            iquv[kk] = iquv[kk] * k_fluence
    if stokes == 'iquv':
        colors = ['k', '#482677FF', '#238A8DDF', '#95D840FF']
        ax1.plot(tval, iquv[0], color=colors[0], label='I', zorder=10)
        ax1.plot(tval, iquv[1], color=colors[1], label='Q', zorder=9)
        ax1.plot(tval, iquv[2], color=colors[2], label='U', zorder=8)
        ax1.plot(tval, iquv[3], color=colors[3], label='V', zorder=7)
    elif stokes == 'ilv':
        colors = ['k', '#33638DFF', '#95D840FF']
        L = np.sqrt(iquv[1]**2 + iquv[2]**2)
        ax1.plot(tval, iquv[0], color=colors[0], label='I', zorder=10)
        ax1.plot(tval, L, color=colors[1], ls='-', label='L', zorder=9)
        ax1.plot(tval, iquv[3], color=colors[2],  label='V', zorder=7)

    ax1.yaxis.set_major_locator(MaxNLocator(nbins=2, integer=True))
    ax1.set_ylim(min(iquv[0]*1.1), max(iquv[0]*1.2))
    if id is not None:
        ax1.text(0.05, 0.95, id, horizontalalignment='left',
                verticalalignment='top', transform=ax1.transAxes)

    ax2 = fig.add_subplot(gss[0, 0], sharex=ax1)
    ax2.plot((t_mask - nsamp/2) * tsamp, PA[t_mask], '.', color='k', alpha=0.5, label='PA')
    # ax2.yaxis.set_major_locator(MultipleLocator(90))
#    ax2.set_ylim(-190,190)
    ax2.yaxis.set_major_locator(MultipleLocator(30))
    ax2.set_ylim(-50,50)
    ax2.grid(b=True, axis='y', color='k', alpha=0.1)

    if waterfall:
        # TODO: reverse frequencies
        waterfall = rebin(D[0], [fscrunch,bscrunch])
        vmin = np.percentile(waterfall, 1.0)
        vmax = np.percentile(waterfall, 99.5)
        ax4 = fig.add_subplot(gss[2,0], sharex=ax1)
        ax4.imshow(waterfall, interpolation='nearest', aspect='auto',
                origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                extent=[tval[0], tval[-1], fmin, fmax])

        if (ii >= nsub - ncols):
            ax4.set_xlabel('Time (ms)')
        else:
            ax4.set_xticklabels([])
        if (ii%ncols == 0):
            if k_fluence is not None:
                ax1.set_ylabel('Flux\n(mJy)')
            ax2.set_ylabel('PA\n(deg)')
            ax4.set_ylabel('Frequency\n(MHz)')
            for ax in (ax1,ax2, ax4):
                ax.yaxis.set_label_coords(-0.15, 0.5)
        else:
            ax2.set_yticklabels([])
            ax4.set_yticklabels([])
        axes = (ax1,ax2,ax4)
    else:
        if (ii >= nsub - ncols):
            ax1.set_xlabel('Time (ms)')
        else:
            ax1.set_xticklabels([])
        axes = (ax1,ax2)

    for ax in axes:
        ax.label_outer()

    if cycle is not None:
        if cycle_color is None:
            cycle_color='w'
        ax1.text(0.95, 0.95, "C{:.0f}".format(cycle),
                horizontalalignment='right',
                verticalalignment='top', transform=ax1.transAxes,
                color=cycle_color, fontweight='bold')

def snippet_plotter(D, tval, fig, gss, ii, t0=None, tot=34, nsub=34,
        nfig=1, ncols=4, nrows=6, k_fluence=None, fmin=1220, fmax=1520,
        cmap='viridis', waterfall=False, id=None, cycle=None, cycle_color=None):
    """
    Plotting I data
    """

    pulse = np.nanmean(D, axis=0)
    if k_fluence is not None:
        pulse = pulse * k_fluence
    colors = ['#080808', '#525252', '#858585', '#C2C2C2']

    ax1 = fig.add_subplot(gss[1,0])
    ax1.plot(tval, pulse, color='k')
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=2, integer=True))
    ax1.set_ylim(min(pulse*1.1), max(pulse*1.2))
    if id is not None:
        ax1.text(0.05, 0.95, id, horizontalalignment='left',
                verticalalignment='top', transform=ax1.transAxes)

    if waterfall:
        #TODO: reverse frequencies
        ax4 = fig.add_subplot(gss[2,0], sharex=ax1)
        vmin = np.percentile(D, 1.0)
        vmax = np.percentile(D, 99.5)
        ax4.imshow(D, interpolation='nearest', aspect='auto', origin='lower',
                cmap=cmap, vmin=vmin, vmax=vmax,
                extent=[tval[0], tval[-1], fmin, fmax])
        axes = [ax1,ax4]

        if (ii >= nsub - ncols):
            ax4.set_xlabel('Time (ms)')
        else:
            ax4.set_xticklabels([])
        if (ii%ncols == 0):
            if k_fluence is not None:
                ax1.set_ylabel('Flux\n(mJy)')
            ax4.set_ylabel('Frequency\n(MHz)')
            for ax in (ax1, ax4):
                ax.yaxis.set_label_coords(-0.15, 0.5)
        else:
            ax4.set_yticklabels([])
    else:
        if (ii >= nsub - ncols):
            ax1.set_xlabel('Time (ms)')
        else:
            ax1.set_xticklabels([])
        axes = [ax1]

    for ax in axes:
        ax.label_outer()

    if cycle is not None:
        if cycle_color is None:
            cycle_color='w'
        ax1.text(0.95, 0.95, "C{:.0f}".format(cycle),
                horizontalalignment='right',
                verticalalignment='top', transform=ax1.transAxes,
                color=cycle_color, fontweight='bold')

# --------------------------------------------------------------------- #
# Input parameters
# --------------------------------------------------------------------- #

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Commands for plotting IQUV data')
    parser.add_argument('-f','--fscrunch', type=int, default=2,
    		help='Factor to scrunch the number of channels')
    parser.add_argument('-b','--bscrunch', type=int, default=2,
    		help='Factor to scrunch the number of time bins')
    parser.add_argument('-n','--nsamp',type=int, default=256,
            help='Number of time samples to waterfall')
    parser.add_argument('-d','--dm',type=float, default=348.75,
            help='DM to dedisperse to')
    parser.add_argument('--nfig',type=int, default=1,
            help='Number of figures to save')
    parser.add_argument('-k','--stokes',type=str, default='ilv',
            help='Stokes to plot (iquv|ilv)')
    parser.add_argument('-c','--cmap',type=str, default='viridis',
            help='Colormap of the waterfall plots')
    parser.add_argument('-w','--waterfall', action='store_true',
            default=False, help='Create waterfall plot')
    parser.add_argument('-s','--show', action='store_true',
            default=False, help='Show waterfall plot')
    parser.add_argument('-norfi','--norfi', action='store_true',
            default=False, help='Do not rfi clean .fil data')
    parser.add_argument('--save_npy', action='store_true',
            default=False,
            help='Save numpy arrays of the plotted dynamic spectra')
    parser.add_argument('--cycle', action='store_true',
            default=False,
            help='Plot a line between bursts from different activity cycles')

    args = parser.parse_args()

    # Parameters:
    nsamp = args.nsamp
    fscrunch = args.fscrunch
    bscrunch = args.bscrunch
    stokes = args.stokes

    # Defining numpy files to plot
    datadir = '/tank/data/FRBs/R3/'
    # fname = '/home/arts/pastor/R3/plots/filenames.txt'
    # file_info  = open(fname, "r")
    # lines = file_info.readlines()
    #
    # fl, fn_xyphase, fn_bandpass, mjds, t0s = [], [], [], [], []
    # for line in lines:
    #     if '#' not in line:
    #         cols = line.split(' ')
    #         mjds.append(float(cols[0]))
    #         fl.append(cols[1])
    #         if cols[1][-4:] == '.npy':
    #             fn_xyphase.append(cols[2])
    #             fn_bandpass.append(cols[3].replace('\n', ''))
    #             t0s.append(None)
    #         elif cols[1][-4:] == '.fil':
    #             fn_xyphase.append(None)
    #             fn_bandpass.append(None)
    #             t0s.append(float(cols[2]))
    # file_info.close()
    # tot = len(fl)
    fname = '/home/arts/pastor/R3/arts_r3_properties.csv'
    burst_data = pd.read_csv(fname)
    tot = len(burst_data.index)
    mjds = np.array(burst_data['detection_mjd'])
    print("PLOTTING", tot, "BURSTS")

    # Example dada header:rebin
    fndada = '/tank/data/FRBs/R3/20200322/iquv/CB00/dada/2020-03-22-10:03:39_0004130611200000.000000.dada'
    header = read_IQUV_dada.read_dada_header(fndada)
    bw = float(header['bw'])
    fmin = header['freq'] - header['bw'] / 2
    fmax = header['freq'] + header['bw'] / 2
    tsamp = header['tsamp'] * 1000. # ms

    sb_generator = triggers.SBGenerator.from_science_case(science_case=4)
    sb_generator.reversed = True

    # Output file
    PA_arr = []

    if args.nfig == 1:
        ncols, nrows = 7,8
        plt_out = ['./R3_IQUV_PA.pdf']
    elif args.nfig == 2:
        ncols, nrows = 5,6
        plt_out = ['./R3_IQUV_PA_{}.pdf'.format(i) for i in range(args.nfig)]
    else :
        args.nfig = 3
        ncols, nrows = 4,5
        plt_out = ['./R3_IQUV_PA_{}.pdf'.format(i) for i in range(args.nfig)]

    # Fluence
    t_res = 0.8192*1e-4 # s
    f_res = 0.1953125 * 1e6 # Hz
    t_cal = t_res*1e4 # s
    nfreq=1536
    Ndish = 10.0
    IAB = True
    NPOL = 2
    BW = 3e8 # Hz
    src = '3C147'
    driftscan = np.load('/tank/data/FRBs/R3/driftscan/'\
            + '2020-08-27-04:45:00.3C147drift00/'\
            + 'CB00_00_downsamp10000_dt0.819.npy')

    CalTools = CalibrationTools(t_res=t_cal, Ndish=Ndish, IAB=IAB, nfreq=nfreq)
    tsys_rms = CalTools.tsys_rms_allfreq(driftscan, off_samp=(0, 200), src=src)
    sefd_rms = CalTools.tsys_to_sefd(tsys_rms)

    k_fluence = np.median(sefd_rms)/np.sqrt(NPOL*f_res*t_res) # mJy
    PA_arr=[]
    phase_arr=[]

    # Defining cycle colors
    if args.cycle:
        cycles = np.unique(get_cycle(mjds, 16.29, ref_mjd=58369.90))
        print("CYCLES", cycles)
        cycle_colors = ['#577590', '#90be6d', '#f9c74f', '#f8961e', '#f94144']


    # ----------------------------------------------------------------- #
    # Starting loop on bursts
    # ----------------------------------------------------------------- #
    #for ii,ff in enumerate(fl[:]):
    for ii,burst in burst_data.iterrows():

        # burst_id = 'A{:02}'.format(ii+1)
        # mjd = mjds[ii]

        burst_id = burst['paper_name']
        mjd = burst['detection_mjd']
        ffil = burst['file_location']
        fnpy = burst['iquv']
        fn_xyphase = burst['xyphase']
        fn_bandpass = burst['bandpass']
        t0 = burst['t_peak']

        print(fnpy, type(fnpy), pd.isnull(fnpy))

        # Distinguising burst cycles
        #cycle = False
        cycle = None
        cycle_color = None
        if args.cycle:
            if (ii == 0) or (mjd - mjds[ii-1] > 3):
                cycle = get_cycle(mjd, 16.29, ref_mjd=58369.90)
                col = np.where(cycles == cycle)[0]
                cycle_color = cycle_colors[int(col)]

        # Plotting
        # Defining subplot number within figure
        if args.nfig == 1:
            jj, fign, nsub = ii, 0, tot
        elif args.nfig == 2:
            if ii < tot//2:
                jj, fign, nsub = ii, 0, ncols*nrows-1
            else:
                jj, fign, nsub = ii - tot//2, 1, tot - (ncols*nrows-1)
        else:
            if ii < ncols*nrows-1:
                jj, fign, nsub = ii, 0, ncols*nrows-1
            elif ii < 2 * (ncols*nrows-1):
                jj, fign, nsub = ii - (ncols*nrows-1), 1, ncols*nrows-1
            else:
                jj, fign, nsub = ii - 2*(ncols*nrows-1), 2, tot - 2*(ncols*nrows-1)

        if jj == 0:
            fig = plt.figure(fign, figsize=(21,27))
            plt.rcParams.update({
                    'font.size': 18,
                    'font.family': 'serif',
                    'axes.labelsize': 14,
                    'axes.titlesize': 18,
                    'xtick.labelsize': 12,
                    'ytick.labelsize': 12,
                    'xtick.direction': 'in',
                    'ytick.direction': 'in',
                    'xtick.top': True,
                    'ytick.right': True,
                    'lines.linewidth': 0.5,
                    'lines.markersize': 5,
                    'legend.fontsize': 18,
                    'legend.borderaxespad': 0,
                    'legend.frameon': False,
                    'legend.loc': 'lower right'})

            gs = gridspec.GridSpec(nrows,ncols, hspace=0.1, wspace=0.1)

        print(burst_id, jj, jj//ncols, jj%ncols, nsub)

        if args.waterfall:
            gss = gridspec.GridSpecFromSubplotSpec(3, 1,
                    subplot_spec=gs[jj//ncols,jj%ncols], hspace=0, wspace=0,
                    height_ratios=[1.5,2,4])
        else:
            gss = gridspec.GridSpecFromSubplotSpec(2, 1,
                    subplot_spec=gs[jj//ncols,jj%ncols], hspace=0, wspace=0,
                    height_ratios=[1,3])

        #if ff[-4:] == '.npy':
        if not pd.isnull(fnpy):
            d = np.load(fnpy) #np.load(ff)
            nfreq,ntime = d.shape[1],d.shape[2]
            ts = nsamp/(2*bscrunch)
            tval = np.arange(-ts, ts) * tsamp * bscrunch

            D, PA, wf, tm = get_pa_iquv(d, fn_xyphase,
                    fn_bandpass=fn_bandpass, nsamp=nsamp,
                    bscrunch=bscrunch, fscrunch=fscrunch, id=burst_id,
                    nfreq=nfreq, ntime=ntime, bw=bw, fmin=fmin, fmax=fmax)
            iquv_plotter(D, PA, tval, fig, gss, jj, wf, tm, k_fluence=k_fluence,
                    nsamp=nsamp, bscrunch=bscrunch, tsamp=tsamp, fmin=fmin,
                    fmax=fmax, fscrunch=fscrunch, tot=tot, nsub=nsub,
                    nfig=args.nfig, ncols=ncols, nrows=nrows, cmap=args.cmap,
                    waterfall=args.waterfall, stokes=stokes, id=burst_id,
                    cycle=cycle, cycle_color=cycle_color)

            # Liam
            aphase = get_phase(mjd, 16.29, ref_mjd=58369.90)
            std_I = np.std(D[0,...,:4500].mean(0))
            sig_arr_time = D[0].mean(0)/std_I
            PA_weighted = (PA[tm]*sig_arr_time[tm]**2/(sig_arr_time[tm]**2).mean()).mean()
#            PAerr = 180./np.pi*(P.mean(0).real**2*sigU**2 + P.mean(0).imag**2*sigQ**2)**0.5 / (2*np.abs(P.mean(0))**2)
            PA_arr.append(PA_weighted)
#            PA_arr_err.append(np.mean(PAerr[t_mask])/np.sqrt(float(len(t_mask))))
            phase_arr.append(aphase)

            if args.save_npy:
                fnout = 'R3_mjd{:.6f}_dedisp348.8'.format(mjd)
                # np_out = '/home/arts/pastor/scripts/arts-analysis/iquv_npy/' \
                #         + fnout
                np_out = '/home/arts/pastor/R3/fluxcal/i_npy/' \
                        + fnout
                print('saving', np_out)
                np.save(np_out, D[0])

            pulse = rebin(D[0], [1,bscrunch]).mean(0)
            (snr, width) = SNRtools.calc_snr_matchedfilter(pulse, widths=range(500))

        else:
            if burst_id in ["A36", "A37", "A38", "A39", "A42", "A43", "A45",
                    "A46", "A47", "A48", "A49", "A50", "A51", "A52"]:
                D, tval, t0 = get_i(ffil, t0, sb_generator, nsamp=nsamp,
                        bscrunch=bscrunch, fscrunch=fscrunch, dm=args.dm,
                        rficlean=True)
            else:
                D, tval, t0 = get_i(ffil, t0, sb_generator, nsamp=nsamp,
                        bscrunch=bscrunch, fscrunch=fscrunch, dm=args.dm,
                        rficlean=not(args.norfi))
            if tval is None:
                continue
            snippet_plotter(D, tval, fig, gss, jj, t0, k_fluence=k_fluence,
                    tot=tot, nfig=args.nfig, fmin=fmin, fmax=fmax,
                    nsub=nsub, ncols=ncols, nrows=nrows, cmap=args.cmap,
                    waterfall=args.waterfall, id=burst_id, cycle=cycle,
                    cycle_color=cycle_color)
            if args.save_npy:
                fnout = 'R3_mjd{:.6f}_dedisp348.8'.format(mjd)
                np_out = '/home/arts/pastor/R3/fluxcal/i_npy/' \
                        + fnout
                print('saving', np_out)
                np.save(np_out, D)
            print(mjd, 'Ifil', np.mean(D), np.median(D), np.std(D))

        # Adding legend
        if jj == nsub-1:
            ll = jj+1
            ax = fig.add_subplot(gs[nrows-1,ncols-1])
            ax.axis('off')
            if stokes == 'iquv':
                colors = ['k', '#482677FF', '#238A8DDF', '#95D840FF']
                lines = [Line2D([0], [0], color=c, linewidth=1, linestyle='-')
                        for c in colors]
                lines.append(plt.plot([], '.', color='k', alpha=0.5)[0])
                labels = ['I', 'Q', 'U', 'V', 'PA']
            elif stokes == 'ilv':
                colors = ['k', '#33638DFF', '#95D840FF']
                lines = [Line2D([0], [0], color=c, linewidth=1, linestyle='-')
                        for c in colors]
                lines.append(plt.plot([], '.', color='k', alpha=0.5)[0])
                labels = ['I', 'L', 'V', 'PA']
            plt.legend(lines, labels)

            # Saving plot
            print('Saving plot to ', plt_out[fign])
            fig.savefig(plt_out[fign], pad_inches=0, bbox_inches='tight')


#    plt.figure()
#    plt.plot(phase_arr, PA_arr,'.')
#    np.save('PAs.npy',PA_arr)

    if args.show:
        plt.show()
