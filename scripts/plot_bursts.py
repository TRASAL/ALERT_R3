#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
import os
import scipy
import math
import argparse
import psrchive
import numpy as np
import matplotlib as mpl
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import (AutoMinorLocator, MaxNLocator, MultipleLocator)
from astropy.time import Time

from get_calib_data import get_data

# --------------------------------------------------------------------------- #
# Plotting function
# --------------------------------------------------------------------------- #

def waterplotter(pulses, spectra, waterfalls, mjds, times, frequencies,
        files=None, snrs=None, site='LOFAR', outfile='./waterfall.pdf',
        ncols=3, cmap='magma'):

    # Plotting
    cmap = mpl.cm.get_cmap(cmap)
    nrows = int(np.ceil(len(args.files)/ncols))
    fig = plt.figure(figsize=(15,15))
    plt.style.use('/home/arts/ines/scripts/arts-analysis/paper.mplstyle')
    plt.rcParams.update({
            'xtick.minor.visible': False,
            'ytick.minor.visible': False})
    gs = gridspec.GridSpec(nrows,ncols, hspace=0.1, wspace=0.1)

    for ii,ff in enumerate(waterfalls):
        pulse = pulses[ii]
        spectrum = spectra[ii]
        waterfall = waterfalls[ii]
        mjd = mjds[ii]
        t = times[ii]
        freqs = frequencies[ii]
        if snrs is not None: snr = snrs[ii]
        else: snr = ''

        if spectrum is not None:
            gss = gridspec.GridSpecFromSubplotSpec(2, 2,
                    subplot_spec=gs[ii//ncols,ii%ncols], hspace=0, wspace=0,
                    height_ratios=[1,3], width_ratios=[3,1])
        else:
            gss = gridspec.GridSpecFromSubplotSpec(2, 1,
                    subplot_spec=gs[ii//ncols,ii%ncols], hspace=0, wspace=0,
                    height_ratios=[1,3])

        # Pulse profile
        ax1 = fig.add_subplot(gss[0, 0])
        if len(pulse.shape) == 1:
            ax1.plot(t, pulse, 'k', lw=1)
            ax1.set_ylim(min(pulse*1.05), max(pulse*1.2))
        else:
            for jj,pp in enumerate(pulse):
                a = (pulse.shape[0]-jj)/pulse.shape[0]
                if jj == 0:
                    ax1.plot(t, pp, 'k', lw=1, alpha = a)
                else:
                    pp = np.average(np.reshape(pp, (len(pp)//3, 3)), axis=1)
                    tt = np.average(np.reshape(t, (len(t)//3, 3)), axis=1)
                    ax1.plot(tt, pp, 'k', lw=1, alpha=a, ls='--')
            ax1.set_ylim(min(pulse[0]*1.05), max(pulse[0]*1.2))
        ax1.text(0.03, 0.90, "{0}{1:02}".format(site[0], ii+1),
                transform=ax1.transAxes, verticalalignment='top')
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))

        # Waterfall
        vmin = np.percentile(waterfall, 1)
        vmax = np.percentile(waterfall, 99.5)
        ax2 = fig.add_subplot(gss[1, 0], sharex=ax1)
        ax2.imshow(waterfall, interpolation='nearest', aspect='auto',
                cmap=cmap, extent=[t[0], t[-1], lowband, highband],
                vmin=vmin, vmax=vmax)
        ax2.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

        # Spectrum
        if spectrum is not None:
            ax3 = fig.add_subplot(gss[1, 1])
            ax3.plot(spectrum, freqs, 'k', lw=1)
            ax3.axvline(x=0, ymin=0, ymax=1e3, color='k', alpha=0.3, ls='--')
            ax3.set_ylim(lowband, highband)
            ax3.set_yticklabels([])
            ax3.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))


        # ticks
        if (ii >= len(args.files) - ncols):
            ticks = np.arange((np.ceil(t[0]/50)+1)*50,
                    (np.ceil(t[-1]/50)-1)*50+1, step=100)
            #ax2.set_xticks(ticks)
            ax2.set_xlabel('Time (ms)')
            ax3.set_xlabel("Flux (Jy)")
        else:
            ax2.set_xticklabels([])
        if (ii%ncols == 0):
            ax1.set_ylabel("Flux (Jy)")
            ax2.set_ylabel('Frequency (MHz)')
        else:
            ax2.set_yticklabels([])

        if files is not None:
            file = files[ii]
            obsid = file.split('/')[-1].split('_')[0].upper()
            t = Time(mjd, format='mjd', scale='utc')
            date = t.datetime.strftime('%Y-%m-%d %H:%M:%S')

            # burst_name obsid, MJD, date, S/N, Fluence
            print("{}{:02} & {} & {:.8f} & {} & {} & \\\\".format(site[0],
                    ii+1, obsid, mjd, date, snr))
    # Saving plot
    print('Saving plot to ', plt_out)
    plt.savefig(plt_out, pad_inches=0, bbox_inches='tight')
    plt.show()


# Parser arguments for inputs and reading them in
parser = argparse.ArgumentParser(description='Commands for tauscat code')
parser.add_argument('files', nargs='+',
					help='The chosen archives')
parser.add_argument('-f','--fscrunch', type=int, default=1,
					help='Factor to scrunch the number of channels')
parser.add_argument('-t','--tscrunch', type=int, default=1,
					help='Factor to scrunch the number of time bins')
parser.add_argument('-s','--save_npy', action='store_true', default=False,
					help='Save .npy arrays of the dynamic spectra')


args = parser.parse_args()

plt_out = '/home/arts/ines/R3/lofar-data/R3_LOFAR_bursts.pdf'

lofar_info = np.genfromtxt('/home/arts/ines/R3/lofar-data/pulses_OBSID_t',
        usecols=(0,1,2,3,4,5), names=True)

pulses = []
spectra = []
waterfalls = []
mjds = []
sites = []
times = []
frequencies = []
snrs = []
dms = []

for ii,ff in enumerate(args.files):
    # Opening archive format snippets (LOFAR)
    if ff[-3:] == '.ar':
        # Data from file
        obsid = int(ff.split('/')[-2].split('_')[0][1:])
        T = int(ff.split('/')[-2].split('_')[1][1:])
        jj = (np.where(lofar_info['OBSID'] == obsid) and
                np.where(lofar_info['T'] == T))[0]
        mjd = (lofar_info['mjd_start'][jj] \
                + lofar_info['t_frb'][jj]/(24*3600))[0]
        snr = lofar_info['SNR'][jj][0]
        #dm = lofar_info['DM'][jj][0]
        dm = 349.0
        print(mjd, snr, dm)


        # Waterfalling
    	wf = psrchive.Archive_load(ff)
        if args.save_npy:
            np_data = wf.get_data()[0,0]
            np_out = ff.replace('.ar', '')
            print("Saving", np_out)
            np.save(np_out, np_data)

    	wf.fscrunch(args.fscrunch)
        if args.tscrunch > 1 :
    	       wf.bscrunch(args.tscrunch)
        wf.set_dispersion_measure(dm)
        wf.dedisperse()
        wf.remove_baseline()
        phase = wf.find_max_phase()
        wf.centre_max_bin()
        wf.rotate_phase(0.3)

        # Low SNR sources
        if snr < 10:
        	wf.fscrunch(2)
        	#wf.bscrunch(2)

        waterfall = wf.get_data()[0,0]

        # Data properties
        nbins = wf.get_nbin()
        nchan = wf.get_nchan()

        # Pulse profile
        wf.fscrunch(nchan)
        pulse = wf.get_data()[0,0,0]

        # Other pulse properties
        centfreq = wf.get_centre_frequency()
        bw = np.abs(wf.get_bandwidth())
        chbw = bw/nchan
        lowband = centfreq - (bw/2)
        highband = centfreq + (bw/2)

        # Additional info
        src = wf.get_source()
        period = (wf.get_Integration(0).get_folding_period()) * 1000.
        site = wf.get_telescope()
        tb = np.arange(nbins) - nbins/2
        t = tb * (period / nbins)

        spectrum = None
        freqs = None

    # Opening archive format snippets (LOFAR)
    elif ff[-6:] == '.calib':
        obsid = int(ff.split('/')[-1].split('_')[0][1:])
        jj = int(int(ff.split('/')[-1].split('_')[1].split('.')[0][3:])-1)
        mjd = (lofar_info['mjd_start'][jj] \
                + lofar_info['t_frb'][jj]/(24*3600))
        snr = lofar_info['SNR'][jj]
        dm = 349.0
        duration = 400
        print(mjd, snr, dm)

        # On-burst-window:
        onp_st = -0.05 #s, wrt burst peak
        onp_en = 0.15 #s, wrt burst peak
        # Plotting window:
        win_st = -0.2
        win_en = 0.3

        # Waterfalling
    	wf = psrchive.Archive_load(ff)

    	wf.fscrunch(args.fscrunch)
        if args.tscrunch > 1 :
    	       wf.bscrunch(args.tscrunch)
        wf.remove_baseline()
        phase = wf.find_max_phase()
        wf.centre_max_bin()

        # Low SNR sources
        if snr < 10:
        	wf.fscrunch(2)
        	wf.bscrunch(2)

        waterfall = wf.get_data()[1,0]
        waterfall = np.flipud(waterfall)

        # Data properties
        nbins = wf.get_nbin()
        nchan = wf.get_nchan()
        T = wf.get_Integration(0).get_duration()*1000 # ms
        dt = T/nbins
        weights = wf.get_weights()*1.
        weights /= np.max(weights)

        # spectrum = np.squeeze(wf.get_data())
        # spectrum = spectrum/1000. # Jy
        # spectrum = spectrum*weights[:,:,np.newaxis]
        # spectrum = np.hstack((spectrum[0,:,:], spectrum[1,:,:], spectrum[2,:,:]))

        # Pulse profile
        wf.fscrunch(nchan)
        pulse = wf.get_data()[1,0,0]
        pulse = pulse/1000. #Jy

        # Centering around burst
        nbinlim = np.int(duration/dt)
        binmin = int(pulse.shape[0]/2 - nbinlim/3)
        binmax = int(pulse.shape[0]/2 + nbinlim/3*2)
        t = np.arange(nbinlim)*dt - 1/3 * duration

        waterfall = waterfall[..., binmin:binmax]
        pulse = pulse[binmin:binmax]

        # burst
        bini = np.where(t>-40)[0][0]
        binf = np.where(t>140)[0][0]

        # Spectrum
        spectrum = np.mean(waterfall[..., bini:binf], axis=1)/1000. #Jy
        spectrum = spectrum*weights[0]
        # print(spectrum.shape, weights.shape)
        # spectrum = np.hstack((spectrum[0,:], spectrum[1,:], spectrum[2,:]))


        # Other pulse properties
        centfreq = wf.get_centre_frequency()
        bw = np.abs(wf.get_bandwidth())
        chbw = bw/nchan
        lowband = centfreq - (bw/2)
        highband = centfreq + (bw/2)
        freqs = np.linspace(lowband, highband, num=nchan)
        freqs = np.flipud(freqs)

        if jj in [5, 6]:
            print("Working on L06")
            freq_arr = [150, 140, 130, 120]
            for kk in range(len(freq_arr)-1):
                chtop = np.where(freqs<freq_arr[kk])[0][0]
                chbot = np.where(freqs<freq_arr[kk+1])[0][0]
                pp = np.mean(waterfall[chtop:chbot, ...], axis=0)/1000. #Jy
                try:
                    pulse = np.append([pulse], [pp], axis=0)
                except ValueError:
                    pulse = np.append(pulse, [pp], axis=0)
            print(pulse.shape)

        # Additional info
        src = wf.get_source()
        period = (wf.get_Integration(0).get_folding_period()) * 1000.
        site = wf.get_telescope()

    pulses.append(pulse)
    spectra.append(spectrum)
    waterfalls.append(waterfall)
    mjds.append(mjd)
    sites.append(site)
    times.append(t)
    frequencies.append(freqs)
    snrs.append(snr)
    dms.append(dm)

files = [f for m, f in sorted(zip(mjds, args.files))]
pulses = [p for m, p in sorted(zip(mjds, pulses))]
spectra = [s for m, s in sorted(zip(mjds, spectra))]
waterfalls = [w for m, w in sorted(zip(mjds, waterfalls))]
sites = [s for m, s in sorted(zip(mjds, sites))]
times = [t for m, t in sorted(zip(mjds, times))]
frequencies = [f for m, f in sorted(zip(mjds, frequencies))]
snrs = [s for m, s in sorted(zip(mjds, snrs))]
mjds.sort()

waterplotter(pulses, spectra, waterfalls, mjds, times, frequencies,
        outfile=plt_out, files=files, snrs=snrs, site='LOFAR')
