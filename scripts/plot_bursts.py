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
from matplotlib.ticker import (FormatStrFormatter, AutoMinorLocator, MultipleLocator)
from astropy.time import Time

# --------------------------------------------------------------------------- #
# Plotting function
# --------------------------------------------------------------------------- #

def waterplotter(pulses, waterfalls, mjds, times, files=None, snrs=None,
        site='LOFAR', outfile='./waterfall.pdf', ncols=3, cmap='magma'):

    plt.rcParams.update({
            'font.size': 12,
            'font.family': 'serif',
            'axes.labelsize': 10,
            'axes.titlesize': 14,
            #'xtick.labelsize': 12,
            #'ytick.labelsize': 12,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.minor.visible': True,
            'ytick.minor.visible': True,
            'xtick.top': True,
            'ytick.right': True,
            'lines.linewidth': 0.5,
            'lines.markersize': 5,
            'legend.fontsize': 6,
            'legend.loc': 'lower right'})
    # Plotting
    cmap = mpl.cm.get_cmap(cmap)
    nrows = int(np.ceil(len(args.files)/ncols))
    fig = plt.figure(figsize=(9,7))
    gs = gridspec.GridSpec(nrows,ncols, hspace=0.05, wspace=0.05)

    for ii,ff in enumerate(waterfalls):
        pulse = pulses[ii]
        waterfall = waterfalls[ii]
        mjd = mjds[ii]
        t = times[ii]
        if snrs is not None: snr = snrs[ii]
        else: snr = ''

        gss = gridspec.GridSpecFromSubplotSpec(2, 1,
                subplot_spec=gs[ii//ncols,ii%ncols], hspace=0, wspace=0,
                height_ratios=[1,3])

        # Pulse profile
        ax1 = fig.add_subplot(gss[0, 0])
        ax1.plot(t, pulse, 'k', lw=1)
        ax1.set_yticks([])
        ax1.text(0.03, 0.90, "{0}{1:02}".format(site[0], ii+1),
                transform=ax1.transAxes, verticalalignment='top')
        # ax1.tick_params(axis='x', which='both', direction='in', bottom=True,
        #         top=True)
        # ax1.tick_params(axis='y', which='both', direction='in', left=True,
        #         right=True)

        ax2 = fig.add_subplot(gss[1, 0], sharex=ax1)
        ax2.imshow(waterfall, interpolation='nearest', aspect='auto',
                cmap=cmap, extent=[t[0], t[-1], lowband, highband])
        print(lowband, highband)

        # ticks
        if (ii >= len(args.files) - ncols):
            ticks = np.arange((np.ceil(t[0]/50)+1)*50,
                    (np.ceil(t[-1]/50)-1)*50+1, step=100)
            #ax2.set_xticks(ticks)
            ax2.set_xlabel('Time (ms)')
        else:
            ax2.set_xticklabels([])
        if (ii%ncols == 0):
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
waterfalls = []
mjds = []
sites = []
times = []
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
        mjd = (lofar_info['mjd_start'][jj] + lofar_info['t_frb'][jj]/(24*3600))[0]
        snr = lofar_info['SNR'][jj][0]
        #dm = lofar_info['DM'][jj][0]
        dm = 349.09
        print(mjd, snr, dm)

        # Waterfall
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

        # Low SNR sources
        if snr < 10:
        	wf.fscrunch(4)
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
        print(period/nbins)

    pulses.append(pulse)
    waterfalls.append(waterfall)
    mjds.append(mjd)
    sites.append(site)
    times.append(t)
    snrs.append(snr)
    dms.append(dm)

files = [f for m, f in sorted(zip(mjds, args.files))]
pulses = [p for m, p in sorted(zip(mjds, pulses))]
waterfalls = [w for m, w in sorted(zip(mjds, waterfalls))]
sites = [s for m, s in sorted(zip(mjds, sites))]
times = [t for m, t in sorted(zip(mjds, times))]
snrs = [s for m, s in sorted(zip(mjds, snrs))]
mjds.sort()

waterplotter(pulses, waterfalls, mjds, times, outfile=plt_out, files=files,
        snrs=snrs, site='LOFAR')
