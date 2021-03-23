from __future__ import print_function
from __future__ import division
import numpy as np
import glob, sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from astropy.time import Time
import astropy.units as u
import datetime
import argparse
from pypulsar.formats import filterbank
import tools
import pol
import read_IQUV_dada
import reader
import triggers
SNRtools = tools.SNR_Tools()

# Usage example:
# python plot_r3_snippet.py /tank/data/FRBs/R3/20200902/filterbank/2020-09-02-22:54:42.FRB180916.J0158+65/CB00_10.0sec_dm0_t05192_sb35_tab00.fil 5.00032768

# Input values
fn_fil = sys.argv[1]

dm=348.75
t0= float(sys.argv[2]) #5.00425984
CB='00'
ntime_plot = 512 # nsamp
bscrunch = 1 # downsamp
fscrunch = 8
#ntime_plot /= bscrunch


sb_generator = triggers.SBGenerator.from_science_case(science_case=4)
sb_generator.reversed = True

# Input files
#fn_fil = '/tank/data/FRBs/R3/20200512/snippet/all/CB00_10.0sec_dm0_t02216_sb35_tab' #00.fil'
# fn_list = ['/tank/data/FRBs/R3/*/snippet/CB00*tab00*',
#         '/tank/data/FRBs/R3/*/*/snippet/all/CB00*tab00*']
# fn_fil = [f for fn in fn_list for f in glob.glob(fn)]
# fn_fil.sort()

fname = '/home/arts/pastor/R3/plots/filenames.txt'
file_info = text_file = open(fname, "r")
lines = text_file.readlines()

# fn_fil, fn_xyphase, mjds, t0s = [], [], [], []
# for line in lines:
#     if '#' not in line:
#         cols = line.split(' ')
#         if cols[0][-4:] == '.fil':
#             fn_fil.append(cols[0])
#             fn_xyphase.append(None)
#             mjds.append(float(cols[1]))
#             t0s.append(float(cols[2]))

#for ii,ff in enumerate(fn_fil):
ii = 0
ff = fn_fil

fig = plt.figure(figsize=(13,9))
ncols, nrows = 1,1
gs = gridspec.GridSpec(nrows,ncols, hspace=0.05, wspace=0.05)


# Reading filterbank file
print(ff)
rawdatafile = filterbank.filterbank(ff)
header = rawdatafile.header

# tstart = header['tstart']  # MJD
# tburst = float(ff.split('/')[-1].split('_')[3].replace('t', '')) * u.s # s
# mjd = tstart + tburst.to("d").value
# mjd = mjds[ii]
# t0 = t0s[ii]

nchans = header['nchans']
fmax = header['fch1'] #1519.50561523
fmin = fmax + nchans * header['foff'] #1219.50561523
nfreq_plot = int(nchans/fscrunch)

f = ff.replace('{}.fil'.format(CB), '')

# Getting pulse peak
full_dm_arr, full_freq_arr, time_res, params = triggers.proc_trigger(
        f, dm, t0, -1,
        ndm=32, mk_plot=False, downsamp=bscrunch,
        beamno=CB, fn_mask=None, nfreq_plot=nfreq_plot,
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

full_freq_arr = np.flip(full_freq_arr, axis=0)
D = full_freq_arr
pulse = np.mean(D, axis=0)
tsamp = time_res * 1000 # ms
tval = np.arange(-ntime_plot/2, ntime_plot/2) * tsamp
tcen = tval[np.argmax(pulse)] * u.ms # ms
t0 += tcen.to('s').value
# mjd += tcen.to('d').value
print("T0", t0)

# SNR width
D = full_freq_arr
D[np.isnan(D)] = np.nanmean(D)
pulse = np.mean(D, axis=0)
tsamp = time_res * 1000
tval = np.arange(-ntime_plot/2, ntime_plot/2) * tsamp

(snr, width) = SNRtools.calc_snr_matchedfilter(pulse, widths=range(500))

print("SNR_boxcar boxcar_width")
print("{} {:.2f} {}".format(ff.split('/')[-1], snr, width))

# Plotting
#print('{} {:.7f}'.format(ff, mjd))
gss = gridspec.GridSpecFromSubplotSpec(2, 1,
        subplot_spec=gs[ii//ncols,ii%ncols], hspace=0, wspace=0,
        height_ratios=[1,3])

ax1 = fig.add_subplot(gss[0,0])
ax1.plot(tval, pulse, 'k')
ax1.axvline(tcen.value, ymin=0, ymax=10)

ax2 = fig.add_subplot(gss[1,0], sharex=ax1)
ax2.imshow(D, interpolation='nearest', aspect='auto',
        extent=[tval[0], tval[-1], fmin, fmax])
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Frequency (MHz)')

#ax1.set_title('{:.7f}'.format(mjd))

plt.show()
