from __future__ import print_function
from __future__ import division
import numpy as np
import glob
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

# --------------------------------------------------------------------- #
# Rebin function
# --------------------------------------------------------------------- #

def rebin(arr, binsz):
    new_shape = [arr.shape[0] //  binsz[0], arr.shape[1] // binsz[1]]
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

def get_pa_iquv(d, nfreq=1536, ntime=1e4,
        fmin=1219.50561523, fmax=1519.50561523):
    """
    Getting PA values from IQUV array after callibration and bandpass correction
    """
    freq = np.arange(fmin, fmax, bw/nfreq)
    xyphase = np.load(fn_xyphase[ii])

    # Bandpass correction
    #d /= bandpass[None, :, None]

    # Frequency mask
    #off_spectrum = d[0, :, :].mean(1) - d[0, :, 4900:5100].mean(1)
    #f_mask = np.where(np.abs(off_spectrum) > np.std(frb_spectrum, axis=-1))[0]
    #f_mask = f_mask[np.where(f_mask<400)]
    #f_mask = np.arange(np.min(f_mask)-10, np.max(f_mask)+10)
    f_mask = range(1532,1536)
    snr_max, width_max = SNRtools.calc_snr_matchedfilter(d[0].mean(0), widths=range(250))
    #get channel weights
    Isubband = d[0].reshape(nfreq//96, 96, ntime).mean(1)
    width_max = max(4,width_max)#hack
    weights_f = []
    for kk in range(Isubband.shape[0]):
        Isub_ = Isubband[kk, :ntime//width_max*width_max]
        Isub_ = Isub_.reshape(-1, width_max).mean(-1)
        Isub_ -= np.median(Isub_)
        snr_subband = Isub_[int(ntime/2./width_max)]/np.std(Isub_[:int(ntime/2./width_max)-5])
        # Create a S/N^2 weights array for full 1536 channel spectrum
        weights_f.append((max(snr_subband,0)*np.ones([96]))**2)


    weights_f = np.array(np.concatenate(weights_f)).flatten()
    weights_f = weights_f[:, None]
    on = d[0, :, 5000-width_max//2:5000+width_max//2].mean(1)
    off = d[0, :, :4500].mean(1)
    frb_spectrum = (on - off)/off
    frb_spectrum[f_mask] = 0.0
    frb_spectrum = frb_spectrum.reshape(-1, 8).mean(-1)

    for jj in range(4):
        off = d[jj, :, :4500].mean(1)
        d[jj] = (d[jj] - off[:, None]) #/ off[:, None]
        d[jj] -= np.median(d[jj].mean(0))
        d[jj] /= np.std(d[jj].mean(0)[:4000])

    d[np.isnan(d)] = 0.0
    d[:, f_mask, :] = 0.0

    # Burst selection
    D = d[:, :, sampmin:sampmax]-d[:,:,:4500, None].mean(-2)
    SS = (d[:, :, 5000-width_max//2:5000+width_max//2].mean(-1))
    #np.save('test%d.npy' % ii, D)

    # Phase correction
    XYdata = D[2]+1j*D[3]
    D[2], D[3] = XYdata.real, XYdata.imag
    P = D[1]+1j*D[2]
    derot_phase = np.load('R3_RM_phase.npy')
    P *= derot_phase[:, None]
    weights_f = 1
    PA = np.angle(np.mean(P*weights_f,0),deg=True)

#    PA = np.mean(np.angle(P,deg=True),0)
#    PA = np.angle(np.mean(D[1]+1j*D[2],0), deg=True) # degrees

    freqArr_Hz = pol.freq_arr*1e6
    # outFilePath = 'out%d.dat' % ii
    # np.savetxt(outFilePath,
    #         np.column_stack((freqArr_Hz, SS[0], SS[1], SS[2],
    #         np.std(d[0,:,:4500],axis=1), np.std(d[1,:,:4500],axis=1),
    #         np.std(d[2,:,:4500],axis=1))))
    #PA[np.where(PA<0)[0]] += 360

    # Temporal mask. index in mask allowed for PA calculation
    std_I = np.std(d[0, ..., 4500:5500].mean(0))
    t_mask = np.where(D[0].mean(0) > 2.0*std_I)[0]
    t_mask = t_mask[np.where(t_mask>50)]

    print('{} {} {}'.format(ff, fn_xyphase[ii], mjds[ii]))
    return D, PA, weights_f, t_mask

def get_i(ff, sb_generator, bscrunch=2, fscrunch=2):

    dm = 349.2
    CB = '00'
    t0 = t0s[ii]

    # Reading filterbank file
    rawdatafile = filterbank.filterbank(ff)
    header = rawdatafile.header

    tstart = header['tstart']  # MJD
    tburst = float(ff.split('/')[-1].split('_')[3].replace('t', '')) * u.s # s
    mjd = tstart + tburst.to("d").value

    nchans = header['nchans']
    fmax = header['fch1'] #1519.50561523
    fmin = fmax + nchans * header['foff'] #1219.50561523
    nfreq_plot = int(nchans/fscrunch)
    ntime_plot = int(nsamp/bscrunch)

    f = ff.replace('00.fil', '')

    # Getting pulse peak
    full_dm_arr_downsamp, full_freq_arr_downsamp, time_res, params = triggers.proc_trigger(
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
            n_iter_frequency=3,
            clean_type='perchannel', freq=1370,
            sb_generator=sb_generator, sb=35)

    D = full_freq_arr_downsamp
    pulse = np.mean(D, axis=0)
    tsamp = time_res * 1000 # ms
    tval = np.arange(-ntime_plot/2, ntime_plot/2) * tsamp * bscrunch
    tcen = tval[np.argmax(pulse)] * u.ms # ms
    #print(tcen)
    # t0 += tcen.to('s').value
    #mjd += tcen.to('d').value

    # Getting pulse data from corrected time
    # full_dm_arr_downsamp, full_freq_arr_downsamp, time_res, params = triggers.proc_trigger(
    #         f, dm, t0, -1,
    #         ndm=32, mk_plot=False, downsamp=bscrunch,
    #         beamno=CB, fn_mask=None, nfreq_plot=nfreq_plot,
    #         ntime_plot=nsamp,
    #         cmap='viridis', cand_no=1, multiproc=False,
    #         rficlean=True, snr_comparison=-1,
    #         outdir='./data', sig_thresh_local=0.0,
    #         subtract_zerodm=False,
    #         threshold_time=3.25, threshold_frequency=2.75,
    #         bin_size=32, n_iter_time=3,
    #         n_iter_frequency=3,
    #         clean_type='perchannel', freq=1370,
    #         sb_generator=sb_generator, sb=35)
    #
    # D = full_freq_arr_downsamp
    # tsamp = time_res * 1000
    # tval = np.arange(-nsamp/2, nsamp/2) * tsamp

    print(mjds[ii], t0s[ii])

    return D, tval

def iquv_plotter(D, PA, tval, gss, ii, weights_f, t_mask,
        bscrunch=2, fscrunch=2, ncols=4, nrows=6, waterfall=False):
    """
    Plotting IQUV data
    """

    colors = ['#577590', '#90be6d', '#f8961e', '#f94144']

    ax1 = fig.add_subplot(gss[1, 0])
    ax1.plot(tval, (D[0]*weights_f).mean(0), color=colors[0], label='I')
    ax1.plot(tval, (D[1]*weights_f).mean(0), color=colors[1], label='Q')
    ax1.plot(tval, (D[2]*weights_f).mean(0), color=colors[2], label='U')
    ax1.plot(tval, (D[3]*weights_f).mean(0), color=colors[3], label='V')
    ax1.yaxis.set_major_locator(MultipleLocator(5))
    ax1.set_yticklabels([])


    ax2 = fig.add_subplot(gss[0, 0], sharex=ax1)
    ax2.plot((t_mask - nsamp/2) * tsamp, PA[t_mask], '.', color='k', alpha=0.5, label='PA')
    PA_arr.append(PA[t_mask])
    ax2.yaxis.set_major_locator(MultipleLocator(90))
    ax2.set_ylim(-190,190)
    ax2.grid(b=True, axis='y', color='k', alpha=0.1)


    # ax3 = fig.add_subplot(gss[0, 1])
    # #ax3.plot(D[0, ..., mask].mean(0), color=colors[0], lw=1)
    # #ax3.plot(D[0].mean(0), color=colors[0], lw=1)
    # ax3.plot(freqArr_Hz[4::8]*1e-6, frb_spectrum, color=colors[0], lw=1)

    if args.waterfall:
        waterfall = rebin(D[0], [fscrunch,bscrunch])

        ax4 = fig.add_subplot(gss[2,0], sharex=ax1)
        ax4.imshow(waterfall, interpolation='nearest', aspect='auto',
                cmap='viridis', extent=[tval[0], tval[-1], fmin, fmax])

        if (ii >= len(fl) - ncols):
            ax4.set_xlabel('Time (ms)')
        else:
            ax4.set_xticklabels([])
        if (ii%ncols == 0):
            ax4.set_ylabel('Frequency (MHz)')
        else:
            ax4.set_yticklabels([])
    else:
        if (ii >= len(fl) - ncols):
            ax1.set_xlabel('Time (ms)')
        else:
            ax1.set_xticklabels([])
    if (ii%ncols != 0):
        ax2.set_yticklabels([])
    else:
        ax2.set_ylabel('PA (deg)')

    for ax in (ax1,ax2,ax4):
        ax.tick_params(axis='x', which='both', direction='in', bottom=True,
                top=True)
        ax.tick_params(axis='y', which='both', direction='in', left=True,
                right=True)
        ax.label_outer()

def snippet_plotter(D, tval, gss, ii, ncols=4, nrows=6, waterfall=False):
    """
    Plotting I data
    """

    pulse = np.mean(D, axis=0)
    colors = ['#577590', '#90be6d', '#f8961e', '#f94144']

    ax1 = fig.add_subplot(gss[1,0])
    ax1.plot(tval, pulse, color=colors[0])
    ax1.yaxis.set_major_locator(MultipleLocator(5))
    ax1.set_yticklabels([])

    if args.waterfall:
        ax4 = fig.add_subplot(gss[2,0], sharex=ax1)
        ax4.imshow(D, interpolation='nearest', aspect='auto',
                extent=[tval[0], tval[-1], fmin, fmax])
        ax4.set_xlabel('Time (ms)')
        ax4.set_ylabel('Frequency (MHz)')

        if (ii >= len(fl) - ncols):
            ax4.set_xlabel('Time (ms)')
        else:
            ax4.set_xticklabels([])
        if (ii%ncols == 0):
            ax4.set_ylabel('Frequency (MHz)')
        else:
            ax4.set_yticklabels([])
    else:
        if (ii >= len(fl) - ncols):
            ax1.set_xlabel('Time (ms)')
        else:
            ax1.set_xticklabels([])

    for ax in (ax1,ax4):
        ax.tick_params(axis='x', which='both', direction='in', bottom=True,
                top=True)
        ax.tick_params(axis='y', which='both', direction='in', left=True,
                right=True)
        ax.label_outer()

# --------------------------------------------------------------------- #
# Input parameters
# --------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='Commands for plotting IQUV data')
# parser.add_argument('files', nargs='+', default=None
# 					help='The chosen files')
parser.add_argument('-f','--fscrunch', type=int, default=2,
		help='Factor to scrunch the number of channels')
parser.add_argument('-b','--bscrunch', type=int, default=2,
		help='Factor to scrunch the number of time bins')
parser.add_argument('-n','--nsamp',type=int, default=256,
        help='Number of time samples to waterfall')
parser.add_argument('-w','--waterfall', action='store_true',
        default=False, help='Show waterfall plot')
parser.add_argument('-s','--show', action='store_true',
        default=False, help='Show waterfall plot')

args = parser.parse_args()


# Parameters:
nsamp = args.nsamp
fscrunch = args.fscrunch
bscrunch = args.bscrunch

# Defining numpy files to plot
datadir = '/tank/data/FRBs/R3/'
# fl_20200322 = glob.glob(datadir+'20200322/iquv/CB00/numpyarr/*dedisp*')
# fl_2020051x = glob.glob(datadir+'2020051*/CB00/iquv/numpyarr/*mjd*dedisp*')
# fl_2020052x = glob.glob(datadir+'2020052*/*/iquv/numpyarr/*dedisp*')
# fl = fl_20200322 + fl_2020051x + fl_2020052x #+ fl_crabtest
# fl.sort()
# fl = np.unique(fl)
fname = '/home/arts/pastor/R3/plots/filenames.txt'
file_info = text_file = open(fname, "r")
lines = text_file.readlines()

fl, fn_xyphase, mjds, t0s = [], [], [], []
for line in lines:
    if '#' not in line:
        cols = line.split(' ')
        fl.append(cols[0])
        if cols[0][-4:] == '.npy':
            fn_xyphase.append(cols[1])
            mjds.append(cols[2])
            t0s.append(None)
        elif cols[0][-4:] == '.fil':
            fn_xyphase.append(None)
            mjds.append(cols[1])
            t0s.append(float(cols[2]))

file_info.close()

# Defining phase calibrator
# fn_xyphase = [datadir+'20200322/iquv/CB00/polcal/xy_phase.npy'] \
#              * len(fl_20200322) \
#              + [datadir+'20200512/CB00/iquv/polcal/xy_phase.npy'] \
#              * len(fl_2020051x) \
#              + [datadir+'20200527/polcal/xy_phase.npy'] \
#              * len(fl_2020052x)#+len(fl_crabtest))

# Defining bandpass
bandpass = np.load('/tank/data/FRBs/FRB200419/20200419/iquv/CB32/polcal/bandpass.npy')

# Example dada header:rebin
fndada = '/tank/data/FRBs/R3/20200322/iquv/CB00/dada/2020-03-22-10:03:39_0004130611200000.000000.dada'
header = read_IQUV_dada.read_dada_header(fndada)
bw = header['bw']
fmin = header['freq'] - header['bw'] / 2
fmax = header['freq'] + header['bw'] / 2
tsamp = header['tsamp'] * 1000. # ms
sampmin = int(5000-nsamp/2)
sampmax = int(5000+nsamp/2)

sb_generator = triggers.SBGenerator.from_science_case(science_case=4)
sb_generator.reversed = True

# Output file
plt_out = './R3_IQUV_PA.pdf'
PA_arr = []

# Plotting

fig = plt.figure(figsize=(21,29.7))
plt.rcParams.update({'font.size': 14,
        'axes.labelsize': 14,
        'axes.titlesize':14,
        'xtick.labelsize':12,
        'ytick.labelsize':12,
        'lines.linewidth':1,
        'lines.markersize':5})
ncols, nrows = 6,5
gs = gridspec.GridSpec(nrows,ncols, hspace=0.05, wspace=0.05)

for ii,ff in enumerate(fl):

    # Plotting
    if args.waterfall:
        gss = gridspec.GridSpecFromSubplotSpec(3, 1,
                subplot_spec=gs[ii//ncols,ii%ncols], hspace=0, wspace=0,
                height_ratios=[1,2,3])
    else:
        gss = gridspec.GridSpecFromSubplotSpec(2, 1,
                subplot_spec=gs[ii//ncols,ii%ncols], hspace=0, wspace=0,
                height_ratios=[1,3])

    if ff[-4:] == '.npy':
        d = np.load(ff)
        nfreq,ntime = d.shape[1],d.shape[2]
        tval = np.arange(-nsamp/2, nsamp/2) * tsamp

        D, PA, wf, tm = get_pa_iquv(d, nfreq=nfreq, ntime=ntime,
                fmin=fmin, fmax=fmax)
        iquv_plotter(D, PA, tval, gss, ii, wf, tm, bscrunch=bscrunch,
                fscrunch=fscrunch, ncols=ncols, nrows=nrows,
                waterfall=args.waterfall)

    elif ff[-4:] == '.fil':
        D, tval = get_i(ff, sb_generator, bscrunch=bscrunch, fscrunch=fscrunch)
        snippet_plotter(D, tval, gss, ii, ncols=ncols, nrows=nrows,
                waterfall=args.waterfall)

np.save('PAs.npy',PA_arr)
# Saving plot
print('Saving plot to ', plt_out)
plt.savefig(plt_out, pad_inches=0, bbox_inches='tight')

if args.show:
    plt.show()
