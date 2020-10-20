from __future__ import print_function
from __future__ import division
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from matplotlib.lines import Line2D
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

# Example usage
# python /home/arts/pastor/scripts/arts-analysis/plot_r3_iquv.py -f 1 -b 1 -w
# -n 512 -s --nfig 3 --save_npy --cmap viridis

# --------------------------------------------------------------------- #
# Rebin function
# --------------------------------------------------------------------- #

def rebin(arr, binsz):
    new_shape = [int(arr.shape[0] //  int(binsz[0])),
            int(arr.shape[1] // int(binsz[1]))]
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

def get_pa_iquv(d, fn_xyphase, nfreq=1536, ntime=1e4, bw=300.,
        fmin=1219.50561523, fmax=1519.50561523):
    """
    Getting PA values from IQUV array after callibration and bandpass correction
    """

    #print('{} {} {}'.format(ff, fn_xyphase[ii], mjds[ii]))
    XYduncal,XYdcal = 0,0
    SS=0
    Dcal_spectrum=0

    freq = np.arange(fmin, fmax, bw/nfreq)
    xyphase = np.load(fn_xyphase)

    # Frequency mask
    f_mask = range(1532,1536)
    snr_max, width_max = SNRtools.calc_snr_matchedfilter(d[0].mean(0),
            widths=range(250))
    #get channel weights
    nsubband_snr=192
    Isubband = d[0].reshape(nfreq//nsubband_snr, nsubband_snr, ntime).mean(1)
    width_max = max(4,width_max)#hack
    weights_f = []

    for kk in range(Isubband.shape[0]):
        Isub_ = Isubband[kk, :ntime//width_max*width_max]
        Isub_ = Isub_.reshape(-1, width_max).mean(-1)
        Isub_ -= np.median(Isub_)
        snr_subband = Isub_[int(ntime/2./width_max)] \
                / np.std(Isub_[:int(ntime/2./width_max)-5])
        weight_kk = (max(snr_subband,0)*np.ones([nsubband_snr]))**2
        weights_f.append((max(snr_subband,0)*np.ones([96]))**2)


    weights_f = np.array(np.concatenate(weights_f)).flatten()
    weights_f = weights_f[:, None]
    on = d[0, :, 5000-width_max//2:5000+width_max//2].mean(1)
    off = d[0, :, :4500].mean(1)
    frb_spectrum = (on - off)/off
    frb_spectrum[f_mask] = 0.0
    frb_spectrum = frb_spectrum.reshape(-1, 16).mean(-1)

    for jj in range(4):
        d[jj] = d[jj].astype(float)

        # SNR
        # d[jj] -= np.median(d[jj], axis=-1)[:, None]
        # d[jj] /= np.std(d[jj])
        # d[jj] -= np.median(d[jj], axis=-1, keepdims=True)
        #
        # sig = np.std(d[jj])
        # dmax = (d[jj].copy()).max()
        # dmed = np.median(d[jj])
        # N = len(d[jj])

        # remove outliers 4 times until there
        # are no events above threshold*sigma
        # thresh=3
        # for ii in range(4):
        #     ind = np.where(np.abs(d[jj]-dmed)<thresh*sig)[0]
        #     sig = np.std(d[jj][ind])
        #     dmed = np.median(d[jj][ind])
        #     d[jj] = d[jj][ind]
        #     N = len(d[jj])

        # d[jj] = (d[jj] - dmed)/(1.048*sig)

        # Plot
        off = d[jj, :, :4500].mean(1)
        d[jj] = (d[jj] - off[:, None]) #/ off[:, None]
        d[jj] -= np.median(d[jj].mean(0)[:4000])
        d[jj] /= np.std(d[jj].mean(0)[:4000])

    d[np.isnan(d)] = 0.0
    d[:, f_mask, :] = 0.0

    # Burst selection
    wm2 = width_max//2
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

    weights_f = 1
    PA = np.angle(np.mean(P*weights_f,0),deg=True)

    freqArr_Hz = pol.freq_arr*1e6

    # Temporal mask. index in mask allowed for PA calculation
    std_I = np.std(d[0, ..., 3900:4900].mean(0))
    t_mask = np.where(D[0].mean(0) > 5.0*std_I)[0]
    t_mask = t_mask[np.where(t_mask>50)]
    #t_mask = np.where(D[0].mean(0).reshape(-1,1).mean(1) > 2.0*std_I)[0]
    return D, PA, weights_f, t_mask

def get_i(ff, t0, sb_generator, nsamp=512, bscrunch=2, fscrunch=2, dm=348.75):

    CB = '00'

    # Reading filterbank file
    rawdatafile = filterbank.filterbank(ff)
    header = rawdatafile.header

    tstart = header['tstart']  # MJD
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

    ntime_plot = int(nsamp/bscrunch)
    p = np.argmax(np.mean(full_freq_arr, axis=0))

    D = full_freq_arr[:, int(p-ntime_plot/2):int(p+ntime_plot/2)]
    pulse = np.mean(D, axis=0)
    tsamp = time_res * 1000 # ms
    tval = np.arange(-ntime_plot/2, ntime_plot/2) * tsamp
    tcen = tval[np.argmax(pulse)] * u.ms # ms
    t0 += tcen.to("s").value
    #print(mjds[ii], tcen, D.shape, t0)
    D = np.flip(D, axis=0)

    # SNR units
    sig = np.std(D)
    dmax = (D.copy()).max()
    dmed = np.median(D)
    N = len(D)

    # remove outliers 4 times until there
    # are no events above threshold*sigma
    # thresh=3
    # for ii in range(4):
    #     ind = np.where(np.abs(D-dmed)<thresh*sig)[0]
    #     sig = np.std(D[ind])
    #     dmed = np.median(D[ind])
    #     D = D[ind]
    #     N = len(D)

    D = (D - dmed)/(1.048*sig)

    return D, tval, t0

def iquv_plotter(D, PA, tval, gss, ii, weights_f, t_mask,
        bscrunch=2, fscrunch=2, tot=34, nsub=34, nfig=1, ncols=4, nrows=6,
        cmap='viridis', waterfall=False, stokes='iquv', id=None):
    """
    Plotting IQUV data
    """

    #colors = ['#577590', '#90be6d', '#f8961e', '#f94144']
    #colors = ['#080808', '#525252', '#858585', '#C2C2C2']

    ax1 = fig.add_subplot(gss[1, 0])
    nfreq, ntime = D.shape[1]//fscrunch, D.shape[2]//bscrunch
    iquv = [rebin(d, [1,bscrunch]).mean(0) for d in D]
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
    ax1.yaxis.set_major_locator(MultipleLocator(5))
    ax1.set_yticklabels([])
    if id is not None:
        ax1.text(0.05, 0.95, id, horizontalalignment='left',
                verticalalignment='top', transform=ax1.transAxes)

    ax2 = fig.add_subplot(gss[0, 0], sharex=ax1)
    ax2.plot((t_mask - nsamp/2) * tsamp, PA[t_mask], '.', color='k', alpha=0.5, label='PA')
    PA_arr.append(PA[t_mask])
    ax2.yaxis.set_major_locator(MultipleLocator(90))
    ax2.set_ylim(-190,190)
    ax2.grid(b=True, axis='y', color='k', alpha=0.1)

    if args.waterfall:
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
            ax4.set_ylabel('Frequency (MHz)')
        else:
            ax4.set_yticklabels([])
        axes = (ax1,ax2,ax4)
    else:
        if (ii >= nsub - ncols):
            ax1.set_xlabel('Time (ms)')
        else:
            ax1.set_xticklabels([])
        axes = (ax1,ax2)
    if (ii%ncols != 0):
        ax2.set_yticklabels([])
    else:
        ax2.set_ylabel('PA')

    for ax in axes:
        # ax.tick_params(axis='x', which='both', direction='in', bottom=True,
        #         top=True)
        # ax.tick_params(axis='y', which='both', direction='in', left=True,
        #         right=True)
        ax.label_outer()

def snippet_plotter(D, tval, gss, ii, t0=None, tot=34, nsub=34,
        nfig=1, ncols=4, nrows=6,
        cmap='viridis', waterfall=False, id=None):
    """
    Plotting I data
    """

    pulse = np.mean(D, axis=0)
    #colors = ['#577590', '#90be6d', '#f8961e', '#f94144']
    colors = ['#080808', '#525252', '#858585', '#C2C2C2']

    ax1 = fig.add_subplot(gss[1,0])
    ax1.plot(tval, pulse, color='k')
    ax1.yaxis.set_major_locator(MultipleLocator(1))
    ax1.set_yticklabels([])
    if id is not None:
        ax1.text(0.05, 0.95, id, horizontalalignment='left',
                verticalalignment='top', transform=ax1.transAxes)
    # if t0 is not None:
    #     ax1.scatter((t0-5)*1e3, np.max(pulse) -1)

    if args.waterfall:
        #TODO: reverse frequencies
        ax4 = fig.add_subplot(gss[2,0], sharex=ax1)
        vmin = np.percentile(D, 1.0)
        vmax = np.percentile(D, 99.5)
        ax4.imshow(D, interpolation='nearest', aspect='auto', origin='lower',
                cmap=cmap, vmin=vmin, vmax=vmax,
                extent=[tval[0], tval[-1], fmin, fmax])
        axes = (ax1,ax4)

        if (ii >= nsub - ncols):
            ax4.set_xlabel('Time (ms)')
        else:
            ax4.set_xticklabels([])
        if (ii%ncols == 0):
            ax4.set_ylabel('Frequency (MHz)')
        else:
            ax4.set_yticklabels([])
    else:
        if (ii >= nsub - ncols):
            ax1.set_xlabel('Time (ms)')
        else:
            ax1.set_xticklabels([])
        axes = (ax1)

    for ax in axes:
        # ax.tick_params(axis='x', which='both', direction='in', bottom=True,
        #         top=True)
        # ax.tick_params(axis='y', which='both', direction='in', left=True,
        #         right=True)
        ax.label_outer()


# --------------------------------------------------------------------- #
# Input parameters
# --------------------------------------------------------------------- #

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Commands for plotting IQUV data')
    # parser.add_argument('files', nargs='+', default=None
    # 					help='The chosen files')
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
    parser.add_argument('--save_npy', action='store_true',
            default=False,
            help='Save numpy arrays of the plotted dynamic spectra')

    args = parser.parse_args()


    # Parameters:
    nsamp = args.nsamp
    fscrunch = args.fscrunch
    bscrunch = args.bscrunch
    stokes = args.stokes

    # Defining numpy files to plot
    datadir = '/tank/data/FRBs/R3/'
    fname = '/home/arts/pastor/R3/plots/filenames.txt'
    file_info  = open(fname, "r")
    lines = file_info.readlines()

    fl, fn_xyphase, mjds, t0s = [], [], [], []
    for line in lines:
        if '#' not in line:
            cols = line.split(' ')
            mjds.append(float(cols[0]))
            fl.append(cols[1])
            if cols[1][-4:] == '.npy':
                fn_xyphase.append(cols[2].replace('\n', ''))
                t0s.append(None)
            elif cols[1][-4:] == '.fil':
                fn_xyphase.append(None)
                t0s.append(float(cols[2]))
    file_info.close()
    tot = len(fl)

    # Defining bandpass
    bandpass = np.load('/tank/data/FRBs/FRB200419/20200419/iquv/CB32/polcal/bandpass.npy')

    # Example dada header:rebin
    fndada = '/tank/data/FRBs/R3/20200322/iquv/CB00/dada/2020-03-22-10:03:39_0004130611200000.000000.dada'
    header = read_IQUV_dada.read_dada_header(fndada)
    bw = float(header['bw'])
    fmin = header['freq'] - header['bw'] / 2
    fmax = header['freq'] + header['bw'] / 2
    tsamp = header['tsamp'] * 1000. # ms
    sampmin = int(5000-nsamp/2)
    sampmax = int(5000+nsamp/2)

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

    for ii,ff in enumerate(fl):

        burst_id = 'A{:02}'.format(ii+1)
        mjd = mjds[ii]

        # Plotting
        # Defining subplot number within figure
        if args.nfig == 1:
            jj, fign, nsub = ii, 0, tot
        elif args.nfig == 2:
            if ii < tot//2:
                jj, fign, nsub = ii, 0, tot//2
            else:
                jj, fign, nsub = ii - tot//2, 1, tot//2 + tot%2
        else:
            if ii < tot//3:
                jj, fign, nsub = ii, 0, tot//3
            elif ii < 2 * tot//3:
                jj, fign, nsub = ii - (tot//3), 1, tot//3
            else:
                jj, fign, nsub = ii - 2 * (tot//3), 2, tot//3

        if jj == 0:
            fig = plt.figure(fign, figsize=(21,29))
            plt.rcParams.update({
                    'font.size': 18,
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
                    'legend.fontsize': 18,
                    'legend.borderaxespad': 0,
                    'legend.frameon': False,
                    'legend.loc': 'lower right'})
            gs = gridspec.GridSpec(nrows,ncols, hspace=0.05, wspace=0.05)

        print(burst_id, jj, jj//ncols, jj%ncols, nsub)

        if args.waterfall:
            gss = gridspec.GridSpecFromSubplotSpec(3, 1,
                    subplot_spec=gs[jj//ncols,jj%ncols], hspace=0, wspace=0,
                    height_ratios=[1.5,2,4])
        else:
            gss = gridspec.GridSpecFromSubplotSpec(2, 1,
                    subplot_spec=gs[jj//ncols,jj%ncols], hspace=0, wspace=0,
                    height_ratios=[1,3])

        if ff[-4:] == '.npy':
            d = np.load(ff)
            nfreq,ntime = d.shape[1],d.shape[2]
            ts = nsamp/(2*bscrunch)
            tval = np.arange(-ts, ts) * tsamp * bscrunch

            D, PA, wf, tm = get_pa_iquv(d, fn_xyphase[ii],
                    nfreq=nfreq, ntime=ntime, bw=bw, fmin=fmin, fmax=fmax)
            iquv_plotter(D, PA, tval, gss, jj, wf, tm, bscrunch=bscrunch,
                    fscrunch=fscrunch, tot=tot, nsub=nsub, nfig=args.nfig,
                    ncols=ncols, nrows=nrows, cmap=args.cmap,
                    waterfall=args.waterfall, stokes=stokes, id=burst_id)
            if args.save_npy:
                fnout = 'R3_mjd{:.6f}_dedisp348.8'.format(mjd)
                # np_out = '/home/arts/pastor/scripts/arts-analysis/iquv_npy/' \
                #         + fnout
                np_out = '/home/arts/pastor/R3/fluxcal/i_npy/' \
                        + fnout
                print('saving', np_out)
                np.save(np_out, D[0])
            print(mjd, 'IQUV', np.mean(D[0]), np.median(D[0]), np.std(D[0]))
            pulse = rebin(D[0], [1,bscrunch]).mean(0)
            (snr, width) = SNRtools.calc_snr_matchedfilter(pulse, widths=range(500))
            print(snr, width)

        if ff[-4:] == '.fil':
            D, tval, t0 = get_i(ff, t0s[ii], sb_generator, nsamp=nsamp,
                    bscrunch=bscrunch, fscrunch=fscrunch, dm=args.dm)
            snippet_plotter(D, tval, gss, jj, t0, tot=tot, nfig=args.nfig,
                    nsub=nsub, ncols=ncols, nrows=nrows, cmap=args.cmap,
                    waterfall=args.waterfall, id=burst_id)
            if args.save_npy:
                fnout = 'R3_mjd{:.6f}_dedisp348.8'.format(mjd)
                # np_out = '/home/arts/pastor/scripts/arts-analysis/iquv_npy/' \
                #         + fnout
                np_out = '/home/arts/pastor/R3/fluxcal/i_npy/' \
                        + fnout
                print('saving', np_out)
                np.save(np_out, D)
            print(mjd, 'Ifil', np.mean(D), np.median(D), np.std(D))


        if jj == nsub-1:
            # Adding legend
            ll = jj+1
            ax = fig.add_subplot(gs[nrows-1,ncols-1])
            #fig.patch.set_visible(False)
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

    #np.save('PAs.npy',PA_arr)

    if args.show:
        plt.show()
