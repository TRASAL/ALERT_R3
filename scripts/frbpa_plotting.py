#!/usr/bin/env python3
import numpy as np
import pandas as pd
import json, logging
import argparse
import pandas as pd
from astropy.time import Time, TimeDelta
from astropy import units as u
import datetime
import pylab as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec
from frbpa.utils import get_phase, get_cycle, get_params
logging_format = '%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)

def sort_dict(dictionary, list):
    sorted_dict = {k: dictionary[k] for k in list if k in dictionary.keys()}
    return sorted_dict

def open_json(data_file):
    with open(data_file, 'r') as f:
        data = json.load(f)

    assert 'obs_duration' in data.keys()
    assert 'bursts' in data.keys()
    assert 'obs_startmjds' in data.keys()

    burst_dict = data['bursts']
    snr_dict = data['snr']
    obs_duration_dict = data['obs_duration']
    obs_startmjds_dict = data['obs_startmjds']
    fmin_dict = data['freq_min']
    fmax_dict = data['freq_max']

    assert len(obs_duration_dict.keys()) == len(obs_startmjds_dict.keys())
    assert len(obs_duration_dict.keys()) < 20
    assert len(burst_dict.keys()) < 10
    assert len(fmin_dict.keys()) ==  len(fmax_dict.keys())

    telescopes = list(obs_duration_dict.keys())

    new_obs_startmjds_dict = {}
    new_obs_duration_dict = {}
    fcen_dict = {}
    for k in obs_startmjds_dict.keys():
        start_times = obs_startmjds_dict[k]
        durations = obs_duration_dict[k]
        fmin = fmin_dict[k]
        fmax = fmax_dict[k]
        #new_start_times = []
        new_durations = []
        for i, t in enumerate(start_times):
            new_durations.append(durations[i]/(3600))
        new_obs_duration_dict[k] = new_durations
        fcen_dict[k] = (fmax + fmin)/2
    obs_duration_dict = new_obs_duration_dict

    # Sorting dictionaries by central frequency
    fcen_dict = {k: v for k, v in sorted(fcen_dict.items(),
            key=lambda item: item[1])}
    burst_dict = sort_dict(burst_dict, fcen_dict.keys())
    snr_dict = sort_dict(snr_dict, fcen_dict.keys())
    obs_duration_dict = sort_dict(obs_duration_dict, fcen_dict.keys())
    obs_startmjds_dict = sort_dict(obs_startmjds_dict, fcen_dict.keys())
    fmin_dict = sort_dict(fmin_dict, fcen_dict.keys())
    fmax_dict = sort_dict(fmax_dict, fcen_dict.keys())

    return burst_dict, snr_dict, obs_duration_dict, obs_startmjds_dict, fmin_dict, fmax_dict, fcen_dict

def open_csv(data_file):
    burst_data = pd.read_csv(data_file)
    burst_data['freq_cen'] = burst_data.loc[: , "freq_min":"freq_max"].mean(1)
    burst_data['key'] = burst_data["telescope"] + '/' + burst_data["receiver"]

    burst_dict = {}
    fmin_dict = {}
    fmax_dict = {}
    fcen_dict = {}
    for k in np.unique(burst_data['key']):
        burst_dict[k] = burst_data.loc[burst_data['key'] == k]['mjd']
        fmin_dict[k] = burst_data.loc[burst_data['key']==k]['freq_min'].iloc[0]
        fmax_dict[k] = burst_data.loc[burst_data['key']==k]['freq_max'].iloc[0]
        fcen_dict[k] = burst_data.loc[burst_data['key']==k]['freq_cen'].iloc[0]

    fcen_dict = {k: v for k, v in sorted(fcen_dict.items(),
            key=lambda item: item[1])}
    burst_dict = sort_dict(burst_dict, fcen_dict.keys())
    fmin_dict = sort_dict(fmin_dict, fcen_dict.keys())
    fmax_dict = sort_dict(fmax_dict, fcen_dict.keys())
    return burst_dict, fmin_dict, fmax_dict, fcen_dict

def make_obs_phase_plot(data_file, period, ref_mjd=58369.30, nbins=40, save=False,
        show=False, log=False, min_freq=200, max_freq=2500):
    """
    Generates burst phase and observation phase distribution plot for a given period.

    :param data_file: json file with data
    :param period: period to use for phase calculation
    :param ref_mjd: reference MJD to use
    :param nbins: number of bins in the phase histogram
    :param save: to save the plot
    :param show: to show the plot
    """

    burst_dict, snr_dict, obs_duration_dict, obs_startmjds_dict, fmin_dict, fmax_dict, fcen_dict = open_json(data_file)

    bursts = []
    for k in burst_dict.keys():
        bursts = bursts + burst_dict[k]

    obs_duration = []
    for k in obs_duration_dict.keys():
        obs_duration = obs_duration + obs_duration_dict[k]

    obs_startmjds = []
    for k in obs_startmjds_dict.keys():
        obs_startmjds = obs_startmjds + obs_startmjds_dict[k]

    assert len(obs_startmjds) == len(obs_duration)

    bursts = np.array(bursts)
    obs_duration = np.array(obs_duration)
    obs_startmjds = np.array(obs_startmjds)

    obs_start_phases = get_phase(obs_startmjds, period, ref_mjd=ref_mjd)
    hist, bin_edges_obs = np.histogram(obs_start_phases, bins=nbins)

    obs_start_phases_dict = {}
    duration_per_phase_dict = {}
    burst_per_phase_dict = {}
    duration_per_phase_tot = np.empty(nbins)
    for k in obs_startmjds_dict.keys():
        obs_start_phases_dict[k] = get_phase(np.array(obs_startmjds_dict[k]),
                                             period, ref_mjd=ref_mjd)
        durations = np.array(obs_duration_dict[k])
        start_phases = obs_start_phases_dict[k]

        d_hist = []
        for i in range(len(bin_edges_obs)):
            if i>0:
                dur = durations[(start_phases < bin_edges_obs[i]) &
                        (start_phases > bin_edges_obs[i-1])].sum()
                d_hist.append(dur)
                duration_per_phase_tot[i-1] += dur
        duration_per_phase_dict[k] = np.array(d_hist)

    obs_duration = np.array(obs_duration)
    duration_hist = []
    for i in range(len(bin_edges_obs)):
        if i>0:
            duration_hist.append(
                    obs_duration[(obs_start_phases < bin_edges_obs[i]) &
                    (obs_start_phases > bin_edges_obs[i-1])].sum())

    duration_hist = np.array(duration_hist)
    bin_mids = (bin_edges_obs[:-1] + bin_edges_obs[1:])/2
    phase_lst = []
    for i,k in enumerate(burst_dict.keys()):
        print("phase list", k, len(burst_dict[k]))
        phase_lst.append(list(get_phase(np.array(burst_dict[k]), period,
                ref_mjd=ref_mjd)))
        burst_per_phase_dict[k], _ = np.histogram(phase_lst[-1],
                bins=nbins, range=(0,1))

    phase_tot = [p for l in phase_lst for p in l]
    phase_tot.sort()
    burst_tot, _ = np.histogram(phase_tot, bins=nbins, range=(0,1))

    # PRINTING AVERAGE RATE PER INSTRUMENT
    for i,k in enumerate(burst_dict.keys()):
        tobs = np.sum(obs_duration_dict[k])
        nbursts = len(burst_dict[k])
        rate = nbursts / tobs
        print("Average rate {}: {:.3f} / h".format(k, rate))

        # off = np.where(burst_per_phase_dict[k] == 0)[0]
        # on = np.where(burst_per_phase_dict[k] > 0)[0]
        # print("Hours Apertif observed TOTAL: {:.2f}".format(
        #         np.sum(duration_per_phase_dict[k])))
        # print("Hours Apertif observed during on phase: {:.2f}".format(
        #         np.sum(duration_per_phase_dict[k][on])))
        # print("Hours Apertif observed during off phase: {:.2f}".format(
        #         np.sum(duration_per_phase_dict[k][off])))

    # DEFINING COLORS
    cm = plt.cm.get_cmap('Spectral_r')

    burst_hist_colors = []
    obs_hist_colors = {}
    if 'uGMRT650' in obs_duration_dict.keys():
        fcen_dict['uGMRT650'] = 1000
    for i,k in enumerate(obs_duration_dict.keys()):
        freq = np.log10(fcen_dict[k])
        col = (np.log10(max_freq)-freq)/(np.log10(max_freq)-np.log10(min_freq))
        color = cm(col)
        print(k, mpl.colors.to_hex(color))
        if k in burst_dict.keys():
            burst_hist_colors.append(color)
        obs_hist_colors[k] = color
    rate_colors = {
            'high': cm((np.log10(max_freq)-np.log10(1800))/(np.log10(max_freq)-np.log10(min_freq))),
            'middle': cm((np.log10(max_freq)-np.log10(500))/(np.log10(max_freq)-np.log10(min_freq))),
            'low': cm((np.log10(max_freq)-np.log10(300))/(np.log10(max_freq)-np.log10(min_freq)))
            }
    if 'uGMRT650' in obs_duration_dict.keys():
        fcen_dict['uGMRT650'] = 650

    # PLOTTING
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9,7),
            gridspec_kw={'height_ratios': [1,1]})
    ax1 = ax[0]
    yhist,xhist,_ = ax1.hist(phase_lst, bins=bin_edges_obs, stacked=True,
            density=False, label=burst_dict.keys(),
            edgecolor='black', linewidth=0.5, color=burst_hist_colors)

    ax1.set_ylabel('N. Bursts')
    ax1.set_xlim(0,1)
    print("YLIM", 0, int(yhist[-1].max()*1.1))
    ax1.set_ylim(0, max(int(yhist[-1].max()*1.1), 4))
    ax1.legend(loc=2)
    ax1.text(-0.07, 0.95, "a", transform=ax1.transAxes, weight='bold')

    ax2 = ax[1]
    cum_ds = np.zeros(nbins)
    for i, k in enumerate(duration_per_phase_dict):
        d = duration_per_phase_dict[k]
        ax2.bar(bin_edges_obs[:-1], d, width=bin_edges_obs[1]-bin_edges_obs[0],
                align='edge', bottom=cum_ds, alpha=1,
                label="{} {:d} MHz".format(k, int(fcen_dict[k])),
                edgecolor='black', linewidth=0.2, color=obs_hist_colors[k])
        cum_ds += d
    ax2.set_xlabel('Phase')
    ax2.set_ylabel('Obs. Duration (h)')
    ax2.legend(loc=2)
    ax2.text(-0.07, 0.95, "b", transform=ax2.transAxes, weight='bold')
    plt.tight_layout()

    if save:
        print('Plot saved: ./burst_obs_phase_hist.png')
        plt.savefig('./burst_obs_phase_hist.png', pad_inches=0,
                bbox_inches='tight', dpi=200)
        plt.savefig('./burst_obs_phase_hist.pdf', pad_inches=0,
                bbox_inches='tight', dpi=200)
    if show:
        plt.show()

    # SAVING COUNTS, OBS_DURATION AND PHASE BIN
    if log:
        print("Writing log")
        dir_out = '/home/ines/Documents/projects/R3/periodicity/burst_phases/'
        with open(dir_out+'counts_per_phase_p{:.2f}.txt'.format(period), 'w') as f:
            f.write("# phase_bin counts chime_counts arts_counts lofar_counts obs_duration chime_duration arts_duration lofar_duration\n")
            for i in range(nbins):
                f.write("{:.3f} {} {} {} {} {:.3f} {:.3f} {:.3f} {:.3f}\n".format(
                        bin_mids[i], burst_tot[i],
                        burst_per_phase_dict["CHIME/FRB"][i],
                        burst_per_phase_dict["Apertif"][i],
                        burst_per_phase_dict["LOFAR"][i],
                        duration_per_phase_tot[i],
                        duration_per_phase_dict["CHIME/FRB"][i],
                        duration_per_phase_dict["Apertif"][i],
                        duration_per_phase_dict["LOFAR"][i]))
        for i,k in enumerate(burst_dict.keys()):
            if k == "CHIME/FRB":
                inst = k.replace("/FRB", "")
            else:
                inst = k
            np.save(dir_out + 'phase_{}_p{:.2f}_f{:.1f}'.format(inst, period,
                    fcen_dict[k]), [burst_dict[k], phase_lst[i]])

def make_obs_phase_plot_csv(data_file, period, ref_mjd=58369.30, nbins=40,
        save=False, show=True, min_freq=900, max_freq=6000):
    """
    Generates burst phase and observation phase distribution plot for a given period.

    :param data_file: csv file with data
    :param period: period to use for phase calculation
    :param ref_mjd: reference MJD to use
    :param nbins: number of bins in the phase histogram
    :param save: to save the plot
    :param show: to show the plot
    """

    burst_dict, fmin_dict, fmax_dict, fcen_dict = open_csv(data_file)

    burst_per_phase_dict = {}
    phase_lst = []
    for i,k in enumerate(burst_dict.keys()):
        print("phase list", k, len(burst_dict[k]))
        phase_lst.append(list(get_phase(np.array(burst_dict[k]), period,
                ref_mjd=ref_mjd)))
        burst_per_phase_dict[k], _ = np.histogram(phase_lst[-1],
                bins=nbins, range=(0,1))

    phase_tot = [p for l in phase_lst for p in l]
    phase_tot.sort()
    burst_tot, _ = np.histogram(phase_tot, bins=nbins, range=(0,1))

    # DEFINING COLORS
    cm = plt.cm.get_cmap('Spectral_r')

    # burst_hist_colors = {}
    burst_hist_colors = []
    for i,k in enumerate(burst_dict.keys()):
        freq = np.log10(fcen_dict[k])
        col = (np.log10(max_freq)-freq)/(np.log10(max_freq)-np.log10(min_freq))
        # freq = fcen_dict[k]
        # col = (max_freq-freq)/(max_freq-min_freq))
        color = cm(col)
        # burst_hist_colors[k] = color
        burst_hist_colors.append(color)

    # PLOTTING
    fig, ax1 = plt.subplots(1, 1, sharex=True, figsize=(9,7))
    yhist,xhist,_ = ax1.hist(phase_lst, bins=nbins, range=(0,1), stacked=True,
            density=False, label=burst_dict.keys(),
            edgecolor='black', linewidth=0.5, color=burst_hist_colors)

    ax1.set_ylabel('N. Bursts')
    ax1.set_xlim(0,1)
    ax1.set_ylim(0, int(yhist[-1].max()*1.1))
    ax1.legend(loc=2)

    if save:
        print('Plot saved: ./burst_obs_phase_hist.png')
        plt.savefig('./burst_obs_phase_hist.png', pad_inches=0,
                bbox_inches='tight', dpi=200)
        plt.savefig('./burst_obs_phase_hist.pdf', pad_inches=0,
                bbox_inches='tight', dpi=200)
    if show:
        plt.show()

def make_obstime_plot(data_file, period, ref_mjd=58369.30, save=False,
        show=False, max_freq=2500, min_freq=200):
    """
    Generates observation exposure plot

    :param data_file: json file with data
    :param period: period to use for phase calculation
    :param ref_mjd: reference MJD to use
    :param cmap: matplotlib colormap to use
    :param save: to save the plot
    :param show: to show the plot
    """

    burst_dict, snr_dict, obs_duration_dict, obs_startmjds_dict, fmin_dict, fmax_dict, fcen_dict = open_json(data_file)

    # Defining duty cycle
    frequency_hours = '%fH' % (24 * period)
    t = Time(ref_mjd, format='mjd')
    t0 = t+((period/2)*u.day)
    tf = datetime.datetime.now()

    t0_low = t+((period/2)*u.day) - (0.16 * period * u.day)
    t0_high = t+((period/2)*u.day) + (0.16 * period * u.day)

    df_period = [t0]
    df_duty_low = [t0_low]
    df_duty_high = [t0_high]
    t_activity, t_low, t_high = t0, t0_low, t0_high
    while t_activity < tf:
        t_activity += period
        t_low += period
        t_high += period
        df_period.append(t_activity)
        df_duty_low.append(t_low)
        df_duty_high.append(t_high)

    n_periods = len(df_period)

    # DEFINING COLORS
    cm = plt.cm.get_cmap('Spectral_r')
    burst_hist_colors = []
    obs_hist_colors = {}
    for i,k in enumerate(obs_duration_dict.keys()):
        freq = np.log10(fcen_dict[k])
        col = (np.log10(max_freq)-freq)/(np.log10(max_freq)-np.log10(min_freq))
        # c = i/len(obs_duration_dict.keys())
        color = cm(col)
        # if k in burst_dict.keys():
        #     burst_hist_colors.append(color)
        obs_hist_colors[k] = color

    # PLOTTING
    fig = plt.figure(figsize=(7,7))
    gs = gridspec.GridSpec(2,1, wspace=0.01, height_ratios=[3,1])

    ax1 = fig.add_subplot(gs[0, 0]) #ax[0]
    for i,k in enumerate(burst_dict.keys()):
        ax1.scatter(burst_dict[k], snr_dict[k],
                color=obs_hist_colors[k], label=k, marker='o', edgecolor='k',
                linewidth=0.5, zorder=10, s=12)

    max_snr = max([m for k in snr_dict.keys()
            for m in snr_dict[k]])*1.1
    ax1.set_ylim(0, max_snr)
    ax1.set_ylabel('SNR')


    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1) #ax[1]
    for i, k in enumerate(obs_duration_dict.keys()):
        #d = duration_per_phase_dict[k]
        # ax.scatter(obs_startmjds_dict[k],
        #         [fcen_dict[k] for i in range(len(obs_startmjds_dict[k]))],
        #         color=obs_hist_colors[i])
        obs_patches = []
        for j,start in enumerate(obs_startmjds_dict[k]):
            obs = Rectangle((start,fmin_dict[k]), obs_duration_dict[k][j]/24,
                    fmax_dict[k]-fmin_dict[k])
            obs_patches.append(obs)
        pc = PatchCollection(obs_patches, facecolor=obs_hist_colors[k],
                alpha=0.7, edgecolor=obs_hist_colors[k], label=k)
        ax2.add_collection(pc)

    max_mjdstart = max([m for k in obs_startmjds_dict.keys()
            for m in obs_startmjds_dict[k]])
    min_mjdstart = min([m for k in obs_startmjds_dict.keys()
            for m in obs_startmjds_dict[k]])
    max_freq = max(fmax_dict.values())+1e3
    min_freq = min(fmin_dict.values())-10
    ax2.set_xlim(int(min_mjdstart-2), int(max_mjdstart+2))
    ax2.set_ylim(min_freq, max_freq)
    ax2.set_yscale('log')
    ax2.set_xlabel('MJD')
    ax2.set_ylabel('Frequency (MHz)')

    # duty cycle
    for low, high in zip(df_duty_low, df_duty_high):
        ax1.axvspan(low.value, high.value, facecolor='#0f0f0f', alpha=0.1)
        ax2.axvspan(low.value, high.value, facecolor='#0f0f0f', alpha=0.1)
    for peak in df_period:
        ax1.vlines(peak.value, [0 for i in range(n_periods)],
                [max_snr for i in range(n_periods)], linestyles='dashed', alpha=0.2)
        ax2.vlines(peak.value, [min_freq for i in range(n_periods)],
                [max_freq for i in range(n_periods)], linestyles='dashed', alpha=0.2)


    ax1.legend()
    plt.show()

def cycle_phase_plot(data_file, period, ref_mjd=58369.30, save=False,
        show=False, min_freq=200, max_freq=2500, nbins=40, obs_t=False):
    """
    Generates plot with bursts detected in cycle vs. phase

    :param data_file: json file with data
    :param period: period to use for phase calculation
    :param ref_mjd: reference MJD to use
    :param cmap: matplotlib colormap to use
    :param save: to save the plot
    :param show: to show the plot
    """

    burst_dict, snr_dict, obs_duration_dict, obs_startmjds_dict, fmin_dict, fmax_dict, fcen_dict = open_json(data_file)

    try:
        frb = data_file.split('/')[-1].split('_')[0]
    except:
        frb = 'FRB'
    # Defining phase and cycle
    l = {}
    c = {}
    for k in burst_dict:
        l[k] = get_phase(burst_dict[k], period, ref_mjd=ref_mjd)
        c[k] = get_cycle(burst_dict[k], period, ref_mjd=ref_mjd)
    n_cycles =  int(max([m for k in c.keys() for m in c[k]]))+1

    obs_start_phases = {}
    obs_start_cycles = {}
    obs_duration_phase = {}
    burst_per_phase_dict = {}
    phase_lst = []
    for i,k in enumerate(burst_dict.keys()):
        obs_start_phases[k] = get_phase(obs_startmjds_dict[k], period,
                ref_mjd=ref_mjd)
        obs_start_cycles[k] = get_cycle(obs_startmjds_dict[k], period,
                ref_mjd=ref_mjd)
        obs_duration_phase[k] = np.array(obs_duration_dict[k])/(24*period)
        phase_lst.append(list(get_phase(np.array(burst_dict[k]), period,
                ref_mjd=ref_mjd)))
        burst_per_phase_dict[k], _ = np.histogram(phase_lst[-1],
                bins=nbins, range=(0,1))

    phase_tot = [p for l in phase_lst for p in l]
    phase_tot.sort()
    burst_tot, _ = np.histogram(phase_tot, bins=nbins, range=(0,1))

    flat_phase_lst = [p for phase in phase_lst for p in phase]
    print(flat_phase_lst)
    print("Phase min:     {:.2f}".format(min(flat_phase_lst)))
    print("Phase max:     {:.2f}".format(max(flat_phase_lst)))
    print("Reference MJD: {}".format(ref_mjd))

    # Defining duty cycle
    frequency_hours = '%fH' % (24 * period)
    t = Time(ref_mjd, format='mjd')
    t0 = t+((period/2)*u.day)
    tf = datetime.datetime.now()

    t0_low = t+((period/2)*u.day) - (2.6 * u.day)
    t0_high = t+((period/2)*u.day) + (2.6 * u.day)

    df_period = [t0]
    df_duty_low = [t0_low]
    df_duty_high = [t0_high]
    t_activity, t_low, t_high = t0, t0_low, t0_high
    while t_activity < tf:
        t_activity += period
        t_low += period
        t_high += period
        df_period.append(t_activity)
        df_duty_low.append(t_low)
        df_duty_high.append(t_high)

    n_periods = len(df_period)

    # DEFINING COLORS
    cm = plt.cm.get_cmap('Spectral_r')
    burst_hist_colors = []
    obs_hist_colors = {}
    for i,k in enumerate(obs_duration_dict.keys()):
        freq = np.log10(fcen_dict[k])
        col = (np.log10(max_freq)-freq)/(np.log10(max_freq)-np.log10(min_freq))
        # c = i/len(obs_duration_dict.keys())
        color = cm(col)
        if k in burst_dict.keys():
            burst_hist_colors.append(color)
        obs_hist_colors[k] = color

    # PLOTTING
    fig = plt.figure(figsize=(6,10))
    gs = gridspec.GridSpec(2,1, hspace=0.01, height_ratios=[1,3])

    # Cycle vs. phase
    ax1 = fig.add_subplot(gs[1, 0]) #ax[0]
    for i,k in enumerate(burst_dict.keys()):
        ax1.scatter(l[k], c[k], color=obs_hist_colors[k],
                edgecolors='k', linewidth=0.5, label=k, zorder=10)
    ax1.hlines(range(n_cycles), [0 for i in range(n_cycles)],
                [1 for i in range(n_cycles)], linestyles='-', alpha=0.1, zorder=0)

    # # Observation times
    if obs_t:
        for i, k in enumerate(obs_duration_dict.keys()):
            obs_patches = []
            for j,s in enumerate(obs_start_phases[k]):
                obs = Rectangle((s, obs_start_cycles[k][j]-0.5),
                        obs_duration_phase[k][j], 1)
                obs_patches.append(obs)
            pc = PatchCollection(obs_patches, facecolor=obs_hist_colors[k],
                    alpha=0.5, edgecolor=obs_hist_colors[k], label=k, zorder=5)
            ax1.add_collection(pc)

    ax1.text(0.05, 0.95, "P = {0} days".format(period),
            transform=ax1.transAxes, verticalalignment='top', fontsize=14)

    ax1.set_xlabel('Phase')
    ax1.set_ylabel('Cycle')
    ax1.set_xlim(0,1)
    ax1.set_ylim(-0.5, n_cycles+0.5)
    ax1.legend()

    # Phase histogram
    ax2 =fig.add_subplot(gs[0, 0])
    yhist,xhist,_ = ax2.hist(phase_lst, bins=nbins, range=(0,1), stacked=True,
            density=False, label=burst_dict.keys(),
            edgecolor='black', linewidth=0.5, color=burst_hist_colors)

    ax2.set_xlim(0,1)
    ax2.set_xticklabels([])
    ax2.set_title(frb)

    if save:
        plt_out = '/home/ines/Documents/projects/repeaters/figures/{}_cycle_p{:.2f}.png'.format(frb, period)
        print("Figure saved", plt_out)
        plt.savefig(plt_out, pad_inches=0, bbox_inches='tight', dpi=200)
    plt.show()

def make_plot_all(data_file, period, ref_mjd=58369.30, nbins=40, save=False,
        show=False, log=False, min_freq=200, max_freq=2500):
    """
    Generates burst phase and observation phase distribution plot for a given period.

    :param data_file: json file with data
    :param period: period to use for phase calculation
    :param ref_mjd: reference MJD to use
    :param nbins: number of bins in the phase histogram
    :param save: to save the plot
    :param show: to show the plot
    """

    burst_dict, snr_dict, obs_duration_dict, obs_startmjds_dict, fmin_dict, fmax_dict, fcen_dict = open_json(data_file)

    bursts = []
    for k in burst_dict.keys():
        bursts = bursts + burst_dict[k]

    obs_duration = []
    for k in obs_duration_dict.keys():
        obs_duration = obs_duration + obs_duration_dict[k]

    obs_startmjds = []
    for k in obs_startmjds_dict.keys():
        obs_startmjds = obs_startmjds + obs_startmjds_dict[k]

    bursts = np.array(bursts)
    obs_duration = np.array(obs_duration)
    obs_startmjds = np.array(obs_startmjds)

    obs_start_phases = get_phase(obs_startmjds, period, ref_mjd=ref_mjd)
    hist, bin_edges_obs = np.histogram(obs_start_phases, bins=nbins)

    obs_start_phases_dict = {}
    duration_per_phase_dict = {}
    burst_per_phase_dict = {}
    duration_per_phase_tot = np.empty(nbins)
    for k in obs_startmjds_dict.keys():
        obs_start_phases_dict[k] = get_phase(np.array(obs_startmjds_dict[k]),
                                             period, ref_mjd=ref_mjd)
        durations = np.array(obs_duration_dict[k])
        start_phases = obs_start_phases_dict[k]

        d_hist = []
        for i in range(len(bin_edges_obs)):
            if i>0:
                dur = durations[(start_phases < bin_edges_obs[i]) &
                        (start_phases > bin_edges_obs[i-1])].sum()
                d_hist.append(dur)
                duration_per_phase_tot[i-1] += dur
        duration_per_phase_dict[k] = np.array(d_hist)

    obs_duration = np.array(obs_duration)
    duration_hist = []
    for i in range(len(bin_edges_obs)):
        if i>0:
            duration_hist.append(
                    obs_duration[(obs_start_phases < bin_edges_obs[i]) &
                    (obs_start_phases > bin_edges_obs[i-1])].sum())

    duration_hist = np.array(duration_hist)
    bin_mids = (bin_edges_obs[:-1] + bin_edges_obs[1:])/2
    phase_lst = []
    for k in burst_dict.keys():
        phase_lst.append(list(get_phase(np.array(burst_dict[k]), period,
                ref_mjd=ref_mjd)))
        burst_per_phase_dict[k], _ = np.histogram(phase_lst[-1],
                bins=nbins, range=(0,1))

    phase_tot = [p for l in phase_lst for p in l]
    phase_tot.sort()
    burst_tot, _ = np.histogram(phase_tot, bins=nbins, range=(0,1))

    # Defining phase and cycle
    l = {}
    c = {}
    for k in burst_dict:
        l[k] = get_phase(burst_dict[k], period, ref_mjd=ref_mjd)
        c[k] = get_cycle(burst_dict[k], period, ref_mjd=ref_mjd)
    n_cycles =  int(max([m for k in c.keys() for m in c[k]]))+1

    obs_start_phases = {}
    obs_start_cycles = {}
    obs_duration_phase = {}
    for k in obs_startmjds_dict.keys():
        obs_start_phases[k] = get_phase(obs_startmjds_dict[k], period,
                ref_mjd=ref_mjd)
        obs_start_cycles[k] = get_cycle(obs_startmjds_dict[k], period,
                ref_mjd=ref_mjd)
        obs_duration_phase[k] = np.array(obs_duration_dict[k])/(24*period)

    # DEFINING COLORS
    cm = plt.cm.get_cmap('Spectral_r')
    burst_hist_colors = []
    obs_hist_colors = {}
    for i,k in enumerate(obs_duration_dict.keys()):
        freq = np.log10(fcen_dict[k])
        col = (np.log10(max_freq)-freq)/(np.log10(max_freq)-np.log10(min_freq))
        # c = i/len(obs_duration_dict.keys())
        color = cm(col)
        if k in burst_dict.keys():
            burst_hist_colors.append(color)
        obs_hist_colors[k] = color

    # PLOTTING
    fig = plt.figure(figsize=(7,7))
    gs = gridspec.GridSpec(3,1, wspace=0.01, height_ratios=[1,2,1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(phase_lst, bins=bin_edges_obs, stacked=True, density=False, label=burst_dict.keys(),
             edgecolor='black', linewidth=0.5, color=burst_hist_colors)

    ax1.set_xlabel('Phase')
    ax1.set_ylabel('N. Bursts')
    ax1.set_xlim(0,1)
    ax1.legend()
    ax1.tick_params(axis='x', which='both', direction='in', bottom=True,
            top=True)
    ax1.tick_params(axis='y', which='both', direction='in', left=True,
            right=True)

    ax2 = fig.add_subplot(gs[2, 0], sharex=ax1)
    cum_ds = np.zeros(nbins)
    for i, k in enumerate(duration_per_phase_dict):
        d = duration_per_phase_dict[k]
        ax2.bar(bin_edges_obs[:-1], d, width=bin_edges_obs[1]-bin_edges_obs[0],
                align='edge', bottom=cum_ds, alpha=1,
                label="{} {:d} MHz".format(k, int(fcen_dict[k])),
                edgecolor='black', linewidth=0.2, color=obs_hist_colors[k])
        cum_ds += d
    ax2.set_xlabel('Phase')
    ax2.set_ylabel('Obs. Duration (h)')
    ax2.legend()
    ax2.tick_params(axis='x', which='both', direction='in', bottom=True,
            top=True)
    ax2.tick_params(axis='y', which='both', direction='in', left=True,
            right=True)
    plt.tight_layout()

    ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)
    for i,k in enumerate(burst_dict.keys()):
        ax3.scatter(l[k], c[k], color=obs_hist_colors[k],
                edgecolors='k', linewidth=0.5, label=k, zorder=10)
    ax3.hlines(range(n_cycles), [0 for i in range(n_cycles)],
                [1 for i in range(n_cycles)], linestyles='-', alpha=0.1, zorder=0)

    for i, k in enumerate(obs_duration_dict.keys()):
        obs_patches = []
        for j,s in enumerate(obs_start_phases[k]):
            obs = Rectangle((s, obs_start_cycles[k][j]-0.5),
                    obs_duration_phase[k][j], 1)
            obs_patches.append(obs)
        pc = PatchCollection(obs_patches, facecolor=obs_hist_colors[k],
                alpha=0.5, edgecolor=obs_hist_colors[k], label=k, zorder=5)
        ax3.add_collection(pc)

    ax3.text(0.05, 0.95, "P = {0} days".format(period),
            transform=ax1.transAxes, verticalalignment='top', fontsize=14)

    ax3.set_ylabel('Cycle')
    ax3.set_ylim(-0.5, n_cycles+0.5)
    ax3.legend()
    ax3.tick_params(axis='x', which='both', direction='in', bottom=True,
            top=True)
    ax3.tick_params(axis='y', which='both', direction='in', left=True,
            right=True)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)

    if save:
        plt.savefig('./R3_obs_detections.png', pad_inches=0,
                bbox_inches='tight', dpi=200)
        plt.savefig('./R3_obs_detections.pdf', pad_inches=0,
                bbox_inches='tight', dpi=200)
    if show:
        plt.show()

def make_snr_plot(data_file, period, ref_mjd=58369.30, save=False,
        show=False, log=False, min_freq=200, max_freq=2500):
    """
    Generates burst phase and observation phase distribution plot for a given period.

    :param data_file: json file with data
    :param period: period to use for phase calculation
    :param ref_mjd: reference MJD to use
    :param nbins: number of bins in the phase histogram
    :param save: to save the plot
    :param show: to show the plot
    """

    burst_dict, snr_dict, obs_duration_dict, obs_startmjds_dict, fmin_dict, fmax_dict, fcen_dict = open_json(data_file)

    # Defining phase and cycle
    l = {}
    c = {}
    for k in burst_dict:
        l[k] = get_phase(burst_dict[k], period, ref_mjd=ref_mjd)
        c[k] = get_cycle(burst_dict[k], period, ref_mjd=ref_mjd)
    n_cycles =  int(max([m for k in c.keys() for m in c[k]]))+1

    obs_start_phases = {}
    obs_start_cycles = {}
    obs_duration_phase = {}
    for k in obs_startmjds_dict.keys():
        obs_start_phases[k] = get_phase(obs_startmjds_dict[k], period, ref_mjd=ref_mjd)
        obs_start_cycles[k] = get_cycle(obs_startmjds_dict[k], period, ref_mjd=ref_mjd)
        obs_duration_phase[k] = np.array(obs_duration_dict[k])/(24*period)

    # DEFINING COLORS
    cm = plt.cm.get_cmap('Spectral_r')
    burst_hist_colors = []
    obs_hist_colors = {}
    for i,k in enumerate(obs_duration_dict.keys()):
        freq = np.log10(fcen_dict[k])
        col = (np.log10(max_freq)-freq)/(np.log10(max_freq)-np.log10(min_freq))
        # c = i/len(obs_duration_dict.keys())
        color = cm(col)
        if k in burst_dict.keys():
            burst_hist_colors.append(color)
        obs_hist_colors[k] = color

    # PLOTTING
    fig = plt.figure(figsize=(7,7))
    gs = gridspec.GridSpec(1,1, wspace=0.01, height_ratios=[1])

    ax1 = fig.add_subplot(gs[0, 0])
    for i,k in enumerate(burst_dict.keys()):
        ax1.scatter(l[k], snr_dict[k], color=obs_hist_colors[k],
                edgecolors='k', linewidth=0.5, label=k, zorder=10)

    ax1.text(0.05, 0.95, "P = {0} days".format(period),
            transform=ax1.transAxes, verticalalignment='top', fontsize=14)
    ax1.set_xlabel('Phase')
    ax1.set_ylabel('SNR')
    ax1.set_xlim(0,1)
    ax1.legend()
    ax1.tick_params(axis='x', which='both', direction='in', bottom=True,
            top=True)
    ax1.tick_params(axis='y', which='both', direction='in', left=True,
            right=True)

    if save:
        plt.savefig('./R3_obs_detections.png', pad_inches=0,
                bbox_inches='tight', dpi=200)
        plt.savefig('./R3_obs_detections.pdf', pad_inches=0,
                bbox_inches='tight', dpi=200)
    if show:
        plt.show()


def make_phase_plot(data_file, period, ref_mjd=None, nbins=40, cmap=None,
        title=None, save=False, show=False):
    """
    Generates burst phase distribution plot at a given period.

    :param data_file: json file with data
    :param period: period to use for phase calculation
    :param ref_mjd: reference MJD to use
    :param nbins: number of bins in the phase histogram
    :param cmap: matplotlib colormap to use
    :param title: title of the plot
    :param save: to save the plot
    :param show: to show the plot
    """
    with open(data_file, 'r') as f:
        data = json.load(f)

    burst_dict = data['bursts']
    all_bursts = []
    for k in burst_dict.keys():
        all_bursts += burst_dict[k]

    if not ref_mjd:
        ref_mjd = np.min(all_bursts)

    l = []
    for k in burst_dict:
        l.append(get_phase(burst_dict[k], period, ref_mjd=ref_mjd))

    refphases = np.linspace(0,1,1000)
    _, bin_edges = np.histogram(refphases, bins=nbins)

    names = burst_dict.keys()
    num_colors = len(names)

    plt.figure(figsize=(7,7))

    if not cmap:
        if num_colors < 20:
            cmap = 'tab20'
            colors = plt.get_cmap(cmap).colors[:num_colors]
        else:
            cmap = 'jet'
            cm = plt.get_cmap(cmap)
            colors = [cm(1.*i/num_colors) for i in range(num_colors)]

    _ = plt.hist(l, bins=bin_edges, stacked=True, density=False, label=names, edgecolor='black',
                 linewidth=0.5, color=colors)
    plt.xlabel('Phase')
    plt.ylabel('No. of Bursts')
    if not title:
        title = f'Burst phases of {len(all_bursts)} bursts at a period of {period} days'
    plt.title(title)
    plt.legend()
    if save:
        plt.savefig('burst_phase_histogram.png', pad_inches=0,
                bbox_inches='tight', dpi=200)
    if show:
        plt.show()

def make_snr_time_plot(data_file, period, width, n_min=0, n_max=30,
        min_freq=200, max_freq=2500, ref_mjd=58369.30,
        cmap=None, title=None, save=False, show=False):

    """
    Generates burst phase distribution plot at a given period.

    :param data_file: json file with data
    :param period: period to use for phase calculation
    :param width: width of the activity window in phase units
    :param ref_mjd: reference MJD to use
    :param cmap: matplotlib colormap to use
    :param title: title of the plot
    :param save: to save the plot
    :param show: to show the plot
    """

    burst_dict, snr_dict, obs_duration_dict, obs_startmjds_dict, fmin_dict, fmax_dict, fcen_dict = open_json(data_file)

    # Defining duty cycle
    frequency_hours = '%fH' % (24 * period)
    t = Time(ref_mjd, format='mjd')
    t0 = t+((period/2)*u.day)
    tf = Time('2020-12-15T00:00:00', format='isot') #datetime.datetime.now()

    t0_low = t+((period/2)*u.day) - (2.6 * u.day)
    t0_high = t+((period/2)*u.day) + (2.6 * u.day)

    df_period = [t0]
    df_duty_low = [t0_low]
    df_duty_high = [t0_high]
    t_activity, t_low, t_high = t0, t0_low, t0_high
    while t_activity <= tf:
        t_activity += period
        t_low += period
        t_high += period
        df_period.append(t_activity)
        df_duty_low.append(t_low)
        df_duty_high.append(t_high)

    n_periods = len(df_period)
    print(n_periods)

    t_min = ref_mjd + period * n_min
    t_max = ref_mjd + period * n_max

    # Instruments to show
    telescopes = ['CHIME/FRB']

    # DEFINING COLORS
    cm = plt.cm.get_cmap('Spectral_r')
    obs_hist_colors = {}
    for i,k in enumerate(obs_duration_dict.keys()):
        freq = np.log10(fcen_dict[k])
        c = (np.log10(max_freq)-freq)/(np.log10(max_freq)-np.log10(min_freq))
        color = cm(c)
        obs_hist_colors[k] = color

    # PLOTTING
    fig = plt.figure(figsize=(15,8))
    gs = gridspec.GridSpec(2,1, hspace=0., height_ratios=[5,1])

    ax1 = fig.add_subplot(gs[0, 0]) #ax[0]
    for i,k in enumerate(telescopes):
        inst = ''.join([i for i in k if not i.isdigit()])
        print(inst, len(burst_dict[k]), len(snr_dict[k]))
        if k in ['Apertif', 'LOFAR']:
            s = 50
        else:
            s = 25
        ax1.scatter(burst_dict[k], snr_dict[k],
                color=obs_hist_colors[k], label="{} {} MHz".format(inst, int(fcen_dict[k])),
                marker='o', s=s, zorder=10, edgecolors='k', linewidth=0.5)

    max_snr = max([m for k in snr_dict.keys()
            for m in snr_dict[k]])*1.1
    ax1.set_ylim(0, max_snr)
    ax1.set_ylabel('SNR')


    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1) #ax[1]
    for i, k in enumerate(telescopes):
        obs_patches = []
        inst = ''.join([i for i in k if not i.isdigit()])
        for j,start in enumerate(obs_startmjds_dict[k]):
            #obs = Rectangle((start,i), max(0.2,obs_duration_dict[k][j]/24), 1)
            obs = Rectangle((start,i), obs_duration_dict[k][j]/24, 1)
            obs_patches.append(obs)
        pc = PatchCollection(obs_patches, facecolor=obs_hist_colors[k],
                alpha=1, edgecolor=obs_hist_colors[k], label=inst, linewidth=0.1)
        ax2.add_collection(pc)

    max_mjdstart = max([m for k in obs_startmjds_dict.keys()
            for m in obs_startmjds_dict[k]])
    min_mjdstart = min([m for k in obs_startmjds_dict.keys()
            for m in obs_startmjds_dict[k]])
    max_f = max(fmax_dict.values())+1e3
    min_f = min(fmin_dict.values())-10
    ax2.set_xlim(t_min, t_max)
    ax2.set_ylim(-0.5, len(telescopes)+1)
    ax2.set_xlabel('MJD')
    #ax2.set_yticks([])
    ax2.set_yticks([i+0.5 for i in range(len(telescopes))])
    ax2.set_yticklabels(telescopes, fontsize=10)
    ax2.tick_params(axis='y', which='both', length=0)

    # duty cycle
    for low, high in zip(df_duty_low, df_duty_high):
        ax1.axvspan(low.value, high.value, facecolor='#0f0f0f', alpha=0.1)
    for ii,peak in enumerate(df_period):
        ax1.vlines(peak.value, 0, max_snr, linestyles='dashed', alpha=0.2)
        if peak.value >= t_min and peak.value <= t_max:
            ax1.text(peak.value, 128, ii, horizontalalignment='center')

    ax1.legend(loc=1)
    plt.setp(ax1.get_xticklabels(), visible=False)
    for ax in (ax1,ax2):
        ax.tick_params(axis='x', which='both', direction='in', bottom=True,
                top=True)
        ax.tick_params(axis='y', which='both', direction='in', left=True,
                right=True)
    plt.show()

    # plt.savefig(plt_out, pad_inches=0, bbox_inches='tight', dpi=200)
    # print('Plot saved:', plt_out)


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Commands for tauscat code')
    parser.add_argument('infile', nargs='?',
    					help='The input json file.')
    parser.add_argument('-P','--period', type=float, default=16.35,
    					help='FRB period in days.')
    parser.add_argument('-W','--width', type=float, default=0.3,
    					help='FRB activity width in phase units.')
    parser.add_argument('-m','--ref_mjd',type=float, default=58369.30,
    					help='Reference MJD.')
                        # Use 58369.30 to center peak on phase 0.5
    parser.add_argument('-n','--nbins',type=int, default=40,
    					help='Number of phase bins.')
    parser.add_argument('-s','--save', action='store_true', default=False,
    					help='Save plots.')
    parser.add_argument('-p','--show_plot', action='store_true',
    			        default=False, help='Show plots.')
    parser.add_argument('-l','--output_log', action='store_true',
    			        default=False, help='Make output log.')
    parser.add_argument('-w','--which', default='phase', type=str,
                        help='Which plot to make. (phase|obstime|cycle|all|snr|time)')

    args = parser.parse_args()

    print(args.infile)

    plt.style.use('/home/ines/.config/matplotlib/stylelib/paper.mplstyle')
    plt.rcParams.update({'legend.fontsize': 8})

    if args.which == 'phase':
        if '.json' in args.infile:
            make_obs_phase_plot(args.infile, args.period, ref_mjd=args.ref_mjd,
                    nbins=args.nbins, save=args.save, show=args.show_plot,
                    log=args.output_log)
        elif '.csv' in args.infile:
            make_obs_phase_plot_csv(args.infile, args.period,
                    ref_mjd=args.ref_mjd, nbins=args.nbins, save=args.save,
                    show=args.show_plot)
    elif args.which == 'obstime':
        make_obstime_plot(args.infile, args.period, ref_mjd=args.ref_mjd,
                save=args.save, show=args.show_plot)
    elif args.which == 'cycle':
        cycle_phase_plot(args.infile, args.period, ref_mjd=args.ref_mjd,
                save=args.save, show=args.show_plot, nbins=args.nbins)
    elif args.which == 'all':
        make_plot_all(args.infile, args.period, ref_mjd=args.ref_mjd,
                nbins=args.nbins, save=args.save, show=args.show_plot)
    elif args.which == 'snr':
        make_snr_plot(args.infile, args.period, ref_mjd=args.ref_mjd,
                save=args.save, show=args.show_plot)
    elif args.which == 'time':
        make_snr_time_plot(args.infile, args.period, args.width,
                n_min=0, n_max=30, ref_mjd=args.ref_mjd)
