import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tqdm, json
from frbpa.search import pr3_search, riptide_search, p4j_search
from frbpa.utils import get_phase

# Defining functions
def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

def half_max_x(x, y):
    """Computes x position of the half maximum of a peak
    """
    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[-1], half)]

# Input files
indir = '/home/ines/Documents/projects/R3/periodicity/'
infile = indir + 'r3all_data.json'
with open(infile, 'r') as f:
    r3_data = json.load(f)

pminsearch = 0.03
pmaxsearch = 20
# pminsearch = 1.57
# pmaxsearch = 65
pres = 10000

# Output files
outdir = '/home/ines/Documents/projects/R3/periodicity/periodograms/'
plt_out = outdir + 'periodograms.pdf'

telescope_groups = ['all', ['Apertif'], ['CHIME'], ['Apertif', 'CHIME']]
periodogram_names = ['all', 'Apertif', 'CHIME', 'CHIME_Apertif']
#telescopes = ['Apertif', 'CHIME']
#telescopes = 'all'
fig = plt.figure(figsize=(13,10))
gs = gridspec.GridSpec(3,len(telescope_groups), hspace=0.05, wspace=0.05)
colors = ['#577590', '#90be6d', '#f8961e', '#f94144']

plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'xtick.top': True,
        'ytick.right': True,
        'lines.linewidth': 0.5,
        'lines.markersize': 5,
        'legend.fontsize': 12,
        # 'legend.borderaxespad': 0,
        # 'legend.frameon': True,
        'legend.loc': 'upper right'})

jj = 1
for ii,telescopes in enumerate(telescope_groups):
    # Opening files
    burst_dict = r3_data['bursts']
    startmjds_dict = r3_data['obs_startmjds']
    duration_dict = r3_data['obs_duration']

    bursts = []
    if telescopes == 'all':
        for k in burst_dict.keys():
            bursts += burst_dict[k]
    else:
        for k in telescopes:
            bursts += burst_dict[k]
    bursts = np.array(bursts)
    unique_days = np.unique(np.round(bursts))

    startmjds = []
    if telescopes == 'all':
        for k in startmjds_dict.keys():
            startmjds += startmjds_dict[k]
    else:
        for k in telescopes:
            startmjds += startmjds_dict[k]
    startmjds = np.array(startmjds)

    durations = []
    if telescopes == 'all':
        for k in duration_dict.keys():
            durations += duration_dict[k]
    else:
        for k in telescopes:
            durations += duration_dict[k]
    durations = np.array(durations)

    #print('Using {} bursts from {}'.format(len(bursts), telescopes))

    #######################
    # Making periodograms
    #######################
    fwhm = 0.2
    title = periodogram_names[ii].replace('_', '+').upper()
    print("{} & {} &".format(title, len(bursts)), end=' ')
    #--------------------------------
    # Pearson chi-square test (PR3)
    #--------------------------------
    print("# PR3 ",telescopes)
    # rch, p_pr3 = pr3_search(bursts=bursts, obs_mjds=startmjds,
    #         obs_durations=durations, pmin=pminsearch, pmax=pmaxsearch, pres=pres)
    # np.save(outdir + periodogram_names[ii] + '_period_pr3', [rch, p_pr3])
    # Opening existing periodograms
    data = np.load(outdir + periodogram_names[ii] + '_period_pr3.npy')
    rch, p_pr3 = data[0], data[1]

    # Maximum close to 16 days
    # index = np.argmax(rch[np.where(p_pr3>10)]) + len(np.where(p_pr3<=10)[0])
    # p_pr3_max = p_pr3[index]
    # # Errors with FWHM
    # ind_peak = np.where((p_pr3>=12.5) & (p_pr3<=20))[0]
    # hmx = half_max_x(p_pr3[ind_peak], rch[ind_peak])
    # fwhm = hmx[1] - hmx[0]
    # half = max(rch[ind_peak])/2.0
    # pmax = hmx[1] - p_pr3_max
    # pmin = p_pr3_max - hmx[0]
    # print("%.2f\\errors{%.2f}{%.2f} &" % (p_pr3_max, pmax, pmin), end=' ')

    # Plotting
    ax1 = fig.add_subplot(gs[0, ii])
    ax1.plot(p_pr3, rch, color=colors[ii], lw=1)
    #ax1.plot(hmx, [half, half])
    ax1.set_ylim(0,35)
    ax1.set_xticklabels([])


    #-----------------------------------------------
    # Narrowest folded profile, Rajwade+2020 (R20)
    #-----------------------------------------------
    print("# R20 ",telescopes)
    # bursts = np.sort(bursts - np.min(bursts))
    # unique_days = np.unique(np.round(bursts))
    # cont_frac, p_r20 = riptide_search(bursts, pmin=pminsearch, pmax=pmaxsearch,
    #         ts_bin_width=1e-6)
    # np.save(outdir + periodogram_names[ii] + '_period_r20', [cont_frac, p_r20])
    # Opening data
    data = np.load(outdir + periodogram_names[ii] + '_period_r20.npy')
    cont_frac, p_r20 = np.flip(data[0]), np.flip(data[1])

    # Maximum close to 16 days
    # index = np.argmax(cont_frac)
    # p_r20_max = p_r20[index]
    # # Errors with FWHM
    # ind_peak = np.where((p_r20>=14) & (p_r20<=19))[0]
    # hmx = half_max_x(p_r20[ind_peak], cont_frac[ind_peak])
    # fwhm = hmx[1] - hmx[0]
    # half = max(cont_frac[ind_peak])/2.0
    # pmax = hmx[1] - p_r20_max
    # pmin = p_r20_max - hmx[0]
    # print("%.2f\\errors{%.2f}{%.2f} &" % (p_r20_max, pmax, pmin), end=' ')

    # Plotting
    ax2 = fig.add_subplot(gs[1, ii])
    ax2.plot(p_r20, cont_frac, color=colors[ii], lw=1)
    #ax2.plot(hmx, [half, half])
    ax2.set_ylim(0,0.8)
    ax2.set_xticklabels([])


    #--------------------------------------------------------
    # QMI based on Euclidean distance for periodogram (P4J)
    #--------------------------------------------------------
    print("# P4J ",telescopes)
    # periodogram, p_p4j = p4j_search(bursts, pmin=pminsearch, pmax=pmaxsearch,
    #         plot=False, save=False, mjd_err=0.1, pres=1e-2)
    # np.save(outdir + periodogram_names[ii] + '_period_p4j', [periodogram,p_p4j])
    # Opening data
    data = np.load(outdir + periodogram_names[ii] + '_period_p4j.npy')
    periodogram, p_p4j = np.flip(data[0]), np.flip(data[1])
    periodogram = np.abs(periodogram)
    periodogram = periodogram-np.median(periodogram)
    periodogram = periodogram/np.max(periodogram)

    # Maximum close to 16 days
    # index = np.argmax(periodogram[np.where((p_p4j>=15.5) & (p_p4j<=17))[0]])
    # p_p4j_max = p_p4j[index + np.where(p_p4j>=15.5)[0][0]]
    # # Errors with FWHM
    # ind_peak = np.where((p_p4j>=15.5) & (p_p4j<=17))[0]
    # hmx = half_max_x(p_p4j[ind_peak], periodogram[ind_peak])
    # fwhm = hmx[1] - hmx[0]
    # half = max(periodogram[ind_peak])/2.0
    # pmax = hmx[1] - p_p4j_max
    # pmin = p_p4j_max - hmx[0]
    # print("%.2f\\errors{%.2f}{%.2f} \\\\" % (p_p4j_max, pmax, pmin))

    # Plotting
    ax3 = fig.add_subplot(gs[2, ii])
    ax3.plot(p_p4j, periodogram,
            color=colors[ii], lw=1, label=periodogram_names[ii])
    #ax3.plot(hmx, [half, half])
    ax3.set_xlabel('Period (days)')
    ax3.set_ylim(-0.5,1.1)

    #######################
    # Plot setup
    #######################

    ax1.set_title(title)

    if ii == 0:
        ax1.set_ylabel(r'Reduced $\chi^2$ (PR3)')
        ax2.set_ylabel(r'Max. cont. fraction (R20)')
        ax3.set_ylabel(r'Normalised QMIEU (P4J)')
    else:
        ax1.set_yticklabels([])
        ax2.set_yticklabels([])
        ax3.set_yticklabels([])

    for jj,ax in enumerate([ax1, ax2, ax3]):
        kk = ii + jj*len(telescope_groups) + 1
        ax.set_xlim(1.57, 65)
        ax.set_xlim(pminsearch, pmaxsearch)
        ax.set_xscale('log')
        ax.text(0.9, 0.9, chr(ord('@')+kk),
                transform=ax.transAxes)
        # ax.vlines(16.35, -1,50, lw=1, color='gray')
        # ax.vlines(8.175, -1,50, linestyle='--', lw=1, color='gray')
        jj += 1
        for n in range(37):
            pnp = (n/0.99727+1/16.28)**-1
            pnm = (n/0.99727-1/16.28)**-1
            ax.vlines(pnp, -10,50, color='gray', linestyles='-',
                    linewidths=0.4)
            ax.vlines(pnm, -10,50, color='gray', linestyles='dotted',
                    linewidths=0.4)
            #print(n, pnp, pnm)
        ax.label_outer()

    #ax3.legend()

print("Saving figure to ", plt_out)
plt.savefig(plt_out, pad_inches=0, bbox_inches='tight', dpi=200)
plt.show()
