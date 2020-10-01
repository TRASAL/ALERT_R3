import numpy as np
import json
from scipy import stats
from scipy.stats import norm
from frbpa.utils import get_phase, get_cycle, get_params
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Defininig functions
def simulate_bursts(obs_start_phases, obs_start_cycles, obs_duration,
        rate=100, mu=0, sigma=1):

    # Simulate the bursts that would be observed
    # given a normal distribution and the observing times
    cycles = np.unique(obs_start_cycles)
    bursts = np.array([])
    for c in cycles:
        index = np.where(obs_start_cycles == c)
        size = abs(int(np.random.normal(loc=rate, scale=rate/5, size=1)))
        r = norm.rvs(loc=mu, scale=sigma, size=size)
        for i in index[0]:
            p = obs_start_phases[i]
            d = obs_duration[i]
            b = r[(np.where((r >= p) & (r <= p+d)))]
            bursts = np.append(bursts, b)
    return bursts

def simulate_ks(data_json, inst1='ARTS', inst2='CHIME',
        outfile='simulated_ks.txt', P=16.28, ref_mjd=58369.30, rate=100,
        mu=0, sigma=1, trials=100):

    with open(data_json, 'r') as f:
        data = json.load(f)

    burst_dict = data['bursts']
    obs_startmjds = data['obs_startmjds']
    obs_duration = data['obs_duration']

    obs_start_phases_1 = get_phase(obs_startmjds[inst1], P, ref_mjd=ref_mjd)
    obs_start_cycles_1 = get_cycle(obs_startmjds[inst1], P, ref_mjd=ref_mjd)
    obs_duration_1 = [d/(3600*24*P) for d in obs_duration[inst1]]

    obs_start_phases_2 = get_phase(obs_startmjds[inst2], P, ref_mjd=ref_mjd)
    obs_start_cycles_2 = get_cycle(obs_startmjds[inst2], P, ref_mjd=ref_mjd)
    obs_duration_2 = [d/(3600*24*P) for d in obs_duration[inst2]]

    if trials == 1:
        bursts_1 = simulate_bursts(obs_start_phases_1, obs_start_cycles_1,
                obs_duration_1, rate=rate, mu=mu, sigma=sigma)
        bursts_2 = simulate_bursts(obs_start_phases_2, obs_start_cycles_2,
                obs_duration_2, rate=rate, mu=mu, sigma=sigma)

        # Comparing samples
        statistic, pvalue = stats.ks_2samp(bursts_1, bursts_2)

    else:
        file = open(outfile, 'w')
        pvalue = np.array([])
        statistic = np.array([])
        file.write("# statistic pvalue bursts_{} bursts_{}\n".format(inst1, inst2))
        for i in range(int(trials)):
            bursts_1 = simulate_bursts(obs_start_phases_1, obs_start_cycles_1,
                    obs_duration_1, rate=rate, mu=mu, sigma=sigma)
            bursts_2 = simulate_bursts(obs_start_phases_2, obs_start_cycles_2,
                    obs_duration_2, rate=rate, mu=mu, sigma=sigma)

            # Comparing samples
            s,p = stats.ks_2samp(bursts_1, bursts_2)
            statistic, pvalue = np.append(statistic, s), np.append(pvalue, p)
            file.write("{:.6f} {:.6e} {} {}\n".format(s, p,
                    len(bursts_1), len(bursts_2)))
        file.close()

    return statistic, pvalue, bursts_1, bursts_2

# Defining variables
P = 16.29
ref_mjd = 58369.9
chime_rate = 0.32*24*P # bursts/cycle

fndir = '/home/ines/Documents/projects/R3/periodicity/burst_phases/'
fnchime = fndir + 'phase_CHIME_p{}_f600.0.npy'.format(P)
fnarts = fndir + 'phase_ARTS_p{}_f1370.0.npy'.format(P)
fnlofar = fndir + 'phase_LOFAR_p{}_f150.0.npy'.format(P)
data_r3 = np.loadtxt('/home/ines/Documents/projects/R3/periodicity/counts_per_phase.txt')
data_dict = {}
data_dict['CHIME'] = np.load(fnchime)
data_dict['ARTS'] = np.load(fnarts)
data_dict['LOFAR'] = np.load(fnlofar)

# Loading data
data_json = '/home/ines/Documents/projects/R3/periodicity/r3all_data.json'

foutdir = '/home/ines/Documents/projects/R3/periodicity/simulated_ks/'

mu,sigma = norm.fit(data_dict['CHIME'][1])

# statistic, pvalue, bursts_arts, bursts_chime = simulate_ks(data_json,
#         inst1='ARTS', inst2='CHIME', P=P,
#         outfile='simulated_ks_CHIME-ARTS.txt',
#         ref_mjd=ref_mjd, rate=chime_rate, mu=mu, sigma=sigma, trials=10000)

#ref_stat, ref_pvalue = stats.ks_2samp(data_arts[1], data_chime[1])

#fig, ax = plt.subplots(3, 1)
fig = plt.figure(figsize=(8,13))
gs = gridspec.GridSpec(3,1, hspace=0.25, wspace=0.01)

for ii,inst in enumerate([['ARTS', 'LOFAR'], ['ARTS', 'CHIME'], ['CHIME', 'LOFAR']]):

    statistic, pvalue, bursts_lofar, bursts_chime = simulate_ks(data_json,
            inst1=inst[0], inst2=inst[1], P=P,
            outfile=foutdir+'simulated_ks_{}_{}-{}.txt'.format(P, inst[0], inst[1]),
            ref_mjd=ref_mjd, rate=chime_rate, mu=mu, sigma=sigma, trials=1e4)
    ref_stat, ref_pvalue = stats.ks_2samp(data_dict[inst[0]][1],
            data_dict[inst[1]][1])
    print(inst, ref_pvalue)

    # sim_data = np.genfromtxt('simulated_ks_CHIME-ARTS.txt', names=True)
    sim_data = np.genfromtxt(foutdir+'simulated_ks_{}_{}-{}.txt'.format(P,
            inst[0], inst[1]), names=True)
    statistic = sim_data['statistic']
    pvalue = sim_data['pvalue']


    # Plotting
    # Simulated bursts
    x = np.linspace(0, 1, 101)
    obs_colors = {'ARTS': 'C2', 'CHIME': 'C1', 'LOFAR': 'C3'}

    # fig1, ax = plt.subplots(1, 1)
    # ax.plot(x, norm.pdf(x, loc=mu, scale=sigma,), '-', c='k', lw=1, label='simulated')
    # ax.hist(bursts_chime, density=False, histtype='stepfilled', color=obs_colors['CHIME'], alpha=0.2, label='{} CHIME'.format(len(bursts_chime)))
    # #ax.hist(bursts_arts, density=False, histtype='stepfilled', color=obs_colors['ARTS'], alpha=0.2, label='{} ARTS'.format(len(bursts_arts)))
    # ax.hist(bursts_lofar, density=False, histtype='stepfilled', color=obs_colors['LOFAR'], alpha=0.2, label='{} LOFAR'.format(len(bursts_lofar)))
    # ax.set_xlim(0,1)
    # ax.set_xlabel('Phase')
    # ax.set_ylabel('N. bursts')
    # plt.legend()

    # Simulated KS
    ax = fig.add_subplot(gs[ii, 0])
    ax.hist(1/pvalue, bins=np.logspace(np.log10(min(1/pvalue)),np.log10(max(1/pvalue)), 100), density=True, histtype='stepfilled', color='C0', alpha=0.4)
    ax.vlines(1/ref_pvalue, ymin=0, ymax=1, colors='k', linestyles='solid')
    s1, s2, s3 = (np.percentile(1/pvalue, 68.27), np.percentile(1/pvalue, 95.45),
            np.percentile(1/pvalue, 99.73))
    ax.vlines([s1,s2,s3], ymin=0, ymax=1, colors='k', alpha=0.3,
            linestyles=['dotted', 'dashdot', 'dashed'])
    ax.text(1/ref_pvalue, 0.45, "pvalue = {:.2e}".format(ref_pvalue),
            rotation='vertical', verticalalignment='top',
            horizontalalignment='right')
    ax.text(s1, 0.45, "68.27 %", rotation='vertical', verticalalignment='top',
            horizontalalignment='right')
    ax.text(s2, 0.45, "95.45 %", rotation='vertical', verticalalignment='top',
            horizontalalignment='right')
    ax.text(s3, 0.45, "99.73 %", rotation='vertical', verticalalignment='top',
            horizontalalignment='right')
    ax.text(0.05, 0.95, '{}-{}'.format(inst[0], inst[1]),
            transform=ax.transAxes, verticalalignment='top')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-8,0.5)
    ax.set_xlim(1,1e9)

    # ax2 = ax[1]
    # ax2.hist(statistic, bins=100, density=True, histtype='stepfilled', color='C0', alpha=0.4)
    # ax2.vlines(ref_stat, ymin=0, ymax=15, colors='k', linestyles='solid')
    # s1, s2, s3 = (np.percentile(statistic, 68.27), np.percentile(statistic, 95.45),
    #         np.percentile(statistic, 99.73))
    # ax2.vlines([s1,s2,s3], ymin=0, ymax=15, colors='k', alpha=0.3,
    #         linestyles=['dotted', 'dashdot', 'dashed'])
    # ax2.text(ref_stat, 11, "stat = {:.2f}".format(ref_stat),
    #         rotation='vertical', verticalalignment='top',
    #         horizontalalignment='right')
    #
    # ax2.set_title('statistic')
    # ax2.set_yscale('log')
    # ax2.set_ylim(1e-2,12)
plt.show()
