import numpy as np
import json
from scipy import stats
from scipy.stats import norm
from frbpa.utils import get_phase, get_cycle, get_params
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tqdm


# Defininig functions
def simulate_gaussian_bursts(obs_start_phases, obs_start_cycles, obs_duration,
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

def simulate_ks(data_json, inst1='Apertif', inst2='CHIME/FRB',
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
        bursts_1 = simulate_gaussian_bursts(obs_start_phases_1, obs_start_cycles_1,
                obs_duration_1, rate=rate, mu=mu, sigma=sigma)
        bursts_2 = simulate_gaussian_bursts(obs_start_phases_2, obs_start_cycles_2,
                obs_duration_2, rate=rate, mu=mu, sigma=sigma)

        # Comparing samples
        statistic, pvalue = stats.ks_2samp(bursts_1, bursts_2)

    else:
        file = open(outfile, 'w')
        pvalue = np.array([])
        statistic = np.array([])
        file.write("# statistic pvalue bursts_{} bursts_{}\n".format(inst1, inst2))
        for i in tqdm.tqdm(range(int(trials))):
            bursts_1 = simulate_gaussian_bursts(obs_start_phases_1, obs_start_cycles_1,
                    obs_duration_1, rate=rate, mu=mu, sigma=sigma)
            bursts_2 = simulate_gaussian_bursts(obs_start_phases_2, obs_start_cycles_2,
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
fnarts = fndir + 'phase_Apertif_p{}_f1370.0.npy'.format(P)
fnlofar = fndir + 'phase_LOFAR_p{}_f150.0.npy'.format(P)
data_r3 = np.loadtxt('/home/ines/Documents/projects/R3/periodicity/counts_per_phase.txt')
data_dict = {}
data_dict['CHIME/FRB'] = np.load(fnchime)
data_dict['Apertif'] = np.load(fnarts)
data_dict['LOFAR'] = np.load(fnlofar)

# Loading data
data_json = '/home/ines/Documents/projects/R3/periodicity/r3all_data_searched.json'

foutdir = '/home/ines/Documents/projects/R3/periodicity/simulated_ks/'

mu,sigma = norm.fit(data_dict['CHIME/FRB'][1])

# statistic, pvalue, bursts_arts, bursts_chime = simulate_ks(data_json,
#         inst1='Apertif', inst2='CHIME/FRB', P=P,
#         outfile='simulated_ks_CHIME/FRB-Apertif.txt',
#         ref_mjd=ref_mjd, rate=chime_rate, mu=mu, sigma=sigma, trials=10000)

#ref_stat, ref_pvalue = stats.ks_2samp(data_arts[1], data_chime[1])

#fig, ax = plt.subplots(3, 1)
fig = plt.figure(figsize=(9,13))
gs = gridspec.GridSpec(3,1, hspace=0.05, wspace=0.01)
plt.style.use('/home/ines/.config/matplotlib/stylelib/paper.mplstyle')

print("Starting simulations")

for ii,inst in enumerate([['Apertif', 'LOFAR'], ['Apertif', 'CHIME/FRB'], ['CHIME/FRB', 'LOFAR']]):

    # statistic, pvalue, bursts_lofar, bursts_chime = simulate_ks(data_json,
    #         inst1=inst[0], inst2=inst[1], P=P,
    #         outfile=foutdir+'simulated_ks_{}_{}-{}.txt'.format(P,
    #         inst[0].replace('/FRB', ''), inst[1].replace('/FRB', '')),
    #         ref_mjd=ref_mjd, rate=chime_rate, mu=mu, sigma=sigma, trials=1e5)
    ref_stat, ref_pvalue = stats.ks_2samp(data_dict[inst[0]][1],
            data_dict[inst[1]][1])
    print(inst, ref_pvalue)

    # sim_data = np.genfromtxt('simulated_ks_CHIME-ARTS.txt', names=True)
    sim_data = np.genfromtxt(foutdir+'simulated_ks_{}_{}-{}.txt'.format(P,
            inst[0].replace("/FRB", ""), inst[1].replace("/FRB", "")),
            names=True)
    statistic = sim_data['statistic']
    pvalue = sim_data['pvalue']

    # Significance of the difference
    percentile = 100 - stats.percentileofscore(pvalue, ref_pvalue)
    print("Percentile", inst, percentile)
    sigma = stats.norm.isf((1-percentile/100)/2.)
    print("Sigma", inst, sigma)


    # Plotting
    # Simulated bursts
    x = np.linspace(0, 1, 101)
    obs_colors = {'Apertif': 'C2', 'CHIME/FRB': 'C1', 'LOFAR': 'C3'}

    # Simulated KS
    ax = fig.add_subplot(gs[ii, 0])
    ax.hist(pvalue,
            bins=np.logspace(np.log10(min(pvalue)),np.log10(max(pvalue)), 100),
            histtype='stepfilled', color='C0', alpha=0.4)
    ax.vlines(ref_pvalue, ymin=0, ymax=1e4, colors='k', linestyles='solid')
    s1, s2, s3 = (np.percentile(pvalue, 100-68.27),
            np.percentile(pvalue, 100-95.45),
            np.percentile(pvalue, 100-99.73))
    ax.vlines([s1,s2,s3], ymin=0, ymax=1e4, colors='k', alpha=0.3,
            linestyles=['dotted', 'dashdot', 'dashed'])
    ax.text(ref_pvalue, 1, "p-value = {:.2e}".format(ref_pvalue),
            rotation='vertical', verticalalignment='bottom',
            horizontalalignment='right')
    ax.text(s1, 1e4, "68.27 %", rotation='vertical', verticalalignment='top',
            horizontalalignment='right')
    ax.text(s2, 1e4, "95.45 %", rotation='vertical', verticalalignment='top',
            horizontalalignment='right')
    ax.text(s3, 1e4, "99.73 %", rotation='vertical', verticalalignment='top',
            horizontalalignment='right')
    ax.text(0.05, 0.9, '{} $-$ {}'.format(inst[0], inst[1]),
            transform=ax.transAxes, color='C0')
    ax.text(0.02, 0.9, chr(ord("a")+ii),
            transform=ax.transAxes, weight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(8e-1,1e4)
    ax.set_xlim(3e-11,1)

    if ii<2:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("p-value")
    ax.set_ylabel("N")

plt_out = "/home/ines/Documents/projects/R3/periodicity/simulated_ks.pdf"
print('Saving figure to ', plt_out)
plt.savefig(plt_out, pad_inches=0, bbox_inches='tight')
plt.show()
