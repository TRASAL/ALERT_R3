import numpy as np
import json
from scipy import stats
from scipy.stats import norm
from frbpa.utils import get_phase, get_cycle, get_params
from frbpa.search import pr3_search, riptide_search, p4j_search
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit


# Defininig functions
def gaussian(x, *p):
    (c, mu, sigma) = p
    res =    c * np.exp( - (x - mu)**2.0 / (2.0 * sigma**2.0) )
    return res

def simulate_bursts(obs_start_phases, obs_start_cycles, obs_duration,
        rate=100, mu=0, sigma=1, P=8.175, ref_mjd=58369.30):

    # Simulate the bursts that would be observed
    # given a normal distribution and the observing times
    cycles = np.unique(obs_start_cycles)
    burst_mjds = np.array([])
    for c in cycles:
        mjd_cycle = ref_mjd + c*P
        index = np.where(obs_start_cycles == c)
        size = abs(int(np.random.normal(loc=rate, scale=rate/5, size=1)))
        r = norm.rvs(loc=mu, scale=sigma, size=size)
        for i in index[0]:
            p = obs_start_phases[i]
            d = obs_duration[i]
            b = r[(np.where((r >= p) & (r <= p+d)))]
            b_mjd = mjd_cycle + b*P
            burst_mjds = np.append(burst_mjds, b_mjd)
    return burst_mjds

def get_average_rate(data_json, inst, P=8.175):
    """Compute average rate in bursts per cycle
    """
    with open(data_json, 'r') as f:
        data = json.load(f)
    bursts = data['bursts'][inst]
    duration = data['obs_duration'][inst]

    nbursts = len(bursts)
    duration_tot = np.sum(duration)/3600
    rate = nbursts/duration_tot * 24 * P
    return rate

def fit_gauss_to_hist(data_json, telescopes, P=8.175, nbins=30,
        ref_mjd=58369.30):

    # Reading input file
    with open(data_json, 'r') as f:
        data = json.load(f)
    burst_dict = data['bursts']
    startmjds_dict = data['obs_startmjds']
    duration_dict = data['obs_duration']

    bursts = []
    if telescopes == 'all':
        for k in burst_dict.keys():
            bursts += burst_dict[k]
    else:
        for k in telescopes:
            bursts += burst_dict[k]
    bursts = np.array(bursts)
    unique_days = np.unique(np.round(bursts))

    burst_phase = get_phase(bursts, period=P, ref_mjd=ref_mjd)
    burst_per_phase, _ = np.histogram(burst_phase, bins=nbins, range=(0,1))
    bin_centre = np.linspace(1/(2*nbins),1-1/(2*nbins), nbins)

    # Fitting histogram to gaussian
    p0 = [14, 0.5, 0.2]
    coeff, var_matrix = curve_fit(gaussian, bin_centre, burst_per_phase,
            p0=p0)
    gauss_fit = gaussian(bin_centre, *coeff)

    # Plotting
    # plt.hist(burst_phase, bins=nbins)
    # plt.plot(bin_centre, burst_per_phase)
    # plt.plot(bin_centre, gauss_fit)
    # plt.xlim(0,1)
    # plt.show()
    return coeff

def plot_obs_sim_periodograms(burst_obs, burst_sim, obs_mjds, obs_durations,
        inst='ARTS'):
    fig = plt.figure(figsize=(13,10))
    gs = gridspec.GridSpec(2,1, hspace=0.05, wspace=0.05)

    # Observed data
    datadir = '/home/ines/Documents/projects/R3/periodicity/periodograms/'
    data = np.load(datadir + inst + '_period_pr3.npy')
    rch_obs, p_pr3_obs = data[0], data[1]

    # Simulated data
    print(type(burst_sim), type(obs_mjds), type(obs_durations))
    rch_sim, p_pr3_sim = pr3_search(bursts=burst_sim,
            obs_mjds=obs_mjds, obs_durations=obs_durations,
            pmin=1.57, pmax=65., pres=1e4)

    # plt.plot(p_pr3_obs, rch_obs, color='k', lw=1)
    # plt.show()

    # Plotting
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(p_pr3_obs, rch_obs, color='k', lw=1)
    ax1.set_xlim(2,17)
    ax1.set_ylim(0,35)
    ax1.set_xticklabels([])
    ax1.text(0.05, 0.95, "Observed",
            transform=ax1.transAxes, verticalalignment='top')

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(p_pr3_sim, rch_sim, color='k', lw=1)
    ax2.set_xlim(2,17)
    ax2.set_ylim(0,35)
    ax2.set_xlabel('Period (days)')
    ax2.text(0.05, 0.95, "Simulated",
            transform=ax2.transAxes, verticalalignment='top')

    plt.show()

# Input parameters
P = 8.175
ref_mjd = 58365.0
telescopes = ['ARTS', 'CHIME']
nbins=20

# Input files
indir = '/home/ines/Documents/projects/R3/periodicity/'
infile = indir + 'r3all_data.json'
telescope_groups = ['all', ['ARTS'], ['CHIME'], ['ARTS', 'CHIME']]
periodogram_names = ['all', 'ARTS', 'CHIME', 'CHIME_ARTS']

ii = telescope_groups.index(telescopes)


# OPening burst data
with open(infile, 'r') as f:
    data = json.load(f)

burst_dict = data['bursts']
obs_startmjds = data['obs_startmjds']
obs_duration = data['obs_duration']

burst_obs_all = []
burst_sim_all = []
obs_startmjds_all = []
obs_duration_all = []

for inst in telescopes:
    obs_start_phases = get_phase(obs_startmjds[inst], P, ref_mjd=ref_mjd)
    obs_start_cycles = get_cycle(obs_startmjds[inst], P, ref_mjd=ref_mjd)
    obs_duration_phase = [d/(3600*24*P) for d in obs_duration[inst]]
    burst_obs = burst_dict[inst]
    burst_obs_all.append(burst_obs)

    # Simulations

    c, mu, sigma = fit_gauss_to_hist(infile, telescopes=[inst], P=P, nbins=nbins,
            ref_mjd=ref_mjd)
    rate = get_average_rate(infile, inst)

    burst_sim = simulate_bursts(obs_start_phases, obs_start_cycles, obs_duration_phase,
            rate=rate, mu=mu, sigma=sigma, P=P, ref_mjd=ref_mjd)
    burst_sim_all.append(burst_sim)

    # plt.hist(get_phase(burst_dict[inst], period=P, ref_mjd=ref_mjd), bins=nbins,
    #         range=(0,1), alpha=0.3, label='Detected')
    # plt.hist(get_phase(bursts, period=P, ref_mjd=ref_mjd), bins=nbins,
    #         range=(0,1), alpha=0.3, label='Simulated')
    # plt.xlim(0,1)
    # plt.legend()
    # plt.show()

burst_obs_all = np.array([b for list in burst_obs_all for b in list])
burst_sim_all = np.array([b for list in burst_sim_all for b in list])
obs_startmjds_all = np.array(obs_startmjds[inst])
obs_duration_all = np.array(obs_duration[inst])

print(burst_obs_all, len(burst_obs_all))

# Periodogram
plot_obs_sim_periodograms(burst_obs_all, burst_sim_all, obs_startmjds_all,
        obs_duration_all, inst=periodogram_names[ii])
