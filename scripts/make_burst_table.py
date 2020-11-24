import numpy as np
import pandas as pd
from astropy.time import Time

print("\n\nApertif burst table\n\n")

fname = '/home/ines/Documents/projects/R3/arts/arts_r3_properties.csv'
burst_data = pd.read_csv(fname)

for ii,burst in burst_data.iterrows():
    if ii>0:
        if (burst['bary_mjd'] - mjdp > 3) :
            print('\\hline')
    bdate = Time(burst['bary_mjd'], format='mjd', scale='utc')
    if np.isnan(burst['drift_rate']):
        burst['drift_rate'] = '...'
        print('A{:02} & {:.8f} & {} & {:.1f} & {:.2f}({:.0f}) & {:.1f} & {} \\\\'.format(
        ii+1, burst['bary_mjd'], bdate.isot, burst['detection_snr'],
        burst['struct_opt_dm'], burst['struct_opt_dm_err']*100,
        burst['fluence_Jyms'], burst['drift_rate']))
    else:
        print('A{:02} & {:.8f} & {} & {:.1f} & {:.2f}({:.0f}) & {:.1f} & {:.2f} \\\\'.format(
        ii+1, burst['bary_mjd'], bdate.isot, burst['detection_snr'],
        burst['struct_opt_dm'], burst['struct_opt_dm_err']*100,
        burst['fluence_Jyms'], burst['drift_rate']))
    mjdp = burst['bary_mjd']
    #print(str(burst['drift_rate_err']))

print("\n\nLOFAR burst table\n\n")

fname = '/home/ines/Documents/projects/R3/lofar/lofar_r3_properties.csv'
burst_data = pd.read_csv(fname)

for ii,burst in burst_data.iterrows():
    if ii<9:
        burst_isot = Time(burst['bary_mjd'], format='mjd', scale='utc')
        burst_isot = burst_isot.isot.replace('T', ' ')
        if np.isnan(burst['tscat']):
            print("{} & {} & {:.8f} & {} & {:.1f} & {:.2f}$\\pm${:.2f} & {:.0f}$\\pm${:.0f} & ... \\\\".format(
                    burst['paper_name'], burst['detection_folder'],
                    burst['bary_mjd'], burst_isot, burst['detection_snr'],
                    burst['gauss_cen_k_dm'], burst['gauss_cen_k_dm_err'],
                    burst['fluence_Jyms'], burst['fluence_err']))
        else:
            print("{} & {} & {:.8f} & {} & {:.1f} & {:.2f}$\\pm${:.2f} & {:.0f}$\\pm${:.0f} & {:.1f}$\\pm${:.1f}\\\\".format(
                    burst['paper_name'], burst['detection_folder'],
                    burst['bary_mjd'], burst_isot, burst['detection_snr'],
                    burst['gauss_cen_k_dm'], burst['gauss_cen_k_dm_err'],
                    burst['fluence_Jyms'], burst['fluence_err'],
                    burst['tscat'], burst['tscat_err']))
