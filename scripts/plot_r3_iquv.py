from __future__ import print_function
from __future__ import division
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.time import Time
import datetime
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

# Defining numpy files to plot
datadir = '/tank/data/FRBs/R3/'
fl_20200322 = glob.glob(datadir+'20200322/iquv/CB00/numpyarr/*dedisp*')
fl_2020051x = glob.glob(datadir+'2020051*/CB00/iquv/numpyarr/*mjd*dedisp*')
fl_2020052x = glob.glob(datadir+'2020052*/*/iquv/numpyarr/*dedisp*')
fl = fl_20200322 + fl_2020051x + fl_2020052x
fl.sort()
fl = np.unique(fl)

print("Plotting following dedispersed files:")
print(fl)

# Defining phase calibrator
fn_xyphase = [datadir+'20200322/iquv/CB00/polcal/xy_phase.npy'] \
             * len(fl_20200322) \
             + [datadir+'20200512/CB00/iquv/polcal/xy_phase.npy'] \
             * len(fl_2020051x) \
             + [datadir+'20200527/polcal/xy_phase.npy'] \
             * len(fl_2020052x)

# Defining bandpass
bandpass = np.load('/tank/data/FRBs/FRB200419/20200419/iquv/CB32/polcal/bandpass.npy')

# Output file
plt_out = '/home/arts/pastor/R3/polcal/R3_IQUV_PA.pdf'

# Plotting
colors = ['#577590', '#90be6d', '#f8961e', '#f94144']

fig = plt.figure(figsize=(29.7,21))
ncols = 6
nrows = 4
gs = gridspec.GridSpec(nrows,ncols, hspace=0.2)

for ii,ff in enumerate(fl):
  #if ii == 2:
    d = np.load(ff)
    xyphase = np.load(fn_xyphase[ii])

    # Bandpass correction
    #d /= bandpass[None, :, None]

    # Frequency mask
    #off_spectrum = d[0, :, :].mean(1) - d[0, :, 4900:5100].mean(1)
    #f_mask = np.where(np.abs(off_spectrum) > np.std(frb_spectrum, axis=-1))[0] 
    #f_mask = f_mask[np.where(f_mask<400)]
    #f_mask = np.arange(np.min(f_mask)-10, np.max(f_mask)+10)
    f_mask = range(1532,1536)
    on = d[0, :, 4950:5050].mean(1)
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
    D = d[..., 4900:5100] 

    # Phase correction
    XYdata = D[2]+1j*D[3]
    XYdata *= np.exp(-1j*(xyphase))[:, None]
    D[2], D[3] = XYdata.real, XYdata.imag
    PA = np.angle(np.mean(D[1]+1j*D[2],0), deg=True) # degrees
    PA[np.where(PA<0)[0]] += 360 

    # Temporal mask. index in mask allowed for PA calculation
    std_I = np.std(d[0, ..., 4500:5500].mean(0))
    t_mask = np.where(D[0].mean(0) > 2*std_I)[0]
    t_mask = t_mask[np.where(t_mask>50)]

    mjd = float(ff.split('/')[-1].split('_')[1].replace('mjd', ''))
    t = Time(mjd, format='mjd', scale='utc')
    date = t.datetime.strftime('%Y-%m-%dT%H:%M:%S')
    print(date)    

    try:
        gss = gridspec.GridSpecFromSubplotSpec(2, 2, 
                subplot_spec=gs[ii//ncols,ii%ncols], hspace=0, wspace=0,
                height_ratios=[2,1])
    except:
        gss = gridspec.GridSpecFromSubplotSpec(2, 2, 
                subplot_spec=gs[0,0], hspace=0, wspace=0,
                height_ratios=[2,1])

    ax1 = fig.add_subplot(gss[0, 0])
    ax1.plot(D[0].mean(0), color=colors[0], lw=1, label='I')
    ax1.plot(D[1].mean(0), color=colors[1], lw=1, label='Q')
    ax1.plot(D[2].mean(0), color=colors[2], lw=1, label='U')
    ax1.plot(D[3].mean(0), color=colors[3], lw=1, label='V') 
    ax1.tick_params(axis='x', which='both', direction='in', bottom=True, 
            top=True)
    ax1.tick_params(axis='y', which='both', direction='in', left=True, 
            right=True)

    ax2 = fig.add_subplot(gss[1, 0], sharex=ax1)
    ax2.plot(t_mask, PA[t_mask], '.', color='k',alpha=0.5, label='PA')
    ax2.yaxis.set_major_locator(MultipleLocator(90))
    ax2.grid(True, axis='y', color='k', alpha=0.1)
    ax2.tick_params(axis='x', which='both', direction='in', bottom=True, 
            top=True)
    ax2.tick_params(axis='y', which='both', direction='in', left=True, 
            right=True)

    ax3 = fig.add_subplot(gss[0, 1])
    #ax3.plot(D[0, ..., mask].mean(0), color=colors[0], lw=1)
    #ax3.plot(D[0].mean(0), color=colors[0], lw=1)
    ax3.plot(frb_spectrum, color=colors[0], lw=1)

    ax1.set_title(date, fontsize=16)
    ax2.set_ylim(0,360)

    for ax in (ax1,ax2):
        ax.label_outer()

# Saving plot
print('Saving plot to ', plt_out)
plt.savefig(plt_out, pad_inches=0, bbox_inches='tight')
plt.show()
