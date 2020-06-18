#!/usr/bin/env python
# ------------------------------------------------------------------------#
#                          get_tscat.py
# Program to calculate the scattering  parameters of pulsars. It estimates
# the  scatter-broadening of the pulse profiles from archive  files. It is
# mainly written to use with the LOFAR HBA data, but it should work on any
# proper PSRFIT archive files.
#
# Usage:
# python get_tscat.py -f <fsc> infile.ar
# <fsc> is the factor to scrunch the number of channels in the archive.
#
# Requirements:
# PSRCHIVE python interface (for this, install PSRCHIVE in 'shared' mode);
# LMFIT; scipy; numpy; matplotlib, statsmodels.
#
# Version 1: (16.10.2018)
# Fits the archives with an analytical function with  tau_sc & W50 as free
# parameters. Estimates the W50 of the profiles in the archive. Calculates
# the scaling index of each of these parameters.
#
# Version 2: (13/06/2019)
# Did several bug fixes. Main improvement is the addition of the MCMC part
# for estimating  alpha and beta.
#
# ** This code cannot fit simultaneously the interpulse and mainpulse. **
#
#
# *** Pending works ***
#
# Template fitting for sub-banded data.
#
# M. A. Krishnakumar
#
#
#
#
# Did some changes so that the input file can have any number of bins, the
# changes are done only for the nchan==1 case.       -- Yogesh Maan
#
# Did some other changes to make it work with low-S/N cases, report output
# parameters in units of bins if needed, -- Yogesh Maan
#
# ------------------------------------------------------------------------#


# imports the required modules
from __future__ import print_function
from __future__ import division
import os
import scipy
import math
import argparse
import psrchive
import lmfit
import numpy as np
from statsmodels import robust
import matplotlib as mpl
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from numpy import array, exp, sin
from psrchive import Integration
from scipy import signal, optimize
from scipy.interpolate import splrep, sproot, splev
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from lmfit import Minimizer, Parameters, report_fit, Model
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# Parser arguments for inputs and reading them in
parser = argparse.ArgumentParser(description='Commands for tauscat code')
parser.add_argument('archive', nargs='+',
					help='The chosen archives')
parser.add_argument('-f','--fscrunch', type=int, default=1,
					help='Factor to scrunch the number of channels')
parser.add_argument('-t','--tscrunch', type=int, default=1,
					help='Factor to scrunch the number of time bins')
parser.add_argument('-m','--model',type=str,
					help='Template profile for fitting')
parser.add_argument('-S','--profsnr',type=int, default=50,
					help='Template profile for fitting')
parser.add_argument('-s','--chansnr',type=int, default=10,
					help='Template profile for fitting')
parser.add_argument('-b','--bins', dest='bins', action='store_true',
			default=False, help='Output fit-params in units of bins')
parser.add_argument('-a','--alpha_beta', dest='alpha_beta', action='store_true',
			default=False, help='Compute alpha and beta. Default: False')

args = parser.parse_args()

# ------------------------------------------------------------------------- #
# The plotter function
# ------------------------------------------------------------------------- #
def plotter(data, fit, x, fr, outfn, mjd=None, cmap='viridis'):
	# Setting plot parameters:
	plt.figure(figsize=(9,7))
	cmap = mpl.cm.get_cmap(cmap)
	fmax=190
	fmin=110
	#color = [cmap(x/len(data)) for x in range(len(data))]
	color = [cmap((fmax-f)/(fmax-fmin)) for f in fr]

	# Plotting
	for i in range(len(data)):
		plt.plot(x, data[i]+fr[i], color=color[i], linestyle='-', lw=0.5)
		plt.plot(x, fit[i]+fr[i], color=color[i], linestyle='-')

	# Saving
	if mjd is not None:
		plt.title(str(mjd))
	plt.xlabel('Time (ms)')
	plt.ylabel('Frequency (MHz)',fontsize=12)
	plt.yticks(fr)
	plt.savefig(outfn, format='eps', bbox_inches='tight')
	print("File saved: ",outfn)

# ------------------------------------------------------------------------- #
# The fitting function for scattering. It uses an analytic Gaussian as a
# template profile and is allowed to fit for its width, height and phase.
# ------------------------------------------------------------------------- #
def func(params, x, data):
	tau = params['tau']
	amp = params['amp']
	sigma = params['sigma']
	phase = params['phase']
	gauss = amp*exp(-(x-phase)**2/(2*sigma**2))
	impulse = scipy.ndimage.interpolation.shift((exp(-(x)/tau)),(phase/4-phase/2))
	impulse /= max(impulse)
	fn = np.convolve(gauss,impulse,'same') / sum(impulse)
	maxfn=max(fn)
	return fn - data

# ------------------------------------------------------------------------- #
# Fitting function for scattering, if given a template profile. Reliable to
# some degree. May require some improvement.
# ------------------------------------------------------------------------- #
def funct(params, x, data, templprof, nbins, case):
	tau = params['tau']
	ampfac = params['ampfac']
	shift = params['shift']
	templ = templprof
	if case == 0:
		impulse = scipy.ndimage.interpolation.shift(exp(-(x)/tau),(nbins/4))
	else:
		impulse = scipy.ndimage.interpolation.shift(exp(-(x)/tau),(nbins/4-nbins/2))
	impulse /= max(impulse)
	fn = np.convolve(templ,impulse,'same') / sum(impulse)
	fn /= ampfac
	fn1 = scipy.ndimage.interpolation.shift(fn,shift)
	return fn1 - data

# ------------------------------------------------------------------------- #
# Function to find the peak phase of the pulse. Outputs only as integer for
# initial guess.
# ------------------------------------------------------------------------- #
def pulsephase(array, maxval):
	for i in range(0,len(array)):
		if array[i] == maxval:
			return i

# ------------------------------------------------------------------------- #
# Function for fitting in MCMC method. Powerlaw in log-log mode, otherwise
# a simple straight-line fit.
# ------------------------------------------------------------------------- #
def f(x, A, B):
	return A*x + B



# Loading the archive for processing
fscfact = args.fscrunch
tsfact = args.tscrunch
for ii,arch in enumerate(args.archive):
	print("Opening " + arch)
	ar = psrchive.Archive_load(arch)

	# Modifying the archive for analysis
	ar.fscrunch(fscfact)
	if tsfact > 1:
	    ar.bscrunch(tsfact)
	ar.tscrunch()
	ar.pscrunch()
	ar.dedisperse()
	ar.remove_baseline()
	ar.centre_max_bin()
	ar.rotate_phase(0.3)
	# Reading various parameters from the archive
	nbins = ar.get_nbin()
	nchan = ar.get_nchan()
	data = ar.get_data()
	print(data.shape)
	print("Archive parameters: nbins={} nchan={}".format(nbins, nchan))
	# Getting the profile S/N ratio
	prof = ar.clone()
	prof.fscrunch()
	snr = prof.get_Profile(0,0,0).snr()
	print("Profile S/N ratio: "+str(snr))
	# The program will work only if the profile S/N ratio is above 50.
	if (nchan > 1):
		if snr < args.profsnr:
			print("Error: Profile S/N ratio is lower than threshold ("+str(int(snr))+" <")
			print(str(int(args.profsnr))+"). Not doing any scattering estimates!!")
			exit(0)
	if (nchan == 1):
		if snr < 6:
			print("Error: Profile S/N ratio is lower than threshold ("+str(int(snr))+" <")
			print("10). Not doing any scattering estimates!!")
			exit(0)

	# Reading the modified parameters of the archive
	centfreq=ar.get_centre_frequency()
	bw=ar.get_bandwidth()
	chbw=bw/nchan
	lowband=centfreq - (bw/2) + (chbw/2)
	highband=centfreq + (bw/2) - (chbw/2)
	mjd_st=ar.get_Integration(0).get_start_time().in_days()
	mjd_end=ar.get_Integration(0).get_end_time().in_days()
	mjd=mjd_st+(mjd_end-mjd_st)/2.
	src = ar.get_source()
	period = (ar.get_Integration(0).get_folding_period()) * 1000.
	site = ar.get_telescope()


	# Opening and reading the template profile, if provided.
	if (nchan > 1) and (args.model != None):
		print("Error: Template profile fitting is not available for multi-channel data.")
		print("Please use the code without -m option for getting the multi-channel fit")
		print(" or frequency scrunch the data to one profile.")
		exit(0)
	templprof=''
	if not (args.model == None):
		templarch = psrchive.Archive_load(args.model)
		templarch.dedisperse()
		templarch.tscrunch()
		templarch.fscrunch()
		templarch.pscrunch()
		templarch.remove_baseline()
		templarch.centre_max_bin()
		templdata = templarch.get_data()
		# **Sub-optimal**. Use of this code for pulsars with Interpulse is yet to be
		# established. So, in the corrent form, it will do fit only the main component
		# and not the interpulse.
		if src=='J1939+2134':
			templprof = templdata.flatten()[:int(float(nbins*(0.6)))].copy()
		else:
			templprof = templdata.flatten()

	# Defining the global variable arrays
	redch = np.zeros(nchan)
	reffreq = np.zeros(nchan)

	if src=='J1939+2134':
		x = np.arange(int(float(nbins*0.6)))
	else:
		x = np.arange(nbins)
	tau = np.zeros(nchan)
	terr = np.zeros(nchan)
	w50 = np.zeros(nchan)
	werr = np.zeros(nchan)
	fr=np.zeros(nchan)
	freq=np.zeros(nchan)
	w_50=np.zeros(nchan)
	w_50err=np.zeros(nchan)
	P_phase = np.zeros(nchan)
	P_phaseErr = np.zeros(nchan)


	#creating the required directories for saving plots & files
	pwd=os.getcwd()
	dirprof=os.path.join(pwd,src+"_"+site+"_profilefit_plots")
	if not os.path.exists(dirprof):
		os.makedirs(dirprof)
	diralp=os.path.join(pwd,src+"_"+site+"_powerlawfit_plots")
	if not os.path.exists(diralp):
		os.makedirs(diralp)
	dirtxt=os.path.join(pwd,src+"_"+site+"_profilefit_results")
	if not os.path.exists(dirtxt):
		os.makedirs(dirtxt)

	# temporary file for storing the results.
	tmpfile=ar.get_source()+str(mjd)+'_temp.txt'

	# ------------------------------------------------------------------------- #
	# Estimating the tau_sc, if the archive has only one channel.
	# ------------------------------------------------------------------------- #
	if (nchan == 1):
		case = 0 # All Pulsars except J1939+2134. Else case=1
		data = ar.get_data() # Reading the data as numpy array
		print(data.shape)
		# Finding the offpulse RMS for using later in chisq normalization
		mean = sum(data[-int(nbins/10):])/(nbins/10)
		offpulse = data[-int(nbins/10):]
		rms = np.std(offpulse)
		if src=='J1939+2134':
			data1 = data.flatten()[:int(float(nbins*0.6))].copy()
			case = 1  # Reading only the 60% of the pulse phase
		else:
			data1 = data.flatten()
			case = 0
		fr = ar.get_centre_frequency()
		result=''
		# If no template profile is provided, doing the fit with function func()
		if (args.model==None):
			atemp = (nbins/25.0)
			btemp = (nbins/50.0)
			ctemp = (nbins/2.0)
			astr  = "(init = {0})".format(atemp)
			bstr  = "(init = {0})".format(btemp)
			params = Parameters()
			params.add('tau', value=atemp, min=1.)
			params.add('amp', value=1., min=0.)
			params.add('sigma', value=btemp, min=1.)
			params.add('phase', value=ctemp, min = 10.)
			minner = Minimizer(func, params, fcn_args=(x, data1),
					nan_policy='omit')
			result = minner.minimize()
			outfile = open(tmpfile,"w+")
			out = lmfit.fit_report(result.params,sort_pars='True')
			outfile.write(out)
			outfile.close()
		# If a template profile is given, do the fit with function funct()
		if not (args.model==None):
			atemp  = (nbins/25.0)
			ashift = (nbins/10.0)
			ampfac = (0.5)
			params = Parameters()
			params.add('tau', value=atemp)
			params.add('shift', value=ashift)
			params.add('ampfac', value=ampfac)
			minner = Minimizer(funct, params,
					fcn_args=(x,data1,templprof,nbins,case), nan_policy='omit')
			result = minner.minimize()
			outfile = open(tmpfile,"w+")
			out = lmfit.fit_report(result.params,sort_pars='True')
			outfile.write(out)
			outfile.close()
		final = data1 + result.residual # The best fit model

		chisq = sum((data1 - final)**2) # Calculating the reduced chisq
		redch = chisq/((int(float(nbins*0.75)))*rms**2) # of the fit
		# Finding the width of the observed pulse profile
		window1 = int((3/100.)*nbins)
		if (window1%2==0):
			window1+= 1
		if window1 <= 3:
			window1 = 5
		psmooth1 = savgol_filter(data1,window1, 3)
		half_max = max(psmooth1)/2.
		p50w = splrep(x, psmooth1 - half_max, k=3)
		roots1 = sproot(p50w)
		w_50 = (roots1[1] - roots1[0]) * period / nbins
		w_50err = w_50 * rms
		# Plotting the profile and the fit to a file
		# plt.figure(figsize=(9, 7))
		# plt.plot(x,data1)
		# plt.plot(x,final,'r')
		# plt.ylabel('%5.1f' % (fr),fontsize=5)
		# plt.xticks([])
		# plt.yticks([])
		# plt.xlim(0,nbins)
		# plt.savefig("{}/{}_{}_{}_scatfit.pdf".format(dirprof, ar.get_source(),
		# 		str(mjd), site), bbox_inches='tight')
		# Reading the fit results from the temporary file
		nlines = len(open(tmpfile).readlines(  )) # Checking for the file.
		if (args.model==None):
			if nlines > 5:
				with open(tmpfile) as fp:
					tmpline = fp.readline().strip().split()
					ampline = fp.readline().strip().split()
					phsline = fp.readline().strip().split()
					sigline = fp.readline().strip().split()
					tauline = fp.readline().strip().split()
					tau = float(tauline[1])
					terr = float(tauline[3])
					w50 = float(sigline[1])
					werr = float(sigline[3])
					freq = fr
			else:
					print("Could not estimate scatter-broadening")
		if not (args.model==None):
			if nlines > 4:
			    with open(tmpfile) as fp:
					tmpline   = fp.readline().strip().split()
					ampline   = fp.readline().strip().split()
					shiftline = fp.readline().strip().split()
					tauline   = fp.readline().strip().split()
					tau = float(tauline[1])
					terr = float(tauline[3])
					w50 = 0.0; werr = 0.0
					freq = fr
			else:
				print("Could not estimate scatter-broadening")
			if (args.bins):
				period = 1.0
				print("period, nbins: ", period,nbins)
				print("output parameters in units of bins")
				w50 *= 2.35482
				werr *= 2.35482
			else:
				print("period, nbins: ", period,nbins)
				print("output parameters in ms")
		# Converting the values to millisecs from bins
		tau  *= (period / nbins)
		terr *= (period / nbins)
		w50  *= (2.35482 * period / float(nbins))
		werr *= (2.35482 * period / nbins)

		os.remove(tmpfile)
		outtmp = "{}/{}_{}_nch{}_scatvals.txt".format(dirtxt, ar.get_source(),
				site, nchan)
		outfile = open(outtmp,"w+")
		outfile.write('# Freq.     Tau    Tau-err    w50    w50-err    red-chi  \n')
		for i in range(0, np.size(tau)):
			print(freq, tau, terr, w50, werr, redch)
			try:
				outfile.write(
						"{:06.5f} {:08.5f} {:06.5f} {:08.5f} {:06.5f} {:04.2f}\n".format(
						freq, tau, terr, w50, werr,	redch))
			except ValueError:
				outfile.write(
						"{} {} {} {} {} {}\n".format(freq, tau, terr, w50, werr,
						redch))
		outfile.close()

		f1 = open("{}_nch{}_b{}_scatresults.txt".format(ar.get_source(), nchan,
				tsfact), 'a')
		f1.write('# MJD       Tau    Tau-err   red-chi   centfreq   bw  site  \n')
		try:
			f1.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.2f} {}\n".format(mjd, tau, terr,
					redch, centfreq, bw, site))
		except ValueError:
			f1.write("{} {} {} {} {} {} {}\n".format(mjd, tau, terr,
					redch, centfreq, bw, site))
		f1.close()
		print('# MJD     Tau  Tau-err     w50  w50-err  red-chi  ')
		print(mjd, '%.4f' % tau, '%.4f' % terr,  '%.4f' %w50, '%.4f' %werr, '%.4f' % redch)
		# End of fitting routine for nchan=1 data

		# Plotting scattering profile
		tmpdirprof = "{}/{}_{}_{}_scatfit.pdf".format(dirprof, ar.get_source(),
				str(mjd), site)
		plt.plot(x * (period / nbins), data1)
		plt.plot(x * (period / nbins), final)
		plotter([data1], [final], x * (period / nbins), [fr], tmpdirprof, mjd=mjd)

	# ------------------------------------------------------------------------- #
	# Estimating tau_sc and its frequency scaling index, alpha. Currently only
	# work with a Gaussian model and not with a template profile.
	# ------------------------------------------------------------------------- #
	if (nchan > 1):
		data_all = [] #np.empty((0,nchan))
		fit_all = [] #np.empty((0,nchan))
		frequency = []
		for i in range(nchan):
			snr = ar.get_Profile(0,0,i).snr() # Finding the SNR of each
			if snr < args.chansnr:
				print("WARNING!!! Removed channel no. "+str(i)+" from fitting")
				print("due to poor S/N ratio ("+str(snr)+" < "+str(args.chansnr)+").")
			data = ''
			if snr > args.chansnr:
				data = ar.get_data()[:,:,i,:].flatten() # taking the ith channel
				#data[np.argwhere(np.isnan(data))] = 0.0
				fr[i] = ar.weighted_frequency(i,0,1) # getting the freq of channel
				frequency.append(fr[i])
				# Finding the offpulse RMS for using later in chisq normalization
				mean = sum(data[-int(nbins/10):])/(nbins/10)
				offpulse = data[-int(nbins/10):]
				rms = np.std(offpulse)

				# Parameters to pass to the fitting function. Choose them wisely.
				atemp = (nbins/20.0)
				btemp = (nbins/50.0)
				ctemp = int((nbins/2.0))
				astr  = "(init = {0})".format(atemp)
				bstr  = "(init = {0})".format(btemp)
				params = Parameters()
				params.add('tau', value=atemp, min=1.)
				params.add('amp', value=1., min=0.)
				params.add('sigma', value=btemp, min=1.)
				params.add('phase', value=ctemp, min = 10.)

				# Fitting the function and writing the results to a file
				minner = Minimizer(func, params, fcn_args=(x, data),
						nan_policy='omit')
				result = minner.minimize()
				outfile = open(tmpfile,"w+")
				out = lmfit.fit_report(result.params,sort_pars='True')
				outfile.write(out)
				outfile.close()

				final = data + result.residual # The best fit model
				chisq = sum((data - final)**2) # Calculating the reduced chisq
				redch[i] = chisq/(nbins*rms**2) # of the fit

				data_all.append(data)
				fit_all.append(final)

				# Finding the width of the observed pulse profile
				window1 = int((3/100.)*nbins)
				if (window1%2==0):
					window1+= 1
				if window1 <= 3:
					window1 = 5
				psmooth1 = savgol_filter(data,window1, 3)
				half_max = max(psmooth1)/2.
				p50w = splrep(x, psmooth1 - half_max, k=3)
				roots1 = sproot(p50w)
				if (len(roots1) < 2):
					w_50[i] = 0; w_50err[i]=0
				else:
					w_50[i] = (roots1[1] - roots1[0]) * period / nbins
					w_50err[i] = w_50[i] * rms

				# Reading the values of tau, tau_err, width of fitted Gaussian from the
				# tempresults.txt file into numpy arrays for further analysis.
				nlines = len(open(tmpfile).readlines(  ))
				if nlines > 5:
					with open(tmpfile) as fp:
						tmpline = fp.readline().strip().split()
						ampline = fp.readline().strip().split()
						phsline = fp.readline().strip().split()
						sigline = fp.readline().strip().split()
						tauline = fp.readline().strip().split()
						tau[i] = float(tauline[1])
						terr[i] = float(tauline[3])
						freq[i] = fr[i]
						w50[i] = float(sigline[1])
						werr[i] = float(sigline[3])
						P_phase[i] = float(phsline[1])
						P_phaseErr[i] = float(phsline[3])
				#os.remove(tmpfile)


			if (args.bins):
				period = 1.0
				print("period, nbins: ", period,nbins)
				print("output parameters in units of bins")
				w50 *= 2.35482
				werr *= 2.35482
			else:
				print("period, nbins: ", period,nbins)
				print("output parameters in ms")
			# Converting the values to millisecs from bins
	  		tau  *= (period / nbins)
			terr *= (period / nbins)
			w50  *= (2.35482 * period / float(nbins))
			werr *= (2.35482 * period / nbins)
		## Converting the results in bins to units of time
		#tau  *= (period / nbins)
		#terr *= (period / nbins)
		#w50  *= (2.35482 * period / nbins)
		#werr *= (2.35482 * period / nbins)

		# Plotting
		tmpdirprof=("{}/{}_{}_{}_scatfit.pdf".format(dirprof, ar.get_source(),
				str(mjd), site))
		plotter(data_all, fit_all, x * (period / nbins), frequency, tmpdirprof,
				mjd=mjd)

		# Removing the badly behaving elements in the results with an ensemble
		# of different conditions.
		mphase = np.median(P_phase)
		mplow = mphase - np.percentile(P_phase,10)
		## Removing data if pulse phase is > (mphase + mplow*2).
		#condition = P_phase < (mphase + mplow*2)
		#tau = np.extract(condition,tau)
		#terr = np.extract(condition,terr)
		#freq = np.extract(condition,freq)
		#w50 = np.extract(condition,w50)
		#werr = np.extract(condition,werr)
		#w_50 = np.extract(condition,w_50)
		#w_50err = np.extract(condition,w_50err)
		#redch = np.extract(condition,redch)
		#P_phase = np.extract(condition,P_phase)
		#P_phaseErr = np.extract(condition,P_phaseErr)
		## Removing data if pulse phase is < (mphase - mplow*2).
		#condition = P_phase > (mphase - mplow*2)
		#tau = np.extract(condition,tau)
		#terr = np.extract(condition,terr)
		#freq = np.extract(condition,freq)
		#w50 = np.extract(condition,w50)
		#werr = np.extract(condition,werr)
		#w_50 = np.extract(condition,w_50)
		#w_50err = np.extract(condition,w_50err)
		#redch = np.extract(condition,redch)
		#P_phase = np.extract(condition,P_phase)
		#P_phaseErr = np.extract(condition,P_phaseErr)
		## Removing data if phaseErr =0.0
		#condition = P_phaseErr > 0.0
		#tau = np.extract(condition,tau)
		#terr = np.extract(condition,terr)
		#freq = np.extract(condition,freq)
		#w50 = np.extract(condition,w50)
		#werr = np.extract(condition,werr)
		#w_50 = np.extract(condition,w_50)
		#w_50err = np.extract(condition,w_50err)
		#redch = np.extract(condition,redch)
		#P_phase = np.extract(condition,P_phase)
		#P_phaseErr = np.extract(condition,P_phaseErr)
		## Cleaning data if phase == intial guess (should never be same!)
		#condition = P_phase != ctemp
		#tau = np.extract(condition,tau)
		#terr = np.extract(condition,terr)
		#freq = np.extract(condition,freq)
		#w50 = np.extract(condition,w50)
		#werr = np.extract(condition,werr)
		#w_50 = np.extract(condition,w_50)
		#w_50err = np.extract(condition,w_50err)
		#redch = np.extract(condition,redch)
		#P_phase = np.extract(condition,P_phase)
		#P_phaseErr = np.extract(condition,P_phaseErr)
		# Remove if tau = 0.0
		condition = tau != 0.
		tau = np.extract(condition,tau)
		terr = np.extract(condition,terr)
		freq = np.extract(condition,freq)
		w50 = np.extract(condition,w50)
		werr = np.extract(condition,werr)
		w_50 = np.extract(condition,w_50)
		w_50err = np.extract(condition,w_50err)
		redch = np.extract(condition,redch)
		P_phase = np.extract(condition,P_phase)
		P_phaseErr = np.extract(condition,P_phaseErr)
		# Removing 'inf' values from data
		pinf = float('+inf')
		condition = redch != pinf
		#tau = np.extract(condition,tau)
		#terr = np.extract(condition,terr)
		taunew = np.extract(condition,tau)
		terrnew = np.extract(condition,terr)
		#freq = np.extract(condition,freq)
		#w50 = np.extract(condition,w50)
		#werr = np.extract(condition,werr)
		#w_50 = np.extract(condition,w_50)
		#w_50err = np.extract(condition,w_50err)
		freqnew = np.extract(condition,freq)
		w50new = np.extract(condition,w50)
		werrnew = np.extract(condition,werr)
		w_50new = np.extract(condition,w_50)
		w_50errnew = np.extract(condition,w_50err)
		#redch = np.extract(condition,redch)
		redchnew = np.extract(condition,redch)
		P_phase = np.extract(condition,P_phase)
		P_phaseErr = np.extract(condition,P_phaseErr)
		## Removing data where tauerr > 0.5 tau
		#erravg = np.std(terr)
		#errmean = np.median(terr)
		#condition = terr <= (tau*0.5)
		#taunew = np.extract(condition,tau)
		#terrnew = np.extract(condition,terr)
		#freqnew = np.extract(condition,freq)
		#w50new = np.extract(condition,w50)
		#werrnew = np.extract(condition,werr)
		#w_50new = np.extract(condition,w_50)
		#w_50errnew = np.extract(condition,w_50err)
		#redchnew = np.extract(condition,redch)
		#P_phase = np.extract(condition,P_phase)
		#P_phaseErr = np.extract(condition,P_phaseErr)
		## Removing data with W50 == 0
		#condition = w50new != 0
		#freqnew = np.extract(condition,freqnew)
		#taunew  = np.extract(condition,taunew)
		#terrnew = np.extract(condition,terrnew)
		#w50new = np.extract(condition,w50new)
		#werrnew = np.extract(condition,werrnew)
		#w_50new = np.extract(condition,w_50new)
		#w_50errnew = np.extract(condition,w_50errnew)
		#redchnew = np.extract(condition,redchnew)
		#P_phase = np.extract(condition,P_phase)
		#P_phaseErr = np.extract(condition,P_phaseErr)

	# ------------------------------------------------------------------------- #
	# Finding the scattering scaling index (alpha) and pulse width scaling index
	# (beta) with a MCMC as explained in KJM17.
	# ------------------------------------------------------------------------- #

		if args.alpha_beta :
			print("Finding alpha and beta")
			# Fitting with a powerlaw model to remove outliers
			powerlaw = lambda x, amp1, alpha: amp1 * (x**alpha)
			fitfunc = lambda p, x: p[0] + p[1] * x
			errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err
			logx = np.log10(freqnew)
			logy = np.log10(taunew)
			logyerr = terrnew / taunew
			pinit = [1.0, -1.0]
			out = optimize.leastsq(errfunc, pinit,args=(logx, logy, logyerr),full_output=1)
			pfinal = out[0]
			covar = out[1]
			alpha = pfinal[1]
			amp1 = 10.0**pfinal[0]
			logy = np.log10(w50new)
			logyerr = werrnew / w50new
			pinit = [1.0, -1.0]
			out = optimize.leastsq(errfunc, pinit,args=(logx, logy, logyerr),full_output=1)
			pfinal = out[0]
			covar = out[1]
			beta = pfinal[1]
			amp2 = 10.0**pfinal[0]

			temp = taunew - powerlaw(freqnew,amp1,alpha)
			tempmed = np.median(temp)
			tempstd = robust.mad(temp)
			# Removing the large outliers using alpha fit
			condition = (temp > (tempmed - 1.5*tempstd)) & (temp < (tempmed + 1.5*tempstd))
			freqnew = np.extract(condition,freqnew)
			taunew  = np.extract(condition,taunew)
			terrnew = np.extract(condition,terrnew)
			w50new = np.extract(condition,w50new)
			werrnew = np.extract(condition,werrnew)
			w_50new = np.extract(condition,w_50new)
			w_50errnew = np.extract(condition,w_50errnew)
			redchnew = np.extract(condition,redchnew)

			temp = w50new - powerlaw(freqnew,amp2,beta)
			tempmed = np.median(temp)
			tempstd = robust.mad(temp)
			# Removing the large outliers using beta fit
			condition = (temp > (tempmed - 1.5*tempstd)) & (temp < (tempmed + 1.5*tempstd))
			freqnew = np.extract(condition,freqnew)
			taunew  = np.extract(condition,taunew)
			terrnew = np.extract(condition,terrnew)
			w50new = np.extract(condition,w50new)
			werrnew = np.extract(condition,werrnew)
			w_50new = np.extract(condition,w_50new)
			w_50errnew = np.extract(condition,w_50errnew)
			redchnew = np.extract(condition,redchnew)

			# Writing the cleaned values of freq, tau, w50_fit, redchi to output file  #
			#tmpdirtxt = dirtxt+"/"+ar.get_source()+"_"+str(mjd)+"_"+site+"_scatvals.txt"
			tmpdirtxt = "{}/{}_{}_nch{}_scatvals.txt".format(dirtxt, ar.get_source(),
					site, nchan)
			outfile = open(tmpdirtxt,"a")
			outfile.write('# MJD  Freq  Tau  Tau-err  w50  w50err  Redch\n')
			for i in range(0, np.size(taunew)):
				outfile.write(
					"{} {:06.4f} {:08.4f} {:06.4f} {:08.4f} {:06.4f} {:04.2f}\n".format(
					mjd, freqnew[i], taunew[i], terrnew[i], w50new[i],
					werrnew[i],	redchnew[i]))
			outfile.close()

			# 100k point MCMC for getting alpha and error bars
			ntau = np.size(taunew)
			len_mcmc = 100000
			# tau_sc part
			tauarr = np.zeros((len_mcmc, ntau))
			alparr = np.zeros(len_mcmc)
			amparr = np.zeros(len_mcmc)
			for i in range(0,len_mcmc):
				tauarr[i] = taunew + (terrnew *4* (np.random.rand(ntau) - 0.5))
				state = (np.any(tauarr[i] <  0.0))
				if (state==False):
					alparr[i],amparr[i] = curve_fit(f,np.log10(freqnew),
											np.log10(tauarr[i]))[0]
			# Getting the medial value of alpha and error bars
			alpha = np.median(alparr)
			alpha_low = abs(alpha*(-1) + np.percentile(alparr,5))
			alpha_high = abs(alpha*(-1) + np.percentile(alparr,95) )
			amp = np.median(10**amparr)
			amp_low =  np.percentile(10**amparr,5)
			amp_high = np.percentile(10**amparr,95)
			t150 = amp * 150**alpha
			t150err = t150 * ((alpha_low + alpha_high)/2) / 5.0
			alphaErr = ( (alpha_low+alpha_high)/2 )

			# W50 part
			w50arr = np.zeros((len_mcmc, ntau))
			beta_arr = np.zeros(len_mcmc)
			beta_amparr = np.zeros(len_mcmc)
			for i in range(0,len_mcmc):
				w50arr[i] = w50new + (werrnew * 4 * (np.random.rand(ntau) - 0.5) )
				state = (np.any(w50arr[i] <  0.0))
				if (state==False):
					beta_arr[i],beta_amparr[i] = curve_fit(f,np.log10(freqnew),
													np.log10(w50arr[i]))[0]
			# Getting the medial value of beta and error bars
			beta = np.median(beta_arr)
			beta_low = abs(beta*(-1) + np.percentile(beta_arr,5))
			beta_high = abs(beta*(-1) + np.percentile(beta_arr,95) )
			beta_amp = np.median(10**beta_amparr)
			betaErr = (beta_low + beta_high)/2
			w50150 = beta_amp * 150**beta
			w50150err = w50150 * ((beta_low + beta_high)/2) / 5.0

			# Printing the results out to a file and in the terminal
			print('# MJD       Tau    Tau-err   alpha  alpha-err   w50 w50err   beta betaerr  centfreq   bw  site  ')
			print(mjd, '%.4f' % t150, '%.4f' % t150err, '%.4f' % alpha, '%.4f' % alphaErr, \
						'%.4f' % w50150, '%.4f' % w50150err, '%.4f' % beta, '%.4f' % betaErr, \
						 '%.1f' % centfreq, '%.1f' % bw, '%s' % site)

			f1 = open("{}_nch{}_b{}_scatresults.txt".format(ar.get_source(),
					nchan, tsfact), 'a')
			f1.write('# MJD       Tau    Tau-err   alpha  alpha_low  alpha_high w50 w50err beta betaerr  centfreq   bw  site  \n')
			f1.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.1f} {:.1f} {}\n".format(mjd, t150, t150err, alpha, alpha_low, alpha_high, w50150, w50150err, beta, betaErr, centfreq, bw, site))
			f1.close()

			# Finding the y-limits for plotting purposes #
			tmin = min(taunew)
			tmax = max(taunew)
			ymin1 = tmin - tmin/10
			ymax1 = tmax + tmax/10
			wmin = min(w50new)
			wmax = max(w50new)
			ymin2 = wmin - wmin/10
			ymax2 = wmax + wmax/10

	    # ------------------------------------------------------------- #
	    # Plotting the fit results of alpha & beta
	    # ------------------------------------------------------------- #

		if args.alpha_beta:
			plt.clf()
			# Plot of alpha and good tau_sc
			plt.subplot(1,2,1)

			plt.plot(freqnew, powerlaw(freqnew, amp, alpha))
			plt.errorbar(freqnew, taunew, yerr=terrnew, fmt='k.')
			plt.xlabel('Frequency (MHz)',fontsize=15)
			plt.ylabel(r'$\tau_{sc}$ (ms)',fontsize=15)
			plt.title(r'$\alpha$ = $%5.2f^{+%.2f}_{-%.2f}$' % (alpha, alpha_high, alpha_low),
					fontsize=15)
			plt.xlim(lowband-(lowband/5),highband+(highband/5))
			plt.ylim(ymin1,ymax1)
			ax = plt.gca()
			ax.set_yscale('log')
			ax.set_xscale('log')
			plt.tick_params(axis='y', which='minor')
			plt.tick_params(axis='x', which='minor')
			ax.yaxis.set_minor_formatter(FormatStrFormatter("%d"))
			ax.xaxis.set_minor_formatter(FormatStrFormatter("%d"))
			plt.rcParams["figure.figsize"] = [6,18]

			# Plot of beta and good pulse widths of fitted Gaussian
			plt.subplot(1,2,2)
			plt.plot(freqnew, powerlaw(freqnew, beta_amp, beta))
			plt.errorbar(freqnew,w50new,yerr=werrnew,fmt='k.')
			plt.xlabel('Frequency (MHz)',fontsize=15)
			plt.ylabel('W50 (ms)',fontsize=15)
			plt.title(r'$\beta$ = $%5.2f^{+%.2f}_{-%.2f}$' % (beta, beta_high, beta_low),
					fontsize=15)
			plt.xlim(lowband-(lowband/5),highband+(highband/5))
			plt.ylim(ymin2,ymax2)
			ax = plt.gca()
			ax.set_yscale('log')
			ax.set_xscale('log')
			plt.tick_params(axis='y', which='minor')
			plt.tick_params(axis='x', which='minor')
			ax.yaxis.set_minor_formatter(FormatStrFormatter("%d"))
			ax.xaxis.set_minor_formatter(FormatStrFormatter("%d"))
			plt.rcParams["figure.figsize"] = [6,10]
			plt.subplots_adjust(wspace = 0.4)
			# Saving the Figure
			plt.savefig(diralp+"/"+ar.get_source()+"_"+str(mjd)+"_"+site+"_alphafit.eps", format='eps',
			orientation='portrait', bbox_inches='tight')
		# End
