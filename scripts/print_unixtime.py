import os

import h5py
import numpy as np
import glob
from datetime import datetime, timedelta
import time 

# Apertif detections
tarr = ([1589208001.096+4387.14,
         1589208001.096+1889.1,
         1589208001.096+1495.37,
         1589193745.992+7153.32,
         1589193745.992+4668.0,
         1589193745.992+2688.72,
         1589193745.992+2317.14,
         1589193745.992+1780.7,
         1589193745.992+4.14925,
         1589182582.3440003+6219.06,
         1589182582.3440003+3610.84])

tarr += 4148*350.*np.abs(1519.7**-2 - 200**-2)

files = glob.glob('/data/projects/COM_ALERT/L*/*s/r3')
files.sort()

for fn in files:
    continue
    files_again = fn[:-2]+'*.h5'
    fn = glob.glob(files_again)[0]
    file = h5py.File(fn, 'r')
    start_time_file=datetime.strptime(file.attrs[u'OBSERVATION_START_UTC'][0:19],'%Y-%m-%dT%H:%M:%S')
    end_time_file=datetime.strptime(file.attrs[u'OBSERVATION_END_UTC'][0:19],'%Y-%m-%dT%H:%M:%S')
    start_time_file_unix=time.mktime(start_time_file.timetuple())
    end_time_file_unix=time.mktime(end_time_file.timetuple())
    for tt in tarr:
        if tt>start_time_file_unix and tt<end_time_file_unix:
            print(tt, fn)

### LOFAR OBSERVATION ###
#files = glob.glob('/data/projects/COM_ALERT/L783*/*s/*P000_bf*.h5')
files = glob.glob('/data/projects/COM_ALERT/L*/*s/*SAP000*S0_P000_bf*.h5')
files.sort()

print("Total of %d files" % len(files))
Ttot=0

obsname = []
startdate = []
startmjd = []
duration = []
nof_stations = []
mode = []

for ii, fn in enumerate(files):
    f = h5py.File(fn,'r')
    if f.attrs['TARGETS'][0] == 'R3':
        obsname.append(f.attrs['OBSERVATION_ID'])
        startdate.append(f.attrs['OBSERVATION_START_UTC'].split('.')[0].replace('T',
                ' '))
        startmjd.append(f.attrs['OBSERVATION_START_MJD'])
        duration.append(f.attrs['TOTAL_INTEGRATION_TIME']) # {:.2f}
        nof_stations.append(f.attrs['OBSERVATION_NOF_STATIONS']) # HBA, HBA0, HBA1

        if file.split('/')[-2] == 'cs':
            m = 'Coherent '
        elif  file.split('/')[-2] == 'is':
            m = 'Incoherent '
        else :
            print('Warning: no CS/IS found')

        dir = file.rsplit('/', 1)[0] + '/*SAP000*.h5'
        bf_files = glob.glob(dir)
        if len(bf_files) == 1:
            m += 'I'
        elif len(bf_files) == 4:
            m += 'IQUV'
        else :
            print('Warning: no stokes found')
        mode.append(m)

        dt = f['SUB_ARRAY_POINTING_000']['BEAM_000'].attrs['SAMPLING_TIME']
        N = f['SUB_ARRAY_POINTING_000']['BEAM_000'].attrs['NOF_SAMPLES']
        ttot = N*dt
        Ttot += ttot
        print("{} {} {} {:.2f}".format(fn, src, start, ttot))

    #print("{} {} {} {}".format(fn, src, start_time_file, expt))

a = np.array([obsname, startdate, startmjd, duration, mode, 
        nof_stations]).transpose()
a = a[a[:,2].argsort()]

# Printing formatted table
for i in range(len(a)):
    print("{} & {} & {:.7f} & {:.2f} & {} & {} & ...\\\\".format(a[i,0], a[i,1], 
            float(a[i,2]), float(a[i,3]), a[i,4], a[i,5]))

print("Total time on R3: {:.2f} s = {:.2f} h".format(Ttot, Ttot/3600))
