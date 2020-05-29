import numpy as np
import os
import glob
from astropy.io import fits
from astropy.table import Table, unique, Column, vstack
import matplotlib.pyplot as plt
from qn_analysis.utils import *
from qn_analysis import variables
from multiprocessing import Pool

rrdir = '/global/cfs/projectdirs/desi/users/jfarr/rrboss_dr12/runs/rr_dr12_randexp/output/'
f_out = variables.OUTDIR+'/results/rr_results/rr_test_randexp.fits'
nproc = 32




f_sdr12q = '/global/projecta/projectdirs/sdss/data/sdss/dr12/boss/qso/DR12Q/Superset_DR12Q.fits'
sdr12q = fits.open(f_sdr12q)

f_spall_dr12 = '/global/projecta/projectdirs/sdss/data/sdss/dr12/boss/spectro/redux/spAll-DR12.fits'
spall_dr12 = fits.open(f_spall_dr12)

f_spall_dr14 = '/global/projecta/projectdirs/sdss/data/sdss/dr15/eboss/spectro/redux/v5_10_0/spAll-v5_10_0.fits'
spall_dr14 = fits.open(f_spall_dr14)

plate = spall_dr12[1].data["PLATE"]
mjd = spall_dr12[1].data["MJD"]
fid = spall_dr12[1].data["FIBERID"]
tid = spall_dr12[1].data["THING_ID"].astype(int)
pmf2tid_dr12 = {(p,m,f):t for p,m,f,t in zip(plate,mjd,fid,tid)}

if os.path.isfile(f_out):
    print('WARN: {} already exists, continuing will overwrite!'.format(f_out))

fi_zbest = glob.glob(rrdir+'/zbest*.fits')
fi_zbest.sort()
fi_rr = glob.glob(rrdir+'/redrock*.h5')
fi_rr.sort()
if len(fi_rr)==len(fi_zbest):
    print('INFO: found {} zbest and redrock files'.format(len(fi_rr)))
else:
    print('WARN: found {} zbest and {} redrock files'.format(len(fi_zbest),len(fi_rr)))

## Check that the zbest and redrock files are aligned.
for i in range(len(fi_rr)):
    assert fi_zbest[i][-15:-6]==fi_rr[i][-13:-4]

## Concatenate output from all zbest files.
rr_data = make_rr_table(fi_zbest[0],fi_rr[0])

#Define a progress- and error-tracking functions.
def log_result(retval):
    results.append(retval)
    N_complete = len(results)
    N_tasks = len(tasks)
    print('INFO: Read {:5d} files out of {:5d}...'.format(N_complete,N_tasks),end='\r')
    return
def log_error(retval):
    print('Error:',retval)
    return

tasks = [(fi_zbest[i],fi_rr[i]) for i in range(1,len(fi_rr))]
#Run the multiprocessing pool
if __name__ == '__main__':
    pool = Pool(processes=nproc)
    results = []
    for task in tasks:
        pool.apply_async(make_rr_table,task,callback=log_result,error_callback=log_error)
    pool.close()
    pool.join()

for result in results:
    rr_data = vstack([rr_data,result])

print('')
print(len(rr_data))
rr_data[:5]

## Make sure that we only have objects that are in sdr12q.
plate,mjd,fiber = targetid2platemjdfiber(rr_data['TARGETID'])
pmf_rr = list(zip(plate,mjd,fiber))
print('{} spectra in RR output'.format(len(pmf_rr)))

# Find spAll entries for objects in SDR12Q
wqso = np.in1d(spall_dr12[1].data["THING_ID"], sdr12q[1].data["THING_ID"]) & (spall_dr12[1].data["THING_ID"]>0)

# Make targetids for all these entries
plates = spall_dr12[1].data['PLATE'][wqso].astype('i8')
mjds = spall_dr12[1].data['MJD'][wqso].astype('i8')
fiberids = spall_dr12[1].data['FIBERID'][wqso].astype('i8')
targetid_sdr12q = platemjdfiber2targetid(plates,mjds,fiberids)

print('{} spectra corresponding to SDR12Q objects in DR12 spAll'.format(len(targetid_sdr12q)))

w = np.in1d(rr_data['TARGETID'],targetid_sdr12q)
print('Of the {} spectra in RR output, {} correspond to SDR12Q objects'.format(len(w),w.sum()))
rr_data = rr_data[w]
print(' -> reduced rr_data table to these {} spectra'.format(len(rr_data)))

w = np.in1d(targetid_sdr12q,rr_data['TARGETID'])
print('Of the {} spectra corresponding to SDR12Q objects, {} in RR output'.format(len(w),w.sum()))
targetid_sdr12q = targetid_sdr12q[w]
print(' -> reduced targetid_sdr12q array to these {} spectra'.format(len(targetid_sdr12q)))

rr_data.sort(['TARGETID'])
targetid_sdr12q.sort()

assert (rr_data['TARGETID'].data==targetid_sdr12q).all()

plate,mjd,fiber = targetid2platemjdfiber(rr_data['TARGETID'])
platemjdfiber = list(zip(plate,mjd,fiber))

rr_thing_id_dr12 = [pmf2tid_dr12[pmf] for pmf in platemjdfiber]

rr_data.add_column(Column(data=rr_thing_id_dr12,name='THING_ID_DR12',dtype='i8'))

rr_data.write(f_out,overwrite=True)
