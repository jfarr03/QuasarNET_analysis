import numpy as np
import os
import glob
from astropy.io import fits
from astropy.table import Table, unique, Column, vstack
import matplotlib.pyplot as plt
from qn_analysis.utils import *
from qn_analysis import variables
from multiprocessing import Pool

################################################################################

reduxdir = '/global/projecta/projectdirs/sdss/data/sdss/dr12/boss/spectro/redux/v5_7_?'
f_out = variables.OUTDIR+'/dr12pipe_results/dr12pipe_sdr12q.fits'
nproc = 32

f_sdr12q = '/global/projecta/projectdirs/sdss/data/sdss/dr12/boss/qso/DR12Q/Superset_DR12Q.fits'
f_spall_dr12 = '/global/projecta/projectdirs/sdss/data/sdss/dr12/boss/spectro/redux/spAll-DR12.fits'
f_spall_dr14 = '/global/projecta/projectdirs/sdss/data/sdss/dr15/eboss/spectro/redux/v5_10_0/spAll-v5_10_0.fits'

## This assumes that there are always 134 fits per spectrum
nfit = 134
## This keeps the top 10 best fits (in terms of reduced chi2)
nfit_keep = 10

################################################################################

## Check to see if the desired output file already exists.
if os.path.isfile(f_out):
    print('WARN: {} already exists, continuing will overwrite!'.format(f_out))

## Open Superset and spAll files.
sdr12q = fits.open(f_sdr12q)
spall_dr12 = fits.open(f_spall_dr12)
spall_dr14 = fits.open(f_spall_dr14)

## Extract data from DR12 spAll.
plate = spall_dr12[1].data["PLATE"]
mjd = spall_dr12[1].data["MJD"]
fid = spall_dr12[1].data["FIBERID"]
tid = spall_dr12[1].data["THING_ID"].astype(int)
pmf2tid_dr12 = {(p,m,f):t for p,m,f,t in zip(plate,mjd,fid,tid)}

## Find the spAll entries corresponding to SDR12Q objects.
w = np.in1d(spall_dr12[1].data['THING_ID'],sdr12q[1].data['THING_ID'])
w &= (spall_dr12[1].data['THING_ID']!=-1)
print('Of the {} spectra in spAll file, {} correspond to SDR12Q objects'.format(len(w),w.sum()))

## Make a data table with the key components in.
plate = spall_dr12[1].data['PLATE'][w].astype('i8')
mjd = spall_dr12[1].data['MJD'][w].astype('i8')
fiberid = spall_dr12[1].data['FIBERID'][w].astype('i8')
targetid = platemjdfiber2targetid(plate,mjd,fiberid)
spectype = spall_dr12[1].data['CLASS'][w]
z = spall_dr12[1].data['Z'][w]
zwarn = spall_dr12[1].data['ZWARNING'][w]
thing_id = [pmf2tid_dr12[(p,m,f)] for p,m,f in zip(plate,mjd,fiberid)]
cols = [targetid,spectype,z,zwarn,thing_id]
colnames = ['TARGETID','SPECTYPE','Z','ZWARN','THING_ID_DR12']
dtypes = ['i8','U8','f8','i8','i8']
table = Table(cols,names=colnames,dtype=dtypes)
print(' -> reduced spAll data to these {} spectra'.format(len(table)))

## Get which files we care about.
fiberid_dict = {}
pm_set = list(set(zip(plate,mjd)))
for p,m in pm_set:
    w = (np.in1d(plate,p) & np.in1d(mjd,m))
    fiberid_dict[(p,m)] = fiberid[w]

def get_spzall_table(p,m,fiberids,nfit,nfit_keep):
    f = glob.glob(reduxdir+'/{}/v5_7_?/spZall-{}-{}.fits'.format(p,p,m))[0]
    return make_spzall_table(f,fiberids,nfit,nfit_keep)

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

tasks = [(p,m,fiberid_dict[(p,m)],nfit,nfit_keep) for (p,m) in pm_set]
#Run the multiprocessing pool
if __name__ == '__main__':
    pool = Pool(processes=nproc)
    results = []
    for task in tasks:
        pool.apply_async(get_spzall_table,task,callback=log_result,error_callback=log_error)
    pool.close()
    pool.join()

## Concatenate the results.
spzall_table = vstack(results)
print('')

## Join the spZall results to the data from spAll.
table = join(table,spzall_table,keys=['TARGETID'],join_type='inner')

## Make sure that we only have objects that are in sdr12q.
plate,mjd,fiber = targetid2platemjdfiber(table['TARGETID'])
pmf = list(zip(plate,mjd,fiber))
print('{} spectra in DR12 pipeline output'.format(len(pmf)))

# Find spAll entries for objects in SDR12Q
wqso = np.in1d(spall_dr12[1].data["THING_ID"], sdr12q[1].data["THING_ID"]) & (spall_dr12[1].data["THING_ID"]>0)

# Make targetids for all these entries
plates = spall_dr12[1].data['PLATE'][wqso].astype('i8')
mjds = spall_dr12[1].data['MJD'][wqso].astype('i8')
fiberids = spall_dr12[1].data['FIBERID'][wqso].astype('i8')
targetid_sdr12q = targetid
print('{} spectra corresponding to SDR12Q objects in DR12 spAll'.format(len(targetid_sdr12q)))

# Check that the targetids line up 100%
w = np.in1d(table['TARGETID'],targetid_sdr12q)
print('Of the {} spectra in DR12 pipeline output, {} correspond to SDR12Q objects'.format(len(w),w.sum()))
table = table[w]
print(' -> reduced table to these {} spectra'.format(len(table)))
w = np.in1d(targetid_sdr12q,table['TARGETID'])
print('Of the {} spectra corresponding to SDR12Q objects, {} in DR12 pipeline output'.format(len(w),w.sum()))
targetid_sdr12q = targetid_sdr12q[w]
print(' -> reduced targetid_sdr12q array to these {} spectra'.format(len(targetid_sdr12q)))
table.sort(['TARGETID'])
targetid_sdr12q.sort()
assert (table['TARGETID'].data==targetid_sdr12q).all()

# Write the output file.
table.write(f_out,overwrite=True)
