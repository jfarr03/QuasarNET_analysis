#!/usr/bin/env python

import glob
from shutil import copytree
from os.path import isdir, mkdir
from variables import *

# Check to see if the base output directory exists. Make it if not.
if not isdir(OUTDIR):
    mkdir(OUTDIR)

# Make sub directories.
for i in ['data', 'outputs', 'qn_models']:
    d = OUTDIR+'/'+i
    if not isdir(d):
        mkdir(d)

## Make data directories.
# Make truth directory.
d = OUTDIR+'/data/truth'
if not isdir(d):
    mkdir(d)

# Make directories for each data type.
for i in ['coadd', 'bestexp', 'randexp']:
    d = OUTDIR+'/data/'+i
    if not isdir(d):
        mkdir(d)
    # Separate full and training data files.
    for i in ['full_datasets', 'training_datasets']:
        sd = d+'/'+i
        if not isdir(sd):
            mkdir(sd)

# Make directories for coadded training data.
for i in COADD_PROP_TRAINSIZES:
    d = OUTDIR+'/data/coadd/training_datasets/prop_{}'.format(i)
    if not isdir(d):
        mkdir(d)
for i in COADD_ABS_TRAINSIZES:
    d = OUTDIR+'/data/coadd/training_datasets/abs_{}'.format(i)
    if not isdir(d):
        mkdir(d)

# Make directories for bestexp training data.
for i in BESTEXP_PROP_TRAINSIZES:
    d = OUTDIR+'/data/bestexp/training_datasets/prop_{}'.format(i)
    if not isdir(d):
        mkdir(d)
for i in BESTEXP_ABS_TRAINSIZES:
    d = OUTDIR+'/data/bestexp/training_datasets/abs_{}'.format(i)
    if not isdir(d):
        mkdir(d)

# Make directories for randexp training data.
for i in RANDEXP_PROP_TRAINSIZES:
    d = OUTDIR+'/data/randexp/training_datasets/prop_{}'.format(i)
    if not isdir(d):
        mkdir(d)
for i in RANDEXP_ABS_TRAINSIZES:
    d = OUTDIR+'/data/randexp/training_datasets/abs_{}'.format(i)
    if not isdir(d):
        mkdir(d)

## Make QN models directories.
# Make directories for setup types.
for i in ['main_setup', 'additional_setups']:
    d = OUTDIR+'/qn_models/'+i
    if not isdir(d):
        mkdir(d)

# Make directories for each data type.
for i in ['coadd', 'bestexp', 'randexp']:
    d = OUTDIR+'/qn_models/main_setup/'+i
    if not isdir(d):
        mkdir(d)

# Make coadded model directories.
dirs = glob.glob(OUTDIR+'/data/coadd/training_datasets/*')
dirnames = [d.split('/')[-1] for d in dirs]
for i in range(len(dirs)):
    _ = copytree(dirs[i],OUTDIR+'/qn_models/main_setup/coadd/'+dirnames[i])

# Make bestexp model directories.
dirs = glob.glob(OUTDIR+'/data/bestexp/training_datasets/*')
dirnames = [d.split('/')[-1] for d in dirs]
for i in range(len(dirs)):
    _ = copytree(dirs[i],OUTDIR+'/qn_models/main_setup/bestexp/'+dirnames[i])

# Make randexp model directories.
dirs = glob.glob(OUTDIR+'/data/randexp/training_datasets/*')
dirnames = [d.split('/')[-1] for d in dirs]
for i in range(len(dirs)):
    _ = copytree(dirs[i],OUTDIR+'/qn_models/main_setup/randexp/'+dirnames[i])

# Make additional setups directories.
for i in ['offset_act', 'nepochs', 'dll_values', 'nchunks']:
    d = OUTDIR+'/qn_models/additional_setups/'+i
    if not isdir(d):
        mkdir(d)

# Make offset activation directories.
for i in OFFSET_ACT_FNS:
    d = OUTDIR+'/qn_models/additional_setups/offset_act/{}'.format(i)
    if not isdir(d):
        mkdir(d)

# Make nepochs directories.
for i in NEPOCH_PROP_TRAINSIZES:
    d = OUTDIR+'/qn_models/additional_setups/nepochs/prop_{}'.format(i)
    if not isdir(d):
        mkdir(d)

# Make dll_values directories.
for i in DLL_VALUES:
    d = OUTDIR+'/qn_models/additional_setups/dll_values/dll_{}'.format(i)
    if not isdir(d):
        mkdir(d)

# Make nchunks directories.
for i in NCHUNK_VALUES:
    d = OUTDIR+'/qn_models/additional_setups/nchunks/nchunks_{}'.format(i)
    if not isdir(d):
        mkdir(d)

# Make QN outputs directory.
d = OUTDIR+'/outputs/qn_outputs'
if not isdir(d):
    mkdir(d)

# Copy the directory structure from qn_models.
dirs = glob.glob(OUTDIR+'/qn_models/*')
dirnames = [d.split('/')[-1] for d in dirs]
for i in range(len(dirs)):
    _ = copytree(dirs[i],OUTDIR+'/qn_outputs/'+dirnames[i])
