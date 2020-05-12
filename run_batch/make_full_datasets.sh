#!/usr/bin/env bash

## Make truth file.
cmd="qn_parse_truth --sdrq /global/projecta/projectdirs/sdss/data/sdss/dr12/boss/qso/DR12Q/Superset_DR12Q.fits --drq /global/projecta/projectdirs/sdss/data/sdss/dr12/boss/qso/DR12Q/DR12Q.fits --out /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/truth/truth_dr12q.fits --mode BOSS"
$cmd

## Make coadded dataset.
cmd="parse_data --spplates /global/cfs/projectdirs/desi/users/jfarr/BOSS_DR12_redux_replica//*/spPlate-*.fits --spall /global/projecta/projectdirs/sdss/data/sdss/dr12/boss/spectro/redux/spAll-DR12.fits --sdrq /global/projecta/projectdirs/sdss/data/sdss/dr12/boss/qso/DR12Q/Superset_DR12Q.fits --out /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/coadd/full_datasets/data_dr12_coadd.fits"
$cmd

## Make bestexp dataset.
cmd="parse_data --spplates /global/cfs/projectdirs/desi/users/jfarr/BOSS_DR12_redux_replica//*/spPlate-*.fits --spall /global/projecta/projectdirs/sdss/data/sdss/dr12/boss/spectro/redux/spAll-DR12.fits --sdrq /global/projecta/projectdirs/sdss/data/sdss/dr12/boss/qso/DR12Q/Superset_DR12Q.fits --out /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/bestexp/full_datasets/data_dr12_bestexp.fits --use-best-exp"
$cmd

## Make randexp dataset.
cmd="parse_data --spplates /global/cfs/projectdirs/desi/users/jfarr/BOSS_DR12_redux_replica//*/spPlate-*.fits --spall /global/projecta/projectdirs/sdss/data/sdss/dr12/boss/spectro/redux/spAll-DR12.fits --sdrq /global/projecta/projectdirs/sdss/data/sdss/dr12/boss/qso/DR12Q/Superset_DR12Q.fits --out /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/randexp/full_datasets/data_dr12_randexp_seed0.fits --use-random-exp --random-seed 0"
$cmd

