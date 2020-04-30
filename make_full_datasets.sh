#!/usr/bin/env bash
source ./variables

## Make coadded dataset.
cmd="parse_data --spplates $REDUXDIR/*/spPlate-*.fits --spall $SPALL --sdrq $SDRQ --out $OUTDIR/data/coadd/full_datasets/data_dr12.fits"
$cmd

## Make bestexp dataset.
cmd="parse_data --spplates $REDUXDIR/*/spPlate-*.fits --spall $SPALL --sdrq $SDRQ --out $OUTDIR/data/bestexp/full_datasets/data_dr12_best_exp.fits --use-best-exp"
$cmd

## Make randexp dataset.
cmd="parse_data --spplates $REDUXDIR/*/spPlate-*.fits --spall $SPALL --sdrq $SDRQ --out $OUTDIR/data/randexp/full_datasets/data_dr12_rand_exp_seed0.fits --use-random-exp --random-seed 0"
$cmd
