#!/usr/bin/env bash
source ./variables

## Load conda environment.
qnet

## Make coadded dataset.
parse_data --spplates $REDUXDIR/*/spPlate-*.fits --spall $SPALL --sdrq $SDRQ --out $OUTDIR/data/full_datasets/data_dr12.fits

## Make bestexp dataset.
parse_data --spplates $REDUXDIR/*/spPlate-*.fits --spall $SPALL --sdrq $SDRQ --out $OUTDIR/data/full_datasets/data_dr12.fits --use-best-exp

## Make randexp dataset.
parse_data --spplates $REDUXDIR/*/spPlate-*.fits --spall $SPALL --sdrq $SDRQ --out $OUTDIR/data/full_datasets/data_dr12.fits --use-random-exp --random-seed 0
