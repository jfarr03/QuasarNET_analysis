#!/usr/bin/env bash
source variables

## Make coadd datasets.
for p in $COADD_PROP_TRAINSIZES; do
  echo "Making coadded training sets with training proportion $p..."
  #echo "splitting $OUTDIR/data/coadd/full_datasets/data_dr12.fits into $COADD_NMODEL splits with training proportion $p (using --auto-independent), and saving to location with prefix $OUTDIR/data/coadd/training_datasets/prop_${p}/data_dr12_train"
  split_data --data $OUTDIR/data/coadd/full_datasets/data_dr12.fits --nsplits $COADD_NMODEL --training-proportion $p --auto-independent --out-prefix $OUTDIR/data/coadd/training_datasets/prop_${p}/data_dr12_train
  echo " -> Done!"
  echo " "
done
for a in $COADD_ABS_TRAINSIZES; do
  echo "Making coadded training sets with training size $a..."
  split_data --data $OUTDIR/data/coadd/full_datasets/data_dr12.fits --nsplits $COADD_NMODEL --training-number $a --auto-independent --out-prefix $OUTDIR/data/coadd/training_datasets/abs_${a}/data_dr12_train
  echo " -> Done!"
  echo " "
done

## Make bestexp datasets.
for p in $BESTEXP_PROP_TRAINSIZES; do
  echo "Making best exposure training sets with training proportion $p..."
  split_data --data $OUTDIR/data/bestexp/full_datasets/data_dr12_best_exp.fits --nsplits $COADD_NMODEL --training-proportion $p --auto-independent --out-prefix $OUTDIR/data/bestexp/training_datasets/prop_${p}/data_dr12_best_exp_train
  echo " -> Done!"
  echo " "
done
for a in $COADD_ABS_TRAINSIZES; do
  echo "Making coadded training sets with training size $a..."
  split_data --data $OUTDIR/data/bestexp/full_datasets/data_dr12_best_exp.fits --nsplits $COADD_NMODEL --training-number $a --auto-independent --out-prefix $OUTDIR/data/bestexp/training_datasets/abs_${a}/data_dr12_best_exp_train
  echo " -> Done!"
  echo " "
done

## Make bestexp datasets.
for p in $RANDEXP_PROP_TRAINSIZES; do
  echo "Making random exposure training sets with training proportion $p..."
  split_data --data $OUTDIR/data/randexp/full_datasets/data_dr12_rand_exp_seed0.fits --nsplits $COADD_NMODEL --training-proportion $p --auto-independent --out-prefix $OUTDIR/data/randexp/training_datasets/prop_${p}/data_dr12_rand_exp_seed0_train
  echo " -> Done!"
  echo " "
done
for a in $COADD_ABS_TRAINSIZES; do
  echo "Making coadded training sets with training size $a..."
  split_data --data $OUTDIR/data/randexp/full_datasets/data_dr12_rand_exp_seed0.fits --nsplits $COADD_NMODEL --training-number $a --auto-independent --out-prefix $OUTDIR/data/randexp/training_datasets/abs_${a}/data_dr12_rand_exp_seed0_train
  echo " -> Done!"
  echo " "
done
