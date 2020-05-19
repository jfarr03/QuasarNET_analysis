#!/usr/bin/env bash

BASE="/global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper/"

# Export all additional models.
for DLL in 0.002 0.0005; do
  MODEL="${BASE}/qn_models/additional_setups/dll_values/dll_${DLL}/qn_train_coadd_indtrain_0_0.h5"
  DATA="${BASE}/data/coadd/full_datasets/data_dr12_coadd_dll${DLL}.fits"
  TRAINING_DATA="${BASE}/data/coadd/training_datasets/prop_0.1/data_dr12_coadd_train_dll${DLL}_indtrain_0_0.fits"
  OUTDIR="${BASE}/outputs/qn_outputs/additional_setups/dll_values/dll_${DLL}/"
  OUTSUFFIX="train_0.1_coadd_0_0_dll${DLL}-test_coadd_dll${DLL}"
  if [ ! -d "$OUTDIR" ]; then
    mkdir $OUTDIR
  fi
  echo "Exporting coadd data, using a coadd data-trained model for training split 0, all using dll=${DLL}"
  cmd="qn_export --model $MODEL --data $DATA --training_data $TRAINING_DATA --dll $DLL --out-dir $OUTDIR --out-suffix $OUTSUFFIX"
  echo $cmd
  echo " "
done

for OAF in linear sigmoid; do
  MODEL="${BASE}/qn_models/additional_setups/offset_act/${OAF}/qn_train_coadd_indtrain_0_0.h5"
  DATA="${BASE}/data/coadd/full_datasets/data_dr12_coadd.fits"
  TRAINING_DATA="${BASE}/data/coadd/training_datasets/prop_0.1/data_dr12_coadd_train_indtrain_0_0.fits"
  DLL=0.001
  OUTDIR="${BASE}/outputs/qn_outputs/additional_setups/offset_act/${OAF}/"
  OUTSUFFIX="train_0.1_coadd_0_0_oaf${OAF}-test_coadd"
  if [ ! -d "$OUTDIR" ]; then
    mkdir $OUTDIR
  fi
  echo "Exporting coadd data, using a coadd data-trained model for training split 0, all using offset activation=${OAF}"
  cmd="qn_export --model $MODEL --data $DATA --training_data $TRAINING_DATA --dll $DLL --out-dir $OUTDIR --out-suffix $OUTSUFFIX"
  echo $cmd
  echo " "
done

for NCHUNKS in 7 10 16 19; do
  MODEL="${BASE}/qn_models/additional_setups/nchunks/nchunks_${NCHUNKS}/qn_train_coadd_indtrain_0_0.h5"
  DATA="${BASE}/data/coadd/full_datasets/data_dr12_coadd.fits"
  TRAINING_DATA="${BASE}/data/coadd/training_datasets/prop_0.1/data_dr12_coadd_train_indtrain_0_0.fits"
  DLL=0.001
  OUTDIR="${BASE}/outputs/qn_outputs/additional_setups/nchunks/nchunks_${NCHUNKS}/"
  OUTSUFFIX="train_0.1_coadd_0_0_nchunks${NCHUNKS}-test_coadd"
  if [ ! -d "$OUTDIR" ]; then
    mkdir $OUTDIR
  fi
  echo "Exporting coadd data, using a coadd data-trained model for training split 0, all using offset activation=${OAF}"
  cmd="qn_export --model $MODEL --data $DATA --training_data $TRAINING_DATA --dll $DLL --out-dir $OUTDIR --out-suffix $OUTSUFFIX"
  echo $cmd
  echo " "
done

# Export all 5% models.
for split in `seq 0 9`; do
  traintype=coadd
  testtype=coadd
  traintype_file=traintype
  if [ $traintype == randexp ]; then
    traintype_file=randexp_seed0
  fi
  testtype_file=testtype
  if [ $testtype == randexp ]; then
    testtype_file=randexp_seed0
  fi

  MODEL="${BASE}/qn_models/main_setup/${traintype}/prop_0.05/model_indtrain_0_${split}/qn_train_${traintype}_indtrain_0_${split}.h5"
  DATA="${BASE}/data/${testtype}/full_datasets/data_dr12_${testtype_file}.fits"
  TRAINING_DATA="${BASE}/data/${traintype}/training_datasets/prop_0.05/data_dr12_${traintype_file}_train_indtrain_0_${split}.fits"
  DLL=0.001
  OUTDIR="${BASE}/outputs/qn_outputs/main_setup/${traintype}/prop_0.05/model_indtrain_0_${split}/"
  OUTSUFFIX="train_0.05_${traintype}_0_${split}-test_${testtype}"
  if [ ! -d "$OUTDIR" ]; then
    mkdir $OUTDIR
  fi

  echo "Exporting $testtype data, using a $traintype data-trained model for training split $split"
  cmd="qn_export --model $MODEL --data $DATA --training_data $TRAINING_DATA --dll $DLL --out-dir $OUTDIR --out-suffix $OUTSUFFIX"
  echo $cmd
  echo " "
done

# Export all 10% models.
for split in `seq 0 9`; do
  for traintype in coadd bestexp randexp; do
    for testtype in coadd bestexp randexp; do
      traintype_file=traintype
      if [ $traintype == randexp ]; then
        traintype_file=randexp_seed0
      fi
      testtype_file=testtype
      if [ $testtype == randexp ]; then
        testtype_file=randexp_seed0
      fi

      MODEL="${BASE}/qn_models/main_setup/${traintype}/prop_0.1/model_indtrain_0_${split}/qn_train_${traintype}_indtrain_0_${split}.h5"
      DATA="${BASE}/data/${testtype}/full_datasets/data_dr12_${testtype_file}.fits"
      TRAINING_DATA="${BASE}/data/${traintype}/training_datasets/prop_0.1/data_dr12_${traintype_file}_train_indtrain_0_${split}.fits"
      DLL=0.001
      OUTDIR="${BASE}/outputs/qn_outputs/main_setup/${traintype}/prop_0.1/model_indtrain_0_${split}/"
      OUTSUFFIX="train_0.1_${traintype}_0_${split}-test_${testtype}"
      if [ ! -d "$OUTDIR" ]; then
        mkdir $OUTDIR
      fi

      echo "Exporting $testtype data, using a $traintype data-trained model for training split $split"
      cmd="qn_export --model $MODEL --data $DATA --training_data $TRAINING_DATA --dll $DLL --out-dir $OUTDIR --out-suffix $OUTSUFFIX"
      echo $cmd
      echo " "
    done
  done
done
