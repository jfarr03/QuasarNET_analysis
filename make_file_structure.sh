#!/usr/bin/env bash
source ./variables

# Check to see if the base output directory exists. Make it if not.
if [ ! -d $OUTDIR ] ; then
    mkdir -p $OUTDIR
fi

# Make sub directories.
for i in data outputs qn_models; do
  d=$OUTDIR/$i
  if [ ! -d $d ] ; then
      mkdir -p $d
  fi
done

## Make data directories.
# Make directories for each data type.
for i in coadd bestexp randexp; do
  d=$OUTDIR/data/$i
  if [ ! -d $d ] ; then
      mkdir -p $d
  fi
  # Separate full and training data files.
  for i in full_datasets training_datasets; do
    sd=$d/$i
    if [ ! -d $sd ] ; then
        mkdir -p $sd
    fi
  done
done

# Make directories for coadded training data.
for i in $COADD_PROP_TRAINSIZES; do
  d=$OUTDIR/data/coadd/training_datasets/prop_$i
  if [ ! -d $d ] ; then
      mkdir -p $d
  fi
  for n in `seq 0 $(( $COADD_NMODEL-1 ))`; do
    sd=$d/model_${n}
    if [ ! -d $sd ] ; then
        mkdir -p $sd
    fi
  done
done
for i in $COADD_ABS_TRAINSIZES; do
  d=$OUTDIR/data/coadd/training_datasets/abs_$i
  if [ ! -d $d ] ; then
      mkdir -p $d
  fi
  for n in `seq 0 $(( $COADD_NMODEL-1 ))`; do
    sd=$d/model_${n}
    if [ ! -d $sd ] ; then
        mkdir -p $sd
    fi
  done
done

# Make directories for bestexp training data.
for i in $BESTEXP_PROP_TRAINSIZES; do
  d=$OUTDIR/data/bestexp/training_datasets/prop_$i
  if [ ! -d $d ] ; then
      mkdir -p $d
  fi
  for n in `seq 0 $(( $BESTEXP_NMODEL-1 ))`; do
    sd=$d/model_${n}
    if [ ! -d $sd ] ; then
        mkdir -p $sd
    fi
  done
done
for i in $BESTEXP_ABS_TRAINSIZES; do
  d=$OUTDIR/data/bestexp/training_datasets/abs_$i
  if [ ! -d $d ] ; then
      mkdir -p $d
  fi
  for n in `seq 0 $(( $BESTEXP_NMODEL-1 ))`; do
    sd=$d/model_${n}
    if [ ! -d $sd ] ; then
        mkdir -p $sd
    fi
  done
done

# Make directories for randexp training data.
for i in $RANDEXP_PROP_TRAINSIZES; do
  d=$OUTDIR/data/randexp/training_datasets/prop_$i
  if [ ! -d $d ] ; then
      mkdir -p $d
  fi
  for n in `seq 0 $(( $RANDEXP_NMODEL-1 ))`; do
    sd=$d/model_${n}
    if [ ! -d $sd ] ; then
        mkdir -p $sd
    fi
  done
done
for i in $RANDEXP_ABS_TRAINSIZES; do
  d=$OUTDIR/data/randexp/training_datasets/abs_$i
  if [ ! -d $d ] ; then
      mkdir -p $d
  fi
  for n in `seq 0 $(( $RANDEXP_NMODEL-1 ))`; do
    sd=$d/model_${n}
    if [ ! -d $sd ] ; then
        mkdir -p $sd
    fi
  done
done


## Make QN models directories.
# Make directories for setup types.
for i in main_setup additional_setups; do
  d=$OUTDIR/qn_models/$i
  if [ ! -d $d ] ; then
      mkdir -p $d
  fi
done

# Make directories for each data type.
for i in coadd bestexp randexp; do
  d=$OUTDIR/qn_models/main_setup/$i
  if [ ! -d $d ] ; then
      mkdir -p $d
  fi
done

# Make coadded model directories.
for i in $COADD_PROP_TRAINSIZES; do
  d=$OUTDIR/qn_models/main_setup/coadd/prop_$i
  if [ ! -d $d ] ; then
      mkdir -p $d
  fi
  for n in `seq 0 $(( $COADD_NMODEL-1 ))`; do
    sd=$d/model_${n}
    if [ ! -d $sd ] ; then
        mkdir -p $sd
    fi
  done
done
for i in $COADD_ABS_TRAINSIZES; do
  d=$OUTDIR/qn_models/main_setup/coadd/abs_$i
  if [ ! -d $d ] ; then
      mkdir -p $d
  fi
  for n in `seq 0 $(( $COADD_NMODEL-1 ))`; do
    sd=$d/model_${n}
    if [ ! -d $sd ] ; then
        mkdir -p $sd
    fi
  done
done

# Make bestexp model directories.
for i in $BESTEXP_PROP_TRAINSIZES; do
  d=$OUTDIR/qn_models/main_setup/bestexp/prop_$i
  if [ ! -d $d ] ; then
      mkdir -p $d
  fi
  for n in `seq 0 $(( $BESTEXP_NMODEL-1 ))`; do
    sd=$d/model_${n}
    if [ ! -d $sd ] ; then
        mkdir -p $sd
    fi
  done
done
for i in $BESTEXP_ABS_TRAINSIZES; do
  d=$OUTDIR/qn_models/main_setup/bestexp/abs_$i
  if [ ! -d $d ] ; then
      mkdir -p $d
  fi
  for n in `seq 0 $(( $BESTEXP_NMODEL-1 ))`; do
    sd=$d/model_${n}
    if [ ! -d $sd ] ; then
        mkdir -p $sd
    fi
  done
done

# Make randexp model directories.
for i in $RANDEXP_PROP_TRAINSIZES; do
  d=$OUTDIR/qn_models/main_setup/randexp/prop_$i
  if [ ! -d $d ] ; then
      mkdir -p $d
  fi
  for n in `seq 0 $(( $RANDEXP_NMODEL-1 ))`; do
    sd=$d/model_${n}
    if [ ! -d $sd ] ; then
        mkdir -p $sd
    fi
  done
done
for i in $RANDEXP_ABS_TRAINSIZES; do
  d=$OUTDIR/qn_models/main_setup/randexp/abs_$i
  if [ ! -d $d ] ; then
      mkdir -p $d
  fi
  for n in `seq 0 $(( $RANDEXP_NMODEL-1 ))`; do
    sd=$d/model_${n}
    if [ ! -d $sd ] ; then
        mkdir -p $sd
    fi
  done
done

# Make additional setups directories.
for i in offset_act nepochs dll_values nchunks; do
  d=$OUTDIR/qn_models/additional_setups/$i
  if [ ! -d $d ] ; then
      mkdir -p $d
  fi
done

# Make offset activation directories.
for i in $OFFSET_ACT_FNS; do
  d=$OUTDIR/qn_models/additional_setups/offset_act/$i
  if [ ! -d $d ] ; then
      mkdir -p $d
  fi
done

# Make nepochs directories.
d=$OUTDIR/qn_models/additional_setups/nepochs/
if [ ! -d $d ] ; then
    mkdir -p $d
fi

# Make dll_values directories.
for i in $DLL_VALUES; do
  d=$OUTDIR/qn_models/additional_setups/dll_values/dll_$i
  if [ ! -d $d ] ; then
      mkdir -p $d
  fi
done

# Make nchunks directories.
for i in $NCHUNK_VALUES; do
  d=$OUTDIR/qn_models/additional_setups/nchunks/nchunks_$i
  if [ ! -d $d ] ; then
      mkdir -p $d
  fi
done


# Make QN outputs directory.
if [ ! -d $OUTDIR/outputs/qn_outputs ] ; then
    mkdir -p $OUTDIR/outputs/qn_outputs
fi

# Copy the directory structure from qn_models.
cp -r $OUTDIR/qn_models/* $OUTDIR/outputs/qn_outputs
