#!/usr/bin/env python

from variables import *
import glob
from os.path import stat
import os

truth = '{}/data/truth/truth_dr12q.fits'.format(OUTDIR)

## Make prepare_models.sh
run_file = 'make_runfiles.sh'

run_file_text = '#!/usr/bin/env bash\n\n'

for p in COADD_PROP_TRAINSIZES:
    run_file_text += 'echo "Preparing for training models on coadded data with training proportion {}..."\n'.format(p)
    train_dir = '{}/data/coadd/training_datasets/prop_{}/'.format(OUTDIR,p)
    train_prefix = 'data_dr12_coadd_train'
    training_sets = glob.glob(train_dir+train_prefix+'_*.fits')
    splits = [training_set.split(train_prefix)[-1][1:-5] for training_set in training_sets]
    splits.sort()
    nhours = PROP_JOB_TIMES[p]
    for split in splits:
        output_dir = '{}/qn_models/main_setup/coadd/prop_{}/model_{}'.format(OUTDIR,p,split)
        output_prefix = 'qn_train_coadd'
        run_file_text += './prepare_single_model.py --training-dir {} --training-prefix {} --split {} --truth {} --nhours {} --output-dir {} --output-prefix {}\n'.format(train_dir,train_prefix,split,truth,nhours,output_dir,output_prefix)
    run_file_text += 'echo " -> Done!"\n'
    run_file_text += 'echo " "\n\n'

for a in COADD_ABS_TRAINSIZES:
    run_file_text += 'echo "Preparing for training models on coadded data with training size {}..."\n'.format(a)
    train_dir = '{}/data/coadd/training_datasets/abs_{}/'.format(OUTDIR,a)
    train_prefix = 'data_dr12_coadd_train'
    training_sets = glob.glob(train_dir+train_prefix+'_*.fits')
    splits = [training_set.split(train_prefix)[-1][1:-5] for training_set in training_sets]
    splits.sort()
    nhours = ABS_JOB_TIMES[a]
    for split in splits:
        output_dir = '{}/qn_models/main_setup/coadd/abs_{}/model_{}'.format(OUTDIR,a,split)
        output_prefix = 'qn_train_coadd'
        run_file_text += './prepare_single_model.py --training-dir {} --training-prefix {} --split {} --truth {} --nhours {} --output-dir {} --output-prefix {}\n'.format(train_dir,train_prefix,split,truth,nhours,output_dir,output_prefix)
    run_file_text += 'echo " -> Done!"\n'
    run_file_text += 'echo " "\n\n'

for p in BESTEXP_PROP_TRAINSIZES:
    run_file_text += 'echo "Preparing for training models on best exposure data with training proportion {}..."\n'.format(p)
    train_dir = '{}/data/bestexp/training_datasets/prop_{}/'.format(OUTDIR,p)
    train_prefix = 'data_dr12_bestexp_train'
    training_sets = glob.glob(train_dir+train_prefix+'_*.fits')
    splits = [training_set.split(train_prefix)[-1][1:-5] for training_set in training_sets]
    splits.sort()
    nhours = PROP_JOB_TIMES[p]
    for split in splits:
        output_dir = '{}/qn_models/main_setup/bestexp/prop_{}/model_{}'.format(OUTDIR,p,split)
        output_prefix = 'qn_train_bestexp'
        run_file_text += './prepare_single_model.py --training-dir {} --training-prefix {} --split {} --truth {} --nhours {} --output-dir {} --output-prefix {}\n'.format(train_dir,train_prefix,split,truth,nhours,output_dir,output_prefix)
    run_file_text += 'echo " -> Done!"\n'
    run_file_text += 'echo " "\n\n'

for a in BESTEXP_ABS_TRAINSIZES:
    run_file_text += 'echo "Preparing for training models on best exposure data with training size {}..."\n'.format(a)
    train_dir = '{}/data/bestexp/training_datasets/abs_{}/'.format(OUTDIR,a)
    train_prefix = 'data_dr12_bestexp_train'
    training_sets = glob.glob(train_dir+train_prefix+'_*.fits')
    splits = [training_set.split(train_prefix)[-1][1:-5] for training_set in training_sets]
    splits.sort()
    nhours = ABS_JOB_TIMES[a]
    for split in splits:
        output_dir = '{}/qn_models/main_setup/bestexp/abs_{}/model_{}'.format(OUTDIR,a,split)
        output_prefix = 'qn_train_bestexp'
        run_file_text += './prepare_single_model.py --training-dir {} --training-prefix {} --split {} --truth {} --nhours {} --output-dir {} --output-prefix {}\n'.format(train_dir,train_prefix,split,truth,nhours,output_dir,output_prefix)
    run_file_text += 'echo " -> Done!"\n'
    run_file_text += 'echo " "\n\n'

for p in RANDEXP_PROP_TRAINSIZES:
    run_file_text += 'echo "Preparing for training models on random exposure data with training proportion {}..."\n'.format(p)
    train_dir = '{}/data/randexp/training_datasets/prop_{}/'.format(OUTDIR,p)
    train_prefix = 'data_dr12_randexp_seed0_train'
    training_sets = glob.glob(train_dir+train_prefix+'_*.fits')
    splits = [training_set.split(train_prefix)[-1][1:-5] for training_set in training_sets]
    splits.sort()
    nhours = PROP_JOB_TIMES[p]
    for split in splits:
        output_dir = '{}/qn_models/main_setup/randexp/prop_{}/model_{}'.format(OUTDIR,p,split)
        output_prefix = 'qn_train_randexp'
        run_file_text += './prepare_single_model.py --training-dir {} --training-prefix {} --split {} --truth {} --nhours {} --output-dir {} --output-prefix {}\n'.format(train_dir,train_prefix,split,truth,nhours,output_dir,output_prefix)
    run_file_text += 'echo " -> Done!"\n'
    run_file_text += 'echo " "\n\n'

for a in RANDEXP_ABS_TRAINSIZES:
    run_file_text += 'echo "Preparing for training models on random exposure data with training size {}..."\n'.format(a)
    train_dir = '{}/data/randexp/training_datasets/abs_{}/'.format(OUTDIR,a)
    train_prefix = 'data_dr12_randexp_seed0_train'
    training_sets = glob.glob(train_dir+train_prefix+'_*.fits')
    splits = [training_set.split(train_prefix)[-1][1:-5] for training_set in training_sets]
    splits.sort()
    nhours = ABS_JOB_TIMES[a]
    for split in splits:
        output_dir = '{}/qn_models/main_setup/randexp/abs_{}/model_{}'.format(OUTDIR,a,split)
        output_prefix = 'qn_train_randexp'
        run_file_text += './prepare_single_model.py --training-dir {} --training-prefix {} --split {} --truth {} --nhours {} --output-dir {} --output-prefix {}\n'.format(train_dir,train_prefix,split,truth,nhours,output_dir,output_prefix)
    run_file_text += 'echo " -> Done!"\n'
    run_file_text += 'echo " "\n\n'

run_file_text += 'echo "Preparing for training models on coadd data with training size {} with varying offset activation functions..."\n'.format(OFFSET_ACT_PROP_TRAINSIZE)
for oaf in OFFSET_ACT_FNS:
    train_dir = '{}/data/coadd/training_datasets/prop_{}/'.format(OUTDIR,OFFSET_ACT_PROP_TRAINSIZE)
    train_prefix = 'data_dr12_coadd_train'
    training_sets = glob.glob(train_dir+train_prefix+'_*.fits')
    splits = [training_set.split(train_prefix)[-1][1:-5] for training_set in training_sets]
    splits.sort()
    split = splits[0]
    nhours = PROP_JOB_TIMES[OFFSET_ACT_PROP_TRAINSIZE]
    output_dir = '{}/qn_models/additional_setups/offset_act/{}'.format(OUTDIR,oaf)
    output_prefix = 'qn_train_coadd'
    run_file_text += './prepare_single_model.py --training-dir {} --training-prefix {} --split {} --truth {} --nhours {} --output-dir {} --output-prefix {} --offset-activation-function {}\n'.format(train_dir,train_prefix,split,truth,nhours,output_dir,output_prefix,oaf)
run_file_text += 'echo " -> Done!"\n'
run_file_text += 'echo " "\n\n'

run_file_text += 'echo "Preparing for training models on coadd data with training size {}, outputting models at each epoch..."\n'.format(p)
for p in NEPOCH_PROP_TRAINSIZES:
    train_dir = '{}/data/coadd/training_datasets/prop_{}/'.format(OUTDIR,p)
    train_prefix = 'data_dr12_coadd_train'
    training_sets = glob.glob(train_dir+train_prefix+'_*.fits')
    splits = [training_set.split(train_prefix)[-1][1:-5] for training_set in training_sets]
    splits.sort()
    split = splits[0]
    nhours = PROP_JOB_TIMES[p]*(NEPOCH_MAX/200)
    output_dir = '{}/qn_models/additional_setups/nepochs/prop_{}/'.format(OUTDIR,p)
    output_prefix = 'qn_train_coadd'
    run_file_text += './prepare_single_model.py --training-dir {} --training-prefix {} --split {} --truth {} --nhours {} --output-dir {} --output-prefix {} --nepochs {} --save-epoch-checkpoints\n'.format(train_dir,train_prefix,split,truth,nhours,output_dir,output_prefix,NEPOCH_MAX)
run_file_text += 'echo " -> Done!"\n'
run_file_text += 'echo " "\n\n'

run_file_text += 'echo "Preparing for training models on coadd data with training size {} with varying dll values..."\n'.format(DLL_PROP_TRAINSIZE)
for dll in DLL_VALUES:
    train_dir = '{}/data/coadd/training_datasets/prop_{}/'.format(OUTDIR,DLL_PROP_TRAINSIZE)
    train_prefix = 'data_dr12_coadd_train'
    training_sets = glob.glob(train_dir+train_prefix+'_*.fits')
    splits = [training_set.split(train_prefix)[-1][1:-5] for training_set in training_sets]
    splits.sort()
    split = splits[0]
    nhours = PROP_JOB_TIMES[DLL_PROP_TRAINSIZE]
    output_dir = '{}/qn_models/additional_setups/dll_values/dll_{}'.format(OUTDIR,dll)
    output_prefix = 'qn_train_coadd'
    run_file_text += './prepare_single_model.py --training-dir {} --training-prefix {} --split {} --truth {} --nhours {} --output-dir {} --output-prefix {} -dll {}\n'.format(train_dir,train_prefix,split,truth,nhours,output_dir,output_prefix,dll)
run_file_text += 'echo " -> Done!"\n'
run_file_text += 'echo " "\n\n'

run_file_text += 'echo "Preparing for training models on coadd data with training size {} with varying nchunk values..."\n'.format(NCHUNK_PROP_TRAINSIZE)
for nchunk in NCHUNK_VALUES:
    train_dir = '{}/data/coadd/training_datasets/prop_{}/'.format(OUTDIR,NCHUNK_PROP_TRAINSIZE)
    train_prefix = 'data_dr12_coadd_train'
    training_sets = glob.glob(train_dir+train_prefix+'_*.fits')
    splits = [training_set.split(train_prefix)[-1][1:-5] for training_set in training_sets]
    splits.sort()
    split = splits[0]
    nhours = PROP_JOB_TIMES[NCHUNK_PROP_TRAINSIZE]
    output_dir = '{}/qn_models/additional_setups/nchunks/nchunks_{}'.format(OUTDIR,nchunk)
    output_prefix = 'qn_train_coadd'
    run_file_text += './prepare_single_model.py --training-dir {} --training-prefix {} --split {} --truth {} --nhours {} --output-dir {} --output-prefix {} --nchunks {}\n'.format(train_dir,train_prefix,split,truth,nhours,output_dir,output_prefix,nchunk)
run_file_text += 'echo " -> Done!"\n'
run_file_text += 'echo " "\n\n'

with open(run_file, 'w') as file:
    file.write(run_file_text)
os.chmod(run_file,stat.S_IRWXU | stat.S_IRWXG)
