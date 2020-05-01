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
    nhours = PROP_JOB_TIMES[p]
    for split in splits:
        output_dir = '{}/qn_models/coadd/prop_{}/model_{}'.format(OUTDIR,p,split)
        output_prefix = 'qn_train'
        run_file_text += './prepare_single_model.py --training_dir {} --train-prefix {} --split {} --truth {} --nhours {} --output-dir {} --output-prefix {}\n'.format(train_dir,train_prefix,split,truth,nhours,output_dir,output_prefix)
    run_file_text += 'echo " -> Done!"\n'
    run_file_text += 'echo " "\n\n'

for a in COADD_ABS_TRAINSIZES:
    run_file_text += 'echo "Preparing for training models on coadded data with training size {}..."\n'.format(a)
    train_dir = '{}/data/coadd/training_datasets/abs_{}/'.format(OUTDIR,p)
    train_prefix = 'data_dr12_coadd_train'
    training_sets = glob.glob(train_dir+train_prefix+'_*.fits')
    splits = [training_set.split(train_prefix)[-1][1:-5] for training_set in training_sets]
    nhours = PROP_JOB_TIMES[p]
    for split in splits:
        output_dir = '{}/qn_models/coadd/abs_{}/model_{}'.format(OUTDIR,a,split)
        output_prefix = 'qn_train'
        run_file_text += './prepare_single_model.py --training_dir {} --train-prefix {} --split {} --truth {} --nhours {} --output-dir {} --output-prefix {}\n'.format(train_dir,train_prefix,split,truth,nhours,output_dir,output_prefix)
    run_file_text += 'echo " -> Done!"\n'
    run_file_text += 'echo " "\n\n'

for p in BESTEXP_PROP_TRAINSIZES:
    run_file_text += 'echo "Preparing for training models on best exposure data with training proportion {}..."\n'.format(p)
    train_dir = '{}/data/bestexp/training_datasets/prop_{}/'.format(OUTDIR,p)
    train_prefix = 'data_dr12_bestexp_train'
    training_sets = glob.glob(train_dir+train_prefix+'_*.fits')
    splits = [training_set.split(train_prefix)[-1][1:-5] for training_set in training_sets]
    nhours = PROP_JOB_TIMES[p]
    for split in splits:
        output_dir = '{}/qn_models/bestexp/prop_{}/model_{}'.format(OUTDIR,p,split)
        output_prefix = 'qn_train_bestexp'
        run_file_text += './prepare_single_model.py --training_dir {} --train-prefix {} --split {} --truth {} --nhours {} --output-dir {} --output-prefix {}\n'.format(train_dir,train_prefix,split,truth,nhours,output_dir,output_prefix)
    run_file_text += 'echo " -> Done!"\n'
    run_file_text += 'echo " "\n\n'

for a in BESTEXP_ABS_TRAINSIZES:
    run_file_text += 'echo "Preparing for training models on best exposure data with training size {}..."\n'.format(a)
    train_dir = '{}/data/bestexp/training_datasets/abs_{}/'.format(OUTDIR,p)
    train_prefix = 'data_dr12_bestexp_train'
    training_sets = glob.glob(train_dir+train_prefix+'_*.fits')
    splits = [training_set.split(train_prefix)[-1][1:-5] for training_set in training_sets]
    nhours = PROP_JOB_TIMES[p]
    for split in splits:
        output_dir = '{}/qn_models/bestexp/abs_{}/model_{}'.format(OUTDIR,a,split)
        output_prefix = 'qn_train_bestexp'
        run_file_text += './prepare_single_model.py --training_dir {} --train-prefix {} --split {} --truth {} --nhours {} --output-dir {} --output-prefix {}\n'.format(train_dir,train_prefix,split,truth,nhours,output_dir,output_prefix)
    run_file_text += 'echo " -> Done!"\n'
    run_file_text += 'echo " "\n\n'

for p in RANDEXP_PROP_TRAINSIZES:
    run_file_text += 'echo "Preparing for training models on random exposure data with training proportion {}..."\n'.format(p)
    train_dir = '{}/data/randexp/training_datasets/prop_{}/'.format(OUTDIR,p)
    train_prefix = 'data_dr12_randexp_train'
    training_sets = glob.glob(train_dir+train_prefix+'_*.fits')
    splits = [training_set.split(train_prefix)[-1][1:-5] for training_set in training_sets]
    nhours = PROP_JOB_TIMES[p]
    for split in splits:
        output_dir = '{}/qn_models/randexp/prop_{}/model_{}'.format(OUTDIR,p,split)
        output_prefix = 'qn_train_randexp'
        run_file_text += './prepare_single_model.py --training_dir {} --train-prefix {} --split {} --truth {} --nhours {} --output-dir {} --output-prefix {}\n'.format(train_dir,train_prefix,split,truth,nhours,output_dir,output_prefix)
    run_file_text += 'echo " -> Done!"\n'
    run_file_text += 'echo " "\n\n'

for a in RANDEXP_ABS_TRAINSIZES:
    run_file_text += 'echo "Preparing for training models on random exposure data with training size {}..."\n'.format(a)
    train_dir = '{}/data/randexp/training_datasets/abs_{}/'.format(OUTDIR,p)
    train_prefix = 'data_dr12_randexp_train'
    training_sets = glob.glob(train_dir+train_prefix+'_*.fits')
    splits = [training_set.split(train_prefix)[-1][1:-5] for training_set in training_sets]
    nhours = PROP_JOB_TIMES[p]
    for split in splits:
        output_dir = '{}/qn_models/randexp/abs_{}/model_{}'.format(OUTDIR,a,split)
        output_prefix = 'qn_train_randexp'
        run_file_text += './prepare_single_model.py --training_dir {} --train-prefix {} --split {} --truth {} --nhours {} --output-dir {} --output-prefix {}\n'.format(train_dir,train_prefix,split,truth,nhours,output_dir,output_prefix)
    run_file_text += 'echo " -> Done!"\n'
    run_file_text += 'echo " "\n\n'



with open(run_file, 'w') as file:
    file.write(run_file_text)
os.chmod(run_file,stat.S_IRWXU | stat.S_IRWXG)
