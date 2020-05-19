#!/usr/bin/env python

from qn_analysis.variables import *
from os.path import stat
import os

run_file = PREPDIR+'/run_batch/make_training_datasets.sh'

run_file_text = '#!/usr/bin/env bash\n\n'

for p in COADD_PROP_TRAINSIZES:
    run_file_text += 'echo "Making coadded training sets with training proportion {}..."\n'.format(p)
    run_file_text += 'split_data --data {}/data/coadd/full_datasets/data_dr12_coadd.fits --nsplits {} --training-proportion {} --auto-independent --out-prefix {}/data/coadd/training_datasets/prop_{}/data_dr12_coadd_train\n'.format(OUTDIR,COADD_NMODEL,p,OUTDIR,p)
    run_file_text += 'echo " -> Done!"\n'
    run_file_text += 'echo " "\n\n'

for a in COADD_ABS_TRAINSIZES:
    run_file_text += 'echo "Making coadded training sets with training size {}..."\n'.format(a)
    run_file_text += 'split_data --data {}/data/coadd/full_datasets/data_dr12_coadd.fits --nsplits {} --training-number {} --auto-independent --out-prefix {}/data/coadd/training_datasets/abs_{}/data_dr12_coadd_train\n'.format(OUTDIR,COADD_NMODEL,a,OUTDIR,a)
    run_file_text += 'echo " -> Done!"\n'
    run_file_text += 'echo " "\n\n'

for p in BESTEXP_PROP_TRAINSIZES:
    run_file_text += 'echo "Making best exposure training sets with training proportion {}..."\n'.format(p)
    run_file_text += 'split_data --data {}/data/bestexp/full_datasets/data_dr12_bestexp.fits --nsplits {} --training-proportion {} --auto-independent --out-prefix {}/data/bestexp/training_datasets/prop_{}/data_dr12_bestexp_train\n'.format(OUTDIR,BESTEXP_NMODEL,p,OUTDIR,p)
    run_file_text += 'echo " -> Done!"\n'
    run_file_text += 'echo " "\n\n'

for a in BESTEXP_ABS_TRAINSIZES:
    run_file_text += 'echo "Making best exposure training sets with training size {}..."\n'.format(a)
    run_file_text += 'split_data --data {}/data/bestexp/full_datasets/data_dr12_bestexp.fits --nsplits {} --training-number {} --auto-independent --out-prefix {}/data/bestexp/training_datasets/abs_{}/data_dr12_bestexp_train\n'.format(OUTDIR,BESTEXP_NMODEL,a,OUTDIR,a)
    run_file_text += 'echo " -> Done!"\n'
    run_file_text += 'echo " "\n\n'

for p in RANDEXP_PROP_TRAINSIZES:
    run_file_text += 'echo "Making random exposure training sets with training proportion {}..."\n'.format(p)
    run_file_text += 'split_data --data {}/data/randexp/full_datasets/data_dr12_randexp_seed0.fits --nsplits {} --training-proportion {} --auto-independent --out-prefix {}/data/randexp/training_datasets/prop_{}/data_dr12_randexp_seed0_train\n'.format(OUTDIR,RANDEXP_NMODEL,p,OUTDIR,p)
    run_file_text += 'echo " -> Done!"\n'
    run_file_text += 'echo " "\n\n'

for a in RANDEXP_ABS_TRAINSIZES:
    run_file_text += 'echo "Making random exposure training sets with training size {}..."\n'.format(a)
    run_file_text += 'split_data --data {}/data/randexp/full_datasets/data_dr12_randexp_seed0.fits --nsplits {} --training-number {} --auto-independent --out-prefix {}/data/randexp/training_datasets/abs_{}/data_dr12_randexp_seed0_train\n'.format(OUTDIR,RANDEXP_NMODEL,a,OUTDIR,a)
    run_file_text += 'echo " -> Done!"\n'
    run_file_text += 'echo " "\n\n'

for dll in DLL_VALUES:
    run_file_text += 'echo "Making coadded training sets with training proportion {}, dll {}..."\n'.format(DLL_PROP_TRAINSIZE,dll)
    run_file_text += 'split_data --data {}/data/coadd/full_datasets/data_dr12_coadd_dll{}.fits --nsplits 1 --training-proportion {} --auto-independent --out-prefix {}/data/coadd/training_datasets/prop_{}/data_dr12_coadd_train_dll{}\n'.format(OUTDIR,dll,DLL_PROP_TRAINSIZE,OUTDIR,DLL_PROP_TRAINSIZE,dll)
    run_file_text += 'echo " -> Done!"\n'
    run_file_text += 'echo " "\n\n'

with open(run_file, 'w') as file:
    file.write(run_file_text)
os.chmod(run_file, stat.S_IRWXU | stat.S_IRWXG)
