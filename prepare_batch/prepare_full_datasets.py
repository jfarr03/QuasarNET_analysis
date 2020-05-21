#!/usr/bin/env python

from qn_analysis.variables import *
from os.path import stat
import os

run_file = PREPDIR+'/run_batch/make_full_datasets.sh'

run_file_text = '#!/usr/bin/env bash\n\n'

run_file_text += '## Make truth file.\n'
run_file_text += 'cmd="qn_parse_truth --sdrq {} --drq {} --out {}/data/truth/truth_dr12q.fits --mode BOSS --nproc 32"\n'.format(SDRQ,DRQ,OUTDIR)
run_file_text += '$cmd\n\n'

run_file_text += '## Make coadded dataset.\n'
run_file_text += 'cmd="parse_data --spplates {}/*/spPlate-*.fits --spall {} --sdrq {} --out {}/data/coadd/full_datasets/data_dr12_coadd.fits --nproc 32"\n'.format(REDUXDIR,SPALL,SDRQ,OUTDIR)
run_file_text += '$cmd\n\n'

run_file_text += '## Make bestexp dataset.\n'
run_file_text += 'cmd="parse_data --spplates {}/*/spPlate-*.fits --spall {} --sdrq {} --out {}/data/bestexp/full_datasets/data_dr12_bestexp.fits --use-best-exp --nproc 32"\n'.format(REDUXDIR,SPALL,SDRQ,OUTDIR)
run_file_text += '$cmd\n\n'

run_file_text += '## Make randexp dataset.\n'
run_file_text += 'cmd="parse_data --spplates {}/*/spPlate-*.fits --spall {} --sdrq {} --out {}/data/randexp/full_datasets/data_dr12_randexp.fits --use-random-exp --random-seed 0 --nproc 32"\n'.format(REDUXDIR,SPALL,SDRQ,OUTDIR)
run_file_text += '$cmd\n\n'

for dll in DLL_VALUES:
    run_file_text += '## Make coadded dataset with dll={}.\n'.format(dll)
    run_file_text += 'cmd="parse_data --spplates {}/*/spPlate-*.fits --spall {} --sdrq {} --out {}/data/coadd/full_datasets/data_dr12_coadd_dll{}.fits --dll {} --nproc 32"\n'.format(REDUXDIR,SPALL,SDRQ,OUTDIR,dll,dll)
    run_file_text += '$cmd\n\n'

with open(run_file, 'w') as file:
    file.write(run_file_text)
os.chmod(run_file, stat.S_IRWXU | stat.S_IRWXG)
