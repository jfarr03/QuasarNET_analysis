#!/usr/bin/env python

import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--training-dir', type=str, required=True)
parser.add_argument('--training-prefix', type=str, required=True)
parser.add_argument('--split', type=str, required=True)
parser.add_argument('--truth', type=str, required=True)

parser.add_argument('--nhours', type=int, required=True)
parser.add_argument('--queue', type=str, required=False, default='regular')

parser.add_argument('--lines', type=str, required=False, default='LYA CIV(1548) CIII(1909) MgII(2796) Hbeta Halpha')
parser.add_argument('--lines-bal', type=str, required=False, default='CIV(1548)')
parser.add_argument('--decay', type=float, required=False, default=0.)
parser.add_argument('--offset-activation-function', type=str, required=False, default='rescaled_simoid')
parser.add_argument('--nepochs', type=int, required=False, default=200)
parser.add_argument('--dll', type=float, required=False, default=1e-4)
parser.add_argument('--nchunks', type=int, required=False, default=13)

parser.add_argument('--output-dir', type=str, required=True)
parser.add_argument('--output-prefix', type=str, required=True)

args = parser.parse_args()

## Check that directories exist for the outputs, make them if not.
print('Output will written to directory {}'.format(args.output_dir))
if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)

run_dir = '{}/run_files'.format(args.output_dir)
print(' -> Job files will be saved to {}'.format(run_dir))
if not os.path.isdir(run_dir):
    os.mkdir(run_dir)

run_script = '{}/run_qn_train_{}.sh'.format(args.output_dir,args.split)
print(' -> Run script will be written to {}'.format(run_script))
if not os.path.isdir(run_script):
    os.mkdir(run_script)

## Write the run file.
run_script_text = ''
run_script_text += '#!/bin/bash -l\n\n'

run_script_text += '#SBATCH --partition {}\n'.format(args.queue)
run_script_text += '#SBATCH --nodes 1\n'
run_script_text += '#SBATCH --time {}:00:00\n'.format(args.nhours)
run_script_text += '#SBATCH --job-name qn-train\n'
run_script_text += '#SBATCH --error "{}/run_files/qn-train-%j.err"\n'.format(args.output_dir)
run_script_text += '#SBATCH --output "{}/run_files/qn-train-%j.out"\n'.format(args.output_dir)
run_script_text += '#SBATCH -C haswell\n'
run_script_text += '#SBATCH -A desi\n\n'

run_script_text += 'source activate qnet\n\n'

run_script_text += 'umask 0002\n'
run_script_text += 'export OMP_NUM_THREADS=64\n\n'

run_script_text += '#Fix to bug in HDF5 (https://www.nersc.gov/users/data-analytics/data-management/i-o-libraries/hdf5-2/h5py/)\n'
run_script_text += 'export HDF5_USE_FILE_LOCKING=FALSE\n\n'

run_script_text += 'command="qn_train --truth {} --data {}/{}_{}.fits --epochs {} --out-prefix {}/{}_{} --lines {} --lines-bal {} --decay {}"\n\n'.format(args.truth,args.training_dir,args.training_prefix,args.split,args.nepochs,args.output_dir,args.output_prefix,args.split,args.lines,args.lines_bal,args.decay)

run_script_text += 'echo "Running command: \$command"\n'
run_script_text += '\$command >& {}/{}_{}.log &\n\n'.format(run_dir,args.output_prefix,args.split)

run_script_text += 'wait\n'
run_script_text += 'date\n'

print(run_script_text)
with open(run_script, 'w') as file:
    file.write(run_script_text)
