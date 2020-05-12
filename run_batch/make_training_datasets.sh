#!/usr/bin/env bash

echo "Making coadded training sets with training proportion 0.9..."
split_data --data /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/coadd/full_datasets/data_dr12_coadd.fits --nsplits 10 --training-proportion 0.9 --auto-independent --out-prefix /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/coadd/training_datasets/prop_0.9/data_dr12_coadd_train
echo " -> Done!"
echo " "

echo "Making coadded training sets with training proportion 0.8..."
split_data --data /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/coadd/full_datasets/data_dr12_coadd.fits --nsplits 10 --training-proportion 0.8 --auto-independent --out-prefix /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/coadd/training_datasets/prop_0.8/data_dr12_coadd_train
echo " -> Done!"
echo " "

echo "Making coadded training sets with training proportion 0.5..."
split_data --data /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/coadd/full_datasets/data_dr12_coadd.fits --nsplits 10 --training-proportion 0.5 --auto-independent --out-prefix /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/coadd/training_datasets/prop_0.5/data_dr12_coadd_train
echo " -> Done!"
echo " "

echo "Making coadded training sets with training proportion 0.2..."
split_data --data /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/coadd/full_datasets/data_dr12_coadd.fits --nsplits 10 --training-proportion 0.2 --auto-independent --out-prefix /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/coadd/training_datasets/prop_0.2/data_dr12_coadd_train
echo " -> Done!"
echo " "

echo "Making coadded training sets with training proportion 0.1..."
split_data --data /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/coadd/full_datasets/data_dr12_coadd.fits --nsplits 10 --training-proportion 0.1 --auto-independent --out-prefix /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/coadd/training_datasets/prop_0.1/data_dr12_coadd_train
echo " -> Done!"
echo " "

echo "Making coadded training sets with training proportion 0.05..."
split_data --data /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/coadd/full_datasets/data_dr12_coadd.fits --nsplits 10 --training-proportion 0.05 --auto-independent --out-prefix /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/coadd/training_datasets/prop_0.05/data_dr12_coadd_train
echo " -> Done!"
echo " "

echo "Making coadded training sets with training proportion 0.025..."
split_data --data /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/coadd/full_datasets/data_dr12_coadd.fits --nsplits 10 --training-proportion 0.025 --auto-independent --out-prefix /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/coadd/training_datasets/prop_0.025/data_dr12_coadd_train
echo " -> Done!"
echo " "

echo "Making coadded training sets with training proportion 0.01..."
split_data --data /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/coadd/full_datasets/data_dr12_coadd.fits --nsplits 10 --training-proportion 0.01 --auto-independent --out-prefix /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/coadd/training_datasets/prop_0.01/data_dr12_coadd_train
echo " -> Done!"
echo " "

echo "Making coadded training sets with training size 100000..."
split_data --data /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/coadd/full_datasets/data_dr12_coadd.fits --nsplits 10 --training-number 100000 --auto-independent --out-prefix /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/coadd/training_datasets/abs_100000/data_dr12_coadd_train
echo " -> Done!"
echo " "

echo "Making coadded training sets with training size 50000..."
split_data --data /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/coadd/full_datasets/data_dr12_coadd.fits --nsplits 10 --training-number 50000 --auto-independent --out-prefix /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/coadd/training_datasets/abs_50000/data_dr12_coadd_train
echo " -> Done!"
echo " "

echo "Making coadded training sets with training size 25000..."
split_data --data /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/coadd/full_datasets/data_dr12_coadd.fits --nsplits 10 --training-number 25000 --auto-independent --out-prefix /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/coadd/training_datasets/abs_25000/data_dr12_coadd_train
echo " -> Done!"
echo " "

echo "Making best exposure training sets with training proportion 0.2..."
split_data --data /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/bestexp/full_datasets/data_dr12_bestexp.fits --nsplits 10 --training-proportion 0.2 --auto-independent --out-prefix /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/bestexp/training_datasets/prop_0.2/data_dr12_bestexp_train
echo " -> Done!"
echo " "

echo "Making best exposure training sets with training proportion 0.1..."
split_data --data /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/bestexp/full_datasets/data_dr12_bestexp.fits --nsplits 10 --training-proportion 0.1 --auto-independent --out-prefix /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/bestexp/training_datasets/prop_0.1/data_dr12_bestexp_train
echo " -> Done!"
echo " "

echo "Making best exposure training sets with training proportion 0.05..."
split_data --data /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/bestexp/full_datasets/data_dr12_bestexp.fits --nsplits 10 --training-proportion 0.05 --auto-independent --out-prefix /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/bestexp/training_datasets/prop_0.05/data_dr12_bestexp_train
echo " -> Done!"
echo " "

echo "Making best exposure training sets with training size 100000..."
split_data --data /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/bestexp/full_datasets/data_dr12_bestexp.fits --nsplits 10 --training-number 100000 --auto-independent --out-prefix /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/bestexp/training_datasets/abs_100000/data_dr12_bestexp_train
echo " -> Done!"
echo " "

echo "Making best exposure training sets with training size 50000..."
split_data --data /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/bestexp/full_datasets/data_dr12_bestexp.fits --nsplits 10 --training-number 50000 --auto-independent --out-prefix /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/bestexp/training_datasets/abs_50000/data_dr12_bestexp_train
echo " -> Done!"
echo " "

echo "Making best exposure training sets with training size 25000..."
split_data --data /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/bestexp/full_datasets/data_dr12_bestexp.fits --nsplits 10 --training-number 25000 --auto-independent --out-prefix /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/bestexp/training_datasets/abs_25000/data_dr12_bestexp_train
echo " -> Done!"
echo " "

echo "Making random exposure training sets with training proportion 0.2..."
split_data --data /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/randexp/full_datasets/data_dr12_randexp_seed0.fits --nsplits 10 --training-proportion 0.2 --auto-independent --out-prefix /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/randexp/training_datasets/prop_0.2/data_dr12_randexp_seed0_train
echo " -> Done!"
echo " "

echo "Making random exposure training sets with training proportion 0.1..."
split_data --data /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/randexp/full_datasets/data_dr12_randexp_seed0.fits --nsplits 10 --training-proportion 0.1 --auto-independent --out-prefix /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/randexp/training_datasets/prop_0.1/data_dr12_randexp_seed0_train
echo " -> Done!"
echo " "

echo "Making random exposure training sets with training proportion 0.05..."
split_data --data /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/randexp/full_datasets/data_dr12_randexp_seed0.fits --nsplits 10 --training-proportion 0.05 --auto-independent --out-prefix /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/randexp/training_datasets/prop_0.05/data_dr12_randexp_seed0_train
echo " -> Done!"
echo " "

echo "Making random exposure training sets with training size 100000..."
split_data --data /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/randexp/full_datasets/data_dr12_randexp_seed0.fits --nsplits 10 --training-number 100000 --auto-independent --out-prefix /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/randexp/training_datasets/abs_100000/data_dr12_randexp_seed0_train
echo " -> Done!"
echo " "

echo "Making random exposure training sets with training size 50000..."
split_data --data /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/randexp/full_datasets/data_dr12_randexp_seed0.fits --nsplits 10 --training-number 50000 --auto-independent --out-prefix /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/randexp/training_datasets/abs_50000/data_dr12_randexp_seed0_train
echo " -> Done!"
echo " "

echo "Making random exposure training sets with training size 25000..."
split_data --data /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/randexp/full_datasets/data_dr12_randexp_seed0.fits --nsplits 10 --training-number 25000 --auto-independent --out-prefix /global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper//data/randexp/training_datasets/abs_25000/data_dr12_randexp_seed0_train
echo " -> Done!"
echo " "

