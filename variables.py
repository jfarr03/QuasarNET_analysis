# Set locations.
REDUXDIR="/global/cfs/projectdirs/desi/users/jfarr/BOSS_DR12_redux_replica/"
SPALL="/global/projecta/projectdirs/sdss/data/sdss/dr12/boss/spectro/redux/spAll-DR12.fits"
SDRQ="/global/projecta/projectdirs/sdss/data/sdss/dr12/boss/qso/DR12Q/Superset_DR12Q.fits"
DRQ="/global/projecta/projectdirs/sdss/data/sdss/dr12/boss/qso/DR12Q/DR12Q.fits"
OUTDIR="/global/cfs/projectdirs/desi/users/jfarr/QuasarNET_paper/"


## Variables for main setup.
# Set parameters for coadded models to train.
COADD_PROP_TRAINSIZES=[0.9, 0.8, 0.5, 0.2, 0.1, 0.05, 0.025, 0.01]
COADD_ABS_TRAINSIZES=[100000, 50000, 25000]
COADD_NMODEL=10

# Set parameters for single exposure (best exp) models to train.
BESTEXP_PROP_TRAINSIZES=[0.2, 0.1, 0.05]
BESTEXP_ABS_TRAINSIZES=[100000, 50000, 25000]
BESTEXP_NMODEL=10

# Set parameters for coadded models to train.
RANDEXP_PROP_TRAINSIZES=[0.2, 0.1, 0.05]
RANDEXP_ABS_TRAINSIZES=[100000, 50000, 25000]
RANDEXP_NMODEL=10


## Variables for additional setups.
# General variables

# Offset activation function choices.
OFFSET_ACT_FNS=['sigmoid', 'linear', 'rescaled_simoid']
OFFSET_ACT_PROP_TRAINSIZE=0.1

# Max number of epochs to look at.
NEPOCH_MAX=300
NEPOCH_PROP_TRAINSIZES=[0.9, 0.5, 0.2, 0.1, 0.05]

# Values of dll to test.
DLL_VALUES=[0.002, 0.001, 0.0005]
DLL_PROP_TRAINSIZE=0.1

# Number of chunks to test.
NCHUNK_VALUES=[7, 10, 13, 16, 19]
NCHUNK_PROP_TRAINSIZE=0.1


## Job time lookup.
PROP_JOB_TIMES = {0.9:   24,
                  0.8:   20,
                  0.5:   12,
                  0.2:   6,
                  0.1:   4,
                  0.05:  2,
                  0.025: 2,
                  0.01:  1,
                  }

ABS_JOB_TIMES = {100000: 6,
                 50000:  4,
                 25000:  2,
                 }
