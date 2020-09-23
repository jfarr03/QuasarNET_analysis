import h5py
import os
import numpy as np
from astropy.io import fits
from astropy.table import Column, join, Table
import matplotlib.pyplot as plt

import glob
from scipy.stats import norm
from quasarnet.io import read_truth, read_data
from quasarnet.utils import process_preds, absorber_IGM

colours = {'C0': '#F5793A',
           'C1': '#A95AA1',
           'C2': '#85C0F9',
           'C3': '#0F2080',
          }

def get_data_filter(data_table,data_used):

    # Determine the filter.
    filt = np.ones(len(data_table)).astype(bool)
    for d in data_used:
        d_filt = (data_table['ISQSO_{}'.format(d)] | True)
        filt = filt & d_filt

    print('INFO: {:6d}/{:6d} ({:3.1%}) spectra used'.format(filt.sum(),len(filt),filt.sum()/len(filt)))

    return filt

def get_data_dict(data_file,truth_file,train_file,nspec=None,nspec_method='first',seed=0):

    ## set nspec to the number of spectra to load or to None for the full sample
    truth = read_truth([truth_file])
    if nspec_method=='first':
        tids_full,X_full,Y_full,z_full,bal_full = read_data([data_file], truth, nspec=nspec)
    elif nspec_method=='random':
        tids_full,X_full,Y_full,z_full,bal_full = read_data([data_file], truth, nspec=None)
        gen = np.random.RandomState(seed=seed)
        w = gen.choice(range(len(tids_full)),size=nspec,replace=False)
        tids_full = tids_full[w]
        X_full = X_full[w,:]
        Y_full = Y_full[w,:]
        z_full = z_full[w]
        bal_full = bal_full[w]
    else:
        raise ValueError('nspec_method value is not recognised')

    ## Get the training data.
    tids_train,X_train,Y_train,z_train,bal_train = read_data([train_file], truth, return_spid=False)

    ## Assess how many spectra from the training data are in the test data.
    in_train = np.in1d(tids_full, tids_train)
    print('INFO: found {} spectra from the training sample in the test sample'.format(in_train.sum()))

    ## to get the validation data, remove the spectra in the training sample from the full sample
    w = ~in_train
    tids_val = tids_full[w]
    X_val = X_full[w]
    Y_val = Y_full[w]
    z_val = z_full[w]
    bal_val = bal_full[w]

    data_dict = {}
    data_dict['full'] = {'tids': tids_full,
                         'X':    X_full,
                         'Y':    Y_full,
                         'z':    z_full,
                         'bal':  bal_full
                        }

    data_dict['train'] = {'tids': tids_train,
                          'X':    X_train,
                          'Y':    Y_train,
                          'z':    z_train,
                          'bal':  bal_train
                         }

    data_dict['val'] = {'tids': tids_val,
                        'X':    X_val,
                        'Y':    Y_val,
                        'z':    z_val,
                        'bal':  bal_val
                        }

    data_dict['files'] = {'data':  data_file,
                          'truth': truth_file,
                          'train': train_file
                         }

    return data_dict

def show_spec(ax,p,data,wave,tid,lines,lines_bal,show_bal=True,ndetect=1,cth=0.8,verbose=False,show_BOSS=False,files=None,show_qn_BOSS=False,show_DESI=False):

    tids_val = data['tids']
    X_val = data['X']
    Y_val = data['Y']
    ztrue_val = data['z']

    if (tids_val==tid).sum()>0:
        ival = np.argmax(tids_val==tid)
    else:
        text = 'WARN: Spectrum for tid {} not found!'.format(tid)
        ax.text(0.5,0.5,text,horizontalalignment='center',verticalalignment='center',transform=ax.transAxes)
        print(text)
        return

    X = X_val[ival:ival+1,:,None]
    Y = Y_val[ival:ival+1,:,None]
    ztrue = ztrue_val[ival]

    c_line, z_line, zbest, c_line_bal, z_line_bal = process_preds(p, lines, lines_bal, wave=wave)
    X_plot = X.reshape(X.shape[1])
    ax.plot(wave.wave_grid, X_plot)

    best_line = np.array(lines)[c_line.argmax(axis=0)].flatten()

    class_dict = {0: 'star', 1: 'gal', 2: 'QSOLZ', 3: 'QSOHZ', 4: 'BAD'}
    true_class = class_dict[np.argmax(Y)]
    isqso = (c_line>cth).sum()>=ndetect

    text = ''
    text += r'ann: class = {}, z = {:1.3f}'.format(true_class,ztrue)
    text += '\n'
    if isqso:
        pred_class = 'QSO'
        text += r'pred: class = {}, z = {:1.3f}'.format(pred_class,zbest[0])
    else:
        pred_class = 'non-QSO'
        text += r'pred: class = {}, z = {:1.3f}'.format(pred_class,zbest[0])
        #title += r'pred: class = {}, z = {}'.format(pred_class,'N/A')

    #title += r'z$_{{pred}}$ = {:1.3f}, z$_{{ann}}$ = {:1.3f}'.format(zbest[0],ztrue[0])
    ax.text(0.98, 0.02, text, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)

    m = X_plot.min()
    M = X_plot.max()
    ax.grid()
    ax.set_ylim(m-2,M+2)
    irrel_lines_text = ''
    ax.set_xlabel(r'$\AA$')

    for il,l in enumerate(lines):

        lam = absorber_IGM[l]*(1+z_line[il])
        w = abs(wave.wave_grid-lam)<100

        if w.sum()!=0:
            m = X_plot[w].min()-1
            M = X_plot[w].max()+1

            if l == best_line[0]:
                ls = 'c--'
            else:
                ls = 'k--'
            ax.plot([lam,lam], [m,M],ls, alpha=0.1+0.9*c_line[il,0])
            text = 'c$_{{{}}}={}$'.format(l,round(c_line[il,0],3))
            text += '\n'
            text += 'z$_{{{}}}={}$'.format(l,round(z_line[il,0],3))
            ax.text(lam,M+0.5,text,alpha=0.1+0.9*c_line[il,0],
                        horizontalalignment='center',verticalalignment='bottom'
                        )

            if l != best_line[0]:
                lam_pred = absorber_IGM[l]*(1+zbest)
                w2 = abs(wave.wave_grid-lam_pred)<100
                if w2.sum()!=0:
                    ax.plot([lam_pred,lam_pred], [m-0.5,m+0.5],'c--')
                    ax.text(lam_pred,m-1.5,'{} pred'.format(l),
                                    horizontalalignment='center')
                else:
                    irrel_lines_text += '{}'.format(l)
                    irrel_lines_text += '\n'

            """if c_line[il]<cth:
                lam_pred = absorber_IGM[l]*(1+zbest)
                if (lam_pred < 10000) * (lam_pred > 3600):
                    ax.plot([lam_pred,lam_pred], [m-1,m],'g--', alpha=0.1+0.9*(1-c_line[il,0]))
                    ax.text(lam_pred,m-1.5,'c$_{{{}}}={}$'.format(l,round(c_line[il,0],3)),
                                    horizontalalignment='center',alpha=0.1+0.9*(1-c_line[il,0]))"""



    ax.text(0.98, 0.98, irrel_lines_text, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize='large', color='c')

    if show_bal:
        for il,l in enumerate(lines_bal):
            lam = absorber_IGM[l]*(1+z_line_bal[il])
            w = abs(wave.wave_grid-lam)<100
            if w.sum()!=0:
                m = X_plot[w].min()-1
                M = X_plot[w].max()+1
                ax.plot([lam,lam], [m,M],'r--', alpha=0.1+0.9*c_line_bal[il,0])
                ax.text(lam,M+2.0,'c$_{{{}}}={}$'.format('BAL'+l,round(c_line_bal[il,0],3)),horizontalalignment='center',alpha=0.1+0.9*c_line_bal[il,0],c='r')

    return

def show_spec_pred(ax,model,data,wave,tid,lines,lines_bal,show_bal=True,ndetect=1,cth=0.8,verbose=False,show_BOSS=False,files=None,show_qn_BOSS=False,show_DESI=False):

    tids_val = data['tids']
    X_val = data['X']
    Y_val = data['Y']
    ztrue_val = data['z']

    if (tids_val==tid).sum()>0:
        ival = np.argmax(tids_val==tid)
    else:
        text = 'WARN: Spectrum for tid {} not found!'.format(tid)
        ax.text(0.5,0.5,text,horizontalalignment='center',verticalalignment='center',transform=ax.transAxes)
        print(text)
        return

    X = X_val[ival:ival+1,:,None]
    Y = Y_val[ival:ival+1,:,None]
    ztrue = ztrue_val[ival]

    p = model.predict(X)
    c_line, z_line, zbest, c_line_bal, z_line_bal = process_preds(p, lines, lines_bal, wave=wave)
    X_plot = X.reshape(X.shape[1])
    ax.plot(wave.wave_grid, X_plot)

    best_line = np.array(lines)[c_line.argmax(axis=0)].flatten()

    class_dict = {0: 'star', 1: 'gal', 2: 'QSOLZ', 3: 'QSOHZ', 4: 'BAD'}
    true_class = class_dict[np.argmax(Y)]
    isqso = (c_line>cth).sum()>=ndetect

    text = ''
    text += r'ann: class = {}, z = {:1.3f}'.format(true_class,ztrue)
    text += '\n'
    if isqso:
        pred_class = 'QSO'
        text += r'pred: class = {}, z = {:1.3f}'.format(pred_class,zbest[0])
    else:
        pred_class = 'non-QSO'
        text += r'pred: class = {}, z = {:1.3f}'.format(pred_class,zbest[0])
        #title += r'pred: class = {}, z = {}'.format(pred_class,'N/A')

    #title += r'z$_{{pred}}$ = {:1.3f}, z$_{{ann}}$ = {:1.3f}'.format(zbest[0],ztrue[0])
    ax.text(0.98, 0.02, text, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)

    m = X_plot.min()
    M = X_plot.max()
    ax.grid()
    ax.set_ylim(m-2,M+2)
    irrel_lines_text = ''
    ax.set_xlabel(r'$\AA$')

    for il,l in enumerate(lines):

        lam = absorber_IGM[l]*(1+z_line[il])
        w = abs(wave.wave_grid-lam)<100

        if w.sum()!=0:
            m = X_plot[w].min()-1
            M = X_plot[w].max()+1

            if l == best_line[0]:
                ls = 'c--'
            else:
                ls = 'k--'
            ax.plot([lam,lam], [m,M],ls, alpha=0.1+0.9*c_line[il,0])
            text = 'c$_{{{}}}={}$'.format(l,round(c_line[il,0],3))
            text += '\n'
            text += 'z$_{{{}}}={}$'.format(l,round(z_line[il,0],3))
            ax.text(lam,M+0.5,text,alpha=0.1+0.9*c_line[il,0],
                        horizontalalignment='center',verticalalignment='bottom'
                        )

            if l != best_line[0]:
                lam_pred = absorber_IGM[l]*(1+zbest)
                w2 = abs(wave.wave_grid-lam_pred)<100
                if w2.sum()!=0:
                    ax.plot([lam_pred,lam_pred], [m-0.5,m+0.5],'c--')
                    ax.text(lam_pred,m-1.5,'{} pred'.format(l),
                                    horizontalalignment='center')
                else:
                    irrel_lines_text += '{}'.format(l)
                    irrel_lines_text += '\n'

            """if c_line[il]<cth:
                lam_pred = absorber_IGM[l]*(1+zbest)
                if (lam_pred < 10000) * (lam_pred > 3600):
                    ax.plot([lam_pred,lam_pred], [m-1,m],'g--', alpha=0.1+0.9*(1-c_line[il,0]))
                    ax.text(lam_pred,m-1.5,'c$_{{{}}}={}$'.format(l,round(c_line[il,0],3)),
                                    horizontalalignment='center',alpha=0.1+0.9*(1-c_line[il,0]))"""



    ax.text(0.98, 0.98, irrel_lines_text, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize='large', color='c')

    if show_bal:
        for il,l in enumerate(lines_bal):
            lam = absorber_IGM[l]*(1+z_line_bal[il])
            w = abs(wave.wave_grid-lam)<100
            if w.sum()!=0:
                m = X_plot[w].min()-1
                M = X_plot[w].max()+1
                ax.plot([lam,lam], [m,M],'r--', alpha=0.1+0.9*c_line_bal[il,0])
                ax.text(lam,M+2.0,'c$_{{{}}}={}$'.format('BAL'+l,round(c_line_bal[il,0],3)),horizontalalignment='center',alpha=0.1+0.9*c_line_bal[il,0],c='r')

    if show_BOSS:
        BOSS_base = '/global/projecta/projectdirs/sdss/data/sdss/dr12/boss/spectro/redux/'

        if files is None:
            raise ValueError('No files supplied to look up BOSS data.')

        h = fits.open(files['truth'])
        w = (h[1].data['TARGETID']==tid)
        try:
            plate = h[1].data['SPID0'][w][0]
            mjd = h[1].data['SPID1'][w][0]
            fiber = h[1].data['SPID2'][w][0]
        except:
            plate = h[1].data['PLATE'][w][0]
            mjd = h[1].data['MJD'][w][0]
            fiber = h[1].data['FIBER'][w][0]
        h.close()
        print(tid,(plate,mjd,fiber))

        f_spPlate = glob.glob(BOSS_base+'/*/{}/spPlate-{}-{}.fits'.format(plate,plate,mjd))
        if len(f_spPlate)>1:
            raise ValueError('More than one plate found with plate={}, mjd={}'.format(plate,mjd))
        elif len(f_spPlate)==1:
            h = fits.open(f_spPlate[0])
            head = h[0].header
            c0 = head["COEFF0"]
            c1 = head["COEFF1"]
            w = h[5].data['FIBERID']==fiber
            fl = h[0].data[w,:][0]
            wave_grid = 10**(c0 + c1*np.arange(fl.shape[0]))
            h.close()

            shift = 15
            ax.plot(wave_grid,fl-shift,color='grey',alpha=0.2,zorder=-1)
            #ax.set_ylim(ax.get_ylim()[0]-shift,ax.get_ylim()[1])
            ax.set_ylim(-shift-5,ax.get_ylim()[1])
        else:
            print('WARN: No plates found for spectrum with tid={}'.format(tid))

    if show_qn_BOSS:

        if files is None:
            raise ValueError('No files supplied to find BOSS QN result.')

        h = fits.open(files['truth'])
        w = (h[1].data['TARGETID']==tid)
        try:
            plate = h[1].data['SPID0'][w][0]
            mjd = h[1].data['SPID1'][w][0]
            fiber = h[1].data['SPID2'][w][0]
        except:
            plate = h[1].data['PLATE'][w][0]
            mjd = h[1].data['MJD'][w][0]
            fiber = h[1].data['FIBER'][w][0]
        h.close()

        h = fits.open('/project/projectdirs/desi/users/jfarr/quasar_classifier_hack/qn_sdr12q.fits')
        w = (h[1].data['PLATE']==plate) & (h[1].data['MJD']==mjd) & (h[1].data['FIBERID']==fiber)
        if w.sum() == 0:
            print('WARN: no QN prediction for pmf=({},{},{}) found'.format(plate,mjd,fiber))
        else:
            text = 'QuasarNET on BOSS data:'
            text += '\n'
            if h[1].data['IS_QSO'][w][0]==1:
                pred_class = 'QSO'
                text += r'pred: class = {}, z = {:1.3f}'.format(pred_class,h[1].data['ZBEST'][w][0])
            else:
                pred_class = 'non-QSO'
                text += r'pred: class = {}, z = {:1.3f}'.format(pred_class,h[1].data['ZBEST'][w][0])

            ax.text(0.02, 0.02, text, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, alpha=0.7)

    if show_DESI:
        DESI_base = '/project/projectdirs/desi/users/jfarr/MiniSV_data/20200219/'

        fi = glob.glob(DESI_base+'coadd*.fits')
        f_choice = None
        for f in fi:
            h = fits.open(f)
            w = h[1].data['TARGETID']==tid
            if w.sum()>0:
                f_choice = f

        if f_choice is None:
            print('WARN: tid {} not found in {}'.format(tid,DESI_base))
        else:
            h = fits.open(f_choice)
            w = h[1].data['TARGETID']==tid
            fl = h[3].data[w,:][0]
            wave_grid = h[2].data
            h.close()

            shift = 15
            ax.plot(wave_grid,fl-shift,color='red',alpha=0.2,zorder=-1)
            #ax.set_ylim(ax.get_ylim()[0]-shift,ax.get_ylim()[1])
            ax.set_ylim(-shift-5,ax.get_ylim()[1])

    return

def get_pmf2tid(f_spall):

    spall = fits.open(f_spall)

    plate = spall[1].data["PLATE"]
    mjd = spall[1].data["MJD"]
    fid = spall[1].data["FIBERID"]
    tid = spall[1].data["THING_ID"].astype(int)

    pmf2tid = {(p,m,f):t for p,m,f,t in zip(plate,mjd,fid,tid)}
    spall.close()

    return pmf2tid

def load_sdrq_data(f_sdrq):

    sdrq = fits.open(f_sdrq)

    ## Convert the VI class labelling.
    isstar = (sdrq[1].data['CLASS_PERSON']==1)
    isgal = (sdrq[1].data['CLASS_PERSON']==4)
    isqso = (sdrq[1].data['CLASS_PERSON']==3) | (sdrq[1].data['CLASS_PERSON']==30)
    objclass = np.array(['']*len(isqso),dtype='U8')
    objclass[isstar] = 'STAR'
    objclass[isgal] = 'GALAXY'
    objclass[isqso] = 'QSO'

    ## Make targetid.
    targetid = platemjdfiber2targetid(sdrq[1].data['PLATE'].astype('i8'),sdrq[1].data['MJD'].astype('i8'),sdrq[1].data['FIBERID'].astype('i8'))

    ## Include only objects that were inspected and don't have thing_id of -1.
    w = (sdrq[1].data['Z_CONF_PERSON']>0) & (sdrq[1].data['THING_ID']!=-1)

    ## Construct an array.
    sdrq_data = list(zip(sdrq[1].data['THING_ID'][w], sdrq[1].data['Z_VI'][w], objclass[w], isqso[w], sdrq[1].data['PLATE'][w], sdrq[1].data['MJD'][w], sdrq[1].data['FIBERID'][w], sdrq[1].data['Z_CONF_PERSON'][w], targetid[w]))
    dtype = [('THING_ID','i8'),('Z','f8'),('CLASS','U8'),('ISQSO','bool'),('PLATE','i8'),('MJD','i8'),('FIBERID','i8'),('ZCONF_PERSON','i8'), ('TARGETID','i8')]
    sdrq_data = np.array(sdrq_data, dtype=dtype)

    return sdrq_data

## Edited version of function of same name at https://github.com/desihub/redrock/blob/master/py/redrock/results.py.
def read_zscan(filename):
    """Read redrock.zfind results from a file.
    """
    # zbest = Table.read(filename, format='hdf5', path='zbest')
    with h5py.File(os.path.expandvars(filename), mode='r') as fx:
        targetids = fx['targetids'].value
        spectypes = list(fx['zscan'].keys())

        """zscan = dict()
        for targetid in targetids:
            zscan[targetid] = dict()
            for spectype in spectypes:
                zscan[targetid][spectype] = dict()

        for spectype in spectypes:
            zchi2 = fx['/zscan/{}/zchi2'.format(spectype)].value
            penalty = fx['/zscan/{}/penalty'.format(spectype)].value
            zcoeff = fx['/zscan/{}/zcoeff'.format(spectype)].value
            redshifts = fx['/zscan/{}/redshifts'.format(spectype)].value
            for i, targetid in enumerate(targetids):
                zscan[targetid][spectype]['redshifts'] = redshifts
                zscan[targetid][spectype]['zchi2'] = zchi2[i]
                zscan[targetid][spectype]['penalty'] = penalty[i]
                zscan[targetid][spectype]['zcoeff'] = zcoeff[i]
                thiszfit = fx['/zfit/{}/zfit'.format(targetid)].value
                ii = (thiszfit['spectype'].astype('U') == spectype)
                thiszfit = Table(thiszfit[ii])
                thiszfit.remove_columns(['targetid', 'znum', 'deltachi2'])
                thiszfit.replace_column('spectype',
                    encode_column(thiszfit['spectype']))
                thiszfit.replace_column('subtype',
                    encode_column(thiszfit['subtype']))
                zscan[targetid][spectype]['zfit'] = thiszfit"""

        zfit = [fx['zfit/{}/zfit'.format(tid)].value for tid in targetids]
        zfit = Table(np.hstack(zfit))
        zfit.replace_column('spectype', encode_column(zfit['spectype']))
        zfit.replace_column('subtype', encode_column(zfit['subtype']))

    return zfit

## Copied from https://github.com/desihub/redrock/blob/master/py/redrock/utils.py.
def encode_column(c):
    """Returns a bytes column encoded into a string column.
    Args:
        c (Table column): a column of a Table.
    Returns:
        array: an array of strings.
    """
    return c.astype((str, c.dtype.itemsize))

def make_rr_table(f_zbest,f_rr):

    ## Make a table from the zbest data.
    zbest = fits.open(f_zbest)
    table = Table(zbest[1].data)
    zbest.close()

    ## Read the rr h5 file.
    zfit = read_zscan(f_rr)

    ## For every targetid, extract the information we want from the h5 file.
    fit_spectype = []
    fit_z = []
    fit_chi2 = []
    fit_zwarn = []
    fit_dof = []
    for tid in table['TARGETID']:
        w = zfit['targetid']==tid
        fit_spectype += [zfit[w]['spectype'].data]
        fit_z += [zfit[w]['z'].data]
        fit_chi2 += [zfit[w]['chi2'].data]
        fit_zwarn += [zfit[w]['zwarn'].data]
        fit_dof += [zfit[w]['npixels'].data - zfit[w]['ncoeff'].data]

    ## Add columns to the table.
    table.add_column(Column(data=fit_spectype,name='FIT_SPECTYPE',dtype='<U8'))
    table.add_column(Column(data=fit_z,name='FIT_Z',dtype=float))
    table.add_column(Column(data=fit_chi2,name='FIT_CHI2',dtype=float))
    table.add_column(Column(data=fit_zwarn,name='FIT_ZWARN',dtype=int))
    table.add_column(Column(data=fit_dof,name='FIT_DOF',dtype=int))

    ## Calculate reduced chi2 values and add to table.
    rchi2 = table['FIT_CHI2']/table['FIT_DOF']
    table.add_column(Column(data=rchi2,name='FIT_RCHI2',dtype=float))

    return table

def make_spzall_table(f,fiberids,nfit,nfit_keep):

    ## Open the spZAll
    spZall = fits.open(f)

    ## Extract the data and reduce to the targetids we want.
    subtable = Table(spZall[1].data)['PLATE','MJD','FIBERID','CLASS','Z','RCHI2','DOF','ZWARNING']
    w = np.in1d(subtable['FIBERID'],fiberids)
    subtable = subtable[w]
    f_targetid = platemjdfiber2targetid(subtable['PLATE'].astype('i8'),subtable['MJD'].astype('i8'),subtable['FIBERID'].astype('i8'))
    nspec = len(set(f_targetid))

    ## Add a column for targetid.
    subtable.add_column(Column(data=f_targetid,name='TARGETID',dtype='i8'))

    assert (f_targetid==subtable['TARGETID'].data).all()

    ## Make arrays with all the data we want in.
    assert len(subtable)==nspec*nfit

    f_targetid_se = f_targetid.reshape((nspec,nfit))[:,0]
    spectype_arr = subtable['CLASS'].reshape((nspec,nfit))[:,:nfit_keep]
    z_arr = subtable['Z'].reshape((nspec,nfit))[:,:nfit_keep]
    rchi2_arr = subtable['RCHI2'].reshape((nspec,nfit))[:,:nfit_keep]
    dof_arr = subtable['DOF'].reshape((nspec,nfit))[:,:nfit_keep]
    zwarn_arr = subtable['ZWARNING'].reshape((nspec,nfit))[:,:nfit_keep]
    chi2_arr = rchi2_arr*dof_arr

    ## Make columns and colnames for the extra data.
    cols = [f_targetid_se,spectype_arr,z_arr,rchi2_arr,dof_arr,zwarn_arr,chi2_arr]
    colnames = ['TARGETID','FIT_SPECTYPE','FIT_Z','FIT_RCHI2','FIT_DOF','FIT_ZWARN','FIT_CHI2']
    dtypes = ['i8','<U8','f8','f8','i8','i8','f8']

    ## Add the resultant onto the stack.
    pm_table = Table(cols,names=colnames,dtype=dtypes)

    spZall.close()

    return pm_table

def load_rr_data(f_rr,mode='BOSS',include_fits=False):

    rr = fits.open(f_rr)

    ## Make a boolean isqso.
    isqso = (rr[1].data['SPECTYPE']=='QSO')

    if mode == 'BOSS':
        obj_id = rr[1].data['THING_ID_DR12']
        spec_id = rr[1].data['TARGETID']
    elif mode == 'DESI':
        obj_id = rr[1].data['TARGETID']
        # Need to come up with a spectrum ID
        spec_id = rr[1].data['TARGETID']

    #rr_data = list(zip(obj_id, rr[1].data['Z'], rr[1].data['SPECTYPE'], isqso, spec_id, rr[1].data['ZWARN']))
    #dtype = [('OBJ_ID','i8'),('Z','f8'),('CLASS','U8'),('ISQSO','bool'),('SPEC_ID','i8'),('ZWARN','i8')]
    #rr_data = np.array(rr_data, dtype=dtype)

    cols = [obj_id, rr[1].data['Z'], rr[1].data['SPECTYPE'], isqso, spec_id, rr[1].data['ZWARN']]
    colnames = ['OBJ_ID','Z','CLASS','ISQSO','SPEC_ID','ZWARN']
    dtypes = ['i8','f8','U8','bool','i8','i8']

    if include_fits:
        cols += [rr[1].data['FIT_SPECTYPE'],rr[1].data['FIT_Z'],rr[1].data['FIT_CHI2'],rr[1].data['FIT_ZWARN'],rr[1].data['FIT_RCHI2']]
        colnames += ['FIT_SPECTYPE','FIT_Z','FIT_CHI2','FIT_ZWARN','FIT_RCHI2']
        dtypes += ['<U8','f8','f8','i8','f8']

    rr_data = Table(cols,names=colnames,dtype=dtypes)

    return rr_data

def load_qn_data(f_qn,n_detect=1,c_th=0.8,include_c=False,include_cbal=False,mode='BOSS',n_lines=6,n_lines_bal=1):

    qn = fits.open(f_qn)

    w = (~(qn[1].data['IN_TRAIN'].astype('bool')))
    data = qn[1].data[w]

    ## Calculate which spectra we think are QSOs.
    isqso = (data['C_LINES']>c_th).sum(axis=1)>n_detect

    ## Convert the QN class labelling.
    objclass = np.array(['']*len(isqso),dtype='U8')
    objclass[isqso] = 'QSO'
    objclass[~isqso] = 'NONQSO'

    if mode == 'BOSS':
        ## Make targetid.
        targetid = platemjdfiber2targetid(data['PLATE'].astype('i8'),data['MJD'].astype('i8'),data['FIBERID'].astype('i8'))
        obj_id = data['THING_ID']
        spec_id = targetid
    elif mode == 'DESI':
        obj_id = data['THING_ID']
        # Need to come up with a spectrum ID
        spec_id = data['THING_ID']

    cols = [obj_id, data['ZBEST'], objclass, isqso, spec_id]
    colnames = ['OBJ_ID','Z','CLASS','ISQSO','SPEC_ID']
    dtypes = ['i8','f8','U8','bool','i8']

    if include_c:
        cols += [data['C_LINES'],data['Z_LINES']]
        colnames += ['C','Z_LINES']
        dtypes += ['f8','f8']
    if include_cbal:
        cols += [data['C_LINES_BAL'],data['Z_LINES_BAL']]
        colnames += ['CBAL','Z_LINES_BAL']
        dtypes += ['f8','f8']

    qn_data = Table(cols,names=colnames,dtype=dtypes)

    return qn_data

##
def load_sq_data(f_sq,p_min=0.32,include_p=False,mode='BOSS'):

    sq = fits.open(f_sq)

    ## Remove duplicated spectra.
    w = (~sq[1].data['duplicated'])
    data = sq[1].data[w]

    isqso = (data['prob']>p_min)

    ## Convert the class labelling.
    objclass = np.array(['']*len(data),dtype='U8')
    objclass[isqso] = 'QSO'
    objclass[~isqso] = 'NONQSO'

    if mode == 'BOSS':
        ## Make targetid.
        targetid = platemjdfiber2targetid(data['PLATE'].astype('i8'),data['MJD'].astype('i8'),data['FIBERID'].astype('i8'))
        obj_id = data['thing_id']
        spec_id = targetid
        z = data['z_try']
    elif mode == 'DESI':
        obj_id = data['TARGETID']
        # Need to come up with a spectrum ID
        spec_id = data['SPECID']
        z = data['Z_TRY']

    if include_p:
        sq_data = list(zip(obj_id, z, objclass, isqso, spec_id, data['prob']))
        dtype = [('OBJ_ID','i8'),('Z','f8'),('CLASS','U8'),('ISQSO','bool'),('SPEC_ID','i8'),('P','f8')]
        sq_data = np.array(sq_data, dtype=dtype)
    else:
        sq_data = list(zip(obj_id, z, objclass, isqso, spec_id))
        dtype = [('OBJ_ID','i8'),('Z','f8'),('CLASS','U8'),('ISQSO','bool'),('SPEC_ID','i8')]
        sq_data = np.array(sq_data, dtype=dtype)

    return sq_data

def reduce_data_to_table(data,truth=None,verbose=True,include_c_qn=False,include_cbal_qn=False,include_p_sq=False,include_fits_rr=False,common_specids=True,spec_ids=None):

    ## If no truth provided, make one from VI data.
    if truth is None:
        print('INFO: No truth provided, using VI in data instead.')
        try:
            test = data['VI']
        except KeyError:
            raise KeyError('No VI data found: check entry is labelled as "VI"!')
        # Cycle through each tid.
        truth = {}
        class metadata:
            pass
        for i,o in enumerate(data['VI']['OBJ_ID']):
            m = metadata()
            setattr(m,'z_conf',data['VI']['ZCONF_PERSON'][i])
            setattr(m,'z',data['VI']['Z'][i])
            setattr(m,'objclass',data['VI']['CLASS'][i])
            truth[o] = m

    ## In each, reduce to the set of spectra that are in the truth dictionary.
    obj_ids_truth = list(truth.keys())
    nonVI_datasets = [c for c in data.keys() if c!='VI']
    for c in nonVI_datasets:
        w = np.in1d(data[c]['OBJ_ID'],obj_ids_truth)
        data[c] = data[c][w]

    ## For each non-VI dataset, also make a table.
    nonVI_tables = []
    for c in nonVI_datasets:
        cols = []
        colnames = []
        # First the ID columns.
        for k in ['OBJ_ID','SPEC_ID']:
            cols += [data[c][k]]
            colnames += [k]
        # Now the data columns.
        for k in ['Z','CLASS','ISQSO']:
            cols += [data[c][k]]
            colnames += ['{}_{}'.format(k,c)]
        # Now optional extras.
        if include_c_qn:
            if 'QN' in c:
                try:
                    cols += [data[c]['C'],data[c]['Z_LINES']]
                    colnames += ['C_{}'.format(c),'Z_LINES_{}'.format(c)]
                except:
                    print('WARN: could not find QuasarNET confidences in {}'.format(c))
        if include_cbal_qn:
            if 'QN' in c:
                try:
                    cols += [data[c]['CBAL'],data[c]['Z_LINES_BAL']]
                    colnames += ['CBAL_{}'.format(c),'Z_LINES_BAL_{}'.format(c)]
                except:
                    print('WARN: could not find QuasarNET BAL confidences in {}'.format(c))
        if include_p_sq:
            if 'SQ' in c:
                try:
                    cols += [data[c]['P']]
                    colnames += ['P_{}'.format(c)]
                except:
                    print('WARN: could not find SQUEzE confidences in {}'.format(c))
        if include_fits_rr:
            if ('RR' in c) or ('PIPE' in c):
                try:
                    cols += [data[c]['FIT_SPECTYPE'],data[c]['FIT_Z'],data[c]['FIT_CHI2'],data[c]['FIT_ZWARN'],data[c]['FIT_RCHI2']]
                    colnames += ['FIT_SPECTYPE_{}'.format(c),'FIT_Z_{}'.format(c),'FIT_CHI2_{}'.format(c),'FIT_ZWARN_{}'.format(c),'FIT_RCHI2_{}'.format(c)]
                except:
                    print('WARN: could not find redrock fit data in {}'.format(c))
        if ('RR' in c) or ('PIPE' in c):
            cols += [data[c]['ZWARN']]
            colnames += ['ZWARN_{}'.format(c)]

        dataset_table = Table(cols,names=colnames)
        if spec_ids is not None:
            w = np.in1d(dataset_table['SPEC_ID'],spec_ids)
            dataset_table = dataset_table[w]
        nonVI_tables.append(dataset_table)

    ## Join each table.
    if common_specids:
        join_type = 'inner'
    else:
        join_type = 'outer'
    table = nonVI_tables[0]
    if len(nonVI_tables)>1:
        for t in nonVI_tables[1:]:
            table = join(table,t,keys=['OBJ_ID','SPEC_ID'],join_type=join_type)

    ## For each SPEC_ID, extract the VI data from the VI dataset.
    new_vi_data = []
    # Dict for converting to text class.
    conv_class = {0: 'BAD', 1: 'STAR', 2: 'GALAXY', 3: 'QSO'}
    for i,obj_id in enumerate(table['OBJ_ID'].data):
        new_vi_data += [(table['SPEC_ID'].data[i],
                         obj_id,
                         truth[obj_id].z_conf,
                         truth[obj_id].z,
                         conv_class[truth[obj_id].objclass],
                         truth[obj_id].objclass==3)]
    dtype = [('SPEC_ID','i8'),('OBJ_ID','i8'),('ZCONF_PERSON','i8'),('Z_VI','f8'),('CLASS_VI','U8'),('ISQSO_VI','bool')]
    new_vi_data = np.array(new_vi_data,dtype=dtype)
    vi_table = Table(new_vi_data)

    ## Join the VI table to the full table.
    table = join(vi_table,table,keys=['OBJ_ID','SPEC_ID'])

    ## Only show a reduced number of digits for redshifts, and other floats.
    ks = [cn for cn in table.colnames if ('Z_' in cn)]
    for k in ks:
        table[k].format = '1.3f'
    if include_c_qn:
        for c in data.keys():
            if 'QN' in c:
                table['C_{}'.format(c)].format = '1.3f'
    if include_cbal_qn:
        for c in data.keys():
            if 'QN' in c:
                table['CBAL_{}'.format(c)].format = '1.3f'
    if include_p_sq:
        for c in data.keys():
            if 'SQ' in c:
                table['P_{}'.format(c)].format = '1.3f'

    return table

def get_w_compare(table,vi_agree,vi_disagree,dv_max=None,dv_min=None,verbose=True):
    """
    Function to compare whether different classifiers agree or disagree with
    VI in terms of QSO/non-QSO classification.

    Input:
     - table: the data table in which the comparison will be carried out.
     - vi_agree: a list of identifiers ('RR','QN','SQ') for which we want to
                 see where they agree with VI in terms of QSO/non-QSO
                 classification.
     - vi_disagree: a list of identifiers ('RR','QN','SQ') for which we want to
                 see where they disagree with VI in terms of QSO/non-QSO
                 classification.
     - dv_max: a maximum velocity error (km/s) to also use when determining
               agreement.
     - verbose: whether to print the results or not.

    Output:
     - w: a boolean array, for which w[i]=True means that, for the spectrum
          corresponding to row i of the table, all of the classifiers listed
          in vi_agree agreed with VI, and all of the classifiers listed in
          vi_disagree disagreed with VI.
    """

    w_agree = np.ones(len(table),dtype='bool')
    for d in vi_agree:
        w_agree_aux = np.ones(len(table),dtype='bool')
        w_agree_aux &= (table['ISQSO_VI']==table['ISQSO_{}'.format(d)])
        if dv_max:
            w_agree_aux &= (300000.*abs(table['Z_VI'] - table['Z_{}'.format(d)])/(1+table['Z_VI']))<=dv_max
        w_agree &= w_agree_aux

    w_disagree = np.ones(len(table),dtype='bool')
    for d in vi_disagree:
        w_disagree_aux = np.ones(len(table),dtype='bool')
        w_disagree_aux &= ~(table['ISQSO_VI']==table['ISQSO_{}'.format(d)])
        if dv_min:
            w_disagree_aux &= (300000.*abs(table['Z_VI'] - table['Z_{}'.format(d)])/(1+table['Z_VI']))>dv_min
        w_disagree &= w_disagree_aux

    w = w_agree*w_disagree

    if verbose:
        print('INFO: for {}/{} ({:.2%}) spectra, {} agree and {} disagree with VI'.format(w.sum(),len(w),w.sum()/len(w),vi_agree,vi_disagree))

    return w

def targetid2platemjdfiber(targetid):
    fiber = targetid % 10000
    mjd = (targetid // 10000) % 100000
    plate = (targetid // (10000 * 100000))
    return (plate, mjd, fiber)

def platemjdfiber2targetid(plate, mjd, fiber):
    return plate*1000000000 + mjd*10000 + fiber

def plot_spectrum(table,p,m,f,pmf2tid,figsize=(10,6),f_rr=None,f_qn=False):

    fig, ax = plt.subplots(1,1,figsize=figsize)
    targetid = platemjdfiber2targetid(p,m,f)

    ## Open the file.
    f_spPlate = '/global/projecta/projectdirs/sdss/staging/dr12/boss/spectro/redux/v5_7_0/{p}/spPlate-{p}-{m}.fits'.format(p=p,m=m)
    #f_spPlate = '/Volumes/external/ohio_data/spPlate-{p}-{m}.fits'.format(p=p,m=m)
    h = fits.open(f_spPlate)

    ## Get the flux spectrum.
    i = np.where(h[5].data['FIBERID']==f)[0][0]
    fl = h[0].data[i,:]

    ## Construct the wavelength grid.
    head = h[0].header
    c0 = head["COEFF0"]
    c1 = head["COEFF1"]
    wave_grid = 10**(c0 + c1*np.arange(fl.shape[0]))

    h.close()

    ## Plot the spectrum.
    ax.plot(wave_grid,fl)

    ## Get the thing_id to read prediction information.
    try:
        tid = pmf2tid[(p,m,f)]
        #tid = 316738985
    except KeyError:
        raise ValueError('pmf not found')

    ## Add information to the title.
    title = 'THING_ID={}, (p,m,f)=({},{},{})'.format(tid,p,m,f)
    try:
        i = np.where(table['THING_ID']==tid)[0][0]
    except IndexError:
        raise ValueError('thing_id {} not found in comparison set'.format(tid))

    ks = [cn[2:] for cn in table.colnames if ('Z_' in cn)]
    for k in ks:
        title += '\n{:2}: class={:8}, z={:.3f}'.format(k,table['CLASS_{}'.format(k)][i],table['Z_{}'.format(k)][i])
    ax.set_title(title)

    if f_rr:
        ## If possible, try to also plot the redrock template.
        try:
            import redrock.templates

            ## Open the redrock output file from earlier.
            rr = fits.open(f_rr)
            zbest = rr[1].data

            ## Load all of the template information.
            templates = dict()
            for filename in redrock.templates.find_templates():
                t = redrock.templates.Template(filename)
                templates[(t.template_type, t.sub_type)] = t

            ## Get data for the spectrum we want.
            i = np.where(zbest['TARGETID']==targetid)[0][0]
            z = zbest['Z'][i]
            spectype = zbest['SPECTYPE'][i].strip()
            subtype = zbest['SUBTYPE'][i].strip()
            fulltype = (spectype, subtype)
            print(templates.keys())
            ncoeff = templates[fulltype].flux.shape[0]
            coeff = zbest['COEFF'][i][0:ncoeff]

            rr.close()

            ## Get the template flux and wavelength, and plot.
            tflux = templates[fulltype].flux.T.dot(coeff)
            twave = templates[fulltype].wave * (1+z)
            ax.plot(twave,tflux,c='k')

        except ModuleNotFoundError:
            pass

    ax.grid()

    return fig, ax

def autolabel_bars(ax,rects,numbers=None,heights=None,percentage=False,above=False,ndpmin=1):
    """Attach a text label above each bar in *rects*, displaying its height."""
    if numbers is not None:
        assert len(rects)==len(heights)
    if numbers is not None:
        assert len(rects)==len(numbers)
    if above:
        sign = 1
        va = 'bottom'
    else:
        sign = -1
        va = 'top'
    for ir,rect in enumerate(rects):
        if heights is not None:
            height = heights[ir]
        else:
            height = rect.get_height()
        if numbers is not None:
            n = numbers[ir]
        else:
            n = height
        if not percentage:
            ax.annotate('{}'.format(n),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, sign*5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va=va, color='k')
        else:
            if n != 0:
                ndp = np.maximum(-int(np.floor(np.log10(n*100))),1)
            else:
                ndp = 1
            ax.annotate(print_format_pct(n,ndpmin=ndpmin),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, sign*5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va=va, color='k')

def print_format_pct(p,ndpmin=1):
    if p != 0:
        ndp = np.maximum(-int(np.floor(np.log10(p*100))),ndpmin)
    else:
        ndp = 1
    return '{:.{ndp}f}%'.format(p*100,ndp=ndp)
