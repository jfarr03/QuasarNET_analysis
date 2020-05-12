import numpy as np
import glob
from scipy.stats import norm
from matplotlib import pyplot as plt
from astropy.io import fits
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from quasarnet.models import QuasarNET, custom_loss
from quasarnet.io import read_truth, read_data, objective
from quasarnet.utils import process_preds, absorber_IGM


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
        
    h = fits.open(train_file)
    tids_train = h[1].data['TARGETID']
    w = np.in1d(tids_full, tids_train)
    X_train = X_full[w]
    Y_train = Y_full[w]
    z_train = z_full[w]
    bal_train = bal_full[w]
    h.close()

    ## to get the validation data, remove the spectra in the training sample from the full sample
    w = ~np.in1d(tids_full, tids_train)
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
