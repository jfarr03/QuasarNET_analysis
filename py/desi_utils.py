import numpy as np
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt

colours = {'C0': '#F5793A',
           'C1': '#A95AA1',
           'C2': '#85C0F9',
           'C3': '#0F2080',
          }

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

def load_rr_data(f_rr):

    rr = fits.open(f_rr)

    ## Make a boolean isqso.
    isqso = (rr[1].data['SPECTYPE']=='QSO')
    
    rr_data = list(zip(rr[1].data['TARGETID'], rr[1].data['Z'], rr[1].data['SPECTYPE'], isqso, rr[1].data['TARGETID'], rr[1].data['ZWARN']))
    dtype = [('THING_ID','i8'),('Z','f8'),('CLASS','U8'),('ISQSO','bool'),('TARGETID','i8'),('ZWARN','i8')]
    rr_data = np.array(rr_data, dtype=dtype)

    return rr_data

def load_qn_data(f_qn,n_lines=1,c_th=0.8,include_cmax=False,include_cmax2=False):

    qn = fits.open(f_qn)
    
    w = (~(qn[1].data['IN_TRAIN'].astype('bool')))
    data = qn[1].data[w]
    
    ## Calculate which spectra we think are QSOs.
    isqso = (data['C_LINES']>c_th).sum(axis=1)>n_lines

    ## Convert the QN class labelling.
    objclass = np.array(['']*len(isqso),dtype='U8')
    objclass[isqso] = 'QSO'
    objclass[~isqso] = 'NONQSO'

    targetid = data['THING_ID']
    
    csort = np.sort(data['C_LINES'],axis=1)
    cmax = csort[:,-1]
    cmax2 = csort[:,-2]

    if include_cmax & include_cmax2:
        qn_data = list(zip(data['ZBEST'], objclass, isqso, targetid, cmax, cmax2))
        dtype = [('Z','f8'),('CLASS','U8'),('ISQSO','bool'),('TARGETID','i8'),('CMAX','f8'),('CMAX2','f8')]
        qn_data = np.array(qn_data, dtype=dtype)
    elif include_cmax2:
        cmax = data['C_LINES'].max(axis=1)
        qn_data = list(zip(data['ZBEST'], objclass, isqso, targetid, cmax2))
        dtype = [('Z','f8'),('CLASS','U8'),('ISQSO','bool'),('TARGETID','i8'),('CMAX2','f8')]
        qn_data = np.array(qn_data, dtype=dtype)
    elif include_cmax:
        cmax = data['C_LINES'].max(axis=1)
        qn_data = list(zip(data['ZBEST'], objclass, isqso, targetid, cmax))
        dtype = [('Z','f8'),('CLASS','U8'),('ISQSO','bool'),('TARGETID','i8'),('CMAX','f8')]
        qn_data = np.array(qn_data, dtype=dtype)
    else:
        qn_data = list(zip(data['ZBEST'], objclass, isqso, targetid))
        dtype = [('Z','f8'),('CLASS','U8'),('ISQSO','bool'),('TARGETID','i8')]
        qn_data = np.array(qn_data, dtype=dtype)
    
    return qn_data

## 
def load_sq_data(f_sq,p_min=0.32,include_p=False):

    sq = fits.open(f_sq)

    ## Remove duplicated spectra.
    w = ~(sq[1].data['DUPLICATED'])
    data = sq[1].data[w]

    isqso = (data['prob']>p_min)

    ## Convert the class labelling.
    objclass = np.array(['']*len(data),dtype='U8')
    objclass[isqso] = 'QSO'
    objclass[~isqso] = 'NONQSO'

    ## Make targetid.
    targetid = data['TARGETID']

    if include_p:
        sq_data = list(zip(data['Z_TRY'], objclass, isqso, targetid, data['PROB']))
        dtype = [('Z','f8'),('CLASS','U8'),('ISQSO','bool'),('TARGETID','i8'),('P','f8')]
        sq_data = np.array(sq_data, dtype=dtype)
    else:
        sq_data = list(zip(data['Z_TRY'], objclass, isqso, targetid))
        dtype = [('Z','f8'),('CLASS','U8'),('ISQSO','bool'),('TARGETID','i8')]
        sq_data = np.array(sq_data, dtype=dtype)
        
    return sq_data

def reduce_data_to_table(data,truth=None,verbose=True,include_cmax_qn=False,include_cmax2_qn=False,include_p_sq=False,common_targetids=True):

    if truth is not None:
        ## In each, reduce to the set of spectra that are in the truth dictionary.
        objids_truth = list(truth.keys())
        nonVI_datasets = [c for c in data.keys() if c!='VI']
        for c in nonVI_datasets:
            w = np.in1d(data[c]['TARGETID'],objids_truth)
            data[c] = data[c][w]
            
        ## Find the set of common targetids in the reduced non-VI datasets.
        ct_set = set(data[nonVI_datasets[0]]['TARGETID'])
        if len(nonVI_datasets)>1:
            for c in nonVI_datasets[1:]:
                ct_set = ct_set.intersection(set(data[c]['TARGETID']))
        common_targetids = np.array(list(ct_set))

        ## In each non-VI dataset, reduce to the set of common targetids.
        ## Sort the data by TARGETID in each of the reduced non-VI datasets.
        for c in nonVI_datasets:
            w = np.in1d(data[c]['TARGETID'],common_targetids)
            data[c] = data[c][w]
            data[c].sort(order='TARGETID')

        if verbose:
            print('INFO: {} common targetids'.format(len(common_targetids)))

        ## Assert that all TARGETID columns are identical before proceeding.
        ref = nonVI_datasets[0]
        for c in nonVI_datasets:
            assert (len(data[ref]['TARGETID'])==len(data[c]['TARGETID']))
            assert (data[ref]['TARGETID']==data[c]['TARGETID']).all()
            
        ## For each TARGETID, extract the VI data from the VI dataset.
        new_vi_data = []

        # Dict for converting to text class.
        conv_class = {0: 'BAD', 1: 'STAR', 2: 'GALAXY', 3: 'QSO'}
        for i,targetid in enumerate(data[ref]['TARGETID']):
            new_vi_data += [(truth[targetid].z_conf,
                             truth[targetid].z,
                             conv_class[truth[targetid].objclass],
                             truth[targetid].objclass==3)]
        dtype = [('ZCONF_PERSON','i8'),('Z_VI','f8'),('CLASS_VI','U8'),('ISQSO_VI','bool')]
        new_vi_data = np.array(new_vi_data,dtype=dtype)

        ## Now make a table by assembling columns.
        cols = []
        colnames = []

        # First the ID columns.
        for k in ['TARGETID']:
            cols += [data[ref][k]]
            colnames += [k]

        # Now add plate, MJD, FIBERID columns.
        #plate,mjd,fiberid = targetid2platemjdfiber(data[ref]['TARGETID'])
        #cols += [plate,mjd,fiberid]
        #colnames += ['PLATE','MJD','FIBERID']

        # Now the VI columns.
        for k in ['ZCONF_PERSON','Z_VI','CLASS_VI','ISQSO_VI']:
            cols += [new_vi_data[k]]
            colnames += [k]

        # Now the data columns.
        for k in ['Z','CLASS','ISQSO']:
            for c in data.keys():
                cols += [data[c][k]]
                colnames += ['{}_{}'.format(k,c)]

        # Now optional extras.
        if include_cmax_qn:
            for c in data.keys():
                if 'QN' in c:
                    cols += [data[c]['CMAX']]
                    colnames += ['CMAX_{}'.format(c)]
        if include_cmax2_qn:
            for c in data.keys():
                if 'QN' in c:
                    cols += [data[c]['CMAX2']]
                    colnames += ['CMAX2_{}'.format(c)]
        if include_p_sq:
            for c in data.keys():
                if 'SQ' in c:
                    cols += [data[c]['P']]
                    colnames += ['P_{}'.format(c)]
        for c in data.keys():
            if 'RR' in c:
                cols += [data[c]['ZWARN']]
                colnames += ['ZWARN_{}'.format(c)]
    else:
        try:
            test = data['VI']
            print('INFO: No truth provided, using VI in data instead.')
        except KeyError:
            raise KeyError('No VI data found: check entry is labelled as "VI"!')

        ## Produce a list of common targetids.
        ct_set = set(data['VI']['TARGETID'])
        for c in data.keys():
            ct_set = ct_set.intersection(set(data[c]['TARGETID']))
        common_tids = np.array(list(ct_set))

        if verbose:
            print('INFO: {} common targetids'.format(len(common_tids)))

        ## Reduce all data so that only these common thing_ids are included
        for c in data.keys():
            w = np.in1d(data[c]['TARGETID'],common_tids)
            data[c] = data[c][w]
            data[c].sort(order='TARGETID')

        ## Assert that all THING_ID columns are identical before proceeding.
        for c in data.keys():
            assert (data['VI']['TARGETID'].shape==data[c]['TARGETID'].shape)
            assert (data['VI']['TARGETID']==data[c]['TARGETID']).all()

        ## Construct a table to make life easier!
        cols = []
        colnames = []
        for k in ['THING_ID','PLATE','MJD','FIBERID','ZCONF_PERSON']:
            cols += [data['VI'][k]]
            colnames += [k]

        ## Add the redshift, class and isqso binary for each classifier.
        for name in ['Z','CLASS','ISQSO']:
            for c in data.keys():
                cols += [data[c][name]]
                colnames += ['{}_{}'.format(name,c)]

        ## Add optional extras.
        if include_cmax_qn:
            for c in data.keys():
                if 'QN' in c:
                    cols += [data[c]['CMAX']]
                    colnames += ['CMAX_{}'.format(c)]
        if include_cmax2_qn:
            for c in data.keys():
                if 'QN' in c:
                    cols += [data[c]['CMAX2']]
                    colnames += ['CMAX2_{}'.format(c)]
        if include_p_sq:
            for c in data.keys():
                if 'SQ' in c:
                    cols += [data[c]['P']]
                    colnames += ['P_{}'.format(c)]

    
    
    table = Table(cols,names=colnames)

    ## Only show a reduced number of digits for redshifts, and other floats.
    ks = [cn for cn in table.colnames if ('Z_' in cn)]
    for k in ks:
        table[k].format = '1.3f'
    if include_cmax_qn:
        for c in data.keys():
            if 'QN' in c:
                table['CMAX_{}'.format(c)].format = '1.3f'
    if include_cmax2_qn:
        for c in data.keys():
            if 'QN' in c:
                table['CMAX2_{}'.format(c)].format = '1.3f'
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
            ndp = np.maximum(-int(np.floor(np.log10(n*100))),1)
            ax.annotate(print_format_pct(n,ndpmin=ndpmin),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, sign*5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va=va, color='k')
            
def print_format_pct(p,ndpmin=1):
    ndp = np.maximum(-int(np.floor(np.log10(p*100))),ndpmin)
    return '{:.{ndp}f}%'.format(p*100,ndp=ndp)