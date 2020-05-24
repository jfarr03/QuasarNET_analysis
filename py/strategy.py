import numpy as np
import copy
import time

from qn_analysis import plot

class Strategy():
    def __init__(self,cf_type,cf_kwargs={},name=None,c=None,ls=None,marker=None):

        self.classifying_fn = get_cf(cf_type,cf_kwargs)
        self.name = name
        self.c = c
        self.ls = ls
        self.marker = marker

        return

    def predict(self,data_table,filter=None,c_kwargs={},class_true_name='CLASS_VI',z_true_name='Z_VI'):

        start = time.time()
        #print('start!')
        temp_data_table = filter_table(data_table,filter)
        #print('checkpoint: filtered table', time.time()-start)
        start = time.time()

        isqso, z = self.classifying_fn(temp_data_table,filter=None,**c_kwargs)
        #print('checkpoint: got isqso,z', time.time()-start)
        start = time.time()

        class_true = copy.deepcopy(temp_data_table[class_true_name].data)
        #print('checkpoint: making class_true', time.time()-start)
        start = time.time()

        z_true = copy.deepcopy(temp_data_table[z_true_name].data)
        #print('checkpoint: making ztrue', time.time()-start)
        start = time.time()

        prediction = Prediction(isqso,z,class_true,z_true)
        #print('checkpoint: making Prediction obj', time.time()-start)
        start = time.time()

        return prediction

class Prediction():
    def __init__(self,isqso,z,class_true,z_true):

        self.isqso = isqso
        self.z = z
        self.class_true = class_true
        self.z_true = z_true

        return

    def get_ishighzqso(self,zcut=2.1):

        ishighzqso = (self.isqso) & (self.z>=zcut)

        return ishighzqso

    def calculate_pur_com(self,dv_max=6000.):

        isqso_truth = (self.class_true=='QSO')
        isgal_truth = (self.class_true=='GALAXY')
        isbad = (self.class_true=='BAD')
        pur, com = plot.get_pur_com(self.isqso,self.z,isqso_truth,isgal_truth,isbad,self.z_true,dv_max=dv_max)

        return pur, com

    def calculate_dv(self,use_abs=False):

        dv = get_dv(self.z,self.z_true,self.z_true,use_abs=use_abs)

        return dv

def get_cf(cf_type,cf_kwargs):

    if cf_type == 'qn':
        return get_cf_qn(**cf_kwargs)
    elif cf_type == 'rr':
        return get_cf_rr(**cf_kwargs)
    elif cf_type == 'sq':
        return get_cf_sq(**cf_kwargs)
    elif cf_type == 'qnorrr':
        return get_cf_qnorrr(**cf_kwargs)
    elif cf_type == 'qnandrr':
        return get_cf_qnandrr(**cf_kwargs)
    elif cf_type == 'qnplusvi':
        return get_cf_qnplusvi(**cf_kwargs)
    elif cf_type == 'rrplusvi':
        return get_cf_rrplusvi(**cf_kwargs)
    elif cf_type == 'rrplusvialt':
        return get_cf_rrplusvialt(**cf_kwargs)
    elif cf_type == 'qnandrrplusvi':
        return get_cf_qnandrrplusvi(**cf_kwargs)
    elif cf_type == 'qnandrrplusviadv':
        return get_cf_qnandrrplusviadv(**cf_kwargs)
    elif cf_type == 'qnplusrrplusvi':
        return get_cf_qnplusrrplusvi(**cf_kwargs)
    else:
        raise ValueError('cf_type value of {} not recognised!'.format(cf_type))

    return

def filter_table(data_table,filter):

    if filter is not None:
        filtered_table = data_table[filter]
    else:
        filtered_table = data_table

    return filtered_table

def get_dv(z1,z2,ztrue,use_abs=False):

    dv = (300000.*(z1-z2)/(1+ztrue))
    if use_abs:
        dv = abs(dv)

    return dv

def get_cf_qn(qn_name='QN',specid_name='SPEC_ID'):

    def cf(data_table,c_th=0.6,n_detect=1,filter=None):

        # Apply filter to the data table.
        temp_data_table = filter_table(data_table,filter)

        # Extract z/isqso values.
        z = copy.deepcopy(temp_data_table['Z_{}'.format(qn_name)].data)
        isqso = ((temp_data_table['C_{}'.format(qn_name)]>c_th).sum(axis=1)>=n_detect)

        return isqso, z

    return cf

def get_cf_sq(sq_name='SQ',specid_name='SPEC_ID'):

    def cf(data_table,p_min=0.32,filter=None):

        # Apply filter to the data table.
        temp_data_table = filter_table(data_table,filter)

        # Extract z/isqso values.
        z = copy.deepcopy(temp_data_table['Z_{}'.format(sq_name)].data)
        isqso = (temp_data_table['P_{}'.format(sq_name)]>p_min)

        return isqso, z

    return cf

def get_cf_rr(rr_name='RR',specid_name='SPEC_ID'):

    def cf(data_table,zwarn=None,filter=None):

        # Apply filter to the data table.
        temp_data_table = filter_table(data_table,filter)

        # Extract z/isqso values.
        z = copy.deepcopy(temp_data_table['Z_{}'.format(rr_name)].data)
        isqso = copy.deepcopy(temp_data_table['ISQSO_{}'.format(rr_name)].data)

        # If zwarn==True, set all spectra with zwarn>0 to have isqso=True.
        # If zwarn==False, set all spectra with zwarn>0 to have isqso=False.
        # Otherwise, leave alone (i.e. ignore zwarn and use best fit).
        if zwarn is not None:
            zwarn_nonzero = (temp_data_table['ZWARN_{}'.format(rr_name)]>0)
            isqso[zwarn_nonzero] = zwarn

        return isqso, z

    return cf

def get_cf_qnorrr(qn_name='QN',rr_name='RR',specid_name='SPEC_ID'):

    def cf(data_table,qn_kwargs={},rr_kwargs={},zchoice='QN',filter=None):

        # Apply filter to the data table.
        temp_data_table = filter_table(data_table,filter)

        # Get classifications from both QN and RR.
        cf_qn = get_cf_qn(qn_name=qn_name,specid_name=specid_name)
        cf_rr = get_cf_rr(rr_name=rr_name,specid_name=specid_name)
        isqso_qn, z_qn = cf_qn(temp_data_table,**qn_kwargs,filter=filter)
        isqso_rr, z_rr = cf_rr(temp_data_table,**rr_kwargs,filter=filter)

        # Combine using |, and choosing z based on zchoice.
        isqso = isqso_qn | isqso_rr
        z = z_rr
        if zchoice=='QN':
            z[isqso_qn] = z_qn[isqso_qn]
        elif zchoice=='RR':
            w_zqn = (~isqso_rr)&isqso_qn
            z[w_zqn] = z_qn[w_zqn]

        return isqso, z

    return cf

def get_cf_qnandrr(qn_name='QN',rr_name='RR',specid_name='SPEC_ID'):

    def cf(data_table,qn_kwargs={},rr_kwargs={},dv_max=6000.,zchoice='QN',filter=None):

        # Apply filter to the data table.
        temp_data_table = filter_table(data_table,filter)

        # Get classifications from both QN and RR.
        cf_qn = get_cf_qn(qn_name=qn_name,specid_name=specid_name)
        cf_rr = get_cf_rr(rr_name=rr_name,specid_name=specid_name)
        isqso_qn, z_qn = cf_qn(temp_data_table,**qn_kwargs,filter=filter)
        isqso_rr, z_rr = cf_rr(temp_data_table,**rr_kwargs,filter=filter)

        # Combine using &, and choosing z based on zchoice.
        dv = get_dv(z_qn,z_rr,temp_data_table['Z_VI'])
        isqso = isqso_qn & isqso_rr & (dv<=dv_max)
        z = z_rr
        if zchoice=='QN':
            z[isqso_qn] = z_qn[isqso_qn]

        return isqso, z

    return cf

def get_cf_qnplusvi(qn_name='QN',specid_name='SPEC_ID'):

    def cf(data_table,c_th_lo=0.1,c_th_hi=0.9,n_detect=1,filter=None):

        # Apply filter to the data table.
        temp_data_table = filter_table(data_table,filter)

        # Get lo and hi classifications from QN.
        cf_qn = get_cf_qn(qn_name=qn_name,specid_name=specid_name)
        isqso_qn_lo, z_qn_lo = cf_qn(temp_data_table,c_th=c_th_lo,n_detect=n_detect,filter=filter)
        isqso_qn_hi, z_qn_hi = cf_qn(temp_data_table,c_th=c_th_hi,n_detect=n_detect,filter=filter)

        # Select to use VI when spectra have middling confidence (i.e. when the
        # lo threshold would give True, but hi would give False).
        use_vi = isqso_qn_lo & (~isqso_qn_hi)
        print('INFO: QN+VI sends {}/{} ({:2.2%}) spectra to VI'.format(use_vi.sum(),len(temp_data_table),use_vi.sum()/len(temp_data_table)))

        # Construct outputs.
        isqso = isqso_qn_hi
        isqso[use_vi] = copy.deepcopy(temp_data_table['ISQSO_VI'].data[use_vi])
        z = z_qn_hi
        z[use_vi] = copy.deepcopy(temp_data_table['Z_VI'].data[use_vi])

        return isqso, z

    return cf

def get_cf_rrplusvi(rr_name='RR',specid_name='SPEC_ID'):

    def cf(data_table,filter=None):

        # Apply filter to the data table.
        temp_data_table = filter_table(data_table,filter)

        # Get classifications from RR.
        cf_rr = get_cf_rr(rr_name=rr_name,specid_name=specid_name)
        isqso_rr, z_rr = cf_rr(temp_data_table,zwarn=False,filter=filter)

        # Select to use VI when a zwarn flag is raised.
        use_vi = (temp_data_table['ZWARN_{}'.format(rr_name)]>0)
        print('INFO: RR+VI sends {}/{} ({:2.1%}) spectra to VI'.format(use_vi.sum(),len(temp_data_table),use_vi.sum()/len(temp_data_table)))

        # Construct outputs.
        isqso = isqso_rr
        isqso[use_vi] = copy.deepcopy(temp_data_table['ISQSO_VI'].data[use_vi])
        z = z_rr
        z[use_vi] = copy.deepcopy(temp_data_table['Z_VI'].data[use_vi])

        return isqso, z

    return cf

def get_cf_rrplusvialt(rr_name='RR',specid_name='SPEC_ID'):

    def cf(data_table,filter=None):

        # Apply filter to the data table.
        temp_data_table = filter_table(data_table,filter)

        # Get classifications from RR.
        cf_rr = get_cf_rr(rr_name=rr_name,specid_name=specid_name)
        isqso_rr, z_rr = cf_rr(temp_data_table,zwarn=False,filter=filter)
        isqso_rr_zwt, z_rr_zwt = cf_rr(temp_data_table,zwarn=True,filter=filter)

        # Select to use VI when a zwarn flag is raised and best fit is QSO.
        use_vi = (isqso_rr_zwt) & (~isqso_rr)
        print('INFO: RR+VI alt. sends {}/{} ({:2.1%}) spectra to VI'.format(use_vi.sum(),len(temp_data_table),use_vi.sum()/len(temp_data_table)))

        # Construct outputs.
        isqso = isqso_rr
        isqso[use_vi] = copy.deepcopy(temp_data_table['ISQSO_VI'].data[use_vi])
        z = z_rr
        z[use_vi] = copy.deepcopy(temp_data_table['Z_VI'].data[use_vi])

        return isqso, z

    return cf

def get_cf_qnandrrplusvi(qn_name='QN',rr_name='RR',specid_name='SPEC_ID'):

    def cf(data_table,qn_kwargs={},rr_kwargs={},dv_max=6000.,zchoice='QN',filter=None):

        # Apply filter to the data table.
        temp_data_table = filter_table(data_table,filter)

        # Get classifications from both QN and RR.
        cf_qn = get_cf_qn(qn_name=qn_name,specid_name=specid_name)
        cf_rr = get_cf_rr(rr_name=rr_name,specid_name=specid_name)
        isqso_qn, z_qn = cf_qn(temp_data_table,**qn_kwargs,filter=filter)
        isqso_rr, z_rr = cf_rr(temp_data_table,**rr_kwargs,filter=filter)

        # Select to use VI when RR and QN disagree.
        dv = get_dv(z_qn,z_rr,temp_data_table['Z_VI'])
        use_vi = ((isqso_qn|isqso_rr) & (~(isqso_qn&isqso_rr))) | ((isqso_qn & isqso_rr) & (dv>dv_max))
        print('INFO: QN&RR+VI sends {}/{} ({:2.1%}) spectra to VI'.format(use_vi.sum(),len(temp_data_table),use_vi.sum()/len(temp_data_table)))

        # Construct outputs.
        isqso = isqso_qn & isqso_rr & (dv<=dv_max)
        isqso[use_vi] = copy.deepcopy(temp_data_table['ISQSO_VI'].data[use_vi])
        z = z_rr
        if zchoice=='QN':
            z[isqso_qn] = z_qn[isqso_qn]
        z[use_vi] = copy.deepcopy(temp_data_table['Z_VI'].data[use_vi])

        return isqso, z

    return cf

def get_cf_qnandrrplusviadv(qn_name='QN',rr_name='RR',specid_name='SPEC_ID'):

    def cf(data_table,c_th_lo=0.1,c_th_hi=0.9,n_detect=1,dv_max=6000.,zchoice='QN',filter=None):

        # Apply filter to the data table.
        temp_data_table = filter_table(data_table,filter)

        # Get classifications from both QN and RR.
        cf_qn = get_cf_qn(qn_name=qn_name,specid_name=specid_name)
        cf_rr = get_cf_rr(rr_name=rr_name,specid_name=specid_name)
        isqso_qn_lo, z_qn_lo = cf_qn(temp_data_table,c_th=c_th_lo,n_detect=n_detect,filter=filter)
        isqso_qn_hi, z_qn_hi = cf_qn(temp_data_table,c_th=c_th_hi,n_detect=n_detect,filter=filter)
        isqso_rr_zwf, z_rr_zwf = cf_rr(temp_data_table,zwarn=False,filter=filter)
        zwarn_nonzero = (temp_data_table['ZWARN_{}'.format(rr_name)]>0)

        # Asks for VI if:
        # 1. RR says QSO without zwarn, and QN says cth_lo<c<cth_hi
        # 2. QN says c>cth_hi but RR gives zwarn.
        use_vi = ((isqso_qn_lo & (~isqso_qn_hi)) & (isqso_rr_zwf)) | (isqso_qn_hi & zwarn_nonzero)
        print('INFO: RR&QN+VI adv. sends {}/{} ({:2.1%}) spectra to VI'.format(use_vi.sum(),len(temp_data_table),use_vi.sum()/len(temp_data_table)))

        # Construct outputs.
        isqso = isqso_qn_hi & isqso_rr_zwf
        isqso[use_vi] = copy.deepcopy(temp_data_table['ISQSO_VI'].data[use_vi])
        z = z_rr_zwf
        if zchoice=='QN':
            z[isqso_qn_hi] = z_qn_hi[isqso_qn_hi]
        z[use_vi] = copy.deepcopy(temp_data_table['Z_VI'].data[use_vi])

        return isqso, z

    return cf

def get_cf_qnplusrrplusvi(qn_name='QN',rr_name='RR',specid_name='SPEC_ID'):

    def cf(data_table,c_th_lo=0.1,c_th_hi=0.9,n_detect=1,dv_max=6000.,zchoice='QN',filter=None):

        # Apply filter to the data table.
        temp_data_table = filter_table(data_table,filter)

        # Get classifications from both QN and RR.
        cf_qn = get_cf_qn(qn_name=qn_name,specid_name=specid_name)
        cf_rr = get_cf_rr(rr_name=rr_name,specid_name=specid_name)
        isqso_qn_lo, z_qn_lo = cf_qn(temp_data_table,c_th=c_th_lo,n_detect=n_detect,filter=filter)
        isqso_qn_hi, z_qn_hi = cf_qn(temp_data_table,c_th=c_th_hi,n_detect=n_detect,filter=filter)
        isqso_rr_zwf, z_rr_zwf = cf_rr(temp_data_table,zwarn=False,filter=filter)
        isqso_rr_zwt, z_rr_zwt = cf_rr(temp_data_table,zwarn=True,filter=filter)
        zwarn_nonzero = (temp_data_table['ZWARN_{}'.format(rr_name)]>0)

        # Asks for VI if:
        # 1. RR raises zwarn, and QN says cth_lo<c<cth_hi
        use_vi = ((isqso_qn_lo & (~isqso_qn_hi)) & zwarn_nonzero)
        print('INFO: RR+QN+VI sends {}/{} ({:2.1%}) spectra to VI'.format(use_vi.sum(),len(temp_data_table),use_vi.sum()/len(temp_data_table)))

        # Construct outputs.
        isqso = (isqso_qn_hi & isqso_rr_zwt) | (isqso_rr_zwf & isqso_qn_lo)
        isqso[use_vi] = copy.deepcopy(temp_data_table['ISQSO_VI'].data[use_vi])
        z = z_rr_zwf
        if zchoice=='QN':
            w_zqn = isqso_qn_hi&(~isqso_rr_zwt)
            z[w_zqn] = z_qn_hi[w_zqn]
        z[use_vi] = copy.deepcopy(temp_data_table['Z_VI'].data[use_vi])

        return isqso, z

    return cf

## Obsolete for now.
def check_specid_array(specid):

    if isinstance(specid,float) or isinstance(specid,int):
        specid = np.array([specid])
    if isinstance(specid,list):
        specid = np.array(specid)

    return specid
