import numpy as np
import copy

class Strategy():
    def __init__(self,data_table,cf_type,cf_kwargs={},name=None,c=None,ls=None,marker=None):

        self.data_table = data_table
        self.classifying_fn = get_cf(data_table,cf_type,cf_kwargs)
        self.name = name
        self.c = c
        self.ls = ls
        self.marker = marker

        return

    def predict(self,filter=None,classification_kwargs={},class_true_name='CLASS_VI',z_true_name='Z_VI'):

        isqso, z = self.classifying_fn(filter=filter,**classification_kwargs)

        class_true = self.data_table[class_true_name]
        z_true = self.data_table[z_true_name]
        prediction = Prediction(isqso,z,class_true,z_true)

        return prediction

class Prediction():
    def __init__(self,isqso,z,class_true,z_true):

        self.isqso = isqso
        self.z = z
        self.class_true = class_true
        self.z_true = z_true

        return

def get_cf(data_table,cf_type,cf_kwargs):

    if cf_type == 'qn':
        return get_cf_qn(data_table,**cf_kwargs)
    if cf_type == 'rr':
        return get_cf_rr(data_table,**cf_kwargs)
    if cf_type == 'sq':
        return get_cf_sq(data_table,**cf_kwargs)
    if cf_type == 'qnorrr':
        return get_cf_qnorrr(data_table,**cf_kwargs)
    if cf_type == 'qnandrr':
        return get_cf_qnandrr(data_table,**cf_kwargs)
    if cf_type == 'qnplusvi':
        return get_cf_qnplusvi(data_table,**cf_kwargs)
    if cf_type == 'rrplusvi':
        return get_cf_rrplusvi(data_table,**cf_kwargs)
    if cf_type == 'qnandrrplusvi':
        return get_cf_qnandrrplusvi(data_table,**cf_kwargs)
    if cf_type == 'qnandrrplusviadv':
        return get_cf_qnandrrplusviadv(data_table,**cf_kwargs)

    return

def filter_table(data_table,filter):

    if filter is not None:
        filtered_table = data_table[filter]
    else:
        filtered_table = data_table

    return filtered_table

def get_dv(z1,z2,ztrue):

    dv = (300000.*abs(z1-z2)/(1+ztrue))

    return dv

def get_cf_qn(data_table,qn_name='QN',specid_name='SPEC_ID'):

    def cf(c_th=0.6,n_detect=1,filter=None):

        # Apply filter to the data table.
        temp_data_table = filter_table(data_table,filter)

        # Extract z/isqso values.
        z = copy.deepcopy(temp_data_table['Z_{}'.format(qn_name)].data)
        isqso = ((temp_data_table['C_{}'.format(qn_name)]>c_th).sum(axis=1)>=n_detect)

        return isqso, z

    return cf

def get_cf_sq(data_table,sq_name='SQ',specid_name='SPEC_ID'):

    def cf(p_min=0.32,filter=None):

        # Apply filter to the data table.
        temp_data_table = filter_table(data_table,filter)

        # Extract z/isqso values.
        z = copy.deepcopy(temp_data_table['Z_{}'.format(sq_name)].data)
        isqso = (temp_data_table['P_{}'.format(sq_name)]>p_min)

        return isqso, z

    return cf

def get_cf_rr(data_table,rr_name='RR',specid_name='SPEC_ID'):

    def cf(zwarn=None,filter=None):

        # Apply filter to the data table.
        temp_data_table = filter_table(data_table,filter)

        # Extract z/isqso values.
        z = copy.deepcopy(data_table['Z_{}'.format(rr_name)].data)
        isqso = copy.deepcopy(data_table['ISQSO_{}'.format(rr_name)].data)

        # If zwarn==True, set all spectra with zwarn>0 to have isqso=True.
        # If zwarn==False, set all spectra with zwarn>0 to have isqso=False.
        # Otherwise, leave alone (i.e. ignore zwarn and use best fit).
        if zwarn is not None:
            zwarn_nonzero = (data_table['ZWARN_{}'.format(rr_name)]>0)
            isqso[zwarn_nonzero] = zwarn

        return isqso, z

    return cf

def get_cf_qnorrr(data_table,qn_name='QN',rr_name='RR',specid_name='SPEC_ID'):

    def cf(qn_kwargs={},rr_kwargs={},zchoice='QN',filter=None):

        # Apply filter to the data table.
        temp_data_table = filter_table(data_table,filter)

        # Get classifications from both QN and RR.
        cf_qn = get_cf_qn(data_table,qn_name=qn_name,specid_name=specid_name)
        cf_rr = get_cf_rr(data_table,rr_name=rr_name,specid_name=specid_name)
        isqso_qn, z_qn = cf_qn(**qn_kwargs,filter=filter)
        isqso_rr, z_rr = cf_rr(**rr_kwargs,filter=filter)

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

def get_cf_qnandrr(data_table,qn_name='QN',rr_name='RR',specid_name='SPEC_ID'):

    def cf(qn_kwargs={},rr_kwargs={},dv_max=6000.,zchoice='QN',filter=None):

        # Apply filter to the data table.
        temp_data_table = filter_table(data_table,filter)

        # Get classifications from both QN and RR.
        cf_qn = get_cf_qn(data_table,qn_name=qn_name,specid_name=specid_name)
        cf_rr = get_cf_rr(data_table,rr_name=rr_name,specid_name=specid_name)
        isqso_qn, z_qn = cf_qn(**qn_kwargs,filter=filter)
        isqso_rr, z_rr = cf_rr(**rr_kwargs,filter=filter)

        # Combine using &, and choosing z based on zchoice.
        dv = get_dv(z_qn,z_rr,data_table['Z_VI'])
        isqso = isqso_qn & isqso_rr & (dv<=dv_max)
        z = z_rr
        if zchoice=='QN':
            z[isqso_qn] = z_qn[isqso_qn]

        return isqso, z

    return cf

def get_cf_qnplusvi(data_table,qn_name='QN',specid_name='SPEC_ID'):

    def cf(c_th_lo=0.1,c_th_hi=0.9,n_detect=1,filter=None):

        # Apply filter to the data table.
        temp_data_table = filter_table(data_table,filter)

        # Get lo and hi classifications from QN.
        cf_qn = get_cf_qn(data_table,qn_name=qn_name,specid_name=specid_name)
        isqso_qn_lo, z_qn_lo = cf_qn(c_th=c_th_lo,n_detect=n_detect,filter=filter)
        isqso_qn_hi, z_qn_hi = cf_qn(c_th=c_th_hi,n_detect=n_detect,filter=filter)

        # Select to use VI when spectra have middling confidence (i.e. when the
        # lo threshold would give True, but hi would give False).
        use_vi = isqso_qn_lo & (~isqso_qn_hi)
        print('INFO: QN+VI sends {}/{} ({:2.2%}) spectra to VI'.format(use_vi.sum(),len(data_table),use_vi.sum()/len(data_table)))

        # Construct outputs.
        isqso = isqso_qn_hi
        isqso[use_vi] = copy.deepcopy(data_table['ISQSO_VI'].data[use_vi])
        z = z_qn_hi
        z[use_vi] = copy.deepcopy(data_table['Z_VI'].data[use_vi])

        return isqso, z

    return cf

def get_cf_rrplusvi(data_table,rr_name='RR',specid_name='SPEC_ID'):

    def cf(filter=None):

        # Apply filter to the data table.
        temp_data_table = filter_table(data_table,filter)

        # Get classifications from RR.
        cf_rr = get_cf_rr(data_table,rr_name=rr_name,specid_name=specid_name)
        isqso_rr, z_rr = cf_rr(zwarn=False,filter=filter)

        # Select to use VI when a zwarn flag is raised.
        use_vi = (data_table['ZWARN_{}'.format(rr_name)]>0)
        print('INFO: RR+VI sends {}/{} ({:2.1%}) spectra to VI'.format(use_vi.sum(),len(data_table),use_vi.sum()/len(data_table)))

        # Construct outputs.
        isqso = isqso_rr
        isqso[use_vi] = copy.deepcopy(data_table['ISQSO_VI'].data[use_vi])
        z = z_rr
        z[use_vi] = copy.deepcopy(data_table['Z_VI'].data[use_vi])

        return isqso, z

    return cf

def get_cf_qnandrrplusvi(data_table,qn_name='QN',rr_name='RR',specid_name='SPEC_ID'):

    def cf(qn_kwargs={},rr_kwargs={},dv_max=6000.,zchoice='QN',filter=None):

        # Apply filter to the data table.
        temp_data_table = filter_table(data_table,filter)

        # Get classifications from both QN and RR.
        cf_qn = get_cf_qn(data_table,qn_name=qn_name,specid_name=specid_name)
        cf_rr = get_cf_rr(data_table,rr_name=rr_name,specid_name=specid_name)
        isqso_qn, z_qn = cf_qn(**qn_kwargs,filter=filter)
        isqso_rr, z_rr = cf_rr(**rr_kwargs,filter=filter)

        # Select to use VI when RR and QN disagree.
        dv = get_dv(z_qn,z_rr,data_table['Z_VI'])
        use_vi = ((isqso_qn|isqso_rr) & (~(isqso_qn&isqso_rr))) | ((isqso_qn & isqso_rr) & (dv>dv_max))
        print('INFO: QN&RR+VI sends {}/{} ({:2.1%}) spectra to VI'.format(use_vi.sum(),len(data_table),use_vi.sum()/len(data_table)))

        # Construct outputs.
        isqso = isqso_qn & isqso_rr & (dv<=dv_max)
        isqso[use_vi] = copy.deepcopy(data_table['ISQSO_VI'].data[use_vi])
        z = z_rr
        if zchoice=='QN':
            z[isqso_qn] = z_qn[isqso_qn]
        z[use_vi] = copy.deepcopy(data_table['Z_VI'].data[use_vi])

        return isqso, z

    return cf

def get_cf_qnandrrplusviadv(data_table,qn_name='QN',rr_name='RR',specid_name='SPEC_ID'):

    def cf(c_th_lo=0.1,c_th_hi=0.9,n_detect=1,dv_max=6000.,zchoice='QN',filter=None):

        # Apply filter to the data table.
        temp_data_table = filter_table(data_table,filter)

        # Get classifications from both QN and RR.
        cf_qn = get_cf_qn(data_table,qn_name=qn_name,specid_name=specid_name)
        cf_rr = get_cf_rr(data_table,rr_name=rr_name,specid_name=specid_name)
        isqso_qn_lo, z_qn_lo = cf_qn(c_th=c_th_lo,n_detect=n_detect,filter=filter)
        isqso_qn_hi, z_qn_hi = cf_qn(c_th=c_th_hi,n_detect=n_detect,filter=filter)
        isqso_rr_zwf, z_rr_zwf = cf_rr(zwarn=False,filter=filter)
        zwarn_nonzero = (data_table['ZWARN_{}'.format(rr_name)]>0)

        # Asks for VI if:
        # 1. RR says QSO without zwarn, and QN says cth_lo<c<cth_hi
        # 2. QN says c>cth_hi but RR gives zwarn.
        use_vi = ((isqso_qn_lo & (~isqso_qn_hi)) & (isqso_rr_zwf)) | (isqso_qn_hi & zwarn_nonzero)
        print('INFO: RR&QN+VI adv. sends {}/{} ({:2.1%}) spectra to VI'.format(use_vi.sum(),len(data_table),use_vi.sum()/len(data_table)))

        # Construct outputs.
        isqso = isqso_qn_hi & isqso_rr_zwf
        isqso[use_vi] = copy.deepcopy(data_table['ISQSO_VI'].data[use_vi])
        z = z_rr_zwf
        if zchoice=='QN':
            z[isqso_qn_hi] = z_qn_hi[isqso_qn_hi]
        z[use_vi] = copy.deepcopy(data_table['Z_VI'].data[use_vi])

        return isqso, z

    return cf

## Obsolete for now.
def check_specid_array(specid):

    if isinstance(specid,float) or isinstance(specid,int):
        specid = np.array([specid])
    if isinstance(specid,list):
        specid = np.array(specid)

    return specid
