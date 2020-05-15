import numpy as np
import copy

class Strategy():
    def __init__(self,classifying_fn,name=None,c=None,ls=None,marker=None):

        self.classifying_fn = classifying_fn
        self.name = name
        self.c = c
        self.ls = ls
        self.marker = marker

        return

    def predict(self,specid,**kwargs):

        isqso, z = self.classifying_fn(specid,**kwargs)

        return isqso, z

def check_specid_array(specid):

    if isinstance(specid,float) or isinstance(specid,int):
        specid = np.array([specid])
    if isinstance(specid,list):
        specid = np.array(specid)

    return specid

def get_dv(z1,z2,ztrue):

    dv = (300000.*abs(z1-z2)/(1+ztrue))

    return dv

def get_cf_qn(data_table,qn_name='QN',specid_name='SPEC_ID'):

    def cf(specid,c_th=0.6,n_detect=1):

        # Set up arrays.
        specid = check_specid_array(specid)
        isqso = np.zeros(len(specid)).astype(bool)
        z = np.zeros(len(specid))

        # For each spectrum:
        for i,s in enumerate(specid):

            # Locate in the table, and extract z/isqso values.
            w = np.in1d(data_table[specid_name],s)
            z[i] = copy.deepcopy(data_table['Z_{}'.format(qn_name)])[w][0]
            isqso[i] = ((data_table['C_{}'.format(qn_name)]>c_th).sum(axis=1)>=n_detect)[w][0]

        return isqso, z

    return cf

def get_cf_sq(data_table,sq_name='SQ',specid_name='SPEC_ID'):

    def cf(specid,p_min=0.32):

        # Set up arrays.
        specid = check_specid_array(specid)
        isqso = np.zeros(len(specid)).astype(bool)
        z = np.zeros(len(specid))

        # For each spectrum:
        for i,s in enumerate(specid):

            # Locate in the table, and extract z/isqso values.
            w = np.in1d(data_table[specid_name],s)
            z[i] = copy.deepcopy(data_table['Z_{}'.format(sq_name)])[w][0]
            isqso[i] = (data_table['P_{}'.format(sq_name)]>p_min)[w][0]

        return isqso, z

    return cf

def get_cf_rr(data_table,rr_name='RR',specid_name='SPEC_ID'):

    def cf(specid,zwarn=None):

        # Set up arrays.
        specid = check_specid_array(specid)
        isqso = np.zeros(len(specid)).astype(bool)
        z = np.zeros(len(specid))

        # For each spectrum:
        for i,s in enumerate(specid):

            # Locate in the table, and extract z/isqso values.
            w = np.in1d(data_table[specid_name],s)
            z[i] = copy.deepcopy(data_table['Z_{}'.format(rr_name)])[w][0]
            isqso[i] = copy.deepcopy(data_table['ISQSO_{}'.format(rr_name)])[w][0]

            # If zwarn==True, set all spectra with zwarn>0 to have isqso=True.
            # If zwarn==False, set all spectra with zwarn>0 to have isqso=False.
            # Otherwise, leave alone (i.e. ignore zwarn and use best fit).
            if zwarn is not None:
                if data_table['ZWARN_{}'.format(rr_name)][w][0]>0:
                    isqso[i] = zwarn

        return isqso, z

    return cf

def get_cf_qnorrr(data_table,qn_name='QN',rr_name='RR',specid_name='SPEC_ID'):

    def cf(specid,qn_kwargs={},rr_kwargs={},zchoice='QN'):

        # Set up arrays.
        specid = check_specid_array(specid)

        # Get classifications from both QN and RR.
        cf_qn = get_cf_qn(data_table,qn_name=qn_name,specid_name=specid_name)
        cf_rr = get_cf_rr(data_table,rr_name=rr_name,specid_name=specid_name)
        isqso_qn, z_qn = cf_qn(specid,**qn_kwargs)
        isqso_rr, z_rr = cf_rr(specid,**rr_kwargs)

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

    def cf(specid,qn_kwargs={},rr_kwargs={},dv_max=None,zchoice='QN'):

        # Set up arrays.
        specid = check_specid_array(specid)

        # Get classifications from both QN and RR.
        cf_qn = get_cf_qn(data_table,qn_name=qn_name,specid_name=specid_name)
        cf_rr = get_cf_rr(data_table,rr_name=rr_name,specid_name=specid_name)
        isqso_qn, z_qn = cf_qn(specid,**qn_kwargs)
        isqso_rr, z_rr = cf_rr(specid,**rr_kwargs)

        # Combine using &, and choosing z based on zchoice.
        isqso = isqso_qn & isqso_rr
        if dv_max is not None:
            dv = get_dv(z_qn,z_rr,data_table['Z_VI'])
            isqso &= (dv_rrqn<=dv_max)
        z = z_rr
        if zchoice=='QN':
            z[isqso_qn] = z_qn[isqso_qn]

        return isqso, z

    return cf

def get_cf_qnplusvi(data_table,qn_name='QN',specid_name='SPEC_ID'):

    def cf(specid,c_th_lo=0.1,c_th_hi=0.9,n_detect=1):

        # Set up arrays.
        specid = check_specid_array(specid)

        # Get lo and hi classifications from QN.
        cf_qn = get_cf_qn(data_table,qn_name=qn_name,specid_name=specid_name)
        isqso_qn_lo, z_qn_lo = cf_qn(specid,c_th=c_th_lo,n_detect=n_detect)
        isqso_qn_hi, z_qn_hi = cf_qn(specid,c_th=c_th_hi,n_detect=n_detect)

        # Select to use VI when spectra have middling confidence (i.e. when the
        # lo threshold would give True, but hi would give False).
        use_vi = isqso_qn_lo & (~isqso_qn_hi)
        print('INFO: QN+VI sends {}/{} ({:2.1%}) spectra to VI'.format(use_vi.sum(),len(data_table),use_vi.sum()/len(data_table)))

        # Construct outputs.
        isqso = isqso_qn_hi
        isqso[use_vi] = copy.deepcopy(data_table['ISQSO_VI'])[use_vi]
        z = z_qn_hi
        z[use_vi] = copy.deepcopy(data_table['Z_VI'])[use_vi]

        return isqso, z

    return cf

def get_cf_rrplusvi(data_table,rr_name='RR',specid_name='SPEC_ID'):

    def cf(specid):

        # Set up arrays.
        specid = check_specid_array(specid)

        # Get classifications from RR.
        cf_rr = get_cf_rr(data_table,rr_name=rr_name,specid_name=specid_name)
        isqso_rr, z_rr = cf_rr(specid,zwarn=False)

        # Select to use VI when a zwarn flag is raised.
        use_vi = (data_table['ZWARN_{}'.format(rr_name)]>0)
        print('INFO: RR+VI sends {}/{} ({:2.1%}) spectra to VI'.format(use_vi.sum(),len(data_table),use_vi.sum()/len(data_table)))

        # Construct outputs.
        isqso = isqso_rr
        isqso[use_vi] = copy.deepcopy(data_table['ISQSO_VI'])[use_vi]
        z = z_rr
        z[use_vi] = copy.deepcopy(data_table['Z_VI'])[use_vi]

        return isqso, z

    return cf

def get_cf_qnandrrplusvi(data_table,qn_name='QN',rr_name='RR',specid_name='SPEC_ID'):

    def cf(specid,qn_kwargs={},rr_kwargs={},dv_max=None,zchoice='QN'):

        # Set up arrays.
        specid = check_specid_array(specid)

        # Get classifications from both QN and RR.
        cf_qn = get_cf_qn(data_table,qn_name=qn_name,specid_name=specid_name)
        cf_rr = get_cf_rr(data_table,rr_name=rr_name,specid_name=specid_name)
        isqso_qn, z_qn = cf_qn(specid,**qn_kwargs)
        isqso_rr, z_rr = cf_rr(specid,**rr_kwargs)

        # Select to use VI when RR and QN disagree.
        dv = get_dv(z_qn,z_rr,data_table['Z_VI'])
        use_vi = ((isqso_qn|isqso_rr) & (~(isqso_qn&isqso_rr))) | ((isqso_qn & isqso_rr) & (dv>dv_max))
        print('INFO: QN&RR+VI sends {}/{} ({:2.1%}) spectra to VI'.format(use_vi.sum(),len(data_table),use_vi.sum()/len(data_table)))

        # Construct outputs.
        isqso = isqso_qn & isqso_rr
        isqso[use_vi] = copy.deepcopy(data_table['ISQSO_VI'])[use_vi]
        z = z_rr
        if zchoice=='QN':
            z[isqso_qn] = z_qn[isqso_qn]
        z[use_vi] = copy.deepcopy(data_table['Z_VI'])[use_vi]

        return isqso, z

    return cf

def get_cf_qnandrrplusviadv(data_table,qn_name='QN',rr_name='RR',specid_name='SPEC_ID'):

    def cf(specid,c_th_lo=0.1,c_th_hi=0.9,n_detect=1,dv_max=None,zchoice='QN'):

        # Set up arrays.
        specid = check_specid_array(specid)

        # Get classifications from both QN and RR.
        cf_qn = get_cf_qn(data_table,qn_name=qn_name,specid_name=specid_name)
        cf_rr = get_cf_rr(data_table,rr_name=rr_name,specid_name=specid_name)
        isqso_qn_lo, z_qn_lo = cf_qn(specid,c_th=c_th_lo,n_detect=n_detect)
        isqso_qn_hi, z_qn_hi = cf_qn(specid,c_th=c_th_hi,n_detect=n_detect)
        isqso_rr, z_rr = cf_rr(specid,zwarn=None)
        isqso_rr_zwf, z_rr_zwf = cf_rr(specid,zwarn=False)

        # Asks for VI if QN says c>cth_hi but RR has a zwarn.
        # Also if RR says QSO without zwarn, and QN says cth_lo<c<cth_hi.
        use_vi = ((isqso_qn_lo & (~isqso_qn_hi)) & (isqso_rr_zwf)) | (isqso_qn_hi & (isqso_rr & (~isqso_rr_zwf)))
        print('INFO: RR&QN+VI adv. sends {}/{} ({:2.1%}) spectra to VI'.format(use_vi.sum(),len(data_table),use_vi.sum()/len(data_table)))

        # Construct outputs.
        isqso = isqso_qn_hi | isqso_rr_zwf
        isqso[use_vi] = copy.deepcopy(data_table['ISQSO_VI'])[use_vi]
        z = z_rr
        if zchoice=='QN':
            z[isqso_qn] = z_qn[isqso_qn]
        z[use_vi] = copy.deepcopy(data_table['Z_VI'])[use_vi]

        return isqso, z

    return cf
