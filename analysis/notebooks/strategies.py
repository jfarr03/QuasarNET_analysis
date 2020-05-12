## Define the strategies
stratdef = {}


## Single classifiers.
# QN definitions.
cth = 0.3
w_qn = data_table['CMAX_QN']>cth
z_qn = copy.deepcopy(data_table['Z_QN'])
stratdef['QN'] = {'w': w_qn, 'z': z_qn}

# RR definitions.
zwarn = data_table['ZWARN_RR']>0
z_rr = copy.deepcopy(data_table['Z_RR'])
w_rr = copy.deepcopy(data_table['ISQSO_RR'])
w_rr[zwarn] = False
stratdef['RR'] = {'w': w_rr, 'z': z_rr}

# RR incl. zwarn definitions.
z_rr = copy.deepcopy(data_table['Z_RR'])
w_rr = copy.deepcopy(data_table['ISQSO_RR'])
stratdef['RRzwarn'] = {'w': w_rr, 'z': z_rr}

# PIPE definitions.
w_pipe = copy.deepcopy(data_table['ISQSO_PIPE'])
z_pipe = copy.deepcopy(data_table['Z_PIPE'])
stratdef['PIPE'] = {'w': w_pipe, 'z': z_pipe}

# SQ definitions.
pmin = 0.32
w_sq = data_table['CMAX_QN']>pmin
z_sq = copy.deepcopy(data_table['Z_SQ'])
stratdef['SQ'] = {'w': w_sq, 'z': z_sq}

# QN 2line definitions.
cth = 0.3
cth2 = 0.3
w_qn2line = (data_table['CMAX_QN']>cth) & (data_table['CMAX2_QN']>cth2)
z_qn2line = copy.deepcopy(data_table['Z_QN'])
stratdef['QN2line'] = {'w': w_qn2line, 'z': z_qn2line}

# RRnew definitions.
zwarn = data_table['ZWARN_RRnew']>0
z_rr = copy.deepcopy(data_table['Z_RRnew'])
w_rr = copy.deepcopy(data_table['ISQSO_RRnew'])
w_rr[zwarn] = False
stratdef['RRnew'] = {'w': w_rr, 'z': z_rr}


## Simple & | combinations of classifiers.
# QN|RR definitions.
w_or = (stratdef['QN']['w'] | stratdef['RR']['w'])
z_or = copy.deepcopy(data_table['Z_RR'])
w_qn_notrr = w_qn&(~w_rr)
z_or[w_qn_notrr] = copy.deepcopy(data_table['Z_QN'])[w_qn_notrr]
stratdef['QN|RR'] = {'w': w_or, 'z': z_or}

# QN&RR definitions
dv_qnrr = (300000.*abs(data_table['Z_QN'] - data_table['Z_RR'])/(1 + data_table['Z_VI']))
w_and = ((stratdef['QN']['w'] & stratdef['RR']['w']) & (dv_qnrr<dv_max))
z_and = copy.deepcopy(data_table['Z_RR'])
stratdef['QN&RR'] = {'w': w_and, 'z': z_and}

# QN|PIPE definitions.
w_or = (stratdef['RR']['w'] | stratdef['PIPE']['w'])
z_or = copy.deepcopy(data_table['Z_PIPE'])
w_qn_notpipe = isqso_qn&(~data_table['ISQSO_PIPE'])
z_or[w_qn_notpipe] = copy.deepcopy(data_table['Z_QN'])[w_qn_notpipe]
stratdef['QN|PIPE'] = {'w': w_or, 'z': z_or}

# QN&PIPE definitions
dv_qnpipe = (300000.*abs(data_table['Z_QN'] - data_table['Z_PIPE'])/(1 + data_table['Z_VI']))
w_and = ((stratdef['RR']['w'] & stratdef['PIPE']['w']) & (dv_qnpipe<dv_max))
z_and = copy.deepcopy(data_table['Z_PIPE'])
stratdef['QN&PIPE'] = {'w': w_and, 'z': z_and}

# QN|RR zQN definitions.
w_or = (stratdef['QN']['w'] | stratdef['RR']['w'])
z_or = data_table['Z_QN']
rr_notqn = w_rr&(~w_qn)
z_or[rr_notqn] = copy.deepcopy(data_table['Z_RR'])[rr_notqn]
stratdef['| zQN'] = {'w': w_or,'z': z_or}

# QN&RR zQN definitions.
dv_qnrr = (300000.*abs(data_table['Z_QN'] - data_table['Z_RR'])/(1 + data_table['Z_VI']))
w_and = ((stratdef['QN']['w'] & stratdef['RR']['w']) & (dv_qnrr<dv_max))
z_and = copy.deepcopy(data_table['Z_QN'])
stratdef['& zQN'] = {'w': w_and,'z': z_and}


## Strategies involving VI.
# RR+VI definitions
zwarn = data_table['ZWARN_RR']>0
vi = np.zeros(len(data_table),dtype=bool)
for b in [1,2,3,4,5,6,10]:
    vi |= (data_table['ZWARN_RR']&(2**b))>0
w_rrplusvi = np.zeros(len(data_table),dtype=bool)
w_rrplusvi[~vi] = copy.deepcopy(data_table['ISQSO_RR'][~vi])
w_rrplusvi[vi] = copy.deepcopy(data_table['ISQSO_VI'][vi])
z_rrplusvi = np.zeros(len(data_table))
z_rrplusvi[~vi] = copy.deepcopy(data_table['Z_RR'][~vi])
z_rrplusvi[vi] = copy.deepcopy(data_table['Z_VI'][zwarn])
print('RR+VI asks for {} VIs'.format(vi.sum()))
stratdef['RR+VI'] = {'w': w_rrplusvi, 'z': z_rrplusvi,}

# QN+VI definitions.
cth_defhi = 0.99
cth_deflo = 0.01
qn_def = (data_table['CMAX_QN']>cth_defhi) | (data_table['CMAX_QN']<cth_deflo)
vi = ~qn_def
w_qnplusvi = np.zeros(len(data_table),dtype=bool)
w_qnplusvi[qn_def] = copy.deepcopy(data_table['ISQSO_QN'][qn_def])
w_qnplusvi[vi] = copy.deepcopy(data_table['ISQSO_VI'][vi])
z_qnplusvi = np.zeros(len(data_table))
z_qnplusvi[qn_def] = copy.deepcopy(data_table['Z_QN'][qn_def])
z_qnplusvi[vi] = copy.deepcopy(data_table['Z_VI'][vi])
print('QN+VI asks for {} VIs'.format(vi.sum()))
stratdef['QN+VI'] = {'w': w_qnplusvi, 'z': z_qnplusvi}

# QN&RR+VI definitions.
cth = 0.3
isqso_qn = data_table['CMAX_QN']>cth
isqso_rr = data_table['ISQSO_RR']
dv_rr_qn = (300000.*abs(data_table['Z_QN'] - data_table['Z_RR'])/(1 + data_table['Z_VI']))
vi = ((isqso_qn|isqso_rr) & (~(isqso_qn&isqso_rr))) | ((isqso_qn & isqso_rr) & (dv_rr_qn>=dv_max))
w_rrplusqnplusvi = np.zeros(len(data_table),dtype=bool)
w_rrplusqnplusvi[isqso_qn&isqso_rr] = np.ones(len(data_table),dtype=bool)[isqso_qn&isqso_rr]
w_rrplusqnplusvi[vi] = copy.deepcopy(data_table['ISQSO_VI'])[vi]
z_rrplusqnplusvi = copy.deepcopy(data_table['Z_RR'])
z_rrplusqnplusvi[isqso_qn] = copy.deepcopy(data_table['Z_QN'])[isqso_qn]
z_rrplusqnplusvi[vi] = copy.deepcopy(data_table['Z_VI'])[vi]
print('QN&RR+VI asks for {} VIs'.format(vi.sum()))
stratdef['QN&RR+VI'] = {'w': w_rrplusqnplusvi, 'z': z_rrplusqnplusvi}

# QN+RR+VI combination 2
zwarn = data_table['ZWARN_RR']>0
isqso_rr = data_table['ISQSO_RR']
cth_defhi = 0.99
cth_deflo = 0.01
qn_ct = (data_table['CMAX_QN']>cth_defhi)
qn_cf = (data_table['CMAX_QN']<cth_deflo)
qn_us = (~qn_ct) & (~qn_cf)
vi = (isqso_rr&(~zwarn)&qn_cf) | (zwarn&qn_us) | (qn_ct&(~isqso_rr)&(~zwarn))
w_rrplusqnplusvi = np.zeros(len(data_table),dtype=bool)
w_rrplusqnplusvi[qn_ct&(~((~isqso_rr)&(~zwarn)))] = np.ones(len(data_table),dtype=bool)[qn_ct&(~((~isqso_rr)&(~zwarn)))]
w_rrplusqnplusvi[qn_us&isqso_rr&(~zwarn)] = np.ones(len(data_table),dtype=bool)[qn_us&isqso_rr&(~zwarn)]
w_rrplusqnplusvi[vi] = copy.deepcopy(data_table['ISQSO_VI'])[vi]
z_rrplusqnplusvi = copy.deepcopy(data_table['Z_RR'])
z_rrplusqnplusvi[qn_ct&(~((~isqso_rr)&(~zwarn)))] = copy.deepcopy(data_table['Z_QN'][qn_ct&(~((~isqso_rr)&(~zwarn)))])
z_rrplusqnplusvi[vi] = copy.deepcopy(data_table['Z_VI'])[vi]
print('QN+RR+VI asks for {} VIs'.format(vi.sum()))
stratdef['QN+RR+VI'] = {'w': w_rrplusqnplusvi, 'z': z_rrplusqnplusvi}
