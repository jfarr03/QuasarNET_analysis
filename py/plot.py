import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.table as mp_table

from qn_analysis import utils

def get_pur_com(isqso_s,z_s,isqso_truth,isgal_truth,isbad,z_truth,zbin=None,dv_max=6000.):

    # Determine which entries are in the z bin.
    in_zbin_zvi = np.ones(z_truth.shape).astype(bool)
    in_zbin_zs = np.ones(z_s.shape).astype(bool)
    if zbin is not None:
        if zbin[0] is not None:
            in_zbin_zvi &= (z_truth>=zbin[0])
            in_zbin_zs &= (z_s>=zbin[0])
        if zbin[1] is not None:
            in_zbin_zvi &= (z_truth<zbin[1])
            in_zbin_zs &= (z_s<zbin[1])

    # See which classifier redshifts are "good".
    zgood = (z_truth>-1.) & (abs(z_s-z_truth) < dv_max*(1+z_truth)/300000.)

    # Calculate the two parts of purity.
    pur_num = (isqso_s & (isqso_truth | isgal_truth) & zgood & (~isbad) & in_zbin_zs).sum()
    pur_denom = (isqso_s & (~isbad) & in_zbin_zs).sum()

    # Calculate the two parts of completeness.
    com_num = (isqso_s & zgood & isqso_truth & in_zbin_zvi).sum()
    com_denom = (isqso_truth & in_zbin_zvi).sum()

    # Add to purity/completeness lists.
    pur = pur_num/pur_denom
    com = com_num/com_denom

    return pur, com

def get_truths(data_table):

    isqso_truth = (data_table['ISQSO_VI'] & (data_table['ZCONF_PERSON']==2))
    isgal_truth = ((data_table['CLASS_VI']=='GALAXY') & (data_table['ZCONF_PERSON']==2))
    isstar_truth = ((data_table['CLASS_VI']=='STAR') & (data_table['ZCONF_PERSON']==2))
    isbad = ((data_table['ZCONF_PERSON']=='BAD') | (data_table['ZCONF_PERSON']!=2))

    return isqso_truth, isgal_truth, isstar_truth, isbad

def get_label_from_zbin(zbin):

    if (zbin[0] is not None) and (zbin[1] is not None):
        zbin_label = r'${} \leq z < {}$'.format(zbin[0],zbin[1])
    elif zbin[0] is None:
        zbin_label = r'$z < {}$'.format(zbin[1])
    elif zbin[1] is None:
        zbin_label = r'$z \geq {}$'.format(zbin[0])
    else:
        zbin_label = None

    return zbin_label

## Function for Figure 1.
def plot_pur_com_vs_z(data_table,strategies,filename=None,zmin=0.,zmax=5.,dz_int=21,dv_max=6000.,nydec=0,figsize=(12,6),ymin=0.93,ymax=1.005):

    # Make figure.
    fig, axs = plt.subplots(1,2,figsize=figsize,sharey=True,squeeze=False)

    # Construct z bins.
    dz_edges = np.linspace(zmin,zmax,dz_int)
    dz_mids = (dz_edges[1:] + dz_edges[:-1])/2.
    dz_widths = (dz_edges[1:] - dz_edges[:-1])

    # Determine truth vectors.
    isqso_truth, isgal_truth, isstar_truth, isbad = get_truths(data_table)

    # For each strategy:
    for j,s in enumerate(strategies.keys()):
        com = []
        pur = []

        # Extract data from strategy.
        isqso_s = strategies[s]['w']
        z_s = strategies[s]['z']

        # For each z bin:
        for i in range(len(dz_edges)-1):

            # Calculate purity and completeness.
            p,c = get_pur_com(isqso_s,z_s,isqso_truth,isgal_truth,isbad,
                data_table['Z_VI'],zbin=(dz_edges[i],dz_edges[i+1]),dv_max=dv_max)

            # Add to purity/completeness lists.
            pur += [p]
            com += [c]

        # Plot the results.
        axs[0,0].step(dz_edges,[pur[0]]+pur,where='pre',label=s,color=strategies[s]['c'],ls=strategies[s]['ls'])
        axs[0,1].step(dz_edges,[com[0]]+com,where='pre',label=s,color=strategies[s]['c'],ls=strategies[s]['ls'])

    # Format the axes.
    for ax in axs.flatten():
        ax.set_xlim(zmin,zmax)
        ax.set_ylim(ymin,ymax)
        ax.axhline(y=1.0,c='lightgrey',zorder=-1)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0,decimals=nydec))

    # Add x labels.
    axs[0,0].set_xlabel(r'$z_\mathrm{classifier}$')
    axs[0,1].set_xlabel(r'$z_\mathrm{VI}$')

    # Add titles.
    axs[0,0].text(0.5,1.05,'Purity',ha='center',va='center',
                  transform=axs[0,0].transAxes,rotation=0)
    axs[0,1].text(0.5,1.05,'Completeness',ha='center',va='center',
                  transform=axs[0,1].transAxes,rotation=0)

    # Add a legend.
    artists = []
    labels = []
    for j,s in enumerate(strategies.keys()):
        artists += [axs[0,0].step([0],[0],where='pre',color=strategies[s]['c'],ls=strategies[s]['ls'])[0]]
        labels += [strategies[s]['n']]
    fig.legend(artists,labels,loc='lower center',borderaxespad=0,bbox_to_anchor=(0.5,0.03),ncol=len(artists))
    rect = (0.0,0.1,1.0,1.0)
    plt.tight_layout(rect=rect)

    # Save.
    if filename is not None:
        plt.savefig(filename)

    return fig, axs

## Function for alternative Figure 1.
def plot_pur_com_vs_cth_zbin(data_table,strategies,filename=None,zbins=[(None,2.1),(2.1,None)],dv_max=6000.,nydec=0,figsize=(12,12),ymin=0.9,ymax=1.):

    fig, axs = plt.subplots(2,2,figsize=figsize,sharex=True,sharey=True,squeeze=False)

    cth_min = 0.0
    cth_max = 1.0
    n_int = 100
    c_th = np.arange(cth_min,cth_max,(1/n_int)*(cth_max-cth_min))

    isqso_truth, isgal_truth, isstar_truth, isbad = get_truths(data_table)

    for j,s in enumerate(strategies.keys()):

        for i,zbin in enumerate(zbins):

            z_s = strategies[s]['z']
            in_zbin_zvi = np.ones(data_table['Z_VI'].shape).astype(bool)
            in_zbin_zs = np.ones(z_s.shape).astype(bool)
            if zbin[0] is not None:
                in_zbin_zvi &= (data_table['Z_VI']>=zbin[0])
                in_zbin_zs &= (z_s>=zbin[0])
            if zbin[1] is not None:
                in_zbin_zvi &= (data_table['Z_VI']<zbin[1])
                in_zbin_zs &= (z_s<zbin[1])

            com = []
            pur = []

            for cth in c_th:

                # Try to use confidences, otherwise use weights.
                try:
                    isqso_s = strategies[s]['confs']>cth
                except KeyError:
                    isqso_s = strategies[s]['w']

                # Calculate purity and completeness.
                p,c = get_pur_com(isqso_s,z_s,isqso_truth,isgal_truth,isbad,
                    data_table['Z_VI'],zbin=zbin,dv_max=dv_max)

                # Add to purity/completeness lists.
                pur += [p]
                com += [c]

            axs[i,0].plot(c_th,pur,color=utils.colours['C0'],ls=strategies[s]['ls'])
            axs[i,1].plot(c_th,com,color=utils.colours['C1'],ls=strategies[s]['ls'])

    for i,zbin in enumerate(zbins):
        zbin_label = get_label_from_zbin(zbin)

        axs[i,0].text(-0.18,0.5,zbin_label,ha='center',va='center',
                      transform=axs[i,0].transAxes,rotation=90)

    axs[0,0].text(0.5,1.05,'Purity',ha='center',va='center',
                  transform=axs[0,0].transAxes,rotation=0)
    axs[0,1].text(0.5,1.05,'Completeness',ha='center',va='center',
                  transform=axs[0,1].transAxes,rotation=0)

    for ax in axs[0,:]:
        ax.set_xlim(0.,1.)
        ax.set_ylim(ymin,ymax)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0,decimals=nydec))
    for ax in axs[1,:]:
        ax.set_xlabel(r'confidence threshold')
        ax.set_xlim(0.,1.)
        ax.set_ylim(ymin,ymax)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0,decimals=nydec))

    artists = []
    labels = []
    for j,s in enumerate(strategies.keys()):
        artists += [axs[0,0].step([0],[0],where='pre',color='grey',ls=strategies[s]['ls'])[0]]
        labels += [strategies[s]['n']]

    fig.legend(artists,labels,loc='lower center',borderaxespad=0,bbox_to_anchor=(0.5,0.03),ncol=len(artists))
    rect = (0,0.15,1.,1.)
    plt.tight_layout(rect=rect)

    # Save.
    if filename is not None:
        plt.savefig(filename)

    return fig, axs

## Function for Figure 2 (3 panel).
def plot_qn_model_compare_3panel(data_table,strategies,filename=None,dv_max=6000.,nydec=1,figsize=(18,6),ymin=0.97,ymax=1.):

    fig, axs = plt.subplots(1,3,figsize=figsize,squeeze=False)

    cth_min = 0.0
    cth_max = 1.0
    n_int = 101
    c_th = np.arange(cth_min,cth_max,(1/n_int)*(cth_max-cth_min))

    n_dv = 51
    dv_bins = np.linspace(-3000.,3000,n_dv)

    isqso_truth, isgal_truth, isstar_truth, isbad = get_truths(data_table)

    for j,s in enumerate(strategies.keys()):

        com = []
        pur = []

        z_s = data_table['Z_{}'.format(s)]

        for cth in c_th:

            # Try to use confidences, otherwise raise error.
            try:
                isqso_s = strategies[s]['confs']>cth
            except KeyError:
                raise KeyError('Confidences not found for strategy {}'.format(s))

            # Calculate purity and completeness.
            p,c = get_pur_com(isqso_s,z_s,isqso_truth,isgal_truth,isbad,
                data_table['Z_VI'],dv_max=dv_max)

            # Add to purity/completeness lists.
            pur += [p]
            com += [c]

        axs[0,0].plot(c_th,pur,label='pur',color=utils.colours['C0'],ls=strategies[s]['ls'])
        axs[0,1].plot(c_th,com,label='com',color=utils.colours['C1'],ls=strategies[s]['ls'])

        ## Plot the dv histogram.
        dv = 300000. * (z_s-data_table['Z_VI']) / (1+data_table['Z_VI'])
        axs[0,2].hist(dv,bins=dv_bins,histtype='step',ls=strategies[s]['ls'],color=utils.colours['C2'])

    for ax in axs[0,:2]:
        ax.set_xlabel(r'$c_{th}$')
        ax.set_xlim(0.,1.)
        ax.set_ylim(ymin,ymax)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0,decimals=nydec))

    axs[0,0].set_ylabel('Purity')
    axs[0,1].set_ylabel('Completeness')
    axs[0,2].axvline(x=0,c='lightgrey',zorder=-1)
    axs[0,2].set_ylabel(r'#')
    axs[0,2].set_xlabel(r'd$v$ [km/s]')
    axs[0,2].set_xlim(-3000.,3000.)

    artists = []
    labels = []
    for j,s in enumerate(strategies.keys()):
        artists += [axs[0,0].plot([0],[0],color='grey',ls=strategies[s]['ls'])[0]]
        labels += [strategies[s]['n']]

    fig.legend(artists,labels,loc='lower center',borderaxespad=0,bbox_to_anchor=(0.5,0.03),ncol=len(artists))
    rect = (0,0.12,1,1.0)
    plt.tight_layout(rect=rect)

    if filename is not None:
        plt.savefig(filename)

    return fig, axs

## Function for Figure 2 (2 panel).
def plot_qn_model_compare_2panel(data_table,strategies,filename=None,dv_max=6000.,nydec=2,figsize=(12,12),ymin=0.98,ymax=1.,verbose=False):

    fig, axs = plt.subplots(2,1,figsize=figsize,squeeze=False)

    cth_min = 0.0
    cth_max = 1.0
    n_int = 100
    c_th = np.arange(cth_min,cth_max,(1/n_int)*(cth_max-cth_min))
    ndetect = 1

    n_dv = 51
    dv_bins = np.linspace(-3000.,3000,n_dv)

    isqso_truth, isgal_truth, isstar_truth, isbad = get_truths(data_table)

    artists = []
    labels = []

    for j,s in enumerate(strategies.keys()):

        com = []
        pur = []

        z_s = data_table['Z_{}'.format(s)]

        for cth in c_th:

            # Try to use confidences, otherwise raise error.
            try:
                isqso_s = strategies[s]['confs']>cth
            except KeyError:
                raise KeyError('Confidences not found for strategy {}'.format(s))

            # Calculate purity and completeness.
            p,c = get_pur_com(isqso_s,z_s,isqso_truth,isgal_truth,isbad,
                data_table['Z_VI'],dv_max=dv_max)

            # Add to purity/completeness lists.
            pur += [p]
            com += [c]

        pur = np.array(pur)
        com = np.array(com)

        ind = np.where(pur>com)[0][0]
        if verbose:
            print('Strategy {}:'.format(s))
            print('Crossover occurs at:')
            print('cth:',c_th[ind-2:ind+2].round(4))
            print('pur:',pur[ind-2:ind+2].round(4))
            print('com:',com[ind-2:ind+2].round(4))

        if j==0:
            labelp = 'purity'
            labelc = 'completeness'
        else:
            labelp = None
            labelc = None

        axs[0,0].plot(c_th,pur,label=labelp,color=utils.colours['C0'],ls=strategies[s]['ls'])
        axs[0,0].plot(c_th,com,label=labelc,color=utils.colours['C1'],ls=strategies[s]['ls'])

        ## Plot the dv histogram.
        dv = 300000. * (z_s-data_table['Z_VI']) / (1+data_table['Z_VI'])
        axs[1,0].hist(dv,bins=dv_bins,histtype='step',ls=strategies[s]['ls'],color=utils.colours['C2'])

        if verbose:
            dv_med = np.median(dv[abs(dv)<dv_max])
            dv_std = np.std(dv[abs(dv)<dv_max])
            print('{} has median velocity error {:3.3f} and standard deviation {:3.3f}\n'.format(s,dv_med,dv_std))

        ## Add objects to list of artists and labels for legend.
        artists += [axs[0,0].plot([0],[0],color='grey',ls=strategies[s]['ls'])[0]]
        labels += [strategies[s]['n']]

    axs[0,0].set_xlabel(r'confidence threshold')
    axs[0,0].set_xlim(0.,1.)
    axs[0,0].set_ylim(ymin,ymax)
    axs[0,0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0,decimals=nydec))
    axs[0,0].legend()

    axs[1,0].axvline(x=0,c='lightgrey',zorder=-1)
    axs[1,0].set_ylabel(r'#')
    axs[1,0].set_xlabel(r'd$v$ [km/s]')
    axs[1,0].set_xlim(-3000.,3000.)

    fig.legend(artists,labels,loc='lower center',borderaxespad=0,bbox_to_anchor=(0.5,0.03),ncol=len(artists))
    rect = (0,0.13,1.,1.)
    plt.tight_layout(rect=rect)

    if filename is not None:
        plt.savefig(filename)

    return fig, axs

## Function for Figure 3.
def plot_qn_model_data_compare(data_table,strategies,filename=None,dv_max=6000.,nydec=0,figsize=(12,12),ymin=0.90,ymax=1.,verbose=False):

    fig, axs = plt.subplots(2,2,figsize=figsize,sharex=True,sharey=True,squeeze=False)

    cth_min = 0.0
    cth_max = 1.0
    n_int = 100
    c_th = np.arange(cth_min,cth_max,(1/n_int)*(cth_max-cth_min))

    isqso_truth, isgal_truth, isstar_truth, isbad = get_truths(data_table)

    # Make sure that we have the right keys.
    keys = np.array(['QN_cc','QN_sc','QN_cs','QN_ss'])
    for s in strategies.keys():
        assert (s in keys)

    for s in strategies.keys():

        j = np.where(keys==s)[0][0]

        com = []
        pur = []

        z_s = data_table['Z_{}'.format(s)]

        for cth in c_th:

            # Try to use confidences, otherwise raise error.
            try:
                isqso_s = strategies[s]['confs']>cth
            except KeyError:
                raise KeyError('Confidences not found for strategy {}'.format(s))

            # Calculate purity and completeness.
            p,c = get_pur_com(isqso_s,z_s,isqso_truth,isgal_truth,isbad,
                data_table['Z_VI'],dv_max=dv_max)

            # Add to purity/completeness lists.
            pur += [p]
            com += [c]

            if (cth==0.5) and verbose:
                print(s)
                w_contaminants = isqso_s & isstar_truth
                w_zerr = isqso_s & (isqso_truth | isgal_truth) & (~zgood)
                print('number of stars is',w_contaminants.sum())
                print('number of zerr is',w_zerr.sum())
                pur_denom = (isqso_s & (~isbad)).sum()
                pur_num = p*pur_denom
                print('number of classified QSOs is',pur_denom)
                print('number of correctly classified QSOs is',pur_num)
                print('')

        pur = np.array(pur)
        com = np.array(com)

        ind = np.where(pur>com)[0][0]
        if verbose:
            print('cth:',c_th[ind-2:ind+2])
            print('pur:',pur[ind-2:ind+2])
            print('com:',com[ind-2:ind+2])
            print('')

        axs[j//2,j%2].plot(c_th,pur,label='purity',color=utils.colours['C0'])
        axs[j//2,j%2].plot(c_th,com,label='completeness',color=utils.colours['C1'])

    axs[0,0].text(-0.22,0.5,'4 exposure (coadded)\ntesting data',ha='center',va='center',
                  transform=axs[0,0].transAxes,rotation=90)
    axs[1,0].text(-0.22,0.5,'single exposure\ntesting data',ha='center',va='center',
                  transform=axs[1,0].transAxes,rotation=90)
    axs[0,0].text(0.5,1.1,'4 exposure (coadded)\ntraining data',ha='center',va='center',
                  transform=axs[0,0].transAxes,rotation=0)
    axs[0,1].text(0.5,1.1,'single exposure\ntraining data',ha='center',va='center',
                  transform=axs[0,1].transAxes,rotation=0)

    for i,ax in enumerate(axs.flatten()):
        if i//2==1:
            ax.set_xlabel(r'confidence threshold')
        ax.grid()
        ax.set_xlim(0.,1.)
        ax.set_ylim(ymin,ymax)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0,decimals=nydec))

    axs[1,1].legend(loc=4)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.1,hspace=0.1)

    if filename is not None:
        plt.savefig(filename)

    return fig, axs

## Function for Figure 4.
def plot_reobservation_performance(data_table,strategies,filename=None,figsize=(12,6),eff_area=None,dv_max=6000.,zcut=2.1,ymin=0.94,xmin=47.,xmax=52.,verbose=False,n_highz_desi=50,nydec=0,filters=None,marker_size=100,strategies_to_plot=None):

    if filters is None:
        filters = {None: np.ones(len(data_table)).astype(bool)}
        strategies = {None: strategies}

    if isinstance(xmin,float) or isinstance(xmin,int):
        if len(filters)>1:
            print('WARN: using same xmin for all panels. Use a dict of values to specify separately.')
        xmin = {filt_name: xmin for filt_name in filters.keys()}

    if isinstance(xmax,float) or isinstance(xmax,int):
        if len(filters)>1:
            print('WARN: using same xmax for all panels. Use a dict of values to specify separately.')
        xmax = {filt_name: xmax for filt_name in filters.keys()}

    if isinstance(n_highz_desi,float) or isinstance(n_highz_desi,int):
        if len(filters)>1:
            print('WARN: using same n_highz_desi for all panels. Use a dict of values to specify separately.')
        n_highz_desi = {filt_name: n_highz_desi for filt_name in filters.keys()}

    fig, axs = plt.subplots(1,len(filters),figsize=figsize,squeeze=False,sharey=True)

    # determine the true classifications
    isqso_truth, isgal_truth, isstar_truth, isbad = get_truths(data_table)
    highz_truth = data_table['Z_VI']>=zcut

    need_colourbar = False
    points_occupied = []

    for k,filt_name in enumerate(filters.keys()):

        filt_strategies = strategies[filt_name]

        filt = filters[filt_name] & (~isbad)
        nhighz_truth = (isqso_truth & highz_truth & filt).sum()

        n_highz_desi_filt = n_highz_desi[filt_name]

        # If no eff_area is provided, normalise such that there are 50/sqd high-z QSOs.
        if eff_area is None:
            eff_area = nhighz_truth/n_highz_desi_filt

        for s in filt_strategies.keys():

            npoints = len(filt_strategies[s]['w'])
            filt_strategies[s]['nhighz_flagged'] = np.zeros(npoints)
            filt_strategies[s]['nhighz_truth_flagged'] = np.zeros(npoints)

            if npoints>1:

                need_colourbar = True

                for i in range(npoints):
                    w = filt_strategies[s]['w'][i] & filt
                    filt_strategies[s]['nhighz_flagged'][i] = (w).sum()
                    filt_strategies[s]['nhighz_truth_flagged'][i] = (isqso_truth&highz_truth&w).sum()

                    if verbose:
                        print(s)
                        print('true hz qsos:',nhighz_truth)
                        print('obj flagged:',filt_strategies[s]['nhighz_flagged'][0])
                        print('true hz qsos flagged:',filt_strategies[s]['nhighz_truth_flagged'][0])
                        print('stars selected:',(isstar_truth&w).sum())
                        print('gal selected:',(isgal_truth&w).sum())
                        print('lowz qso selected:',(isqso_truth&(~highz_truth)&w).sum())
                        print('')

            else:
                w = filt_strategies[s]['w'][0] & filt
                filt_strategies[s]['nhighz_flagged'][0] = (w).sum()
                filt_strategies[s]['nhighz_truth_flagged'][0] = (isqso_truth&highz_truth&w).sum()

                if (('RR' in s) or ('PIPE' in s)) and verbose:
                    zwarn = data_table['ZWARN_{}'.format(s)]>0
                    print(s)
                    print('true hz qsos:',nhighz_truth)
                    print('obj flagged:',filt_strategies[s]['nhighz_flagged'][0])
                    print('true hz qsos flagged:',filt_strategies[s]['nhighz_truth_flagged'][0])
                    print('obj with zwarn:',(zwarn).sum())
                    print('true hz qsos with zwarn:',(highz_truth&isqso_truth&zwarn).sum())
                    print('true hz qsos with zwarn missed:',(highz_truth&isqso_truth&zwarn&(~filt_strategies[s]['w'][0])).sum())
                    print('stars selected:',(isstar_truth&w).sum())
                    print('gal selected:',(isgal_truth&w).sum())
                    print('lowz qso selected:',(isqso_truth&(~highz_truth)&w).sum())
                    print('')

        if strategies_to_plot is None:

            for s in filt_strategies.keys():

                reobs_dens = filt_strategies[s]['nhighz_flagged']/eff_area
                pli = filt_strategies[s]['nhighz_truth_flagged']/nhighz_truth

                if len(reobs_dens)>1:
                    axs[0,k].plot(reobs_dens,pli,c='grey',marker=filt_strategies[s]['marker'],label=s,zorder=2,ms=np.sqrt(marker_size))
                    points = axs[0,k].scatter(reobs_dens,pli,c=filt_strategies[s]['color'],marker=filt_strategies[s]['marker'],s=marker_size,zorder=3)
                else:
                    while (reobs_dens,pli) in points_occupied:
                        reobs_dens *= 1.004
                    points = axs[0,k].scatter(reobs_dens,pli,c=filt_strategies[s]['color'],marker=filt_strategies[s]['marker'],s=marker_size,label=s,zorder=3)
                    points_occupied += [(reobs_dens,pli)]

        else:

            for s in strategies_to_plot[filt_name].keys():

                nhighz_flagged = []
                nhighz_truth_flagged = []

                for ss in strategies_to_plot[filtname][s]:

                    nhighz_flagged.append(filt_strategies[s]['nhighz_flagged'])
                    nhighz_truth_flagged.append(filt_strategies[s]['nhighz_truth_flagged'])

                nhighz_flagged = np.vstack(nhighz_flagged)
                nhighz_truth_flagged = np.vstack(nhighz_truth_flagged)

                reobs_dens = nhighz_flagged/(eff_area*len(strategies_to_plot[filt_name][s]))
                pli = nhighz_truth_flagged/(nhighz_truth*len(strategies_to_plot[filt_name][s]))

                ## THEN NEED To PLOT

        axs[0,k].axvline(x=n_highz_desi_filt,c='k',zorder=1,ls='--')

        axs[0,k].set_xlim(xmin[filt_name],xmax[filt_name])
        axs[0,k].grid()
        axs[0,k].set_axisbelow(True)

        # Shaded region.
        x = np.linspace(n_highz_desi_filt*0.0,n_highz_desi_filt*2.0,101)
        y = x/n_highz_desi_filt
        y[x>n_highz_desi_filt] = 1.
        axs[0,k].fill_between(x,y,np.ones_like(y)*1.1,edgecolor='darkgrey',facecolor='none',hatch='\\')

    axs[0,0].set_ylabel(r'fraction of high-$z$ QSOs reobserved')
    axs[0,0].set_ylim(ymin,1.0+0.05*(1.-ymin))
    axs[0,0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0,decimals=nydec))

    xlabel = r'number density of fibers allocated to reobservations [sqd$^{-1}$]'
    if len(filters) == 1:
        axs[0,0].legend(loc=4,ncol=1)
        axs[0,0].set_xlabel(xlabel)
    else:
        artists = []
        labels = []
        filt_strategies = strategies[filt_name]
        try:
            markers = {filt_strategies[k]['n']: filt_strategies[k]['marker'] for k in filt_strategies.keys()}
        except KeyError:
            markers = {k.split(' ')[0]: filt_strategies[k]['marker'] for k in filt_strategies.keys()}
        for s in markers.keys():
            ## Add marker shapes for strategy
            artists += [axs[0,0].scatter([],[],color='grey',marker=markers[s],s=marker_size)]
            labels += ['{}'.format(s)]
        fig.legend(artists,labels,loc='lower center',borderaxespad=0,bbox_to_anchor=(0.5,0.03),ncol=len(artists))
        rect = (0,0.15,1.,1.)
        plt.tight_layout(rect=rect)
        fig.text(0.5,0.14,xlabel,ha='center',va='center')
        for k, filt_name in enumerate(filters.keys()):
            axs[0,k].set_title(filt_name)

    if need_colourbar:
        # Colour bar
        fig.subplots_adjust(right=0.87)
        cbar_ax = fig.add_axes([0.89, 0.15, 0.02, 0.7])
        cb = fig.colorbar(points,label=r'$c_{th}$',cax=cbar_ax)
        cb.mappable.set_clim(vmin=0.,vmax=1.)

    plt.savefig(filename)

    return fig, axs

## Function for Figure 5.
def plot_catalogue_performance(data_table,strategies,filename=None,figsize=(12,6),zbins=[(0.9,2.1),(2.1,None)],desi_nqso=[1.3*10**6,0.8*10**6],dv_max=6000.,show_correctwrongzbin=False,verbose=False,nydec=0,ymax=0.1):

    fig, axs = plt.subplots(1,len(zbins),figsize=figsize,sharey=True,squeeze=False)

    # determine the true classifications
    isqso_truth, isgal_truth, isstar_truth, isbad = get_truths(data_table)

    for i,zbin in enumerate(zbins):

        for s in strategies.keys():

            z_s = strategies[s]['z']
            w_s = strategies[s]['w']

            in_zbin_zvi = np.ones(data_table['Z_VI'].shape).astype(bool)
            in_zbin_zs = np.ones(z_s.shape).astype(bool)
            if zbin[0] is not None:
                in_zbin_zvi &= (data_table['Z_VI']>=zbin[0])
                in_zbin_zs &= (z_s>=zbin[0])
            if zbin[1] is not None:
                in_zbin_zvi &= (data_table['Z_VI']<zbin[1])
                in_zbin_zs &= (z_s<zbin[1])

            zgood = (data_table['Z_VI']>-1) & (abs(z_s-data_table['Z_VI']) < dv_max*(1+data_table['Z_VI'])/300000.)

            strategies[s]['ncat'] = (w_s & in_zbin_zs & (~isbad)).sum()

            strategies[s]['nstar'] = (w_s & in_zbin_zs & isstar_truth).sum()
            strategies[s]['ngalwrongz'] = (w_s & ~zgood & in_zbin_zs & isgal_truth).sum()
            strategies[s]['nqsowrongz'] = (w_s & ~zgood & in_zbin_zs & isqso_truth).sum()
            strategies[s]['ncorrectwrongzbin'] = (w_s & zgood & in_zbin_zs & (isqso_truth | isgal_truth) & ~in_zbin_zvi).sum()

            strategies[s]['nwrong'] = (strategies[s]['nstar'] + strategies[s]['ngalwrongz']
                                        + strategies[s]['nqsowrongz'] + strategies[s]['ncorrectwrongzbin'])

            com_num = (w_s & zgood & in_zbin_zs & isqso_truth & in_zbin_zvi).sum()
            com_denom = (isqso_truth & in_zbin_zvi).sum()
            strategies[s]['completeness'] = com_num/com_denom

            if ('RR' in s) and verbose:
                print(s)
                for k in strategies[s]:
                    if (k[0]=='n') or (k=='completeness'):
                        print(k,strategies[s][k])
                nlostqso = (~w_s & isqso_truth & in_zbin_zvi).sum()
                print(nlostqso,com_denom,nlostqso/com_denom)
                print('')

        nwrong = np.array([strategies[s]['nwrong'] for s in strategies.keys()])
        ncat = np.array([strategies[s]['ncat'] for s in strategies.keys()])

        pstar = np.array([strategies[s]['nstar'] for s in strategies.keys()])/ncat
        pgalwrongz = np.array([strategies[s]['ngalwrongz'] for s in strategies.keys()])/ncat
        pqsowrongz = np.array([strategies[s]['nqsowrongz'] for s in strategies.keys()])/ncat
        pcorrectwrongzbin = np.array([strategies[s]['ncorrectwrongzbin'] for s in strategies.keys()])/ncat

        completeness = np.array([strategies[s]['completeness'] for s in strategies.keys()])

        axs[0,i].bar(range(len(strategies)),pstar,color=utils.colours['C0'],label='star',width=0.5)
        axs[0,i].bar(range(len(strategies)),pgalwrongz,bottom=pstar,color=utils.colours['C1'],label='galaxy w.\nwrong $z$',width=0.5)
        bars = axs[0,i].bar(range(len(strategies)),pqsowrongz,bottom=pstar+pgalwrongz,color=utils.colours['C2'],label='QSO w.\nwrong $z$',width=0.5)
        if show_correctwrongzbin:
            axs[0,i].bar(range(len(strategies)),pcorrectwrongzbin,bottom=pstar+pgalwrongz+pqsowrongz,color=utils.colours['C3'],label='correct w.\nwrong $z$-bin',width=0.5)

        bar_heights = pstar+pgalwrongz+pqsowrongz
        if show_correctwrongzbin:
            bar_heights += pcorrectwrongzbin

        utils.autolabel_bars(axs[0,i],bars,numbers=bar_heights,heights=bar_heights,percentage=True,above=True)
        DESI_ncat_presents = []
        for j,c in enumerate(completeness):
            pcon = nwrong[j]/ncat[j]
            DESI_ncat = c * desi_nqso[i]/(1 - pcon)
            DESI_ncat_present = (round(DESI_ncat * 10**-6,3))
            DESI_ncat_presents.append(DESI_ncat_present)

        axs[0,i].set_xlabel('classification strategy',labelpad=10)
        axs[0,i].set_xticks(range(len(strategies)))
        axs[0,i].set_xlim(-0.5,len(strategies)-0.5)
        slabels = []
        for s in strategies.keys():
            try:
                slabels += [strategies[s]['label']]
            except KeyError:
                slabels += [s]
        axs[0,i].set_xticklabels(slabels)

        axs[0,i].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0,decimals=nydec))
        axs[0,i].set_ylim(0,ymax)
        zbin_label = get_label_from_zbin(zbin)
        axs[0,i].text(0.5,1.05,zbin_label,ha='center',va='center',transform=axs[0,i].transAxes)

        cell_text = []
        extrarow = False
        for s in slabels:
            if '\n' in s:
                extrarow = True
        if extrarow:
            cell_text.append(['']*len(completeness))
        cell_text.append(['{:2.1%}'.format(c) for c in completeness])
        cell_text.append(['{:1.3f}'.format(c) for c in DESI_ncat_presents])

        rowLabels = []
        if extrarow:
            rowLabels += ['']
        if i==0:
            rowLabels += ['completeness:','estimated DESI\ncatalogue size\n[million QSOs]:']
        else:
            rowLabels += ['','']
        table = mp_table.table(cellText=cell_text,
                      rowLabels=rowLabels,
                      colLabels=['' for s in strategies.keys()],
                      loc='bottom',
                      ax=axs[0,i],
                      edges='open',
                      cellLoc='center',
                      rowLoc='right',
                      in_layout=True)
        table.scale(1,6)

    offset_label = -0.1
    axs[0,0].set_ylabel('contamination of\nQSO catalogue')
    axs[0,1].legend()

    rect = (0.07,0.2,1.0,1.0)
    plt.tight_layout(rect=rect)
    if filename is not None:
        plt.savefig(filename)

    return fig, axs
