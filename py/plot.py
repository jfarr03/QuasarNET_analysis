    ## Same plot as above but colouring classifiers differently

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def get_pur_com(isqso_c,z_c,isqso_truth,isgal_truth,isbad,z_truth,zbin=None,dv_max=6000.):

    # Determine which entries are in the z bin.
    in_zbin_zvi = np.ones(z_truth.shape).astype(bool)
    in_zbin_zc = np.ones(z_c.shape).astype(bool)
    if zbin is not None:
        if zbin[0] is not None:
            in_zbin_zvi &= (z_truth>=zbin[0])
            in_zbin_zc &= (z_c>=zbin[0])
        if zbin[1] is not None:
            in_zbin_zvi &= (z_truth<zbin[1])
            in_zbin_zc &= (z_c<zbin[1])

    # See which classifier redshifts are "good".
    zgood = (z_truth>-1.) & (abs(z_c-z_truth) < dv_max*(1+z_truth)/300000.)

    # Calculate the two parts of purity.
    pur_num = (isqso_c & (isqso_truth | isgal_truth) & zgood & (~isbad) & in_zbin_zc).sum()
    pur_denom = (isqso_c & (~isbad) & in_zbin_zc).sum()

    # Calculate the two parts of completeness.
    com_num = (isqso_c & zgood & isqso_truth & in_zbin_zvi).sum()
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

## Function for Figure 1.
def plot_pur_com_vs_z(data_table,strategies,filename=None,zmin=0.,zmax=5.,dz_int=21,dv_max=6000.,nydec=0,figsize=(12,6),ymin=0.93,ymax=1.005):

    # Make figure.
    fig, axs = plt.subplots(1,2,figsize=figsize,sharey=True)

    # Construct z bins.
    dz_edges = np.linspace(zmin,zmax,dz_int)
    dz_mids = (dz_edges[1:] + dz_edges[:-1])/2.
    dz_widths = (dz_edges[1:] - dz_edges[:-1])

    # Determine truth vectors.
    isqso_truth, isgal_truth, isstar_truth, isbad = get_truths(data_table)

    # For each strategy:
    for j,c in enumerate(strategies.keys()):
        com = []
        pur = []

        # Extract data from strategy.
        isqso_c = strategies[c]['w']
        z_c = strategies[c]['z']

        # For each z bin:
        for i in range(len(dz_edges)-1):

            # Calculate purity and completeness.
            p,c = get_pur_com(isqso_c,z_c,isqso_truth,isgal_truth,isbad,
                data_table['Z_VI'],zbin=(dz_edges[i],dz_edges[i+1]),dv_max=dv_max)

            # Add to purity/completeness lists.
            pur += [p]
            com += [c]

        # Plot the results.
        axs[0].step(dz_edges,[pur[0]]+pur,where='pre',label=c,color=strategies[c]['c'],ls=strategies[c]['ls'])
        axs[1].step(dz_edges,[com[0]]+com,where='pre',label=c,color=strategies[c]['c'],ls=strategies[c]['ls'])

    # Format the axes.
    for ax in axs:
        ax.set_xlim(zmin,zmax)
        ax.set_ylim(ymin,ymax)
        ax.axhline(y=1.0,c='lightgrey',zorder=-1)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0,decimals=nydec))

    # Add x labels.
    axs[0].set_xlabel(r'$z_\mathrm{classifier}$')
    axs[1].set_xlabel(r'$z_\mathrm{VI}$')

    # Add titles.
    axs[0].text(0.5,1.05,'Purity',ha='center',va='center',
                  transform=axs[0].transAxes,rotation=0)
    axs[1].text(0.5,1.05,'Completeness',ha='center',va='center',
                  transform=axs[1].transAxes,rotation=0)

    # Add a legend.
    artists = []
    labels = []
    for j,s in enumerate(strategies.keys()):
        artists += [axs[0].step([0],[0],where='pre',color=strategies[s]['c'],ls=strategies[s]['ls'])[0]]
        labels += [strategies[s]['n']]
    fig.legend(artists,labels,loc='lower center',borderaxespad=0,bbox_to_anchor=(0.5,0.03),ncol=len(artists))
    rect = (0.0,0.1,1.0,1.0)
    plt.tight_layout(rect=rect)

    # Save.
    if filename is not None:
        plt.savefig(filename)

    return

## Function for alternative Figure 1.
def plot_pur_com_vs_cth_zbin(data_table,strategies,filename=None,zbins=[(None,2.1),(2.1,None)],dv_max=6000.,nydec=0,figsize=(12,12),ymin=0.9,ymax=1.):

    fig, axs = plt.subplots(2,2,figsize=figsize,sharex=True,sharey=True)

    cth_min = 0.0
    cth_max = 1.0
    n_int = 100
    c_th = np.arange(cth_min,cth_max,(1/n_int)*(cth_max-cth_min))

    isqso_truth, isgal_truth, isstar_truth, isbad = get_truths(data_table)

    for j,s in enumerate(strategies.keys()):

        for i,zbin in enumerate(zbins):

            z_c = strategies[s]['z']
            in_zbin_zvi = np.ones(data_table['Z_VI'].shape).astype(bool)
            in_zbin_zc = np.ones(z_c.shape).astype(bool)
            if zbin[0] is not None:
                in_zbin_zvi &= (data_table['Z_VI']>=zbin[0])
                in_zbin_zc &= (z_c>=zbin[0])
            if zbin[1] is not None:
                in_zbin_zvi &= (data_table['Z_VI']<zbin[1])
                in_zbin_zc &= (z_c<zbin[1])

            com = []
            pur = []

            for cth in c_th:

                # Try to use confidences, otherwise use weights.
                try:
                    isqso_c = strategies[s]['confs']>cth
                except KeyError:
                    isqso_c = strategies[s]['w']

                # Calculate purity and completeness.
                p,c = get_pur_com(isqso_c,z_c,isqso_truth,isgal_truth,isbad,
                    data_table['Z_VI'],zbin=zbin,dv_max=dv_max)

                # Add to purity/completeness lists.
                pur += [p]
                com += [c]

            axs[i,0].plot(c_th,pur,color=utils.colours['C0'],ls=strategies[c]['ls'])
            axs[i,1].plot(c_th,com,color=utils.colours['C1'],ls=strategies[c]['ls'])

    for i,zbin in enumerate(zbins):
        zbin_label = r'$z$'
        if zbin[0] is not None:
            zbin_label = r'${}\geq$'.format(zbin[0]) + zbin_label
        if zbin[1] is not None:
            zbin_label = zbin_label + r'$<{}$'.format(zbin[1])

        axs[i,0].text(-0.18,0.5,zbin_label,ha='center',va='center',
                      transform=axs[1,0].transAxes,rotation=90)

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
    rect = (0,0.13,1.,1.)
    plt.tight_layout(rect=rect)

    # Save.
    if filename is not None:
        plt.savefig(filename)

    return

## Function for Figure 2 (3 panel).
def plot_qn_model_compare_3panel(data_table,strategies,filename=None,dv_max=6000.,nydec=1,figsize=(12,12),ymin=0.97,ymax=1.):

    fig, axs = plt.subplots(1,3,figsize=figsize)

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

        z_c = data_table['Z_{}'.format(s)]

        for cth in c_th:

            # Try to use confidences, otherwise use weights.
            try:
                isqso_c = strategies[s]['confs']>cth
            except KeyError:
                raise KeyError('Confidences not found for strategy {}'.format(s))

            # Calculate purity and completeness.
            p,c = get_pur_com(isqso_c,z_c,isqso_truth,isgal_truth,isbad,
                data_table['Z_VI'],dv_max=dv_max)

            # Add to purity/completeness lists.
            pur += [p]
            com += [c]

        axs[0].plot(c_th,pur,label='pur',color=utils.colours['C0'],ls=strategies[s]['ls'])
        axs[1].plot(c_th,com,label='com',color=utils.colours['C1'],ls=strategies[s]['ls'])

        ## Plot the dv histogram.
        dv = 300000. * (z_c-data_table['Z_VI']) / (1+data_table['Z_VI'])
        axs[2].hist(dv,bins=dv_bins,histtype='step',ls=strategies[s]['ls'],color=utils.colours['C2'])

    for ax in axs[:2]:
        ax.set_xlabel(r'$c_{th}$')
        ax.set_xlim(0.,1.)
        ax.set_ylim(ymin,ymax)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0,decimals=nydec))

    axs[0].set_ylabel('Purity')
    axs[1].set_ylabel('Completeness')
    axs[2].axvline(x=0,c='lightgrey',zorder=-1)
    axs[2].set_ylabel(r'#')
    axs[2].set_xlabel(r'd$v$ [km/s]')
    axs[2].set_xlim(-3000.,3000.)

    artists = []
    labels = []
    for j,s in enumerate(strategies.keys()):
        artists += [axs[0].plot([0],[0],color='grey',ls=strategies[s]['ls'])[0]]
        labels += [strategies[s]['n']]

    fig.legend(artists,labels,loc='lower center',borderaxespad=0,bbox_to_anchor=(0.5,0.03),ncol=len(artists))
    rect = (0,0.12,1,1.0)
    plt.tight_layout(rect=rect)

    if filename is not None:
        plt.savefig(filename)

    return

## Function for Figure 2 (2 panel).
def plot_qn_model_compare_2panel(data_table,strategies,filename=None,dv_max=6000.,nydec=2,figsize=(12,12),ymin=0.98,ymax=1.,verbose=False):

    fig, axs = plt.subplots(2,1,figsize=figsize)

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

        z_c = data_table['Z_{}'.format(s)]

        for cth in c_th:

            # Try to use confidences, otherwise use weights.
            try:
                isqso_c = strategies[s]['confs']>cth
            except KeyError:
                raise KeyError('Confidences not found for strategy {}'.format(s))

            # Calculate purity and completeness.
            p,c = get_pur_com(isqso_c,z_c,isqso_truth,isgal_truth,isbad,
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

        axs[0].plot(c_th,pur,label=labelp,color=utils.colours['C0'],ls=strategies[s]['ls'])
        axs[1].plot(c_th,com,label=labelc,color=utils.colours['C1'],ls=strategies[s]['ls'])

        ## Plot the dv histogram.
        dv = 300000. * (z_c-data_table['Z_VI']) / (1+data_table['Z_VI'])
        axs[2].hist(dv,bins=dv_bins,histtype='step',ls=strategies[s]['ls'],color=utils.colours['C2'])

        if verbose:
            dv_med = np.median(dv[abs(dv)<dv_max])
            dv_std = np.std(dv[abs(dv)<dv_max])
            print('{} has median velocity error {:3.3f} and standard deviation {:3.3f}\n'.format(c,dv_med,dv_std))

        ## Add objects to list of artists and labels for legend.
        artists += [axs[0].plot([0],[0],color='grey',ls=strategies[s]['ls'])[0]]
        labels += [strategies[s]['n']]

    axs[0].set_xlabel(r'confidence threshold')
    axs[0].set_xlim(0.,1.)
    axs[0].set_ylim(ymin,ymax)
    axs[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0,decimals=nydec))
    axs[0].legend()

    axs[1].axvline(x=0,c='lightgrey',zorder=-1)
    axs[1].set_ylabel(r'#')
    axs[1].set_xlabel(r'd$v$ [km/s]')
    axs[1].set_xlim(-3000.,3000.)

    fig.legend(artists,labels,loc='lower center',borderaxespad=0,bbox_to_anchor=(0.5,0.03),ncol=len(artists))
    rect = (0,0.13,1.,1.)
    plt.tight_layout(rect=rect)

    if filename is not None:
        plt.savefig(filename)

    return

## Function for Figure 3.
def plot_qn_model_data_compare():
    return

## Function for Figure 4.
def plot_reobservation_performance():
    return

## Function for Figure 5.
def plot_catalogue_performance():
    return
