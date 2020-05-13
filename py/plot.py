    ## Same plot as above but colouring classifiers differently

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

## Function for Figure 1.
def plot_pur_com_vs_z(data_table,strategies,filename=None,zmin=0.,zmax=5.,dz_int=21,dv_max=6000.,nydec=0,figsize=(12,6),ymin=0.93,ymax=1.005):

    # Make figure.
    fig, axs = plt.subplots(1,2,figsize=figsize,sharey=True)

    # Construct z bins.
    dz_edges = np.linspace(zmin,zmax,dz_int)
    dz_mids = (dz_edges[1:] + dz_edges[:-1])/2.
    dz_widths = (dz_edges[1:] - dz_edges[:-1])

    # Determine truth vectors.
    isqso_truth = (data_table['ISQSO_VI'] & (data_table['ZCONF_PERSON']==2))
    isgal_truth = (data_table['CLASS_VI']=='GALAXY')
    isbad = ((data_table['CLASS_VI']=='BAD') | (data_table['ZCONF_PERSON']!=2))

    # For each strategy:
    for j,c in enumerate(strategies.keys()):
        com = []
        pur = []

        # Extract data from strategy.
        isqso_c = strategies[c]['w']
        z_c = strategies[c]['z']
        zgood = (data_table['Z_VI']>0) & (abs(z_c-data_table['Z_VI']) < dv_max*(1+data_table['Z_VI'])/300000.)

        # For each z bin:
        for i in range(len(dz_edges)-1):

            # Determine which entries are in the z bin.
            in_zbin_zvi = (data_table['Z_VI']>=dz_edges[i]) & (data_table['Z_VI']<dz_edges[i+1])
            in_zbin_zc = (z_c>=dz_edges[i]) & (z_c<dz_edges[i+1])

            # Calculate the two parts of purity.
            pur_num = (isqso_c & (isqso_truth | isgal_truth) & zgood & ~isbad & in_zbin_zc).sum()
            pur_denom = (isqso_c & (~isbad) & in_zbin_zc).sum()

            # Calculate the two parts of completeness.
            com_num = (isqso_c & zgood & isqso_truth & in_zbin_zvi).sum()
            com_denom = (isqso_truth & in_zbin_zvi).sum()

            # Add to purity/completeness lists.
            pur += [pur_num/pur_denom]
            com += [com_num/com_denom]

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

    isqso_truth = (data_table['ISQSO_VI'] & (data_table['ZCONF_PERSON']==2))
    isgal_truth = (data_table['CLASS_VI']=='GALAXY')
    isbad = ((data_table['ZCONF_PERSON']=='BAD') | (data_table['ZCONF_PERSON']!=2))

    for j,s in enumerate(strategies.keys()):

        for i,zbin in enumerate(zbins):

            z_c = strategies[s]['z']
            in_zbin_zvi = np.ones(z_c.shape).astype(bool)
            if zbin[0] is not None:
                in_zbin_zvi &= (data_table['Z_VI']>=zbin[0])
                in_zbin_zc &= (z_c>=zbin[0])
            if zbin[1] is not None:
                in_zbin_zvi &= (data_table['Z_VI']<zbin[1])
                in_zbin_zc &= (z_c<zbin[1])

            com = []
            pur = []

            zgood = (data_table['Z_VI']>-1) & (abs(z_c-data_table['Z_VI']) < dv_max*(1+data_table['Z_VI'])/300000.)

            for cth in c_th:

                # Try to use confidences, otherwise use weights.
                try:
                    isqso_c = strategies[s]['confs']>cth
                except KeyError:
                    isqso_c = strategies[s]['w']

                # Calculate the two parts of purity.
                pur_num = (isqso_c & (isqso_truth | isgal_truth) & zgood & ~isbad & in_zbin_zc).sum()
                pur_denom = (isqso_c & (~isbad) & in_zbin_zc).sum()

                # Calculate the two parts of completeness.
                com_num = (isqso_c & zgood & isqso_truth & in_zbin_zvi).sum()
                com_denom = (isqso_truth & in_zbin_zvi).sum()

                # Add to purity/completeness lists.
                pur += [pur_num/pur_denom]
                com += [com_num/com_denom]

            axs[i,0].plot(c_th,pur,color=colours['C0'],ls=strategies[c]['ls'])
            axs[i,1].plot(c_th,com,color=colours['C1'],ls=strategies[c]['ls'])

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

## Function for Figure 2.
def plot_qn_model_compare():
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
