from pylab import *
import numpy as np
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
rcParams["mathtext.fontset"]='cm'
import os

def plot(z, kmax, Pkmc):
    folder_derivative = '/tigress/ab4671/Quijote/results/derivatives/'
    suffix      = '%.2f_z=%s.txt'%(kmax, z)
    f_out       = folder_derivative + 'mean_Pk%s_%s.png' % (Pkmc,suffix)
    realizations = 500

    x_min, x_max = 8e-3, kmax+0.01
    y_min, y_max = 5e2, 5e4

    fig = figure(figsize=(17,6))

    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)    

    for ax,par,par2 in zip([ax1,ax2,ax3,ax4,ax5,ax6],
                           [r'$\Omega_{m}$',r'$\Omega_{b}$', r'$h$' ,r'$n_{s}$', r'$\sigma_8$', r'$M_\nu$'],
                           ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu']):

        ax.patch.set_alpha(0.2)
        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlim([x_min,x_max])
        ax.set_ylim([y_min,y_max])
        
        ax.set_title(par,fontsize=18)
        
        if ax in [ax1,ax4]:      ax.set_ylabel(r'$P_m\,[h^{-3}{\rm Mpc^3}]$',fontsize=18)
        if ax in [ax4,ax5,ax6]:  ax.set_xlabel(r'$k\,[h{\rm Mpc}^{-1}]$',fontsize=18)
            
        if par2 == 'Mnu':
            sims = ['fiducial_ZA'] + ['Mnu_p','Mnu_pp','Mnu_ppp']
            colors = ['k','r','orangered','lightcoral']
        else:
            sims = ['fiducial'] + [par2+'_p', par2+'_m']
            colors = ['k','r','b']
            
        for si,sim in enumerate(sims):
            c = colors[si]
            
            f1 = os.path.join(folder_derivative, sim, 'mean_Pk_%s_%d_'%(Pkmc,realizations) + suffix)

            # read data
            data = np.loadtxt(f1,unpack=False);  X, Y = data[:,0], data[:,1]*1e10    # note the scaling of Pk to match the analysis (FIXME could amke this a parameter)

            # plot the data
            ax.plot(X, Y, linestyle='-', c=c)
            
    ax1.plot([x_min,x_max],[-10,-10],label=r'fid',c='k')
    ax1.plot([x_min,x_max],[-10,-10],label=r'+',c='r')
    ax1.plot([x_min,x_max],[-10,-10],label=r'-',c='b')
    
    #legend
    ax1.legend(loc=0,prop={'size':13.5},ncol=1,frameon=True) 

    subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.0)
    for ax in [ax1,ax2,ax3]:
        ax.xaxis.set_major_formatter( NullFormatter() )

    savefig(f_out, bbox_inches='tight')
    close(fig)
    
    plot_fid(z, kmax, Pkmc)

    
def plot_fid(z, kmax, Pkmc):
    folder_derivative = '/tigress/ab4671/Quijote/results/derivatives/'
    suffix      = '%.2f_z=%s.txt'%(kmax, z)
    f_out       = folder_derivative + 'mean_Pk%s_%s.pdf' % (Pkmc,suffix)
    realizations = 500

    x_min, x_max = 8e-3, kmax+0.01
    y_min, y_max = 5e2, 5e4

    fig, ax = subplots(1,1,figsize=(5,5))

    if 1:

        ax.patch.set_alpha(0.2)
        ax.set_xscale('log')
        ax.set_yscale('log')

        #ax.set_xlim([x_min,x_max])
        #ax.set_ylim([y_min,y_max])
        
        ax.set_ylabel(r'$P_m$',fontsize=15)
        ax.set_xlabel(r'$k\,[h{\rm Mpc}^{-1}]$',fontsize=15)
        
        sims = ['fiducial']
        colors = ['k']
            
        for si,sim in enumerate(sims):
            c = colors[si]
            
            f1 = os.path.join(folder_derivative, sim, 'mean_Pk_%s_%d_'%(Pkmc,realizations) + suffix)

            # read data
            data = np.loadtxt(f1,unpack=False);  X, Y = data[:,0], data[:,1]*1e10    # note the scaling of Pk to match the analysis (FIXME could amke this a parameter)

            # plot the data
            ax.plot(X, Y, linestyle='-', c=c)

    savefig(f_out[:-4]+'_fid'+f_out[-4:], bbox_inches='tight')
    close(fig)
