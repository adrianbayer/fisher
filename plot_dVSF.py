from pylab import *
import numpy as np
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
rcParams["mathtext.fontset"]='cm'

Mnu_scaling = 10    # scale y axis of Mnu plots

# plot1 is for a single choice of Mnu deirvative
def plot1(z, Rmin, Rmax, VSFmc, delta_th_void, VSF_bins, Mnu_str):
    folder_derivative = '/tigress/ab4671/Quijote/results/derivatives/'
    suffix      = '%.1e_%.1e_%.1f_%d_z=%s.txt'%(Rmin, Rmax, delta_th_void, VSF_bins, z)
    f_out       = folder_derivative + 'derivatives_VSF%s_%s%s' % (VSFmc, suffix[:-4], Mnu_str)
    n_realizations = [300,400,500]

    x_min, x_max = Rmin, Rmax
    y_min, y_max = 8e-10, 8e-6 #1.1e-9, 5e-5
    if delta_th_void == 0.7:
        y_min, y_max = 8e-11, 8e-5

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
        #ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlim([x_min,x_max])
        ax.set_ylim([y_min,y_max])

        if ax in [ax4,ax5,ax6]:  ax.set_xlabel(r'$R\,[h^{-1}{\rm Mpc}]$',fontsize=18)

        for realizations,c in zip(n_realizations,['r','b','k']):

            if par2=='Mnu':
                f1 = folder_derivative + 'VSF/derivative_%s%s_VSF%s_%d_%s'%(par2,Mnu_str,VSFmc,realizations,suffix)
                if Mnu_scaling != 1:
                    ax.set_ylabel(r'$%d\times\partial {\rm VSF}/\partial$'%Mnu_scaling + '%s'%par,fontsize=18)
                else:
                    ax.set_ylabel(r'$\partial {\rm VSF}/\partial$' + '%s'%par,fontsize=18)
            else:
                f1 = folder_derivative + 'VSF/derivative_%s_VSF%s_%d_%s'%(par2,VSFmc,realizations,suffix)
                ax.set_ylabel(r'$\partial {\rm VSF}/\partial$' + '%s'%par,fontsize=18)

            # read data
            data = np.loadtxt(f1,unpack=False);  X, der = data[:,0], data[:,1]
            if par2=='Mnu':  der *= Mnu_scaling

            # plot the data: solid lines (positive values) and dashed lines (negative values)
            sign, X1, der1 = np.sign(der[0]), np.array(X[0]), np.array(der[0])
            for i in range(1,len(X)):
                X1, der1 = np.hstack([X1, X[i]]), np.hstack([der1, der[i]])
                if np.sign(der[i])!=sign:
                    der1[-1] *= -1
                    if sign==1.0:  ax.plot(X1, der1,linestyle='-', marker='None',c=c)
                    else:          ax.plot(X1,-der1,linestyle='--',marker='None',c=c)
                    sign, X1, der1 = np.sign(der[i]), np.array(X[i]), np.array(der[i])
            ax.plot(X1, der1,linestyle='-', marker='None',c=c)
            ax.plot(X1,-der1,linestyle='--',marker='None',c=c)

    ax1.plot([x_min,x_max],[-10,-10],label=r'$300\,\,{\rm realizations}$',c='r')
    ax1.plot([x_min,x_max],[-10,-10],label=r'$400\,\,{\rm realizations}$',c='b')
    ax1.plot([x_min,x_max],[-10,-10],label=r'$500\,\,{\rm realizations}$',c='k')

    #legend
    ax1.legend(loc=0,prop={'size':13.5},ncol=1,frameon=True) 

    subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.0)
    for ax in [ax1,ax2,ax3]:
        ax.xaxis.set_major_formatter( NullFormatter() )

    savefig(f_out+'.png', bbox_inches='tight')
    savefig(f_out+'.pdf', bbox_inches='tight')
    close(fig)

def plot_Mnu_derivs(z, Rmin, Rmax, VSFmc, delta_th_void, VSF_bins):
    folder_derivative = '/tigress/ab4671/Quijote/results/derivatives/'
    suffix      = '%.1e_%.1e_%.1f_%d_z=%s.txt'%(Rmin, Rmax, delta_th_void, VSF_bins, z)
    f_out       = folder_derivative + 'derivatives_Mnu_VSF%s_%s' % (VSFmc, suffix[:-4])
    n_realizations = [300,400,500]

    x_min, x_max = Rmin, Rmax
    y_min, y_max = 8e-10, 8e-6 #1.1e-9, 5e-5
    if delta_th_void == 0.7:
         y_min, y_max = 8e-11, 8e-5

    fig = figure(figsize=(17,6))

    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)    

    for ax,par,par2 in zip([ax1,ax2,ax3,ax4,ax5,ax6],
                           ['11','12','13','21','22','3'],
                           ['_0.1-0.0','_0.2-0.0','_0.4-0.0','_0.2-0.1-0.0','_0.4-0.2-0.0','_0.4-0.2-0.1-0.0']):

        ax.patch.set_alpha(0.2)
        #ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlim([x_min,x_max])
        ax.set_ylim([y_min,y_max])
        
        ax.set_title(par)

        if ax in [ax4,ax5,ax6]:  ax.set_xlabel(r'$R\,[h^{-1}{\rm Mpc}]$',fontsize=18)
        if ax in [ax1,ax4]:      
            if Mnu_scaling != 1:
                ax.set_ylabel(r'$%d\times\partial {\rm VSF}/\partial M_\nu$'%Mnu_scaling,fontsize=18)
            else:
                ax.set_ylabel(r'$\partial {\rm VSF}/\partial M_\nu$',fontsize=18)
                
        for realizations,c in zip(n_realizations,['r','b','k']):
            
            f1 = folder_derivative + 'VSF/derivative_Mnu%s_VSF%s_%d_%s'%(par2,VSFmc,realizations,suffix)
            
            # read data
            data = np.loadtxt(f1,unpack=False);  X, der = data[:,0], data[:,1]
            der *= Mnu_scaling

            # plot the data: solid lines (positive values) and dashed lines (negative values)
            sign, X1, der1 = np.sign(der[0]), np.array(X[0]), np.array(der[0])
            for i in range(1,len(X)):
                X1, der1 = np.hstack([X1, X[i]]), np.hstack([der1, der[i]])
                if np.sign(der[i])!=sign:
                    der1[-1] *= -1
                    if sign==1.0:  ax.plot(X1, der1,linestyle='-', marker='None',c=c)
                    else:          ax.plot(X1,-der1,linestyle='--',marker='None',c=c)
                    sign, X1, der1 = np.sign(der[i]), np.array(X[i]), np.array(der[i])
            ax.plot(X1, der1,linestyle='-', marker='None',c=c)
            ax.plot(X1,-der1,linestyle='--',marker='None',c=c)

    ax1.plot([x_min,x_max],[-10,-10],label=r'$300\,\,{\rm realizations}$',c='r')
    ax1.plot([x_min,x_max],[-10,-10],label=r'$400\,\,{\rm realizations}$',c='b')
    ax1.plot([x_min,x_max],[-10,-10],label=r'$500\,\,{\rm realizations}$',c='k')

    #legend
    ax1.legend(loc=0,prop={'size':13.5},ncol=1,frameon=True) 

    subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.0)
    for ax in [ax1,ax2,ax3]:
        ax.xaxis.set_major_formatter( NullFormatter() )

    savefig(f_out+'.png', bbox_inches='tight')
    savefig(f_out+'.pdf', bbox_inches='tight')
    close(fig)

def plot(z, Rmin, Rmax, VSFmc, delta_th_void, VSF_bins):
    for Mnu_str in ['','_0.1-0.0','_0.2-0.0','_0.4-0.0','_0.2-0.1-0.0','_0.4-0.2-0.0','_0.4-0.2-0.1-0.0']:
        plot1(z, Rmin, Rmax, VSFmc, delta_th_void, VSF_bins, Mnu_str)
    plot_Mnu_derivs(z, Rmin, Rmax, VSFmc, delta_th_void, VSF_bins)
