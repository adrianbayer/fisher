from pylab import *
import numpy as np
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
rcParams["mathtext.fontset"]='cm'

Mnu_scaling = 10   # scale y axis of Mnu plots

# this is the delta between the p and m cosmology
delta_pm_cp = {'Om':0.3275-0.3075, 'Ob2':0.051-0.047, 'h':0.6911-0.6511, 'ns':0.9824-0.9424, 's8':0.849-0.819, 'Mnu':0}

# log plot with solid for positive and dashed for negative (der is dependent variable)
def plot_pos_neg(ax, X, der, c, alpha=1):
    sign, X1, der1 = np.sign(der[0]), np.array(X[0]), np.array(der[0])
    for i in range(1,len(X)):
        X1, der1 = np.hstack([X1, X[i]]), np.hstack([der1, der[i]])
        if np.sign(der[i])!=sign:
            der1[-1] *= -1
            if sign==1.0:  ax.plot(X1, der1,linestyle='-', marker='None',c=c)
            else:          ax.plot(X1,-der1,linestyle='--',marker='None',c=c)
            sign, X1, der1 = np.sign(der[i]), np.array(X[i]), np.array(der[i])
    ax.plot(X1, der1,linestyle='-', marker='None',c=c,alpha=alpha)
    ax.plot(X1,-der1,linestyle='--',marker='None',c=c,alpha=alpha)
    

def plot1(z, Nmin, Nmax, HMF_bins, Mnu_str):
    folder_derivative = '/tigress/ab4671/Quijote/results/derivatives/'
    suffix      = '%.1e_%.1e_%d_z=%s.txt'%(Nmin, Nmax, HMF_bins, z)
    f_out       = folder_derivative + 'derivatives_HMF_%s%s' % (suffix[:-4], Mnu_str)
    n_realizations = [300,400,500]
    
    mp_fid = 656562367483.9242
    x_min, x_max = Nmin * mp_fid, Nmax * mp_fid
    y_min, y_max = 2e-8, 3e-3  #1e-9, 5e-3 #5e-1, 2e6

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

        if ax in [ax4,ax5,ax6]:  ax.set_xlabel(r'$M\,[h^{-1}M_\odot]$',fontsize=18)   
        for realizations,c in zip(n_realizations,['r','b','k']):

            if par2=='Mnu':
                f1 = folder_derivative + 'HMF/derivative_%s%s_HMF_%d_%s'%(par2,Mnu_str,realizations,suffix)
                if Mnu_scaling != 1:
                    ax.set_ylabel(r'$%d\times\partial {\rm HMF}/\partial$'%Mnu_scaling + '%s'%par,fontsize=18)
                else:
                    ax.set_ylabel(r'$\partial {\rm HMF}/\partial$' + '%s'%par,fontsize=18)
            else:
                f1 = folder_derivative + 'HMF/derivative_%s_HMF_%d_%s'%(par2,realizations,suffix)
                ax.set_ylabel(r'$\partial {\rm HMF}/\partial$' + '%s'%par,fontsize=18)

            # read data
            data = np.loadtxt(f1,unpack=False);  X, der = data[:,0]*mp_fid, data[:,1]   # convert N to M
            if par2=='Mnu':  der *= Mnu_scaling

            # plot the data: solid lines (positive values) and dashed lines (negative values)
            plot_pos_neg(ax, X, der, c)
            
            
            
            # PLOT THE UNCORRECTED HMF DERIV TOO? 
            # ALSO PLOT the direct M derivative?
            if 0:
                #if par2=='Mnu':   #  my filenaming for Mnu doens't work... meh... just plot Om... FIXME
                #    f1N = folder_derivative + 'HMF/derivativeN_%s_HMF_%d_%s'%(par2,realizations,suffix)             
                if par2=='Om':
                    f1N = folder_derivative + 'HMF/derivativeN_%s_HMF_%d_%s'%(par2,realizations,suffix)
                    f1M = folder_derivative + 'HMF/derivativeM_%s_HMF_%d_%s'%(par2,realizations,suffix)

                    # read data
                    dataN = np.loadtxt(f1N,unpack=False);  XN, derN = dataN[:,0]*mp_fid, dataN[:,1]   # convert N to M (even for uncorrected deirvative we want this to plot on same plot)
                    dataM = np.loadtxt(f1M,unpack=False);  XM, derM = dataM[:,0], dataM[:,1]          # already in mass units
                    
                    # plot the data: solid lines (positive values) and dashed lines (negative values)
                    plot_pos_neg(ax, XN, derN, c, alpha=0.2)    # make slightly transparent to distinguish.
                    plot_pos_neg(ax, XM, derM, c, alpha=0.5)    # make slightly transparent to distinguish.
                    
            
        # plot theory
        camb_class = 'camb'
        for model,c in zip(['P3ST', 'P3Tinker'], ['g','y']):
            if par2=='Mnu':
                #"""
                for ci,delta_factor_str in enumerate(['','2','4','8','16'][:1]):
                    delta_factor = int(delta_factor_str) if len(delta_factor_str) > 0 else 1
                    theory_p = np.loadtxt('theory/%s/%s/mean_HMF_Mnu_p%s_simbin.txt' % (model, camb_class, delta_factor_str))    # FIXME remove simbin!
                    theory_f = np.loadtxt('theory/%s/%s/mean_HMF_fiducial_simbin.txt' % (model, camb_class))
                    M_p, HMF_theory_p = theory_p[:,0], theory_p[:,1]
                    M_f, HMF_theory_f = theory_f[:,0], theory_f[:,1]
                    assert((M_p == M_f).all())

                    der_theory = (HMF_theory_p - HMF_theory_f) / 0.1 * delta_factor
                    
                    """   this section not generalized for delta factor
                    theory_ppp = np.loadtxt('theory/%s/mean_HMF_Mnu_ppp.txt' % (model, camb_class))
                    theory_pp = np.loadtxt('theory/%s/mean_HMF_Mnu_pp.txt' % (model, camb_class))
                    theory_f = np.loadtxt('theory/%s/mean_HMF_fiducial.txt' % (model, camb_class))
                    M_ppp, HMF_theory_ppp = theory_ppp[:,0], theory_ppp[:,1]
                    M_pp, HMF_theory_pp = theory_pp[:,0], theory_pp[:,1]
                    M_f, HMF_theory_f = theory_f[:,0], theory_f[:,1]
                    assert((M_pp == M_f).all())

                    der_theory = (- HMF_theory_ppp + 4 * HMF_theory_pp - 3 * HMF_theory_f) / (2*0.2)
                    """

                    der_theory *= Mnu_scaling
                    ccc = ['y','orange','r']
                    plot_pos_neg(ax, M_p, der_theory, c)#ccc[ci])
                
            else:
                # plot the different deltas
                for ci,delta_factor_str in enumerate(['','2','4','8','16'][:1]):
                    delta_factor = int(delta_factor_str) if len(delta_factor_str) > 0 else 1
                    theory_p = np.loadtxt('theory/%s/%s/mean_HMF_%s_p%s_simbin.txt'% (model, camb_class, par2, delta_factor_str))
                    theory_m = np.loadtxt('theory/%s/%s/mean_HMF_%s_m%s_simbin.txt'% (model, camb_class, par2, delta_factor_str))
                    M_p, HMF_theory_p = theory_p[:,0], theory_p[:,1]
                    M_m, HMF_theory_m = theory_m[:,0], theory_m[:,1]
                    assert((M_p == M_m).all())

                    der_theory = (HMF_theory_p - HMF_theory_m) / delta_pm_cp[par2] * delta_factor

                    ccc = ['y','orange','r']
                    plot_pos_neg(ax, M_p, der_theory, c)#ccc[ci])
            
            
            
        
    ax1.plot([x_min,x_max],[-10,-10],label=r'ST',c='g')
    ax1.plot([x_min,x_max],[-10,-10],label=r'Tinker',c='y')
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

    
def plot_Mnu_derivs(z, Nmin, Nmax, HMF_bins):
    folder_derivative = '/tigress/ab4671/Quijote/results/derivatives/'
    suffix      = '%.1e_%.1e_%d_z=%s.txt'%(Nmin, Nmax, HMF_bins, z)
    f_out       = folder_derivative + 'derivatives_Mnu_HMF_%s' % (suffix[:-4])
    n_realizations = [300,400,500]
    
    mp_fid = 656562367483.9242
    x_min, x_max = Nmin * mp_fid, Nmax * mp_fid
    y_min, y_max = 2e-8, 3e-3  #1e-9, 5e-3 #5e-1, 2e6

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
        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlim([x_min,x_max])
        ax.set_ylim([y_min,y_max])
        
        ax.set_title(par)

        if ax in [ax4,ax5,ax6]:  ax.set_xlabel(r'$M\,[h^{-1}M_\odot]$',fontsize=18)  
        if ax in [ax1,ax4]:      
            if Mnu_scaling != 1:
                ax.set_ylabel(r'$%d\times\partial {\rm HMF}/\partial M_\nu$'%Mnu_scaling,fontsize=18)
            else:
                ax.set_ylabel(r'$\partial {\rm HMF}/\partial M_\nu$',fontsize=18)
        
        for realizations,c in zip(n_realizations,['r','b','k']):

            f1 = folder_derivative + 'HMF/derivative_Mnu%s_HMF_%d_%s'%(par2,realizations,suffix)
      
            # read data
            data = np.loadtxt(f1,unpack=False);  X, der = data[:,0]*mp_fid, data[:,1]   # convert N to M
            der *= Mnu_scaling

            # plot the data: solid lines (positive values) and dashed lines (negative values)
            plot_pos_neg(ax, X, der, c)
            
        # plot theory
        camb_class = 'camb'
        for model,c,c2 in zip(['P3ST', 'P3Tinker'], ['green','y'], ['darkgreen','orange']):
            
            theory_ppp = np.loadtxt('theory/%s/%s/mean_HMF_Mnu_ppp.txt' % (model, camb_class))
            theory_pp = np.loadtxt('theory/%s/%s/mean_HMF_Mnu_pp.txt' % (model, camb_class))
            theory_p = np.loadtxt('theory/%s/%s/mean_HMF_Mnu_p.txt' % (model, camb_class))
            theory_f = np.loadtxt('theory/%s/%s/mean_HMF_fiducial.txt' % (model, camb_class))
            
            M_ppp, HMF_theory_ppp = theory_ppp[:,0], theory_ppp[:,1]
            M_pp, HMF_theory_pp = theory_pp[:,0], theory_pp[:,1]
            M_p, HMF_theory_p = theory_p[:,0], theory_p[:,1]
            M_f, HMF_theory_f = theory_f[:,0], theory_f[:,1]
            assert((M_ppp == M_f).all())
            assert((M_pp == M_f).all())
            assert((M_p == M_f).all())
                
            der_theory11 = (HMF_theory_p - HMF_theory_f) / 0.1

            # compute the theory derivative using the same FDM stencil as the simulation:
            if par == '11':
                der_theory = (HMF_theory_p - HMF_theory_f) / 0.1
            elif par == '12':
                der_theory = (HMF_theory_pp - HMF_theory_f) / 0.2
            elif par == '13':
                der_theory = (HMF_theory_ppp - HMF_theory_f) / 0.4
            elif par == '21':
                der_theory = (-HMF_theory_pp + 4*HMF_theory_p - 3*HMF_theory_f) / (2*0.1)
            elif par == '22':
                der_theory = (-HMF_theory_ppp + 4*HMF_theory_pp - 3*HMF_theory_f) / (2*0.2)
            elif par == '3':
                der_theory = (HMF_theory_ppp - 12*HMF_theory_pp + 32*HMF_theory_p - 21*HMF_theory_f) / (12*0.1)
            
            der_theory11 *= Mnu_scaling
            der_theory *= Mnu_scaling
            plot_pos_neg(ax, M_p, der_theory11, c)
            plot_pos_neg(ax, M_p, der_theory, c2)
        
    ax1.plot([x_min,x_max],[-10,-10],label=r'ST 11',c='green')
    ax1.plot([x_min,x_max],[-10,-10],label=r'Tinker 11',c='y')
    ax1.plot([x_min,x_max],[-10,-10],label=r'ST',c='darkgreen')
    ax1.plot([x_min,x_max],[-10,-10],label=r'Tinker',c='orange')
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
    
def plot(z, Nmin, Nmax, HMF_bins):
    for Mnu_str in ['','_0.1-0.0','_0.2-0.0','_0.4-0.0','_0.2-0.1-0.0','_0.4-0.2-0.0','_0.4-0.2-0.1-0.0']:
        plot1(z, Nmin, Nmax, HMF_bins, Mnu_str)
    plot_Mnu_derivs(z, Nmin, Nmax, HMF_bins)
