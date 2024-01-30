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
def plot_pos_neg(ax, X, der, c):
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
    

def plot1(z, kmax, Pkmc, Mnu_str):
    folder_derivative = '/tigress/ab4671/Quijote/results/derivatives/'
    suffix      = '%.2f_z=%s.txt'%(kmax, z)
    f_out       = folder_derivative + 'derivatives_Pk%s_%s%s' % (Pkmc, suffix[:-4], Mnu_str)
    n_realizations = [300,400,500]

    x_min, x_max = 8e-3, kmax+0.01
    y_min, y_max = 5e1, 4e5

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

        if ax in [ax4,ax5,ax6]:  ax.set_xlabel(r'$k\,[h{\rm Mpc}^{-1}]$',fontsize=18)

        for realizations,c in zip(n_realizations,['r','b','k']):

            if par2=='Mnu':
                f1 = folder_derivative + 'Pk%s/derivative_%s%s_Pk_%s_%d_%s'%(Pkmc,par2,Mnu_str,Pkmc,realizations,suffix)
                if Mnu_scaling != 1:
                    ax.set_ylabel(r'$%d\times\partial P_{%s}/\partial$'%(Mnu_scaling,Pkmc) + '%s'%par,fontsize=18)
                else:
                    ax.set_ylabel(r'$\partial P_{%s}/\partial$'%Pkmc + '%s'%par,fontsize=18)
            else:
                f1 = folder_derivative + 'Pk%s/derivative_%s_Pk_%s_%d_%s'%(Pkmc,par2,Pkmc,realizations,suffix)
                ax.set_ylabel(r'$\partial P_{%s}/\partial$'%Pkmc + '%s'%par,fontsize=18)

            # read data
            data = np.loadtxt(f1,unpack=False);  X, der = data[:,0], data[:,1]*1e10    # note the scaling of Pk to match the analysis (FIXME could amke this a parameter)
            if par2=='Mnu':  der *= Mnu_scaling

            # plot the data: solid lines (positive values) and dashed lines (negative values)
            plot_pos_neg(ax, X, der, c)
            
        # plot theory
        if 0 and Pkmc=='m':
            camb_class = 'camb'
            halofit_version = 'takahashi'    # for CAMB only
            if par2=='Mnu':
                if camb_class == 'camb':
                    # plot the different delta options
                    for ci,delta_factor_str in enumerate(['','2','4','8','16'][::2]):
                        delta_factor = int(delta_factor_str) if len(delta_factor_str) > 0 else 1
                        theory_p = np.loadtxt('theory/Pm/camb/mean_Pkm_%s_Mnu_p%s.txt' % (halofit_version, delta_factor_str))
                        theory_f = np.loadtxt('theory/Pm/camb/mean_Pkm_%s_fiducial.txt' % halofit_version)
                        k_p, Pm_theory_p = theory_p[:,0], theory_p[:,1]
                        k_f, Pm_theory_f = theory_f[:,0], theory_f[:,1]
                        assert((k_f == k_p).all())
                        der_theory = (Pm_theory_p - Pm_theory_f) / 0.1 * delta_factor

                        der_theory *= Mnu_scaling
                        ccc = ['y','orange','r']
                        plot_pos_neg(ax, k_p, der_theory, ccc[ci])

                    # also plot the 3 deirvative using /4,8,16
                    theory_ppp = np.loadtxt('theory/Pm/camb/mean_Pkm_%s_Mnu_p%s.txt' % (halofit_version, 4))
                    theory_pp = np.loadtxt('theory/Pm/camb/mean_Pkm_%s_Mnu_p%s.txt' % (halofit_version, 8))
                    theory_p = np.loadtxt('theory/Pm/camb/mean_Pkm_%s_Mnu_p%s.txt' % (halofit_version, 16))
                    theory_f = np.loadtxt('theory/Pm/camb/mean_Pkm_%s_fiducial.txt' % halofit_version)
                    k_ppp, Pm_theory_ppp = theory_ppp[:,0], theory_ppp[:,1]
                    k_pp, Pm_theory_pp = theory_pp[:,0], theory_pp[:,1]
                    k_p, Pm_theory_p = theory_p[:,0], theory_p[:,1]
                    k_f, Pm_theory_f = theory_f[:,0], theory_f[:,1]

                    der_theory = (Pm_theory_ppp - 12*Pm_theory_pp + 32*Pm_theory_p - 21*Pm_theory_f) / (12*0.1/16)

                    der_theory *= Mnu_scaling
                    plot_pos_neg(ax, k_p, der_theory, 'm')


                elif camb_class == 'class':
                    theory_p = np.loadtxt('theory/Pm/class/Mnu_p00_pk_nl.dat')
                    theory_f = np.loadtxt('theory/Pm/class/fiducial00_pk_nl.dat')
                    k_p, Pm_theory_p = theory_p[:,0], theory_p[:,1]
                    k_f, Pm_theory_f = theory_f[:,0], theory_f[:,1]
                    k_p = k_p[5:-5]    # this should ensure p is withing f
                    Pm_theory_p = Pm_theory_p[5:-5]
                    from scipy.interpolate import interp1d
                    Pm_theory_f_interp = interp1d(k_f, Pm_theory_f)(k_p)
                    der_theory = (Pm_theory_p - Pm_theory_f_interp) / 0.1

                    der_theory *= Mnu_scaling
                    plot_pos_neg(ax, k_p, der_theory, 'y')

            else:
                if camb_class == 'camb':
                    # plot the different delta options
                    for ci,delta_factor_str in enumerate(['','2','4','8','16'][::2]):
                        delta_factor = int(delta_factor_str) if len(delta_factor_str) > 0 else 1
                        theory_p = np.loadtxt('theory/Pm/camb/mean_Pkm_%s_%s_p%s.txt' % (halofit_version, par2, delta_factor_str))
                        theory_m = np.loadtxt('theory/Pm/camb/mean_Pkm_%s_%s_m%s.txt' % (halofit_version, par2, delta_factor_str))
                        k_p, Pm_theory_p = theory_p[:,0], theory_p[:,1]
                        k_m, Pm_theory_m = theory_m[:,0], theory_m[:,1]
                        assert((k_m == k_p).all())
                        der_theory = (Pm_theory_p - Pm_theory_m) / delta_pm_cp[par2] * delta_factor

                        ccc = ['y','orange','r']
                        plot_pos_neg(ax, k_p, der_theory, ccc[ci])

                elif camb_class == 'class':
                    theory_p = np.loadtxt('theory/Pm/class/%s_p00_pk_nl.dat'%par2)
                    theory_m = np.loadtxt('theory/Pm/class/%s_m00_pk_nl.dat'%par2)
                    k_p, Pm_theory_p = theory_p[:,0], theory_p[:,1]
                    k_m, Pm_theory_m = theory_m[:,0], theory_m[:,1]
                    k_p = k_p[5:-5]              # this should ensure p is within m
                    Pm_theory_p = Pm_theory_p[5:-5]
                    from scipy.interpolate import interp1d
                    Pm_theory_m_interp = interp1d(k_m, Pm_theory_m)(k_p)
                    der_theory = (Pm_theory_p - Pm_theory_m_interp) / delta_pm_cp[par2]

                    plot_pos_neg(ax, k_p, der_theory, 'y')

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
    
# this is a 6 image plot of the 6 Mnu derivative options
def plot_Mnn_derivs(z, kmax, Pkmc):
    folder_derivative = '/tigress/ab4671/Quijote/results/derivatives/'
    suffix      = '%.2f_z=%s.txt'%(kmax, z)
    f_out       = folder_derivative + 'derivatives_Mnu_Pk%s_%s' % (Pkmc,suffix[:-4])
    n_realizations = [300,400,500]

    x_min, x_max = 8e-3, kmax+0.01
    y_min, y_max = 5e1, 4e5

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

        if ax in [ax4,ax5,ax6]:  ax.set_xlabel(r'$k\,[h{\rm Mpc}^{-1}]$',fontsize=18)
        if ax in [ax1,ax4]:      
            if Mnu_scaling != 1:
                ax.set_ylabel(r'$%d\times\partial P_{%s}/\partial M_\nu$'%(Mnu_scaling,Pkmc),fontsize=18)
            else:
                ax.set_ylabel(r'$\partial P_{%s}/\partial M_\nu$'%Pkmc,fontsize=18)

        for realizations,c in zip(n_realizations,['r','b','k']):

            f1 = folder_derivative + 'Pk%s/derivative_Mnu%s_Pk_%s_%d_%s'%(Pkmc,par2,Pkmc,realizations,suffix)


            # read data
            data = np.loadtxt(f1,unpack=False);  X, der = data[:,0], data[:,1]*1e10    # note the scaling of Pk to match the analysis (FIXME could amke this a parameter)
            der *= Mnu_scaling

            # plot the data: solid lines (positive values) and dashed lines (negative values)
            plot_pos_neg(ax, X, der, c)
            
        # plot theory
        camb_class = 'camb'
        halofit_version = 'takahashi'    # for CAMB only
        
        if camb_class == 'camb' and Pkmc=='m':
            
            
            theory_f = np.loadtxt('theory/Pm/camb/mean_Pkm_%s_fiducial.txt' % halofit_version)
            theory_p = np.loadtxt('theory/Pm/camb/mean_Pkm_%s_Mnu_p.txt' % halofit_version)
            theory_pp = np.loadtxt('theory/Pm/camb/mean_Pkm_%s_Mnu_pp.txt' % halofit_version)
            theory_ppp = np.loadtxt('theory/Pm/camb/mean_Pkm_%s_Mnu_ppp.txt' % halofit_version)
            k_f, Pm_theory_f = theory_f[:,0], theory_f[:,1]
            k_p, Pm_theory_p = theory_p[:,0], theory_p[:,1]
            k_pp, Pm_theory_pp = theory_pp[:,0], theory_pp[:,1]
            k_ppp, Pm_theory_ppp = theory_ppp[:,0], theory_ppp[:,1]
            
            assert((k_f == k_p).all())
            assert((k_f == k_pp).all())
            assert((k_f == k_ppp).all())
            
            der_theory_11 = (Pm_theory_p - Pm_theory_f) / 0.1
            
            # compute the theory derivative using the same FDM stencil as the simulation:
            if par == '11':
                der_theory = (Pm_theory_p - Pm_theory_f) / 0.1
            elif par == '12':
                der_theory = (Pm_theory_pp - Pm_theory_f) / 0.2
            elif par == '13':
                der_theory = (Pm_theory_ppp - Pm_theory_f) / 0.4
            elif par == '21':
                der_theory = (-Pm_theory_pp + 4*Pm_theory_p - 3*Pm_theory_f) / (2*0.1)
            elif par == '22':
                der_theory = (-Pm_theory_ppp + 4*Pm_theory_pp - 3*Pm_theory_f) / (2*0.2)
            elif par == '3':
                der_theory = (Pm_theory_ppp - 12*Pm_theory_pp + 32*Pm_theory_p - 21*Pm_theory_f) / (12*0.1)
            
            
        elif camb_class == 'class' and Pkmc=='m':                                      # FIXME: This only compute 11 derivative
            theory_ppp = np.loadtxt('theory/Pm/class/Mnu_ppp00_pk_nl.dat')
            theory_pp = np.loadtxt('theory/Pm/class/Mnu_pp00_pk_nl.dat')
            theory_p = np.loadtxt('theory/Pm/class/Mnu_p00_pk_nl.dat')
            theory_f = np.loadtxt('theory/Pm/class/fiducial00_pk_nl.dat')
            
            k_ppp, Pm_theory_ppp = theory_ppp[:,0], theory_ppp[:,1]
            k_pp, Pm_theory_pp = theory_pp[:,0], theory_pp[:,1]
            k_p, Pm_theory_p = theory_p[:,0], theory_p[:,1]
            k_f, Pm_theory_f = theory_f[:,0], theory_f[:,1]
            
            # interp evth onto _p. 
            #k_ppp = k_ppp[15:-15]    # this should ensure p is withing f
            #k_pp = k_pp[15:-15]    # this should ensure p is withing f
            k_p = k_p[5:-50]    # this should ensure p is withing f
            
            #Pm_theory_ppp = Pm_theory_ppp[15:-15]
            #Pm_theory_pp = Pm_theory_pp[15:-15]
            Pm_theory_p = Pm_theory_p[5:-50]
            
            from scipy.interpolate import interp1d
            
            Pm_theory_ppp_interp = interp1d(k_ppp, Pm_theory_ppp)(k_p)
            Pm_theory_pp_interp = interp1d(k_pp, Pm_theory_pp)(k_p)
            Pm_theory_f_interp = interp1d(k_f, Pm_theory_f)(k_p)
            
            der_theory_11 = (Pm_theory_p - Pm_theory_f_interp) / 0.1
            
            # compute the theory derivative using the same FDM stencil as the simulation:
            if par == '11':
                der_theory = (Pm_theory_p - Pm_theory_f_interp) / 0.1
            elif par == '12':
                der_theory = (Pm_theory_pp_interp - Pm_theory_f_interp) / 0.2
            elif par == '13':
                der_theory = (Pm_theory_ppp_interp - Pm_theory_f_interp) / 0.4
            elif par == '21':
                der_theory = (-Pm_theory_pp_interp + 4*Pm_theory_p - 3*Pm_theory_f_interp) / (2*0.1)
            elif par == '22':
                der_theory = (-Pm_theory_ppp_interp + 4*Pm_theory_pp_interp - 3*Pm_theory_f_interp) / (2*0.2)
            elif par == '3':
                der_theory = (Pm_theory_ppp_interp - 12*Pm_theory_pp_interp + 32*Pm_theory_p - 21*Pm_theory_f_interp) / (12*0.1)
        
        der_theory_11 *= Mnu_scaling
        der_theory *= Mnu_scaling
        plot_pos_neg(ax, k_p, der_theory_11, 'y')
        plot_pos_neg(ax, k_p, der_theory, 'orange')

    ax1.plot([x_min,x_max],[-10,-10],label=r'%s %s (11)' % (camb_class, halofit_version), c='y')
    ax1.plot([x_min,x_max],[-10,-10],label=r'%s %s' % (camb_class, halofit_version), c='orange')
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

def plot(z, kmax, Pkmc):
    for Mnu_str in ['','_0.1-0.0','_0.2-0.0','_0.4-0.0','_0.2-0.1-0.0','_0.4-0.2-0.0','_0.4-0.2-0.1-0.0']:
        plot1(z, kmax, Pkmc, Mnu_str)
    plot_Mnn_derivs(z, kmax, Pkmc)
