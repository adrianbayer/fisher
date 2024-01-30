from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
rcParams["mathtext.fontset"]='cm'

# This function takes a subFisher and computes the ellipse parameters
def ellipse_params(subCov):
    a2 = 0.5*(subCov[0,0]+subCov[1,1]) + np.sqrt(0.25*(subCov[0,0]-subCov[1,1])**2 + subCov[0,1]**2)
    a  = np.sqrt(a2)
    b2 = 0.5*(subCov[0,0]+subCov[1,1]) - np.sqrt(0.25*(subCov[0,0]-subCov[1,1])**2 + subCov[0,1]**2)
    b  = np.sqrt(b2)
    theta = 0.5*np.arctan2(2.0*subCov[0,1],(subCov[0,0]-subCov[1,1]))
    return a,b,theta

# This function plots the ellipses
def plot_ellipses(ax, fiducial_x, fiducial_y, a, b, theta, c='r'):
    e1 = Ellipse(xy=(fiducial_x,fiducial_y), width=1.52*a, height=1.52*b,
                 angle=theta*360.0/(2.0*np.pi))
    e2 = Ellipse(xy=(fiducial_x,fiducial_y), width=2.48*a, height=2.48*b,
                 angle=theta*360.0/(2.0*np.pi))

    for e in [e1,e2]:
        ax.add_artist(e)
        if e==e1:  alpha = 0.7
        if e==e2:  alpha = 0.4
        e.set_alpha(alpha)
        e.set_facecolor(c)


def plot(realizations_der, realizations_Cov, z, kmax, Pkmc, Nmin, Nmax, HMF_bins, Rmin, Rmax, VSFmc, delta_th_void, VSF_bins):
    fig = figure(figsize=(7,7))
    ############################################ INPUT #########################################
    folder_fisher = '/tigress/ab4671/Quijote/results/Fisher/'
    f1 = folder_fisher + 'Fisher_%d_%d_Pk%s_%.2f_z=%s.npy' % (realizations_der, realizations_Cov, Pkmc, kmax, z)
    f2 = folder_fisher + 'Fisher_%d_%d_HMF_%.1e_%.1e_%d_z=%s.npy' % (realizations_der, realizations_Cov, Nmin, Nmax, HMF_bins, z)
    f3 = folder_fisher + 'Fisher_%d_%d_VSF%s_%.1e_%.1e_%.1f_%d_z=%s.npy' % (realizations_der, realizations_Cov, VSFmc, Rmin, Rmax, delta_th_void, VSF_bins, z)
    f4 = folder_fisher + 'Fisher_%d_%d_Pk%s_%.2f_HMF_%.1e_%.1e_%d_VSF%s_%.1e_%.1e_%.1f_%d_z=%s.npy' % (realizations_der, realizations_Cov, Pkmc, kmax,  Nmin, Nmax, HMF_bins, VSFmc, Rmin, Rmax, delta_th_void, VSF_bins, z)
    
    # this is the main fisher we plot (change if needed) I just use this to fix axes if plotting different fisher data
    if delta_th_void == 0.5:
        f_main = folder_fisher + 'Fisher_%d_%d_Pk%s_%.2f_HMF_%.1e_%.1e_%d_VSF_%.1e_%.1e_%d_z=%s.npy' % (500, 15000, Pkmc, 0.5,  30, 7000, 15, 6.5, 53, 18, 0)
    elif delta_th_void == 0.7:
        f_main = folder_fisher + 'Fisher_%d_%d_Pk%s_%.2f_HMF_%.1e_%.1e_%d_VSF%s_%.1e_%.1e_%.1f_%d_z=%s.npy' % (500, 15000, Pkmc, 0.5,  30, 7000, 15, VSFmc, 10.4, 30, delta_th_void, 15, 0)
    
    f_out = f4[:-4] + 'Mnu_vs_s8'

    # define the parameters and their fiducial value
    parameter_label = [r'$\Omega_{m}$', r'$\Omega_{b}$', r'$h$', r'$n_{s}$', r'$\sigma_8$', r'$M_\nu\,({\rm eV})$']
    fiducial        = [0.3175,              0.049,               0.67,   0.96,           0.834,         0.0]
    mp_fid = 656562367483.9242
    ############################################################################################


    # read the inverse Fisher matrix and find the number of parameters
    Fisher1 = np.load(f1);  Cov1 = np.linalg.inv(Fisher1)
    Fisher2 = np.load(f2);  Cov2 = np.linalg.inv(Fisher2)
    Fisher3 = np.load(f3);  Cov3 = np.linalg.inv(Fisher3)
    Fisher4 = np.load(f4);  Cov4 = np.linalg.inv(Fisher4)
    Fisher_main = np.load(f_main);  Cov_main = np.linalg.inv(Fisher_main)

    # find the number of parameters
    parameters = Cov1.shape[0]

    colors = ['r', 'b', 'g', 'k']
    Cov    = [Cov1, Cov2, Cov3, Cov4]


    ax1 = fig.add_subplot(1, 1, 1)

    i,j = 4,5

    # set the x- and y- limits of the subplot
    x2_max = np.max([Cov1[i,i], Cov2[i,i], Cov3[i,i]])
    y2_max = np.max([Cov1[j,j], Cov2[j,j], Cov3[j,j]])
    #x_range = np.sqrt(Cov4[i,i])*10.0
    #y_range = np.sqrt(Cov4[j,j])*40.0
    x_range = 0.105#0.064
    y_range = 2.05#2.05
    ax1.set_xlim([fiducial[i]-x_range, fiducial[i]+x_range])
    ax1.set_ylim([fiducial[j]-y_range, fiducial[j]+y_range])

    # compute the ellipses area: to improve visualization we plot first largest ellipses
    areas = np.array([Cov1[i,i]*Cov1[j,j], Cov2[i,i]*Cov2[j,j],
                      Cov3[i,i]*Cov3[j,j], Cov4[i,i]*Cov4[j,j]])
    indexes = np.argsort(areas)[::-1]
    #indexes = np.arange(4)

    # plot the ellipses
    for k in range(len(indexes)):
        Cov_aux = Cov[indexes[k]]
        subCov = np.array([[Cov_aux[i,i], Cov_aux[i,j]], [Cov_aux[j,i], Cov_aux[j,j]]])
        a,b,theta = ellipse_params(subCov)
        plot_ellipses(ax1, fiducial[i], fiducial[j], a, b, theta, c=colors[indexes[k]])

        ax1.set_xlabel(parameter_label[i], fontsize=18*1.4)
        ax1.set_ylabel(parameter_label[j], fontsize=18*1.4)


    p1,=ax1.plot([-10,-9],[-10,-9],linestyle='-',marker='None',c=colors[0],alpha=0.7,lw=7)
    p2,=ax1.plot([-10,-9],[-10,-9],linestyle='-',marker='None',c=colors[1],alpha=0.7,lw=7)
    p3,=ax1.plot([-10,-9],[-10,-9],linestyle='-',marker='None',c=colors[2],alpha=0.7,lw=7)
    p4,=ax1.plot([-10,-9],[-10,-9],linestyle='-',marker='None',c=colors[3],alpha=0.7,lw=7)

    # use 'main' cov for consitent axes accross all plots 
    x_range = np.sqrt(Cov_main[i,i])*4.0/3.0/1. * 1.5
    y_range = np.sqrt(Cov_main[j,j])*4.0/2

    polygon = Rectangle((fiducial[i]-x_range, fiducial[j]-y_range), 2*x_range, 2*y_range,
                        edgecolor='k',lw=0.5,fill=False)
    ax1.add_artist(polygon)

    #ax3=axes([fiducial[i]-x_range, fiducial[j]-y_range, 2*x_range, 2*y_range])

    ax2 = plt.axes([0,0,1,1])
    #ip = InsetPosition(ax1, [0.055,0.36,0.25,0.25])
    #ip = InsetPosition(ax1, [0.74,0.74,0.25,0.25])
    ip = InsetPosition(ax1, [0.28,0.06,0.25,0.24])
    #ip = InsetPosition(ax1, [0.55,0.72,0.25,0.25])
    ax2.set_axes_locator(ip)
    mark_inset(ax1, ax2, loc1=1, loc2=2, fc="none",ec="0.5")

    ax2.set_xticks([0.832, 0.836])
    ax2.set_yticks([-0.03,-0.02,-0.01,0,0.01,0.02,0.03])
    plt.setp(ax2.get_xticklabels(), fontsize=12)
    plt.setp(ax2.get_yticklabels(), fontsize=12)


    ax2.set_xlim([fiducial[i]-x_range, fiducial[i]+x_range])
    ax2.set_ylim([fiducial[j]-y_range, fiducial[j]+y_range])

    # compute the ellipses area: to improve visualization we plot first largest ellipses
    areas = np.array([Cov1[i,i]*Cov1[j,j], Cov2[i,i]*Cov2[j,j],
                      Cov3[i,i]*Cov3[j,j], Cov4[i,i]*Cov4[j,j]])
    indexes = np.argsort(areas)[::-1]
    #indexes = np.arange(4)

    # plot the ellipses
    for k in range(len(indexes)):
        Cov_aux = Cov[indexes[k]]
        subCov = np.array([[Cov_aux[i,i], Cov_aux[i,j]], [Cov_aux[j,i], Cov_aux[j,j]]])
        a,b,theta = ellipse_params(subCov)
        plot_ellipses(ax2, fiducial[i], fiducial[j], a, b, theta, c=colors[indexes[k]])

    # legend
    leg = ax1.legend([p1,p2,p3,p4],
                     [r"$P_{%s}$"%Pkmc,
                      r"${\rm HMF}$",
                      r"${\rm VSF}$",
                      r"${\rm ALL}$"],
                     loc='right',prop={'size':18},ncol=1,frameon=False)
    leg.get_frame().set_edgecolor('k')

    
    # add title if this is one of the robustness plots and shrink plot (redo legend size too)
    if realizations_Cov < 15000 or realizations_der < 500 or kmax < 0.5 or Nmin > 40 or Rmax < 29:
        fig.set_figheight(fig.get_figheight()/1.6)
        fig.set_figwidth(fig.get_figwidth()/1.6)
        leg = ax1.legend([p1,p2,p3,p4],
                         [r"$P_{%s}$"%Pkmc,
                          r"${\rm HMF}$",
                          r"${\rm VSF}$",
                          r"${\rm ALL}$"],
                         loc='right',prop={'size':16},ncol=1,frameon=True)
        leg.get_frame().set_edgecolor('k')
    
        if realizations_Cov < 15000:
            ax1.set_title(r"$N_{\rm cov} = %d$" % realizations_Cov, fontsize=17)
        if realizations_der < 500:
            ax1.set_title(r"$N_{\rm der} = %d$" % realizations_der, fontsize=17)
        if kmax < 0.5:
            ax1.set_title(r"$k_{\rm max}  = %.1f h {\rm Mpc}^{-1}$" % kmax, fontsize=17)
        if Nmin > 40:
            Mmin = Nmin * mp_fid
            ax1.set_title(r"$N_{\rm min}  = %.1f\times10^{%d}\, h^{-1} M_\odot}$" % (Mmin/10**ceil(np.log10(Mmin)-1), ceil(np.log10(Mmin))-1), fontsize=17)
        if Rmax < 10:
            ax1.set_title(r"$R_{\rm max}  = %.1f h^{-1} {\rm Mpc}$" % Rmax, fontsize=17)
            
    ax1.tick_params(axis = 'both', which = 'major', labelsize = 18)
    ax2.tick_params(axis = 'both', which = 'minor', labelsize = 18)
    
    subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=0.0)
    #plt.show()
    savefig(f_out+'.png', bbox_inches='tight')
    savefig(f_out+'.pdf', bbox_inches='tight')
    close(fig)



