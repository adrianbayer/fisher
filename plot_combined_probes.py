from pylab import *
import numpy as np
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
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
    fig = figure(figsize=(14,14))
    ############################################ INPUT #########################################
    folder_fisher = '/tigress/ab4671/Quijote/results/Fisher/'
    f1 = folder_fisher + 'Fisher_%d_%d_Pk%s_%.2f_HMF_%.1e_%.1e_%d_z=%s.npy' % (realizations_der, realizations_Cov, Pkmc, kmax, Nmin, Nmax, HMF_bins, z)
    f2 = folder_fisher + 'Fisher_%d_%d_Pk%s_%.2f_VSF%s_%.1e_%.1e_%.1f_%d_z=%s.npy' % (realizations_der, realizations_Cov, Pkmc, kmax, VSFmc, Rmin, Rmax, delta_th_void, VSF_bins, z)
    f3 = folder_fisher + 'Fisher_%d_%d_HMF_%.1e_%.1e_%d_VSF%s_%.1e_%.1e_%.1f_%d_z=%s.npy' % (realizations_der, realizations_Cov, Nmin, Nmax, HMF_bins, VSFmc, Rmin, Rmax, delta_th_void, VSF_bins, z)
    f4 = folder_fisher + 'Fisher_%d_%d_Pk%s_%.2f_HMF_%.1e_%.1e_%d_VSF%s_%.1e_%.1e_%.1f_%d_z=%s.npy' % (realizations_der, realizations_Cov, Pkmc, kmax, Nmin, Nmax, HMF_bins, VSFmc, Rmin, Rmax, delta_th_void, VSF_bins, z)

    f_out = f4[:-4] + '_2'

    # define the parameters and their fiducial value
    parameter_label = [r'$\Omega_{m}$', r'$\Omega_{b}$', r'$h$', r'$n_{s}$', r'$\sigma_8$', r'$M_\nu\,({\rm eV})$']
    fiducial        = [0.3175,              0.049,               0.67,   0.96,           0.834,         0.0]
    ############################################################################################


    # read the Fisher and compute the covariance matrix
    Fisher1 = np.load(f1);  Cov1 = np.linalg.inv(Fisher1)
    Fisher2 = np.load(f2);  Cov2 = np.linalg.inv(Fisher2)
    Fisher3 = np.load(f3);  Cov3 = np.linalg.inv(Fisher3)
    Fisher4 = np.load(f4);  Cov4 = np.linalg.inv(Fisher4)

    # find the number of parameters
    parameters = Cov1.shape[0]

    colors = ['fuchsia', 'gold', 'c', 'k']
    Cov    = [Cov1, Cov2, Cov3, Cov4]

    # do a loop over the different subpanels and plot contours
    for i in range(parameters-1):
        for j in range(i+1, parameters):

            number = (parameters-1)*(j-1) + i + 1

            ax1 = fig.add_subplot(parameters-1, parameters-1, number)
            ax1.patch.set_alpha(0.5)

            # set the x- and y- limits of the subplot
            x2_max = np.max([Cov1[i,i], Cov2[i,i], Cov3[i,i]])
            y2_max = np.max([Cov1[j,j], Cov2[j,j], Cov3[j,j]])
            x_range = np.sqrt(x2_max)*1.5
            y_range = np.sqrt(y2_max)*1.5
            ax1.set_xlim([fiducial[i]-x_range, fiducial[i]+x_range])
            ax1.set_ylim([fiducial[j]-y_range, fiducial[j]+y_range])

            # compute the ellipses area: to improve visualization we plot first largest ellipses
            areas = np.array([Cov1[i,i]*Cov1[j,j], Cov2[i,i]*Cov2[j,j],
                              Cov3[i,i]*Cov3[j,j], Cov4[i,i]*Cov4[j,j]])
            indexes = np.argsort(areas)[::-1]

            # plot the ellipses
            for k in range(len(indexes)):
                Cov_aux = Cov[indexes[k]]
                subCov = np.array([[Cov_aux[i,i], Cov_aux[i,j]], [Cov_aux[j,i], Cov_aux[j,j]]])
                a,b,theta = ellipse_params(subCov)
                plot_ellipses(ax1, fiducial[i], fiducial[j], a, b, theta, c=colors[indexes[k]])

            # clean the x- and y- axes to make the figure nice
            if i>0 and j<(parameters-1):
                ax1.xaxis.set_major_formatter( NullFormatter() ) #unset x label
                ax1.yaxis.set_major_formatter( NullFormatter() ) #unset y label 

            if j==parameters-1:
                ax1.set_xlabel(parameter_label[i], fontsize=18)
                if i>0:
                    ax1.yaxis.set_major_formatter( NullFormatter() ) #unset y label 
                for label in ax1.get_xticklabels():
                    label.set_rotation(-60)

            if i==0:
                ax1.set_ylabel(parameter_label[j], fontsize=18)
                if j<parameters-1:
                    ax1.xaxis.set_major_formatter( NullFormatter() ) #unset x label

    p1,=ax1.plot([-10,-9],[-10,-9],linestyle='-',marker='None',c=colors[0],alpha=0.7,lw=7)
    p2,=ax1.plot([-10,-9],[-10,-9],linestyle='-',marker='None',c=colors[1],alpha=0.7,lw=7)
    p3,=ax1.plot([-10,-9],[-10,-9],linestyle='-',marker='None',c=colors[2],alpha=0.7,lw=7)
    p4,=ax1.plot([-10,-9],[-10,-9],linestyle='-',marker='None',c=colors[3],alpha=0.7,lw=7)

    # legend
    leg = ax1.legend([p1,p2,p3,p4],
                     [r"$P_{%s}+{\rm HMF}$"%Pkmc,
                      r"$P_{%s}+{\rm VSF}$"%Pkmc,
                      r"${\rm HMF}+{\rm VSF}$",
                      r"$P_{%s}+{\rm HMF}+{\rm VSF}$"%Pkmc],
                     loc=(-.6,4.05),prop={'size':20},ncol=1,frameon=True)
    leg.get_frame().set_edgecolor('k')

    subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=0.0)
    savefig(f_out+'.png', bbox_inches='tight')
    savefig(f_out+'.pdf', bbox_inches='tight')
    close(fig)



