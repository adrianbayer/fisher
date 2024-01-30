from pylab import *
import numpy as np
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
rcParams["mathtext.fontset"]='cm'

minimum = -1.0#0.2
maximum = 1.0

# do a loop over the different covariances with different number of realizations
def plot(realizations, z, kmax, Pkmc, Nmin, Nmax, HMF_bins, Rmin, Rmax, VSFmc, delta_th_void, VSF_bins):

    fig=figure(figsize=(12,9)) #give dimensions to the figure
    ax1=fig.add_subplot(111) 

    #ax1.set_xlabel(r'$k_1\in[0-0.5]\,h{\rm Mpc}^{-1}\,\,+\,\,M_1>6\times10^{13}\,h^{-1}M_\odot\,\,+\,\,R_1\in[5-33]\,h^{-1}{\rm Mpc}$',fontsize=16)
    #ax1.set_ylabel(r'$k_2\in[0-0.5]\,h{\rm Mpc}^{-1}\,\,+\,\,M_2>6\times10^{13}\,h^{-1}M_\odot\,\,+\,\,R_2\in[5-33]\,h^{-1}{\rm Mpc}$',fontsize=16)
    
    folder_covariance = '/tigress/ab4671/Quijote/results/covariance/'
    #f1    = folder_covariance + 'Cov_norm_15000_Pk_0.50_HMF_1.0e+02_1.0e+04_15_VSF_4.0e+00_3.3e+01_23_[1, 2, 3, 5, 7, 9]_z=0.txt'
    #f1    = folder_covariance + 'Cov_15000_Pkm_0.05_Pkc_0.20_HMF_1.0e+02_1.0e+04_15_VSF_53.4_6.5_19_z=0.txt'
    #f1    = folder_covariance + 'Cov_norm_15000_Pkm_0.05_HMF_1.0e+02_1.0e+04_15_VSF_53.4_6.5_19_z=0.txt'
    #f1    = folder_covariance + 'Cov_norm_15000_Pkm_0.50_HMF_1.0e+02_1.0e+04_15_VSF_53.4_6.5_19_z=0.txt'
    f1 = folder_covariance + 'Cov_norm_%d_Pk%s_%.2f_HMF_%.1e_%.1e_%d_VSF%s_%.1e_%.1e_%.1f_%d_z=%s.txt' % (realizations, Pkmc, kmax,  Nmin, Nmax, HMF_bins, VSFmc, Rmin, Rmax, delta_th_void, VSF_bins, z)
    f_out = f1[:-4]

    k1,k2,Cov = np.loadtxt(f1,unpack=True) 
    bins = int(round(np.sqrt(len(Cov))))
    Cov = np.reshape(Cov, (bins,bins))
    k  = k2[:bins]
    
    Pkm_bins = bins - HMF_bins - VSF_bins

    cax = ax1.imshow(Cov,cmap='seismic',origin='lower',
                     vmin=minimum,vmax=maximum)
    cbar = fig.colorbar(cax, ax=ax1) #in ax2 colorbar of ax1
    cbar.set_label(r"${\rm Corr}(O_\alpha, O_\beta)$",#"/\sqrt{{\rm Var}(d_1){\rm Var}(d_2)}$",
                   fontsize=20,labelpad=0)
    cbar.ax.tick_params(labelsize=13)  #to change size of ticks

    ax1.xaxis.set_major_formatter( NullFormatter() )   #unset x label 
    ax1.yaxis.set_major_formatter( NullFormatter() )   #unset y label 
    
    ax1.set_xticks([Pkm_bins-0.5, Pkm_bins+HMF_bins-0.5])#, Pkm_bins+HMF_bins+VSF_bins])
    ax1.set_yticks([Pkm_bins-0.5, Pkm_bins+HMF_bins-0.5])#, Pkm_bins+HMF_bins+VSF_bins])
    
    #ax1.set_xlabel(r'$k_1\in[0-0.5]\,h{\rm Mpc}^{-1}\,\,+\,\,M_1>6\times10^{13}\,h^{-1}M_\odot\,\,+\,\,R_1\in[5-33]\,h^{-1}{\rm Mpc}$',fontsize=16)
    #ax1.set_ylabel(r'$k_2\in[0-0.5]\,h{\rm Mpc}^{-1}\,\,+\,\,M_2>6\times10^{13}\,h^{-1}M_\odot\,\,+\,\,R_2\in[5-33]\,h^{-1}{\rm Mpc}$',fontsize=16)
    ax1.set_xlabel(r'                            $P_%s$                        ${\rm HMF}$     ${\rm VSF}$'%Pkmc, fontsize=20)
    ax1.set_ylabel(r'                            $P_%s$                        ${\rm HMF}$     ${\rm VSF}$'%Pkmc, fontsize=20)
    
    ax1.set_title(r'${\rm Correlation \,\, matrix:}\,\,P_m\,+\,{\rm HMF}\,+\,{\rm VSF}$',position=(0.5,1.02),size=22)

    savefig(f_out+'.png', bbox_inches='tight')
    savefig(f_out+'.pdf', bbox_inches='tight')
    close(fig)
