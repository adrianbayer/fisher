# This scripts computes:
# 1) The covariance matrix for all probes (P_m(k) + HMF + VSF)
# 2) The mean of all statistics for all cosmologies
# 3) The derivatives of the statistics with respect to the cosmological parameters
# 4) The Fisher matrix for the considered probes
from mpi4py import MPI
import numpy as np
from scipy.linalg import block_diag
import sys,os
import itertools
#sys.path.append('/home/fvillaescusa/data/pdf_information/analysis/git_repo/library')
import analysis_library as AL

###### MPI DEFINITIONS ######
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

##################################### INPUT ###########################################
# folders with the data and output files
root_data    = '/projects/QUIJOTE/'
root_results = '/tigress/ab4671/Quijote/results'

# general parameters
parameters       = ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu']
BoxSize          = 1000.0  #Mpc/h
z                = 0
snapnum          = {0:4}[z]       #z=0
realizations_Cov = 15000   #number of realizations for the covariance
realizations_der = 500     #number of realizations for the derivatives
Volume           = 1.0     #(Gpc/h)^3 (for rescaling Fisher)
diag_CovM_all    = 0    # only use the diagonal components of C?
diag_CovM_cross  = 0    # just reomce cross correaltion (i.e. between different probes)

# parameters of the Pk_m
kmax_m     = 0.5 #h/Mpc
folder_Pkm = root_data + 'Pk/matter/'
Pkmc = 'm'                          # while code says Pkm, you cna choose to use c instead.

# parameters of the VSF
# note this exactly corresponds to the (reversed) bin edges of the VSF already in the catalogue. don't change this
# note I reverse to get increasing Radii order. I also make sure to reverse the VSFwhen reading it in analyse_library.py
# FIXME really one should just get Radii directly from the void catalogue, but this shouldn't change, so I won't modify the code
# for now this is just used for filename Rmin and Rmax.
#Radii       = np.array([41, 39, 37, 35, 33, 31, 29, 27, 25, 23, 21, 19, 17, 
#                        15, 13, 11, 9, 7, 5], dtype=np.float32)[:7:-1]*1000.0/768 #Mpc/h   [:7:-1]

Radii       = np.arange(41,4,-1, dtype=np.float32)[:17:-1][3:]*1000.0/768 #Mpc/h   [:7:-1]     #[:14:-1][:]  [:17:-1][:]  [:17:-1][1:]
VSFmc       = 'm' 
delta_th_void = 0.7
group_bins_VSF = 1
if delta_th_void == 0.5:
    folder_VSF = root_data + 'Voids/'
else:
    folder_VSF = root_data + 'Voids2/'
    
# parameters of the HMF
Nmin       = 30.0    #minimum number of CDM particles in a halo
Nmax       = 7000.0  #maximum number of CDM particles in a halo
HMF_bins   = 15       #number of bins in the HMF
folder_HMF = root_data + 'Halos/'

#######################################################################################

# find the corresponding redshift
z = {4:0, 3:0.5, 2:1, 1:2, 0:3}[snapnum]

for do_Pkm, do_VSF, do_HMF in list(itertools.product(*([0,1],[0,1],[0,1])))[1:]:

    ######################## COMPUTE/READ FULL COVARIANCE #############################
    # read/compute the covariance of all the probes (Pk+HMF+VSF)
    # bins is an array with the number of bins in each statistics
    # X is an array with the value of the statistics in each bin
    # Cov is the covariance matrix with size bins x bins
    bins, X, Cov = AL.covariance(realizations_Cov, BoxSize, snapnum, root_data, root_results, 
                                 kmax_m, Pkmc, folder_Pkm,
                                 Radii, VSFmc, delta_th_void, folder_VSF, group_bins_VSF,
                                 HMF_bins, Nmin, Nmax, folder_HMF) 
    ###################################################################################
    
    ########################## COMPUTE ALL DERIVATIVES ################################
    # compute the mean values of the different statistics for all cosmologies and 
    # compute the derivatives of the statistics with respect to the parameters
    AL.derivatives(realizations_der, BoxSize, snapnum, root_data, root_results, 
                   ['Pkm','HMF','VSF'],                 # a little ugly, but I'm still using Pkm for the suffix, even if Pc
                   kmax_m, Pkmc, folder_Pkm,
                   Radii, VSFmc, delta_th_void, folder_VSF, group_bins_VSF,
                   HMF_bins, Nmin, Nmax, folder_HMF)
    ###################################################################################
    
    #if myrank>0:  sys.exit() #here finishes the parallelism 
    if myrank == 0:

        ######################## COMPUTE INVERSE OF (SUB)-COVARIANCE ######################
        # find the (sub)covariance of the considered observables; invert it
        Cov  = AL.subcovariance(Cov, bins, do_Pkm, do_HMF, do_VSF)
        if diag_CovM_all:
            COV = np.zeros_like(Cov)
            np.fill_diagonal(COV,np.diag(Cov))
            Cov = COV
        elif diag_CovM_cross:
            if   do_Pkm and not do_HMF and not do_VSF: bin_cuts = [bins[0]]
            elif not do_Pkm and do_HMF and not do_VSF: bin_cuts = [bins[1]]
            elif not do_Pkm and not do_HMF and do_VSF: bin_cuts = [bins[2]]
            elif do_Pkm and do_HMF and not do_VSF: bin_cuts = [bins[0],bins[0]+bins[1]]
            elif do_Pkm and not do_HMF and do_VSF: bin_cuts = [bins[0],bins[0]+bins[2]]
            elif not do_Pkm and do_HMF and do_VSF: bin_cuts = [bins[1],bins[1]+bins[2]]
            elif do_Pkm and do_HMF and do_VSF:     bin_cuts = np.cumsum(bins)
            
            Cov_blocks = []
            bc0 = 0
            for bc in bin_cuts:
                Cov_blocks.append(Cov[bc0:bc,bc0:bc])
                bc0 = bc
            Cov = block_diag(*Cov_blocks)
        ICov = AL.Inv_Cov(Cov)
        ###################################################################################

        ################################# GENERAL THINGS ##################################
        # find the k-bins, M-bins and R-bins and the number of cosmo parameter
        km = X[np.arange(0,                np.sum(bins[:1]))] #k-modes for P_m(k)
        N  = X[np.arange(np.sum(bins[:1]), np.sum(bins[:2]))] #number of particles bins
        R  = X[np.arange(np.sum(bins[:2]), np.sum(bins[:3]))] #radii bins
        all_bins   = Cov.shape[0]    #number of bins in the (sub)-covariance matrix
        params_num = len(parameters) #number of cosmological parameters

        # find the different suffixes
        suffix_Pkm = 'Pk_%s_%d_%.2f_z=%s.txt'%(Pkmc, realizations_der, kmax_m, z)
        suffix_HMF = 'HMF_%d_%.1e_%.1e_%d_z=%s.txt'%(realizations_der, Nmin, Nmax, HMF_bins, z)
        suffix_VSF = 'VSF%s_%d_%.1e_%.1e_%.1f_%d_z=%s.txt'\
                     %(VSFmc, realizations_der, Radii[0], Radii[-1], delta_th_void, int((len(Radii)-1)/group_bins_VSF), z)

        # read the HMF of the fiducial cosmology
        #f = '%s/fiducial_NCV/mean_HMF_%d_%.1e_%.1e_%d_z=%s.txt'\
        #    %(root_results, realizations_der, Nmin, Nmax, HMF_bins, z)
        #N_fiducial, HMF_fiducial, dHMF_fiducial = np.loadtxt(f, unpack=True)
        #if not(np.allclose(N_fiducial, N, rtol=1e-8, atol=1e-10)):  
        #    raise Exception('N-values differ in the fiducial HMF!!!')
        ###################################################################################

        ############################## READ DERIVATIVES ###################################
        # define the matrix containing the derivatives
        derivative = np.zeros((params_num, all_bins), dtype=np.float64)

        # do a loop over all the parameters
        for i,parameter in enumerate(parameters):

            # temporary array storing the derivatives
            derivat = np.array([], dtype=np.float64)

            if do_Pkm:  #read the P_m(k) derivatives (ONLY IMPLEMENTED FOR Pm)
                f = '%s/derivatives/Pk%s/derivative_%s_%s'%(root_results, 'm', parameter, suffix_Pkm)              # FIXME this works fine, but is ugly... Pc derivs are stroed in Pkm folder
                if parameter=='Mnu':
                    #f = '%s/derivatives/Pkm/derivative_Mnu_0.4-0.2-0.1-0.0_%s'%(root_results, suffix_Pkm)
                    #f = '%s/derivatives/Pkm/derivative_Mnu_0.1-0.0_%s'%(root_results, suffix_Pkm)
                    f = '%s/derivatives/Pkm/derivative_Mnu_0.4-0.2-0.0_%s'%(root_results, suffix_Pkm)
                k_der, der_Pk = np.loadtxt(f, unpack=True)
                if not(np.allclose(k_der, km, rtol=1e-8, atol=1e-10)):  
                    raise Exception('k-values differ in the Pk derivatives!!!')
                
                # use theory deirvtaive: let's just do Mnu, as that is the only quesitonable one
                if 0:
                    if parameter == 'Mnu':
                        halofit_version = 'takahashi'
                        delta_factor_str = ''
                        delta_factor = int(delta_factor_str) if len(delta_factor_str) > 0 else 1
                        theory_p = np.loadtxt('theory/Pm/camb/mean_Pkm_%s_Mnu_p%s.txt' % (halofit_version, delta_factor_str))
                        theory_f = np.loadtxt('theory/Pm/camb/mean_Pkm_%s_fiducial.txt' % halofit_version)
                        k_p, Pm_theory_p = theory_p[:,0], theory_p[:,1]
                        k_f, Pm_theory_f = theory_f[:,0], theory_f[:,1]
                        assert((k_f == k_p).all())
                        der_theory = (Pm_theory_p - Pm_theory_f) / 0.1 * delta_factor

                        #interp onto k_der from simulation
                        from scipy.interpolate import interp1d
                        der_Pk = interp1d(k_f, der_theory)(k_der)
                    
                derivat = np.hstack([derivat, der_Pk])

            if do_HMF:  #read the HMF derivatives
                f = '%s/derivatives/HMF/derivative_%s_%s'%(root_results, parameter, suffix_HMF)
                if parameter=='Mnu':
                    #f = '%s/derivatives/HMF/derivative_Mnu_0.4-0.2-0.1-0.0_%s'%(root_results, suffix_HMF)
                    #f = '%s/derivatives/HMF/derivative_Mnu_0.1-0.0_%s'%(root_results, suffix_HMF)
                    f = '%s/derivatives/HMF/derivative_Mnu_0.4-0.2-0.0_%s'%(root_results, suffix_HMF)
                if parameter=='Om':
                    pass
                    #f = '%s/derivatives/HMF/derivativeM_%s_%s'%(root_results, parameter, suffix_HMF)
                N_der, der_HMF = np.loadtxt(f, unpack=True)
                if not(np.allclose(N_der, N, rtol=1e-8, atol=1e-10)):  
                    raise Exception('N-values differ in the HMF derivatives!!!')
                
                
                # use theory deirvtaive: let's just do Mnu, as that is the only quesitonable one
                if 0:
                    #if parameter == 'Mnu':
                    #    pass
                    if parameter == 'Mnu':
                        delta_factor_str = ''
                        delta_factor = int(delta_factor_str) if len(delta_factor_str) > 0 else 1
                        theory_p = np.loadtxt('theory/%s/%s/mean_HMF_Mnu_p%s_simbin.txt' % ('P3Tinker', 'camb', delta_factor_str))
                        theory_f = np.loadtxt('theory/%s/%s/mean_HMF_fiducial_simbin.txt' % ('P3Tinker', 'camb'))
                        mp_fid = 656562367483.9242
                        N_p, HMF_theory_p = theory_p[:,0]/mp_fid, theory_p[:,1]   # this isn't really N, but give correction has been applied N mean M/mp_fid for fisher...
                        N_f, HMF_theory_f = theory_f[:,0]/mp_fid, theory_f[:,1]
                        assert((N_p == N_f).all())

                        der_theory = (HMF_theory_p - HMF_theory_f) / 0.1 * delta_factor
                        print(parameter, der_HMF/der_theory)
                        
                        #interp onto N_der from simulation
                        from scipy.interpolate import interp1d
                        #der_HMF = interp1d(N_f, der_theory)(N_der)
                        #der_HMF = np.exp(interp1d(np.log(N_f), np.log(der_theory))(np.log(N_der)))
                        der_HMF = der_theory   # use this when using simbin matched theory dataset
                    elif 0:#if parameter != 'ns' and parameter != 'Ob2':
                        delta_factor_str = ''
                        delta_factor = int(delta_factor_str) if len(delta_factor_str) > 0 else 1
                        theory_p = np.loadtxt('theory/%s/%s/mean_HMF_%s_p%s_simbin.txt' % ('P3Tinker', 'camb', parameter, delta_factor_str))
                        theory_m = np.loadtxt('theory/%s/%s/mean_HMF_%s_m%s_simbin.txt' % ('P3Tinker', 'camb', parameter, delta_factor_str))
                        mp_fid = 656562367483.9242
                        N_p, HMF_theory_p = theory_p[:,0]/mp_fid, theory_p[:,1]   # this isn't really N, but give correction has been applied N mean M/mp_fid for fisher...
                        N_m, HMF_theory_m = theory_m[:,0]/mp_fid, theory_m[:,1]
                        assert((N_p == N_m).all())

                        delta_pm_cp = {'Om':0.3275-0.3075, 'Ob2':0.051-0.047, 'h':0.6911-0.6511, 'ns':0.9824-0.9424, 's8':0.849-0.819}
                        der_theory = (HMF_theory_p - HMF_theory_m) / delta_pm_cp[parameter] * delta_factor
                        
                        #interp onto N_der from simulation
                        from scipy.interpolate import interp1d
                        #der_HMF = interp1d(N_m, der_theory)(N_der)
                        #der_HMF = np.exp(interp1d(np.log(N_m), np.log(der_theory))(np.log(N_der)))
                        print(parameter, der_HMF/der_theory)
                        der_HMF = der_theory   # use this when using simbin matched theory dataset
                
                
                derivat = np.hstack([derivat, der_HMF])

            if do_VSF:  #read the VSF derivatives
                f = '%s/derivatives/VSF/derivative_%s_%s'%(root_results, parameter, suffix_VSF)
                if parameter=='Mnu':
                    #f = '%s/derivatives/VSF/derivative_Mnu_0.4-0.2-0.1-0.0_%s'%(root_results, suffix_VSF)
                    #f = '%s/derivatives/VSF/derivative_Mnu_0.1-0.0_%s'%(root_results, suffix_VSF)
                    f = '%s/derivatives/VSF/derivative_Mnu_0.4-0.2-0.0_%s'%(root_results, suffix_VSF)
                R_der, der_VSF = np.loadtxt(f, unpack=True)
                if not(np.allclose(R_der, R, rtol=1e-8, atol=1e-10)):
                    raise Exception('R-values differ in the VSF derivatives!!!')
                derivat = np.hstack([derivat, der_VSF])

            derivative[i] = derivat
        ###################################################################################

        #################################### FISHER #######################################
        # compute the Fisher matrix
        Fisher = np.zeros((params_num, params_num), dtype=np.float64)
        for i in range(params_num):
            for j in range(i, params_num):
                if i==j:
                    Fisher[i,j] = np.dot(derivative[i], np.dot(ICov, derivative[i]))
                else:
                    Fisher[i,j] = 0.5*(np.dot(derivative[i], np.dot(ICov, derivative[j])) + \
                                       np.dot(derivative[j], np.dot(ICov, derivative[i])))
                    Fisher[j,i] = Fisher[i,j]
        Fisher *= Volume

        CAMB_Fisher = np.array([
            [2.13080592e+05, -1.20573100e+06, 1.48016560e+05, 2.93458548e+04, 
             -2.06713944e+04, -1.65766154e+03],
            [-1.20573100e+06, 1.35133806e+07, -2.18303421e+05, -1.26270926e+04,
             -1.61514959e+04, -5.92496230e+04],
            [ 1.48016560e+05, -2.18303421e+05,  2.03038428e+05, -1.38685185e+04,
              -1.61497519e+04, -1.55300001e+03],
            [ 2.93458548e+04, -1.26270926e+04, -1.38685185e+04,  1.02172866e+05,
              -6.36387231e+03, -5.65461481e+03],
            [-2.06713944e+04, -1.61514959e+04, -1.61497519e+04, -6.36387231e+03,
             2.29958884e+04,  6.30418193e+03],
            [-1.65766154e+03, -5.92496230e+04, -1.55300001e+03, -5.65461481e+03,
             6.30418193e+03,  2.27796421e+03]])

        # compute the marginalized error on the parameters
        IFisher = np.linalg.inv(Fisher)
        for i in range(params_num):
            print ('Error on %03s = %.5f'%(parameters[i], np.sqrt(IFisher[i,i])))
        
        # save the marginalized errors (just for easy copying)
        os.makedirs('results/errors', exist_ok=1)
        fout = 'results/errors/errors_%d_%d'%(realizations_der, realizations_Cov)
        if do_Pkm:  fout += '_Pk%s_%.2f'%(Pkmc, kmax_m)
        if do_HMF:  fout += '_HMF_%.1e_%.1e_%d'%(Nmin, Nmax, HMF_bins)
        if do_VSF:  fout += '_VSF%s_%.1e_%.1e_%.1f_%d'%(VSFmc, Radii[0], Radii[-1], delta_th_void, (len(Radii)-1)/group_bins_VSF)    # would be nicer to us len -1 for consitency with HMF
        if diag_CovM_all: fout += '_diag_CovM_all'
        elif diag_CovM_cross: fout += '_diag_CovM_cross'
        fout += '_z=%s.txt'%z
        np.savetxt(fout, np.diag(np.sqrt(IFisher)))

        # save results to file
        os.makedirs('results/Fisher', exist_ok=1)
        fout = 'results/Fisher/Fisher_%d_%d'%(realizations_der, realizations_Cov)
        if do_Pkm:  fout += '_Pk%s_%.2f'%(Pkmc, kmax_m)
        if do_HMF:  fout += '_HMF_%.1e_%.1e_%d'%(Nmin, Nmax, HMF_bins)
        if do_VSF:  fout += '_VSF%s_%.1e_%.1e_%.1f_%d'%(VSFmc, Radii[0], Radii[-1], delta_th_void, (len(Radii)-1)/group_bins_VSF)    # would be nicer to us len -1 for consitency with HMF
        # note the fisher is overwirtten each time regardless of diag CovM...
        #if diag_CovM_all: fout += '_diag_CovM_all'
        #elif diag_CovM_cross: fout += '_diag_CovM_cross'
        fout += '_z=%s.npy'%z
        print(fout)
        np.save(fout, Fisher)
        ###################################################################################
        
        # Save errors as we cut out bins:
        #"""
        if 0:#realizations_der == 500 and realizations_Cov == 15000 and myrank == 0:
            from copy import deepcopy
            if do_Pkm:
                Fdiag_arr = np.empty((len(km)-1, params_num))
                IFdiag_arr = np.empty((len(km)-1, params_num))
                _derivative = deepcopy(derivative)
                _Cov = deepcopy(Cov)
                for a in range(len(km)-1):

                    if a > 0:
                        del_id = len(km) - a
                        _derivative = np.delete(_derivative, del_id, axis=1)    # delet along the bin axis (not the parameter axis)
                        _Cov = np.delete(_Cov, del_id, 0)
                        _Cov = np.delete(_Cov, del_id, 1)

                    _ICov = AL.Inv_Cov(_Cov)

                    for i in range(params_num):
                        for j in range(i, params_num):
                            if i==j:
                                Fisher[i,j] = np.dot(_derivative[i], np.dot(_ICov, _derivative[i]))
                            else:
                                Fisher[i,j] = 0.5*(np.dot(_derivative[i], np.dot(_ICov, _derivative[j])) + \
                                                   np.dot(_derivative[j], np.dot(_ICov, _derivative[i])))
                                Fisher[j,i] = Fisher[i,j]

                    Fdiag_arr[a,:] = np.diag(Fisher)
                    IFdiag_arr[a,:] = np.diag(np.linalg.inv(Fisher))
                np.save(fout[:-4] + '_Fdiag_removebinsPk.npy', Fdiag_arr)
                np.save(fout[:-4] + '_IFdiag_removebinsPk.npy', IFdiag_arr)

            if do_HMF:
                Fdiag_arr = np.empty((len(N)-1, params_num))
                IFdiag_arr = np.empty((len(N)-1, params_num))
                _derivative = deepcopy(derivative)
                _Cov = deepcopy(Cov)
                for a in range(len(N)-1):
                    if a > 0:
                        del_id = len(N) - a
                        if do_Pkm:
                            del_id += len(km)
                        _derivative = np.delete(_derivative, del_id, axis=1)    # delet along the bin axis (not the parameter axis)
                        _Cov = np.delete(_Cov, del_id, 0)
                        _Cov = np.delete(_Cov, del_id, 1)

                    _ICov = AL.Inv_Cov(_Cov)

                    for i in range(params_num):
                        for j in range(i, params_num):
                            if i==j:
                                Fisher[i,j] = np.dot(_derivative[i], np.dot(_ICov, _derivative[i]))
                            else:
                                Fisher[i,j] = 0.5*(np.dot(_derivative[i], np.dot(_ICov, _derivative[j])) + \
                                                   np.dot(_derivative[j], np.dot(_ICov, _derivative[i])))
                                Fisher[j,i] = Fisher[i,j]

                    Fdiag_arr[a,:] = np.diag(Fisher)
                    IFdiag_arr[a,:] = np.diag(np.linalg.inv(Fisher))
                np.save(fout[:-4] + '_Fdiag_removebinsHMF.npy', Fdiag_arr)
                np.save(fout[:-4] + '_IFdiag_removebinsHMF.npy', IFdiag_arr)

                # remove bins from other direction (i.e. change Mmin)
                Fdiag_arr = np.empty((len(N)-1, params_num))
                IFdiag_arr = np.empty((len(N)-1, params_num))
                _derivative = deepcopy(derivative)
                _Cov = deepcopy(Cov)
                for a in range(len(N)-1):
                    if a > 0:
                        del_id = 0   # in this case after removal, the removal index is the same! just the first (0th) bin
                        if do_Pkm:
                            del_id += len(km)
                        _derivative = np.delete(_derivative, del_id, axis=1)    # delet along the bin axis (not the parameter axis)
                        _Cov = np.delete(_Cov, del_id, 0)
                        _Cov = np.delete(_Cov, del_id, 1)

                    _ICov = AL.Inv_Cov(_Cov)

                    for i in range(params_num):
                        for j in range(i, params_num):
                            if i==j:
                                Fisher[i,j] = np.dot(_derivative[i], np.dot(_ICov, _derivative[i]))
                            else:
                                Fisher[i,j] = 0.5*(np.dot(_derivative[i], np.dot(_ICov, _derivative[j])) + \
                                                   np.dot(_derivative[j], np.dot(_ICov, _derivative[i])))
                                Fisher[j,i] = Fisher[i,j]

                    Fdiag_arr[a,:] = np.diag(Fisher)
                    IFdiag_arr[a,:] = np.diag(np.linalg.inv(Fisher))
                np.save(fout[:-4] + '_Fdiag_removebinsHMFbkwd.npy', Fdiag_arr)
                np.save(fout[:-4] + '_IFdiag_removebinsHMFbkwd.npy', IFdiag_arr)

            if do_VSF:
                Fdiag_arr = np.empty((len(R)-1, params_num))
                IFdiag_arr = np.empty((len(R)-1, params_num))
                _derivative = deepcopy(derivative)
                _Cov = deepcopy(Cov)
                for a in range(len(R)-1):

                    if a > 0:
                        del_id = len(R) - a
                        if do_Pkm:
                            del_id += len(km)
                        if do_HMF:
                            del_id += len(N)
                        _derivative = np.delete(_derivative, del_id, axis=1)    # delet along the bin axis (not the parameter axis)
                        _Cov = np.delete(_Cov, del_id, 0)
                        _Cov = np.delete(_Cov, del_id, 1)

                    _ICov = AL.Inv_Cov(_Cov)

                    for i in range(params_num):
                        for j in range(i, params_num):
                            if i==j:
                                Fisher[i,j] = np.dot(_derivative[i], np.dot(_ICov, _derivative[i]))
                            else:
                                Fisher[i,j] = 0.5*(np.dot(_derivative[i], np.dot(_ICov, _derivative[j])) + \
                                                   np.dot(_derivative[j], np.dot(_ICov, _derivative[i])))
                                Fisher[j,i] = Fisher[i,j]

                    Fdiag_arr[a,:] = np.diag(Fisher)
                    IFdiag_arr[a,:] = np.diag(np.linalg.inv(Fisher))
                np.save(fout[:-4] + '_Fdiag_removebinsVSF.npy', Fdiag_arr)
                np.save(fout[:-4] + '_IFdiag_removebinsVSF.npy', IFdiag_arr)

                # remove frmo other direction I.e. change Rmin
                Fdiag_arr = np.empty((len(R)-1, params_num))
                IFdiag_arr = np.empty((len(R)-1, params_num))
                _derivative = deepcopy(derivative)
                _Cov = deepcopy(Cov)
                for a in range(len(R)-1):

                    if a > 0:
                        del_id = 0
                        if do_Pkm:
                            del_id += len(km)
                        if do_HMF:
                            del_id += len(N)
                        _derivative = np.delete(_derivative, del_id, axis=1)    # delet along the bin axis (not the parameter axis)
                        _Cov = np.delete(_Cov, del_id, 0)
                        _Cov = np.delete(_Cov, del_id, 1)

                    _ICov = AL.Inv_Cov(_Cov)

                    for i in range(params_num):
                        for j in range(i, params_num):
                            if i==j:
                                Fisher[i,j] = np.dot(_derivative[i], np.dot(_ICov, _derivative[i]))
                            else:
                                Fisher[i,j] = 0.5*(np.dot(_derivative[i], np.dot(_ICov, _derivative[j])) + \
                                                   np.dot(_derivative[j], np.dot(_ICov, _derivative[i])))
                                Fisher[j,i] = Fisher[i,j]

                    Fdiag_arr[a,:] = np.diag(Fisher)
                    IFdiag_arr[a,:] = np.diag(np.linalg.inv(Fisher))
                np.save(fout[:-4] + '_Fdiag_removebinsVSFbkwd.npy', Fdiag_arr)
                np.save(fout[:-4] + '_IFdiag_removebinsVSFbkwd.npy', IFdiag_arr)
        #"""
        ###################################################################################

# make the plots
if 1 and realizations_der == 500 and realizations_Cov == 15000 and myrank == 0:
    if 1:
        import plot_Pk
        import plot_HMF
        import plot_VSF

        plot_Pk.plot(z, kmax_m, Pkmc)
        plot_HMF.plot(z, Nmin, Nmax, HMF_bins)
        plot_VSF.plot(z, Radii[0], Radii[-1], VSFmc, delta_th_void, (len(Radii)-1)/group_bins_VSF)
    if 0:
        import plot_dPk
        import plot_dHMF
        import plot_dVSF

        if 1:#try:           # this plots multiple realization options at a time, while above code only runs for one, so need to have done all runs for this to work
            plot_dPk.plot(z, kmax_m, Pkmc)
            plot_dHMF.plot(z, Nmin, Nmax, HMF_bins)
            plot_dVSF.plot(z, Radii[0], Radii[-1], VSFmc, delta_th_void, (len(Radii)-1)/group_bins_VSF)
        #except OSError:     
        #    print("Files missing for the deirvative plot for different number of realizations.")

        import plot_covariance

        plot_covariance.plot(realizations_Cov, z, kmax_m, Pkmc, Nmin, Nmax, HMF_bins, Radii[0], Radii[-1], VSFmc, delta_th_void, (len(Radii)-1)/group_bins_VSF)
        
    import plot_individual_probes
    import plot_combined_probes


    plot_individual_probes.plot(realizations_der, realizations_Cov, z, kmax_m,  Pkmc, Nmin, Nmax, HMF_bins, Radii[0], Radii[-1], VSFmc, delta_th_void, (len(Radii)-1)/group_bins_VSF)
    plot_combined_probes.plot(realizations_der, realizations_Cov, z, kmax_m, Pkmc, Nmin, Nmax, HMF_bins, Radii[0], Radii[-1], VSFmc, delta_th_void, (len(Radii)-1)/group_bins_VSF)
    import plot_Mnu_s8
    plot_Mnu_s8.plot(realizations_der, realizations_Cov, z, kmax_m, Pkmc, Nmin, Nmax, HMF_bins, Radii[0], Radii[-1], VSFmc, delta_th_void, (len(Radii)-1)/group_bins_VSF)
