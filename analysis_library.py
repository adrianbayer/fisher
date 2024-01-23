# This library computes the covariance of each observable and the derivatives wrt
# the different cosmological parameters
from mpi4py import MPI
import numpy as np
import sys,os,h5py
import readfof

###### MPI DEFINITIONS ######
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()


#######################################################################################
# This routine takes a covariance matrix and computes its inverse and conditional number
def Inv_Cov(Cov):

    
    # find eigenvalues and eigenvector of the covariance
    v1,w1 = np.linalg.eig(Cov)
    
    # compute the inverse of the covariance
    ICov = np.linalg.inv(Cov)

    # find eigenvalues and eigenvector of the covariance
    v2,w2 = np.linalg.eig(ICov)

    #np.savetxt('eigenvalues.txt', 
    #           np.transpose([np.arange(elements), np.sort(v1), np.sort(v2)]))
    
    # check the product of the covariance and its inverse gives the identity matrix
    Equal = np.allclose(np.dot(Cov, ICov), np.eye(Cov.shape[0]))
    
    if 1:#not Equal:
        print ('\n####################################################')
        print ('Max eigenvalue    Cov = %.3e'%np.max(v1))
        print ('Min eigenvalue    Cov = %.3e'%np.min(v1))
        print ('Condition number  Cov = %.3e'%(np.max(v1)/np.min(v1)))
        print (' ')
        print ('Max eigenvalue   ICov = %.3e'%np.max(v2))
        print ('Min eigenvalue   ICov = %.3e'%np.min(v2))
        print ('Condition number ICov = %.3e'%(np.max(v2)/np.min(v2)))

        print ('\nHas the inverse been properly found?',Equal)#,np.diag(np.dot(Cov, ICov)))
        print ('####################################################\n')
    
    return ICov

# This routine computes the subcovariance of the considered statistics given 
# the full covariance
def subcovariance(Cov, bins, do_Pkm, do_HMF, do_VSF):
    indexes_Pkm = np.arange(0,                np.sum(bins[:1])) #Pkm indexes
    indexes_HMF = np.arange(np.sum(bins[:1]), np.sum(bins[:2])) #HMF indexes
    indexes_VSF = np.arange(np.sum(bins[:2]), np.sum(bins[:3])) #VSF indexes
    
    # find the indexes of the subcovariance
    indexes = np.array([], dtype=np.int32)
    if do_Pkm: indexes = np.hstack([indexes, indexes_Pkm])
    if do_HMF: indexes = np.hstack([indexes, indexes_HMF])
    if do_VSF: indexes = np.hstack([indexes, indexes_VSF])
    all_bins = len(indexes)

    # fill the subcovariance
    new_Cov  = np.zeros((all_bins,all_bins), dtype=np.float64)
    for i,id1 in enumerate(indexes):
        for j,id2 in enumerate(indexes):
            new_Cov[i,j] = Cov[id1,id2]

    return new_Cov

# This routine reads and returns the covariance matrix
def read_covariance(f_Cov):
    f           = open(f_Cov, 'r') #read the header: number of bins of each probe
    bins_probes = np.array(f.readline().split()[1:], dtype=np.int32);  f.close()
    X1, X2, Cov = np.loadtxt(f_Cov, unpack=True) #read covariance
    bins        = int(round(np.sqrt(len(X1))))   #find the number of bins
    Cov         = np.reshape(Cov, (bins,bins))   #reshape covariance
    X           = X2[:bins]                      #get the values of the X-axis
    return bins_probes, X, Cov

# This routine computes the covariance matrix of all probes
def covariance(realizations, BoxSize, snapnum, root_data, root_results,
               kmax_m, Pkmc, folder_Pkm,                           #Pkm parameters
               Radii, VSFmc, delta_th_void, folder_VSF, group_bins_VSF,                           #VSF parameters    
               HMF_bins, Nmin, Nmax, folder_HMF):            #HMF parameters

    # find redshift and define the working folders
    z = {4:0, 3:0.5, 2:1, 1:2, 0:3}[snapnum]
    folder = '/%s/covariance'%root_results
    if myrank==0: os.makedirs(folder, exist_ok=1)
    comm.Barrier()

    # perform checks and compute binning
    Pkm_bins, mean_km   = binning_Pk(folder_Pkm,  z, kmax_m)  #Pkm
    bins_HMF, dN, Nmean = binning_HMF(Nmin, Nmax, HMF_bins)   #HMF
    VSF_bins, dR, Rmean = binning_VSF(folder_VSF, z, Radii, VSFmc, delta_th_void, group_bins_VSF)                  #VSF

    # find the vector with the X-values (only needed to save the covariances)
    mean = np.hstack([mean_km, Nmean, Rmean])

    # find the suffix name
    suffix = '%d_Pk%s_%.2f_HMF_%.1e_%.1e_%d_VSF%s_%.1e_%.1e_%.1f_%d_z=%s.txt'\
        %(realizations, Pkmc, kmax_m, Nmin, Nmax, HMF_bins, VSFmc, Radii[0], Radii[-1], delta_th_void,
          (len(Radii)-1)/group_bins_VSF, z)

    # compute the total number of bins
    bins = Pkm_bins + HMF_bins + VSF_bins
    if myrank==0:
        print ('\n%d bins: %d(Pk%s) + %d(HMF) + %d(VSF)\n'\
            %(bins, Pkm_bins, Pkmc, HMF_bins, VSF_bins))

    # find the name of the output files; if they exist, just read them
    fout  = '%s/Cov_%s'%(folder,suffix)
    fout1 = '%s/Cov_norm_%s'%(folder,suffix)
    if os.path.exists(fout):  return read_covariance(fout)

    ########### read all the data ############
    if myrank==0:  print ('Reading data at z=%s...'%z)

    # define the array hosting all the data
    data_p = np.zeros((realizations,bins), dtype=np.float64) #data each cpu reads
    data   = np.zeros((realizations,bins), dtype=np.float64) #Matrix with all data

    # when using less than 15000 relaizations we want to choose which relaizations to use
    # for specific choices of 10000 and 5000 us the start and end respectively (i.e. for 500 start from realization 1000)
    realization0 = 0
    if realizations < 7500:
        realization0 = 15000 - realizations
    
    # each cpu reads its corresponding data
    numbers = np.where(( realization0+np.arange(realizations) )%nprocs==myrank)[0]
    for i in numbers:
        #if i%1000==0:  print (i)
        print (i)
        Pkm_folder = '%s/fiducial/%d'%(folder_Pkm,i)
        HMF_folder = '%s/fiducial/%d'%(folder_HMF,i)
        VSF_folder = '%s/fiducial/%d'%(folder_VSF,i)
        dumb = np.array([], dtype=np.float64)
        dumb = np.hstack([dumb, read_Pkm_data(Pkm_folder,z,kmax_m,Pkmc)])
        dumb = np.hstack([dumb, read_HMF_data(HMF_folder,snapnum,bins_HMF,dN,BoxSize)])
        dumb = np.hstack([dumb, read_VSF_data(VSF_folder,z,Radii,VSFmc,delta_th_void,group_bins_VSF)])
        data_p[i] = dumb

    # join the data of the different cpus into a single matrix (only for master)
    comm.Reduce(data_p, data, root=0);  del data_p

    # compute the mean and std of the data
    data_mean, data_std = np.mean(data, axis=0), np.std(data,  axis=0)
    
    ############################################

    ########## compute the covariance ##########
    if myrank>0:  return 0,0,0
    print ('Computing the covariance at z=%s...'%z)

    # define the arrays containing the covariance and correlation matrices
    Cov = np.zeros((bins,bins), dtype=np.float64)
    Cor = np.zeros((bins,bins), dtype=np.float64)

    # compute the covariance matrix
    for i in range(bins):
        for j in range(i,bins):
            Cov[i,j] = np.sum((data[:,i]-data_mean[i])*(data[:,j]-data_mean[j])) 
            if j>i:  Cov[j,i] = Cov[i,j]
    Cov /= (realizations-1.0)

    # compute the normalized covariance matrix
    f = open(fout, 'w');  g = open(fout1, 'w')
    f.write('# %d %d %d\n'%(Pkm_bins, HMF_bins, VSF_bins)) #header (#-bins)
    g.write('# %d %d %d\n'%(Pkm_bins, HMF_bins, VSF_bins)) #header (#-bins)
    for i in range(bins):
        for j in range(bins):
            Cor[i,j] = Cov[i,j]/np.sqrt(Cov[i,i]*Cov[j,j])
            f.write('%.18e %.18e %.18e\n'%(mean[i], mean[j], Cov[i,j]))
            g.write('%.18e %.18e %.18e\n'%(mean[i], mean[j], Cor[i,j]))
    f.close();  g.close()

    # read and return the covariance
    return read_covariance(fout)
#######################################################################################
# This function computes the means, i.e. it computes the probe (by binning the HMF for example)
def means(cosmo,                                   # specify the cosmology to compute the mean for
          realizations, BoxSize, snapnum, root_data, root_results, probes,
          kmax_m, Pkmc, folder_Pkm,                      #Pkm parameters
          Radii, VSFmc, delta_th_void, folder_VSF, group_bins_VSF,      #VSF parameters    
          num_HMF_bins, Nmin, Nmax, folder_HMF):   #HMF parameters

    # find the corresponding redshift
    z = {4:0, 3:0.5, 2:1, 1:2, 0:3}[snapnum] 
    
    # perform checks and compute binning
    num_Pkm_bins, mean_km    = binning_Pk(folder_Pkm, z, kmax_m)      #Pkm
    bins_HMF, dN, mean_N     = binning_HMF(Nmin, Nmax, num_HMF_bins)  #HMF
    num_VSF_bins, dR, mean_R = binning_VSF(folder_VSF, z, Radii, VSFmc, delta_th_void, group_bins_VSF)   #VSF
    
    # find the suffixes (same for mean and deriv)
    suffix_Pkm = 'Pk_%s_%d_%.2f_z=%s.txt'%(Pkmc, realizations, kmax_m, z)
    suffix_HMF = 'HMF_%d_%.1e_%.1e_%d_z=%s.txt'%(realizations, Nmin,Nmax,num_HMF_bins, z)
    suffix_VSF = 'VSF%s_%d_%.1e_%.1e_%.1f_%d_z=%s.txt'%(VSFmc, realizations, Radii[0], Radii[-1], delta_th_void,
                                                 (len(Radii)-1)/group_bins_VSF, z)
    
    # create output folder if it does not exist
    folder = '%s/derivatives/%s'%(root_results, cosmo)
    if myrank==0: os.makedirs(folder, exist_ok=1)

    # do a loop over the different probes
    for probe in probes:

        comm.Barrier() #synchronize threads

        if   probe=='Pkm':  bins, mean, suffix = num_Pkm_bins, mean_km, suffix_Pkm
        elif probe=='HMF':  bins, mean, suffix = num_HMF_bins, mean_N,  suffix_HMF
        elif probe=='VSF':  bins, mean, suffix = num_VSF_bins, mean_R,  suffix_VSF

        # find output file name
        fout = '%s/mean_%s'%(folder,suffix)
        if os.path.exists(fout):  continue

        # define the array hosting the data
        data_p = np.zeros((realizations,bins), dtype=np.float64) 
        data   = np.zeros((realizations,bins), dtype=np.float64) 

        # do a loop over the different realizations
        count, count_p = np.array([0]), np.array([0])
        numbers = np.where(np.arange(realizations)%nprocs==myrank)[0]
        for i in numbers:
            Pkm_folder = os.path.join(folder_Pkm,cosmo,str(i))
            HMF_folder = os.path.join(folder_HMF,cosmo,str(i))
            VSF_folder = os.path.join(folder_VSF,cosmo,str(i))

            if   probe=='Pkm':  
                data_p[i] = read_Pkm_data(Pkm_folder, z, kmax_m, Pkmc)
            elif probe=='HMF':
                data_p[i] = read_HMF_data(HMF_folder, snapnum, bins_HMF, dN, BoxSize)
            elif probe=='VSF':
                data_p[i] = read_VSF_data(VSF_folder, z, Radii, VSFmc, delta_th_void, group_bins_VSF)
            count_p[0] += 1

        # join all data into a single matrix (only for master)
        comm.Reduce(data_p,  data,  root=0)
        comm.Reduce(count_p, count, root=0)
        #if myrank>0:  continue   # changes this because of below HMF M bin computaton
        if myrank == 0:
            # save results to file (only master)
            data_mean, data_std = np.mean(data, axis=0), np.std(data, axis=0)
            np.savetxt(fout, np.transpose([mean, data_mean, data_std]))
            print ('%d realizations found for %s probe %s'%(count,cosmo,probe))
        
        # for HMF;Om also compute the deirvative wrt M directly:
        if probe == 'HMF' and 'Om' in cosmo:
            
            mp_fid = 656562367483.9242
            Mmin = Nmin * mp_fid
            Mmax = Nmax * mp_fid
            
            # define the M bins 
            Mbins_HMF, dM, mean_M = binning_HMF_Mbin(Mmin, Mmax, num_HMF_bins)
            bins, mean, suffix = num_HMF_bins, mean_M,  suffix_HMF
            
            # just copy above basically:
            # find output file name [not the M in fname now]
            fout = '%s/meanM_%s'%(folder,suffix)
            if os.path.exists(fout):  continue

            # define the array hosting the data
            data_p = np.zeros((realizations,bins), dtype=np.float64) 
            data   = np.zeros((realizations,bins), dtype=np.float64) 

            # do a loop over the different realizations
            count, count_p = np.array([0]), np.array([0])
            numbers = np.where(np.arange(realizations)%nprocs==myrank)[0]
            for i in numbers:
                HMF_folder = os.path.join(folder_HMF,cosmo,str(i)) 
                data_p[i] = read_HMF_data_Mbin(HMF_folder, snapnum, Mbins_HMF, dM, BoxSize)
                count_p[0] += 1

            # join all data into a single matrix (only for master)
            comm.Reduce(data_p,  data,  root=0)
            comm.Reduce(count_p, count, root=0)
            if myrank>0:  continue

            # save results to file (only master)
            data_mean, data_std = np.mean(data, axis=0), np.std(data, axis=0)
            np.savetxt(fout, np.transpose([mean, data_mean, data_std]))

#######################################################################################

#######################################################################################
# This functions computes the derivatives with respect to the different parameters
def derivatives(realizations, BoxSize, snapnum, root_data, root_results, probes, 
                kmax_m, Pkmc, folder_Pkm,                      #Pkm parameters
                Radii, VSFmc, delta_th_void, folder_VSF, group_bins_VSF,      #VSF parameters    
                num_HMF_bins, Nmin, Nmax, folder_HMF):   #HMF parameters

    #cosmologies = ['Om_p/',  'Ob_p/',  'Ob2_p/', 'h_p/', 'ns_p/', 's8_p/', 
    #               'Om_m/',  'Ob_m/',  'Ob2_m/', 'h_m/', 'ns_m/', 's8_m/', 
    #               'Mnu_p/', 'Mnu_pp/', 'Mnu_ppp/', 'fiducial/']
    cosmologies = ['Om_p/',  'Ob2_p/', 'h_p/', 'ns_p/', 's8_p/', 
                   'Om_m/',  'Ob2_m/', 'h_m/', 'ns_m/', 's8_m/', 
                   'Mnu_p/', 'Mnu_pp/', 'Mnu_ppp/', 
                   'fiducial/', 'fiducial_ZA/']
    
    #parameters = ['Om',  'Ob',   'Ob2',  'h',   'ns',  's8',   'Mnu']
    #diffs      = [0.01,  0.001,  0.002,  0.02,  0.02,  0.015,  0.10]
    parameters = ['Om',  'Ob2',  'h',   'ns',  's8',   'Mnu']
    diffs      = [0.01,  0.002,  0.02,  0.02,  0.015,  0.10]
    params_fid = {'Om':0.3175, 'Ob2':0.049,  'h':0.6711, 'ns':0.9624, 's8':0.834, 'Mnu':0}
    # find the corresponding redshift
    z = {4:0, 3:0.5, 2:1, 1:2, 0:3}[snapnum] 

    # find the suffixes
    suffix_Pkm = 'Pk_%s_%d_%.2f_z=%s.txt'%(Pkmc, realizations, kmax_m, z)
    suffix_HMF_tmp = 'HMF_%d_%.1e_%.1e_%d_z=%s.txt'                             # useful to define for the HMF correction
    suffix_HMF = suffix_HMF_tmp % (realizations, Nmin,Nmax,num_HMF_bins, z)
    suffix_VSF = 'VSF%s_%d_%.1e_%.1e_%.1f_%d_z=%s.txt'%(VSFmc, realizations, Radii[0], Radii[-1], delta_th_void,
                                                 (len(Radii)-1)/group_bins_VSF, z)

    # create the folder that will contain the derivatives (and means)
    folder = '%s/derivatives/'%root_results
    if myrank==0: os.makedirs(folder, exist_ok=1)
        
    ##### compute means ####
    # do a loop over the different cosmologies
    for cosmo in cosmologies:
        means(cosmo,                                   # specify the cosmology to compute the mean for
              realizations, BoxSize, snapnum, root_data, root_results, probes,
              kmax_m, Pkmc, folder_Pkm,                      # Pkm parameters
              Radii, VSFmc, delta_th_void, folder_VSF, group_bins_VSF,      # VSF parameters    
              num_HMF_bins, Nmin, Nmax, folder_HMF)    # HMF parameters
    
    # also compute the mean for fiducial using all 15k sims (needed for HMF correction)
    # FIXME would be more effificient to do this when computing the covariance matrix
    # [NOTE: this is now unused and commented out]
    #means('fiducial',
    #      15000, BoxSize, snapnum, root_data, root_results, ['HMF'],
    #      kmax_m, folder_Pkm,                      # Pkm parameters
    #      Radii, VSFmc, delta_th_void, folder_VSF, group_bins_VSF,      # VSF parameters    
    #      num_HMF_bins, Nmin, Nmax, folder_HMF)    # HMF parameters
    ###########################

    ##### derivatives #####
    if myrank>0:  return 0  #only master does this
    
    # do a loop over the different probes
    for probe in probes:

        # create output folder if it does not exists
        folder = '%s/derivatives/%s'%(root_results,probe)
        os.makedirs(folder, exist_ok=1)

        if   probe=='Pkm': suffix = suffix_Pkm
        elif probe=='HMF': suffix = suffix_HMF
        elif probe=='VSF': suffix = suffix_VSF
            
        for parameter,diff in zip(parameters,diffs):
            
            # find name of output file
            fout = '%s/derivative_%s_%s'%(folder, parameter, suffix)
            if os.path.exists(fout) and probe != 'HMF':  continue

            if parameter=='Mnu':
                f0 = '%s/derivatives/fiducial_ZA/mean_%s'%(root_results, suffix)
                f1 = '%s/derivatives/Mnu_p/mean_%s'%(root_results,    suffix)
                f2 = '%s/derivatives/Mnu_pp/mean_%s'%(root_results,   suffix)
                f4 = '%s/derivatives/Mnu_ppp/mean_%s'%(root_results,  suffix)
                X, Y0, dY0 = np.loadtxt(f0, unpack=True)
                X, Y1, dY1 = np.loadtxt(f1, unpack=True)
                X, Y2, dY2 = np.loadtxt(f2, unpack=True)
                X, Y4, dY4 = np.loadtxt(f4, unpack=True)
                
                deriv11 = (Y1 - Y0)/(1.0*diff)
                deriv12 = (Y2 - Y0)/(2.0*diff)
                deriv13 = (Y4 - Y0)/(4.0*diff)
                deriv21 = (4.0*Y1 - 3.0*Y0 - Y2)/(2.0*diff)
                deriv22 = (4.0*Y2 - 3.0*Y0 - Y4)/(2.0*2.0*diff)
                deriv3 = (Y4 - 12.0*Y2 + 32.0*Y1 - 21.0*Y0)/(12.0*diff)
                
                # save corrected version
                if probe == 'HMF':
                    # first save uncorrected version
                    np.savetxt('%s/derivativeN_Mnu_0.1-0.0_%s'%(folder,suffix), 
                           np.transpose([X, deriv11]))
                    np.savetxt('%s/derivativeN_Mnu_0.2-0.0_%s'%(folder,suffix), 
                               np.transpose([X, deriv12]))
                    np.savetxt('%s/derivativeN_Mnu_0.4-0.0_%s'%(folder,suffix), 
                               np.transpose([X, deriv13]))
                    np.savetxt('%s/derivativeN_Mnu_0.2-0.1-0.0_%s'%(folder,suffix), 
                               np.transpose([X, deriv21]))
                    np.savetxt('%s/derivativeN_Mnu_0.4-0.2-0.0_%s'%(folder,suffix), 
                               np.transpose([X, deriv22]))
                    np.savetxt('%s/derivativeN_Mnu_0.4-0.2-0.1-0.0_%s'%(folder,suffix),    # this will be corrected later, otherwise it wont be
                               np.transpose([X, deriv3]))

                    # apply N->M bin correction
                    dHMFdlnN = HMF_deriv_correction('%s/derivatives/fiducial_ZA/mean_%s'%(root_results, suffix))       # using ZA for neutrinos, maybe change? FIXME
                    correction = dHMFdlnN / (params_fid['Om'] * 93.14 * params_fid['h']**2)

                    deriv11 += correction
                    deriv12 += correction
                    deriv13 += correction
                    deriv21 += correction
                    deriv22 += correction
                    deriv3 += correction

                np.savetxt('%s/derivative_Mnu_0.1-0.0_%s'%(folder,suffix), 
                            np.transpose([X, deriv11]))
                np.savetxt('%s/derivative_Mnu_0.2-0.0_%s'%(folder,suffix), 
                           np.transpose([X, deriv12]))
                np.savetxt('%s/derivative_Mnu_0.4-0.0_%s'%(folder,suffix), 
                           np.transpose([X, deriv13]))
                np.savetxt('%s/derivative_Mnu_0.2-0.1-0.0_%s'%(folder,suffix), 
                           np.transpose([X, deriv21]))
                np.savetxt('%s/derivative_Mnu_0.4-0.2-0.0_%s'%(folder,suffix), 
                           np.transpose([X, deriv22]))
                np.savetxt('%s/derivative_Mnu_0.4-0.2-0.1-0.0_%s'%(folder,suffix),    # this will be corrected later, otherwise it wont be
                            np.transpose([X, deriv3]))
                
                deriv = deriv22

            else:
                f1 = '%s/derivatives/%s_m/mean_%s'%(root_results, parameter, suffix)
                f2 = '%s/derivatives/%s_p/mean_%s'%(root_results, parameter, suffix)
                X, Ym, dYm = np.loadtxt(f1, unpack=True)
                X, Yp, dYp = np.loadtxt(f2, unpack=True)
                deriv = (Yp - Ym)/(2.0*diff)
            
            # apply the HMF correction to Om
            if probe == 'HMF' and parameter == 'Om':
                np.savetxt('%s/derivativeN_%s_%s'%(folder, parameter, suffix), 
                       np.transpose([X, deriv]))
                dHMFdlnN = HMF_deriv_correction('%s/derivatives/fiducial/mean_%s'%(root_results, suffix))
                #dHMFdlnN = HMF_deriv_correction('%s/derivatives/fiducial/mean_%s'%(root_results, suffix_HMF_tmp%(15000, Nmin,Nmax,num_HMF_bins, z)))  # use 15k fiducial sims for Om correction
                correction = - dHMFdlnN / params_fid['Om'] 
                deriv += correction
                
                # also compute the HMF;Om derivative directly from Mbinning (meanM):
                # use double letter to avoid overwrting the corrected 'derivative'
                ff1 = '%s/derivatives/%s_m/meanM_%s'%(root_results, parameter, suffix)
                ff2 = '%s/derivatives/%s_p/meanM_%s'%(root_results, parameter, suffix)
                XX, YYm, dYYm = np.loadtxt(ff1, unpack=True)
                XX, YYp, dYYp = np.loadtxt(ff2, unpack=True)
                dderiv = (YYp - YYm)/(2.0*diff)
                np.savetxt('%s/derivativeM_%s_%s'%(folder, parameter, suffix), 
                       np.transpose([XX, dderiv]))
            
            np.savetxt('%s/derivative_%s_%s'%(folder, parameter, suffix), 
                       np.transpose([X, deriv]))
#######################################################################################



#######################################################################################
######################################## Pk ###########################################
# This routine determines the number of bins until kmax
def binning_Pk(folder,z,kmax):
    if kmax==0:  raise Exception('kmax have to be larger than 0')
    fin = os.path.join(folder,'fiducial/0/Pk_m_z=%s.txt'%z)         #take the first realization
    k, Pk = np.loadtxt(fin, unpack=True)          #read its power spectrum
    indexes = np.where(k<=kmax)[0]
    return len(indexes), k[indexes]

# This routine reads the Pk_m of a given realization
def read_Pkm_data(folder,z,kmax,Pkmc):
    if   Pkmc == 'c' and 'Mnu' in folder: fin = os.path.join(folder,'Pk_cb_z=%s.txt'%z)
    else:                                 fin = os.path.join(folder,'Pk_m_z=%s.txt'%z)
    k, Pk_p = np.loadtxt(fin, unpack=True)
    indexes = np.where(k<kmax)[0]
    return Pk_p[indexes] / 1e10

#######################################################################################
#######################################################################################

#######################################################################################
####################################### VSF ###########################################
# This function computes the binning of the VSF... everything is hardcoded into the catalogue, so careful
# lots of asseritons to check

def _read_VSF_data(folder, z, Radii, VSFmc, delta_th_void, group_bins=1):   
    """
    folder must include 'cosmo/sim#/'
    VSFmc: compute VSf for 'm' or 'c'? Only applies for if folder has Mnu... otherwise always m
    """
    if VSFmc == 'c' and 'Mnu' not in folder:
        _VSFmc = 'm'
    else:
        _VSFmc = VSFmc
    
    if delta_th_void == 0.5:
        fin = os.path.join(folder, 'void_catalogue_%s_z=%s.hdf5'%(_VSFmc,z))
        # FIXME: assert this is really 0.5
    elif delta_th_void == 0.3:# or delta_th_void == 0.7:
        fin = os.path.join(folder, 'void_catalogue_%s_%.1f_z=%s.hdf5'%(_VSFmc,delta_th_void,z))
    elif delta_th_void == 0.7:   # i will assume we want the new data with extra bins... really could be more general... FIXME
        fin = os.path.join(folder, 'void_catalogue_%s_%.1f_new_z=%s.hdf5'%(_VSFmc,delta_th_void,z))

    f = h5py.File(fin, 'r')
    Rmean = f['VSF_Rbins'][:]
    VSF_p = f['VSF'][:];  f.close()
    
    # the catalogue uses bins of decreasing R, 
    # so check if this is indeed the case and if so reverse
    if Rmean[0] > Rmean[1]:
        Rmean = Rmean[::-1]
        VSF_p = VSF_p[::-1]

    if len(Radii)-1 != len(Rmean) and group_bins != 1:
        raise Exception("Cutting and grouping bins in one analysis not supported.")
    
    # if you cut Radii:
    if len(Radii)-1 < len(Rmean): 
        
        bin_check = 0
        for ii in range( len(Rmean) - (len(Radii)-1) ):
            if ((Radii[1:] + Radii[:-1])/2 == Rmean[ii : len(Radii)-1+ii]).all():    # incase you cut rmin and rmax... shift along to check which bins
                Rmean = Rmean[ii : len(Radii)-1+ii]
                VSF_p = VSF_p[ii : len(Radii)-1+ii]
                bin_check = 1
                break
        if not bin_check:
            raise Exception("Input bins don't match bins in catalogue.")
        
    # group adjacent bins to investigate effects of using less bins
    if group_bins > 1:
        if len(Radii) % 2 == 0:
            raise Exception("WARNING. Best to use an odd number of bin edges (or even number of bins) when grouping bins. \
                             Otherwise final bin cut off.")
        
        Rmean = np.mean(Rmean.reshape(-1,group_bins), axis=-1) # new bin centers are just the mean of each set of grouped bin centers
        VSF_p = np.mean(VSF_p.reshape(-1,group_bins), axis=-1) # take mean, as this rescales the dR in the VSF denominator appropriately
    
    return Rmean, VSF_p
    
def binning_VSF(folder, z, Radii, VSFmc, delta_th_void, group_bins=1): 
    fin = os.path.join(folder,'fiducial/0')    # as with Pk, a quirk is that folder for binning function means the Void dir so choose a random thing to get binnning (which should be consistent)
    Rmean = _read_VSF_data(fin, z, Radii, VSFmc, delta_th_void, group_bins)[0]
    dR    = Rmean[1:] - Rmean[:-1]       #size of the bin
    return len(Rmean), dR, Rmean

def read_VSF_data(folder, z, Radii, VSFmc, delta_th_void, group_bins=1):
    VSF_p = _read_VSF_data(folder, z, Radii, VSFmc, delta_th_void, group_bins)[1]
    return VSF_p
#######################################################################################
#######################################################################################

#######################################################################################
####################################### HMF ###########################################
# This function computes the binning of the HMF
def binning_HMF(Nmin, Nmax, HMF_bins):
    if HMF_bins==0 or Nmin==0 or Nmax==0:
        raise Exception('HMF_bins, Nmin and Nmax have to be larger than 0')
    bins_HMF = np.logspace(np.log10(Nmin), np.log10(Nmax), HMF_bins+1)
    dN       = bins_HMF[1:] - bins_HMF[:-1]       #size of the bin
    Nmean    = 0.5*(bins_HMF[1:] + bins_HMF[:-1]) #mean of the bin
    return bins_HMF,dN,Nmean

# This routine computes the HMF for a given realization
def read_HMF_data(snapdir, snapnum, bins_HMF, dN, BoxSize):
    FoF     = readfof.FoF_catalog(snapdir,snapnum,long_ids=False,
                                  swap=False,SFR=False,read_IDs=False)
    mass    = FoF.GroupMass*1e10  #Msun/h  
    part    = FoF.GroupLen        #number of particles in the halo
    p_mass  = mass[0]/part[0]     #mass of a single particle in Msun/h
    mass    = p_mass*(part*(1.0-part**(-0.6))) #corect FoF masses
    
    dlogN = np.log(bins_HMF[1:]) - np.log(bins_HMF[:-1])
    return np.histogram(part, bins=bins_HMF)[0]/(dlogN*BoxSize**3)#*1e12

########################

# This function computes the binning of the HMF in Mbins!
def binning_HMF_Mbin(Mmin, Mmax, HMF_bins):
    if HMF_bins==0 or Mmin==0 or Mmax==0:
        raise Exception('HMF_bins, Mmin and Mmax have to be larger than 0')
    bins_HMF = np.logspace(np.log10(Mmin), np.log10(Mmax), HMF_bins+1)
    dM       = bins_HMF[1:] - bins_HMF[:-1]       #size of the bin
    Mmean    = 0.5*(bins_HMF[1:] + bins_HMF[:-1]) #mean of the bin
    return bins_HMF,dM,Mmean

# This routine computes the HMF for a given realization in Mbins
def read_HMF_data_Mbin(snapdir, snapnum, bins_HMF, dM, BoxSize):
    FoF     = readfof.FoF_catalog(snapdir,snapnum,long_ids=False,
                                  swap=False,SFR=False,read_IDs=False)
    mass    = FoF.GroupMass*1e10  #Msun/h  
    part    = FoF.GroupLen        #number of particles in the halo
    p_mass  = mass[0]/part[0]     #mass of a single particle in Msun/h
    mass    = p_mass*(part*(1.0-part**(-0.6))) #corect FoF masses
    
    print('N')
    
    dlogM = np.log(bins_HMF[1:]) - np.log(bins_HMF[:-1])
    return np.histogram(mass, bins=bins_HMF)[0]/(dlogM*BoxSize**3)#*1e12

#######################################################################################
#######################################################################################

from scipy.interpolate import UnivariateSpline as Spline

def HMF_deriv_correction_spline(f):   # f should be the fiducial measn filename.
    """ N is HMF bin centers (in N unites), H is HMF """
    N, H, _ = np.loadtxt(f, unpack=True)
    
    logN = np.log(N)
    logH = np.log(H)
    
    f_logH = Spline(logN, logH)                # splin fit in logs for goot fit
    f_H = lambda logN : np.exp(f_logH(logN))

    # compute derivative
    f_dlogH_dlogN = f_logH.derivative()
    # use chain rule
    f_dH_dlogN = lambda logN : f_dlogH_dlogN(logN) * f_H(logN)
    # evaluate on the deriv bin centers
    dH_dlogN = f_dH_dlogN(logN)
    
    return dH_dlogN   # need to multiple by dmp/dtheta to get totatl coerrection after!
    
def HMF_deriv_correction_FDM(f):   # f should be the fiducial measn filename.
    """ N is HMF bin centers (in N unites), H is HMF """
    N, H, _ = np.loadtxt(f, unpack=True)
    
    logN = np.log(N)
    logH = np.log(H)

    # compute derivative
    dlogH_dlogN = (logH[1:] - logH[:-1]) / (logN[1:] - logN[:-1])
    dlogH_dlogN = np.hstack((dlogH_dlogN, dlogH_dlogN[-1]))                # duplicate final entry for final bin useing bkwd diff scheme
    
    dH_dlogN = dlogH_dlogN * np.exp(logH)
    
    return dH_dlogN   # need to multiple by dmp/dtheta to get totatl coerrection after!
    
    
def HMF_deriv_correction_central(f):   # f should be the fiducial measn filename.
    """ N is HMF bin centers (in N unites), H is HMF """
    N, H, _ = np.loadtxt(f, unpack=True)
    
    logN = np.log(N)
    logH = np.log(H)
    
    dlogH_dlogN = np.empty_like(logN)

    # compute derivative using CDS for all except edge bins:
    dlogH_dlogN[1:-1] = (logH[2:] - logH[:-2]) / (logN[2:] - logN[:-2])
    
    # FDS on left bin
    dlogH_dlogN[0] = (logH[1] - logH[0]) / (logN[1] - logN[0])                # duplicate final entry for final bin useing bkwd diff scheme
    
    # BDS on right bin
    dlogH_dlogN[-1] = (logH[-1] - logH[-2]) / (logN[-1] - logN[-2])
    
    dH_dlogN = dlogH_dlogN * np.exp(logH)
    
    return dH_dlogN   # need to multiple by dmp/dtheta to get totatl coerrection after!

def HMF_deriv_correction(f):   # f should be the fiducial measn filename.
    """ N is HMF bin centers (in N unites), H is HMF """
    N, H, _ = np.loadtxt(f, unpack=True)
    
    logN = np.log(N)
    logH = np.log(H)
    
    f_logH = Spline(logN, logH)                # spline fit in logs for goot fit
    f_H = lambda logN : np.exp(f_logH(logN))

    # compute derivative
    f_dlogH_dlogN = f_logH.derivative()
    # use chain rule
    f_dH_dlogN = lambda logN : f_dlogH_dlogN(logN) * f_H(logN)
    # evaluate on the deriv bin centers
    dH_dlogN = f_dH_dlogN(logN)
    
    # use FDM on leftmost bin
    dH_dlogN[0] = (logH[1] - logH[0]) / (logN[1] - logN[0]) * np.exp(logH[0])
    
    return dH_dlogN   # need to multiply by dmp/dtheta to get totatl coerrection after!
