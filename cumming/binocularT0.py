import os
import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import Dataset
import NDNT.utils as utils
#from NDNT.utils import download_file, ensure_dir
from copy import deepcopy
import h5py
from NTdatasets.sensory_base import SensoryBase

class binocular_singleT(SensoryBase):

    def __init__(self, expt_num=None, time_embed=0, num_lags=12, skip_lags=0, verbose=True, **kwargs):
        """
        Args: 
            expt_num: the experiment index
            time_embed: whether to time-embed the stimulus or not
            num_lags: the number of lags to use in time-embedding
            skip_lags: shift stim to throw out early lags
            filename: currently the pre-processed matlab file from Dan's old-style format
            **kwargs: non-dataset specific arguments that get passed into SensoryBase

            Inherited (but needed from SensoryBase init):
                datadir, 
                time_embed=2,  # 0 is no time embedding, 1 is time_embedding with get_item, 2 is pre-time_embedded
                include_MUs = False,
                drift_interval = None,
            """

        assert expt_num is not None, "Binocular experiment number needed (expt_n)."
        filename = 'Bexpt'+ str(expt_num) + '.mat'

        # call parent constructor
        super().__init__(
            filename, 
            num_lags=num_lags, time_embed=time_embed,
            **kwargs)

        self.dt = 0.01 #100Hz
        self.upsample = 1
        self.robs_upsample = None
        self.dfs_upsample = None
        self.stim_upsample = None
        self.spike_times = np.zeros((0,2))

        if verbose:
            print( "Loading", self.datadir + filename)

        # Store stimulus trimmed to 36 - 36 binocular configuration
        stim_trim = np.concatenate( (np.arange(3,39), np.arange(45,81)))
        Bmatdat = h5py.File(self.datadir + filename, 'r')
        self.Bstim = np.transpose(Bmatdat['stimulus'])[:, stim_trim]

        self.dims = [1, 72, 1, 1]
        self.divide_stim=False
        # Note Bstim is stored as numpy

        # Responses
        RobsSU = np.transpose(Bmatdat['binned_SU'])
        dfsSU = np.transpose(Bmatdat['data_filters'])
        self.NT, self.numSUs = RobsSU.shape

        RobsMU = np.transpose(Bmatdat['binned_MUA'])
        self.numMUs = RobsMU.shape[1]
    
        if self.include_MUs:
            self.NC = self.numSUs + self.numMUs

            dfsMU = np.transpose(Bmatdat['data_filters_MUA'])

            self.robs = torch.tensor(
                np.concatenate( (RobsSU, RobsMU), axis=1 ),
                dtype=torch.float32 )
            self.dfs = torch.tensor( 
                np.concatenate( (dfsSU, dfsMU), axis=1 ),
                dtype=torch.float32 )
        else:
            self.NC = self.numSUs
            self.robs = torch.tensor(RobsSU, dtype=torch.float32 )
            self.dfs = torch.tensor(dfsSU, dtype=torch.float32 )

        # used_inds and XV
        used_inds = np.add(np.array(Bmatdat['used_inds'], dtype=np.int32)[0,:], -1) # note adjustment for python v matlab indexing
        # implement within datafilters:
        df_mult = np.zeros([self.NT,1], dtype=np.float32)
        df_mult[used_inds] = 1.0
        self.dfs *= df_mult

        #Trial Starts and Ends
        start_times_refs = Bmatdat['trial_data']['start_times'][0]
        end_times_refs = Bmatdat['trial_data']['end_times'][0]

        trial_starts = np.zeros(len(start_times_refs))
        trial_ends = np.zeros(len(end_times_refs))

        for i in range(len(start_times_refs)):
            trial_starts[i] = Bmatdat[start_times_refs[i]][0][0] #getting data out of mat file structure is strange sometimes
            trial_ends[i] = Bmatdat[end_times_refs[i]][0][0]

        #Spike Times (only SUs)
        num_probes = len(Bmatdat['spike_data']['SU_spk_times'][0])
        spike_times_tmp = np.zeros((0,2))
        cell_num = 0
        n_filter = 0

        for i in range(num_probes):
            
            SU_spike_times_ref = Bmatdat['spike_data']['SU_spk_times'][0][i]
            unlabeled_spike_times = Bmatdat[SU_spike_times_ref][0]
            
            if np.isscalar(unlabeled_spike_times):
                # probes with no data have a 0 instead of the spike times array
                continue
            
            n_filter += 1
            if np.sum(Bmatdat['data_filters'][n_filter-1]) == 0:
                #somtimes cell is excluded entirely in dfs, but still has spike times
                continue

            spike_num = len(unlabeled_spike_times)
            cell_nums = cell_num*np.ones(spike_num) #for labeling spikes
            cell_num += 1
            
            labeled_spike_times = np.stack((cell_nums,unlabeled_spike_times), axis=1)
            spike_times_tmp = np.concatenate((spike_times_tmp,labeled_spike_times),axis=0)
        

        print(' Spike Times Processing')
        Ntrials = len(Bmatdat['time_data']['trial_flip_inds'])
        for i in range(Ntrials): #loop over number of trials
            while np.isnan(trial_starts[i]) or np.isnan(trial_ends[i]): #at least one experiment had an extra nan end time that messed up training
                trial_starts = np.delete(trial_starts, i)
                trial_ends = np.delete(trial_ends, i)

            t0 = trial_starts[i]
            t_end = trial_ends[i]
            spk_tr_inds = np.where((spike_times_tmp[:,1]>=t0) & (spike_times_tmp[:,1]<=t_end))[0]#find spikes in a trial
            spike_times_tmp[spk_tr_inds,1] -= trial_starts[i]# subtract the trial start time from those spike times
            
            spike_tot_times = spike_times_tmp[spk_tr_inds]
            spike_tot_times[:,1] += (Bmatdat['time_data']['trial_flip_inds'][i]-1)*self.dt #add trial start times relative to time bins 
            
            self.spike_times = np.concatenate([self.spike_times,deepcopy(spike_tot_times)], axis=0)
        
        # for i in np.unique(self.spike_times[:,0]): #debugging
        #     inds = np.where(self.spike_times[:,0]==i)[0]
        #     sum = len(inds)
        #     print("number of spikes for cell %i: %i" % (i, sum))

        # self.Ui_analog = Bmatdat['Ui_analog'][:,0]  # these are automatically in register
        # self.XiA_analog = Bmatdat['XiA_analog'][:,0]
        # self.XiB_analog = Bmatdat['XiB_analog'][:,0]
        # # two cross-validation datasets -- for now combine
        # self.Xi_analog = self.XiA_analog+self.XiB_analog  # since they are non-overlapping, will make 1 in both places

        # # Derive full-dataset Ui and Xi from analog values
        # self.used_inds = used_inds
        # self.train_inds = np.intersect1d(used_inds, np.where(self.Ui_analog > 0)[0])
        # self.val_inds = np.intersect1d(used_inds, np.where(self.Xi_analog > 0)[0])
        # self.val_indsA = np.intersect1d(used_inds, np.where(self.XiA_analog > 0)[0])
        # self.val_indsB = np.intersect1d(used_inds, np.where(self.XiB_analog > 0)[0])

        dispt_raw = np.transpose(Bmatdat['all_disps'])[:,0]
        # this has the actual disparity values, which are at the resolution of single bars, and centered around the neurons
        # disparity (sometime shifted to drive neurons well)
        # Sometimes a slightly disparity is used, so it helps to round the values at some resolution
        self.dispt = np.round(dispt_raw*100)/100
        # Fix expt10
        if expt_num == 10:  # make the uncommon disparity (at the extreme) into uncorrelated, which is it anyway...
            print('  dispt-fix for expt 10') 
            self.dispt[self.dispt > 0.5] = -1005

        self.frs = np.transpose(Bmatdat['all_Frs'])[:,0]
        self.corrt = np.transpose(Bmatdat['all_corrs'])[:,0]
        # Make dispt consistent with corrt (early experiments had dispt labeled incorrectly)
        corr_funny = np.where((self.corrt == 0) & (self.dispt != -1005))[0]
        if len(corr_funny) > 0:
            print( "Warning: %d indices have corr=0 but labeled disparity."%len(corr_funny) )
            self.dispt[corr_funny] = -1005

        self.disp_list = np.unique(self.dispt)
        # where it is -1009 this corresponds to a blank frame
        # where it is -1005 this corresponds to uncorrelated images between the eyes

        # if Bmatdat['rep_inds'] is None:
        #     #rep_inds = [None]*numSUs
        #     rep_inds = None
        # elif len(Bmatdat['rep_inds'][0][0]) < 10:
        #     rep_inds = None
        # else:
        #     rep_inds = []
        #     for cc in range(self.numSUs):
        #         rep_inds.append( np.add(Bmatdat['rep_inds'][0][cc], -1) ) 
        # self.rep_inds = rep_inds

        if verbose:
            print( "Expt %d: %d SUs, %d total units, %d out of %d time points used."%(expt_num, self.numSUs, self.NC, len(used_inds), self.NT))

        self.prepare_stim( time_embed=time_embed, skip_lags=skip_lags, num_lags=num_lags, verbose=verbose)
    # END binocular_single.__init__

    def prepare_stim( self, time_embed=0, skip_lags=None, num_lags=None, verbose=True ):
        """
        Prepare stimulus for dataset.

        Args:
            time_embed: whether to time-embed the stimulus or not
            skip_lags: shift stim to throw out early lags
            num_lags: the number of lags to use in time-embedding

        Returns:
            None
        """
        if skip_lags is not None:  
            self.skip_lags = skip_lags
            
        # Shift stimulus by skip_lags (note this was prev multiplied by DF so will be valid)
        stim = deepcopy(self.Bstim)
        assert self.skip_lags >= 0, "Negative skip_lags does not make sense"
        if self.skip_lags > 0:
            stim[self.skip_lags:, :] = deepcopy( stim[:-self.skip_lags, :] )
            stim[:self.skip_lags, :] = 0.0

        self.stim_dims = deepcopy(self.dims)
        if time_embed == 0:
            self.stim = torch.tensor( self.Bstim, dtype=torch.float32 )
            self.stim_dims = deepcopy(self.dims)
        else:
            if num_lags is None:
                # then read from dataset (already set):
                num_lags = self.num_lags
            self.stim = self.time_embedding( stim=stim, nlags=num_lags, verbose=verbose )
            # This will return a torch-tensor
            self.stim_dims[3] = num_lags
    # END binocular_single.prepare_stim()

    def separate_eyes(self, val=True):
        """
        Separate the stimulus into left and right eyes.

        Args:
            val: whether to separate the stimulus or not

        Returns:
            None
        """
        NX = self.stim_dims[1]//2
        stim = self.stim.reshape([self.NT, 2*NX, self.num_lags])
        self.stimL = stim[:, :NX, :].reshape([self.NT, -1])
        self.stimR = stim[:, NX:, :].reshape([self.NT, -1])
        self.divide_stim = val

    def __getitem__(self, idx):

        idx = self.index_to_array(idx,self.NT)
        if self.upsample > 1:
            new_index = (np.repeat(idx[:, None]*self.upsample, self.upsample, axis=1)+ np.arange(self.upsample)[None,:]).reshape([-1])

            stim = self.stim_upsample
            robs = self.robs_upsample
            dfs = self.dfs_upsample
        else:
            stim = self.stim
            robs = self.robs
            dfs = self.dfs

        if len(self.cells_out) == 0:
            out = {
                'stim': stim[idx, :], 
                'robs': robs[idx, :],
                'dfs': dfs[idx, :]}
            #if self.speckled:
            #    out['Mval'] = self.Mval[idx, :]
            #    out['Mtrn'] = self.Mtrn[idx, :]
        else:
            robs_tmp =  robs[:, self.cells_out]
            dfs_tmp =  dfs[:, self.cells_out]
            out = {
                    'stim': stim[idx, :], 
                    'robs': robs_tmp[idx, :],
                    'dfs': dfs_tmp[idx, :]}
            
        if self.divide_stim:
            out['stimL'] = self.stimL[idx, :]
            out['stimR'] = self.stimR[idx, :]

        if self.Xdrift is not None:
            out['Xdrift'] = self.Xdrift[idx, :]

        #if len(self.covariates) > 0:
        #   self.append_covariates( out, idx)

        return out
    # END binocular_single.__getitem()

    def set_upsample(self, frac):
        """
        This sets upsample flag and generates a higher-time resolution Robs. Note it will also automatically scale up
        the num_lags associated with the dataset, but note that this will have to be adjusted in other places too.
        
        Args:
            frac: integer amount of upsampling past frame resolution

        Returns:
            None, but modifies self.upsample and self.robs_upsample
        """
        assert frac >= 1, "frac must be a positive integer"

        upsample_mult = frac/self.upsample
        if upsample_mult > 1:
            print( "  Upsampling by %d: changing num_lags with dataset to %d"%(frac, self.num_lags*upsample_mult) )
            
        orig_lags = int(self.num_lags/self.upsample)
        self.num_lags = int(self.num_lags*upsample_mult)

        self.upsample = frac
        if frac == 1:
            self.robs_upsample = None
            self.dfs_upsample = None
            self.stim_upsample = None
            return
        else:
            assert self.spike_times[:,0] is not None, "No spike time information in dataset."
        
        dt = self.dt/frac
        self.robs_upsample = np.zeros([self.NT*frac, self.NC], dtype=np.uint8 )

        for cc in range(self.NC):
            a = np.where(self.spike_times[:,0] == cc)[0] 
            if len(a) > 0:
                robs_up = np.histogram(self.spike_times[a,1], np.arange(self.NT*frac+1)*dt)[0]
                # print(robs_up.shape, self.robs_upsample[:, cc].shape)
                self.robs_upsample[:, cc] = robs_up.astype(np.uint8)

        if not self.time_embed:
            self.stim_upsample = np.repeat(self.stim,frac,axis=0)
        else:
            self.stim_upsample = self.time_embedding(np.repeat(self.stim[:,::orig_lags],frac,axis=0))
            self.stim_dims[3] = self.num_lags
        
        self.dfs_upsample = np.repeat(self.dfs,frac,axis=0)

        if self.device is None:
            device = torch.device("cpu")
        else:
            device = self.device

        if type(self.robs_upsample) != torch.Tensor:
            self.robs_upsample = torch.tensor(self.robs_upsample, dtype=torch.float32, device=device)
        
    