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
        filename = 'B2Texpt'+ str(expt_num) + '.mat'

        # call parent constructor
        super().__init__(
            filename, 
            num_lags=num_lags, time_embed=time_embed,
            **kwargs)

        self.dt = 0.01 #100Hz
        self.upsample = 1
        self.robs_upsample = None
        self.dfs_upsample = None

        if verbose:
            print( "Loading", self.datadir + filename)

        # Store stimulus trimmed to 36 - 36 binocular configuration
        stim_trim = np.concatenate( (np.arange(3,39), np.arange(45,81)))
        #Bmatdat = h5py.File(self.datadir + filename, 'r')
        Bmatdat = sio.loadmat(self.datadir + filename, squeeze_me=True)
        #self.Bstim = np.transpose(Bmatdat['stim'])[:, stim_trim]
        self.Bstim = np.array(Bmatdat['stim'], dtype=np.float32)[:, stim_trim]

        self.dims = [1, 72, 1, 1]
        self.divide_stim = False

        # Responses
        self.unit_index = np.array(Bmatdat['unit_raw_index'], dtype=int)
        self.spk_times = np.array(Bmatdat['spk_times'], dtype=np.float32)
        self.spk_ids = np.array(Bmatdat['spk_ids'], dtype=int)

        RobsSU = np.array(Bmatdat['RobsSU'], dtype=np.float32)
        dfsSU = np.array(Bmatdat['SUdata_filter'], dtype=np.float32)
        self.NT, self.numSUs = RobsSU.shape

        RobsMU = np.array(Bmatdat['RobsMU'], dtype=np.float32)
        self.numMUs = RobsMU.shape[1]
    
        if self.include_MUs:
            self.NC = self.numSUs + self.numMUs

            dfsMU = np.transpose(Bmatdat['MUdata_filter'])

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
        used_inds = np.add(np.array(Bmatdat['used_inds'], dtype=np.int32), -1) # note adjustment for python v matlab indexing
        # implement within datafilters:
        df_mult = np.zeros([self.NT,1], dtype=np.float32)
        df_mult[used_inds] = 1.0
        self.dfs *= df_mult

        # Hard-code block_inds since each trial is 3 sec = 300 time points
        for ii in range(0, self.NT, 300):
            self.block_inds.append(np.arange(ii, min(ii+300, self.NT)))

        # Spike times (only SUs)
        self.Ui_analog = Bmatdat['Ui_analog'].squeeze()  # these are automatically in register
        self.XiA_analog = Bmatdat['XiA_analog'].squeeze()
        self.XiB_analog = Bmatdat['XiB_analog'].squeeze()
        # also combine two cross-validation datasets
        self.Xi_analog = self.XiA_analog+self.XiB_analog  # since they are non-overlapping, will make 1 in both places

        # # Derive full-dataset Ui and Xi from analog values
        self.used_inds = used_inds
        self.train_inds = np.intersect1d(used_inds, np.where(self.Ui_analog > 0)[0])
        self.val_inds = np.intersect1d(used_inds, np.where(self.Xi_analog > 0)[0])
        self.val_indsA = np.intersect1d(used_inds, np.where(self.XiA_analog > 0)[0])
        self.val_indsB = np.intersect1d(used_inds, np.where(self.XiB_analog > 0)[0])

        dispt_raw = np.transpose(Bmatdat['all_disps'])
        # this has the actual disparity values, which are at the resolution of single bars, and centered around the neurons
        # disparity (sometime shifted to drive neurons well)
        # Sometimes a slightly disparity is used, so it helps to round the values at some resolution
        self.dispt = np.round(dispt_raw*100)/100
        # Fix expt10
        if expt_num == 10:  # make the uncommon disparity (at the extreme) into uncorrelated, which is it anyway...
            print('  dispt-fix for expt 10') 
            self.dispt[self.dispt > 0.5] = -1005

        self.frs = np.transpose(Bmatdat['all_frs'])
        self.corrt = np.transpose(Bmatdat['all_corrs'])
        # Make dispt consistent with corrt (early experiments had dispt labeled incorrectly)
        corr_funny = np.where((self.corrt == 0) & (self.dispt != -1005))[0]
        if len(corr_funny) > 0:
            print( "Warning: %d indices have corr=0 but labeled disparity."%len(corr_funny) )
            self.dispt[corr_funny] = -1005

        self.disp_list = np.unique(self.dispt)
        # where it is -1009 this corresponds to a blank frame
        # where it is -1005 this corresponds to uncorrelated images between the eyes

        if not 'rep_inds' in Bmatdat:
            #rep_inds = [None]*numSUs
            rep_inds = None
        elif Bmatdat['rep_inds'][0].shape[0] < 10:  # check first cell rep_inds to make sure valid
            print("Warning: valid rep_inds not found in dataset.")
            rep_inds = None
        else:
            rep_inds = []
            for cc in range(self.numSUs):
                rep_inds.append( np.add(Bmatdat['rep_inds'][cc], -1) ) 
        self.rep_inds = rep_inds

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
        assert self.skip_lags >= 0, "Negative skip_lags does not make sense"

        if self.upsample == 1:
            stim = deepcopy(self.Bstim)
            skip_lags = self.skip_lags
        else:
            stim = np.repeat(deepcopy(self.Bstim), self.upsample, axis=0)
            skip_lags = self.skip_lags*self.upsample

        if skip_lags > 0:
            stim[skip_lags:, :] = deepcopy( stim[:-skip_lags, :] )
            stim[skip_lags, :] = 0.0

        self.stim_dims = deepcopy(self.dims)
        if time_embed == 0:
            self.stim = torch.tensor( self.Bstim, dtype=torch.float32 )
        else:
            if num_lags is None: # then read from dataset
                num_lags = self.num_lags
            num_lags = num_lags*self.upsample
            self.stim = self.time_embedding( stim=stim, nlags=num_lags, verbose=verbose )
            # This will return a torch-tensor
            self.stim_dims[3] = num_lags

        # Also compute frame-switch regressors
        self.compute_frame_switch_regressors()
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
        print('WARNING: separate_eyes() is not fully tested for upsample > 1')
        stim = self.stim.reshape([self.NT*self.upsample, 2*NX, self.num_lags])
        self.stimL = stim[:, :NX, :].reshape([self.NT, -1])
        self.stimR = stim[:, NX:, :].reshape([self.NT, -1])
        self.divide_stim = val
    # END binocular_single.separate_eyes()

    def __getitem__(self, idx):

        idx = self.index_to_array(idx,self.NT)
        if self.upsample > 1:
            idx = (np.repeat(idx[:, None]*self.upsample, self.upsample, axis=1)+ np.arange(self.upsample)[None,:]).reshape([-1])
            robs = self.robs_upsample
            dfs = self.dfs_upsample
        else:
            robs = self.robs
            dfs = self.dfs

        if len(self.cells_out) == 0:
            out = {
                'stim': self.stim[idx, :],
                'robs': robs[idx, :],
                'dfs': dfs[idx, :]}
            #if self.speckled:
            #    out['Mval'] = self.Mval[idx, :]
            #    out['Mtrn'] = self.Mtrn[idx, :]
        else:
            robs_tmp =  robs[:, self.cells_out]
            dfs_tmp =  dfs[:, self.cells_out]
            out = {
                    'stim': self.stim[idx, :], 
                    'robs': robs_tmp[idx, :],
                    'dfs': dfs_tmp[idx, :]}
            
        if self.divide_stim:
            out['stimL'] = self.stimL[idx, :]
            out['stimR'] = self.stimR[idx, :]

        if self.Xdrift is not None:
            if self.upsample > 1:
                out['Xdrift'] = self.Xdrift[idx//self.upsample, :]
            else:
                out['Xdrift'] = self.Xdrift[idx, :]

        out['Xframe_switch'] = self.frame_switch_mat[idx, :]
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
        self.upsample = frac

        if upsample_mult > 1:
            print( "  Upsampling by %d: changing num_lags with dataset to %d"%(frac, self.num_lags*frac) )
        elif upsample_mult < 1:
            print( "  Downsampling by %d: changing num_lags with dataset to %d"%(frac, self.num_lags*frac) )

        if upsample_mult != 1:
            self.prepare_stim(time_embed=self.time_embed, verbose=False)

        if frac == 1:
            # No upsampling needed
            self.dfs_upsample = None
            self.robs_upsample = None
            return
        else:
            assert self.spk_times is not None, "No spike time information in dataset."
        
        self.dfs_upsample = np.repeat(self.dfs, frac, axis=0)

        # Generate new robs at higher time resolution
        dt = self.dt/frac

        self.robs_upsample = np.zeros([self.NT*frac, self.NC], dtype=np.uint8 )
        for cc in range(self.NC):
            a = np.where(self.spk_ids[:] == cc+1)[0] 
            if len(a) > 0:
                robs_up = np.histogram(self.spk_times[a], bins=np.arange(self.NT*frac+1)*dt)[0]
                # print(robs_up.shape, self.robs_upsample[:, cc].shape)
                self.robs_upsample[:, cc] = robs_up.astype(np.uint8)

        # this is handled now in prepare_stim
        #if not self.time_embed:
        #    self.stim_upsample = np.repeat(self.stim,frac,axis=0)
        #else:
        #    self.stim_upsample = self.time_embedding(np.repeat(self.stim[:,::orig_lags],frac,axis=0))
        #    self.stim_dims[3] = self.num_lags

        if self.device is None:
            device = torch.device("cpu")
        else:
            device = self.device

        if type(self.robs_upsample) != torch.Tensor:
            self.robs_upsample = torch.tensor(self.robs_upsample, dtype=torch.float32, device=device)
    # END binocular_single.set_upsample()

    def compute_frame_switch_regressors(self):
        """
        """
        from NTdatasets.cumming.BinocUtils import disparity_matrix

        # Derive switches of disparity and frame-time itself
        dmat = np.repeat(disparity_matrix( self.dispt, self.corrt ), self.upsample, axis=0)
        disp_switches = np.expand_dims(np.concatenate( (np.sum(abs(np.diff(dmat, axis=0)),axis=1), [0]), axis=0), axis=1)/2
        disp_switches[np.where(self.frs == 1)[0]] = 0.0

        # Need to time-embed this with number of lags in betwen fr3 (3*upsample-1)
        switch_mat = utils.create_time_embedding( disp_switches, 3*self.upsample-1)  # leaves last lag out

        # Add regressors for frame switches
        if self.upsample > 1:
            frame_switches = np.zeros([self.NT*self.upsample, 1], dtype=np.float32)
            frame_switches[np.arange(0, self.NT*self.upsample, self.upsample), 0] = 1.0
            if self.upsample > 2:
                switch_mat = np.concatenate(
                    (switch_mat, utils.create_time_embedding( frame_switches, self.upsample-1)), axis=1) 
            else:
                switch_mat = np.concatenate((switch_mat, frame_switches), axis=1) 
        print(switch_mat.shape)
        # Need blank regressor? (it will show up in disparity)
        #blanks = dmat[:, -1][:, None]
        #tmat = np.concatenate( (blanks, switches), axis=1 )
        self.frame_switch_mat = torch.tensor(switch_mat, dtype=torch.float32)
    # END binocular_single.compute_frame_switch_regressors()