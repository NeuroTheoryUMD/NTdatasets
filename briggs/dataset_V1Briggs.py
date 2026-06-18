import os
import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import Dataset
from copy import deepcopy
import h5py

import sys
#sys.path.append('/Users/dbutts/GitCode')
sys.path.append('/home/dbutts/Code')
import NDNT.utils as utils
from NTdatasets.sensory_base import SensoryBase


class Vision2Dsimple(SensoryBase):
    """Test 2-D V1 dataset using Mseq Briggs data. Nothing fancy here, but good to test the basics
    """

    def __init__(
        self,
        datafile, 
        num_lags=10,
        include_opto = 0,
        time_embed = False,
        reps_to_use = None,
        device = None):
        """
        include_opto = 0 means just non-opto dataset, 1 means just opto_dataset, 2 means both
        """

        if time_embed:
            time_embed = 1
        else:
            time_embed = 0
        if device is None:
            device = torch.device('cpu') # default is to store on CPU   

        super().__init__(
            filenames=datafile, datadir='', device=device,
            time_embed=time_embed, num_lags=num_lags, include_MUs=False, 
            drift_interval=None, block_sample=False)

        #self.datafile = datafile
        #if device is None:
        #    device = torch.device('cpu') # default is to store on CPU
        #self.device = device
        #self.num_lags = num_lags

        # Internal variables
        #self.test_inds = None
        #self.val_inds = None
        #self.train_inds = None
        #self.block_inds = None
        self.NX = 16

        self.stim_dims = [1, self.NX, self.NX, 1]

        # Read data into memory -- no need to keep file open
        with h5py.File(datafile, 'r') as f:

            self.stim = torch.tensor(np.array(f['stim']), dtype=torch.float32, device=self.device).t()
            self.num_units = np.array(f['NCs'][:,0], dtype=np.int64)
            # Handle Robs
            if include_opto != 1:
                self.robs = torch.tensor(f['Robs'], dtype=torch.float32, device=self.device)
            else:
                self.robs = torch.tensor(f['RobsOpto'], dtype=torch.float32, device=self.device)
            if include_opto == 2:
                self.robs = torch.cat( (self.robs, torch.tensor(f['RobsOpto'], dtype=torch.float32, device=self.device)), axis=1)
            self.robs = self.robs.t()  # matlab reverses order

            # Process data and DFs
            self.trial_starts = np.array(f['trial_starts'][:,0], dtype=np.int64)

        # Make datafilters to match Robs but take out lags at beginning of each fixation
        NT, NC = self.robs.shape
        assert NC == np.sum(self.num_units), "num units do not match: %d -> %d"%(np.sum(self.num_units), NC)
        self.data_filters = torch.ones( [NT, NC], dtype=torch.float32, device=self.device )

        if reps_to_use is None:
            reps_to_discard = []
        else:
            reps_to_discard = list(set(list(range(4)))-set(reps_to_use))

        rstim = self.stim
        for rep in range(4):
            if rep > 0:
                self.stim = torch.cat( (self.stim, rstim), axis=0 )
            if rep in reps_to_discard:
                if rep == 3:
                    indx = np.arange( self.trial_starts[rep], NT )
                else:
                    indx = np.arange( self.trial_starts[rep], self.trial_starts[rep+1] )
                self.data_filters[indx, :] = 0
            else:
                self.data_filters[range(self.trial_starts[rep], self.trial_starts[rep]+num_lags), :] = 0

        self.time_embed = time_embed
        if time_embed:
            #self.Xstim = torch.tensor( 
            #    utils.create_time_embedding( self.stim, [num_lags, self.NX, self.NX]),
            #    dtype=torch.float32, device=self.device)
            self.Xstim = self.time_embedding( self.stim, nlags = num_lags, verbose=True ).to(device)
            #, dtype=torch.float32, device=self.device)

            self.stim_dims[3] = num_lags

        self.num_samples = self.robs.shape[0]
    # END Vision2Dsimple.__init__
 
    def crossval_setup(self, folds=5, random_gen=False, test_set=False, block_size = None):
        """This sets the cross-validation indices up We can add featuers here. Many ways to do this
        but will stick to some standard for now. It sets the internal indices, which can be read out
        directly or with helper functions. Perhaps helper_functions is the best way....
        
        Inputs: 
            random_gen: whether to pick random trials for validation or uniformly distributed
            test_set: whether to set aside first an n-fold test set, and then within the rest n-fold train/val sets
        Outputs:
            None: sets internal variables test_inds, train_inds, val_inds
        """
        #from NDNT.utils import fold_sample
        test_fixes = []
        tfixes = []
        vfixes = []

        # If not time-embedded, then forced to use temporally-contiguous blocks: need block size
        if not self.time_embed:
            if block_size is None:
                block_size = self.num_lags*10
                print( "Temporally contiguous blocks needed: using block-size of %d"%block_size )
        # Can break dataset into contiguous blocks (time embedded or not -- really should)
        if block_size is not None:
            self.num_samples = self.robs.shape[0]//block_size
            # Make block_list
            self.block_inds = []
            for nn in range(self.num_samples-1):
                self.block_inds.append(list(range(block_size*nn, block_size*(nn+1))))
            # for this case, make last block slightly bigger to get all data
            self.block_inds.append(list(range(block_size*nn, self.robs.shape[0])))
        
        tr_inds = np.arange(self.num_samples)
        vfix1, tfix1 = utils.fold_sample(len(tr_inds), folds, random_gen=random_gen)
        if test_set:
            test_fixes += list(tr_inds[vfix1])
            vfix2, tfix2 = utils.fold_sample(len(tfix1), folds, random_gen=random_gen)
            vfixes += list(tr_inds[tfix1[vfix2]])
            tfixes += list(tr_inds[tfix1[tfix2]])
        else:
            vfixes += list(tr_inds[vfix1])
            tfixes += list(tr_inds[tfix1])

        # If block-size greater than 1, translate back into 
        self.val_inds = np.array(vfixes, dtype='int64')
        self.train_inds = np.array(tfixes, dtype='int64')
        if test_set:
           self.test_inds = np.array(test_fixes, dtype='int64')
    # END Vision2Dsimple.crossval_setup

    def __getitem__(self, index):
        
        idx = self.index_to_array(index, len(self))

        # Translate into blocks if using temporally contigous blocks
        if len(self.block_inds) == 0:
            if self.time_embed:
                stim = self.Xstim[idx, :]  
            else:
                stim = self.stim[idx, :]
            robs = self.robs[idx, :]
            dfs = self.robs[idx, :]
        else:
            # need to do block-by-block with lags accounted for (with extra dfs)
            accum_stim, accum_robs, accum_dfs = [], [], []
            for nn in idx:
                if self.time_embed:
                    accum_stim.append(self.Xstim[self.block_inds[nn], :])
                else:
                    accum_stim.append(self.stim[self.block_inds[nn], :])
                accum_robs.append(self.robs[self.block_inds[nn], :])

                accum_dfs.append(self.dfs[self.block_inds[nn], :])
                accum_dfs[-1][:self.num_lags,:] = 0

                stim = torch.cat(accum_stim, axis=0)
                robs = torch.cat(accum_robs, axis=0)
                dfs = torch.cat(accum_dfs, axis=0)

        return {'stim': stim, 'robs': robs, 'dfs': dfs}
    # END Vision2Dsimple.__get_item__

    def __len__(self):
        #if self.time_embed:
        return self.stim.shape[0]
        #else:
        #    return len(self.block_inds)
