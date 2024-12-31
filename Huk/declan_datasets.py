import os
import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import Dataset
import NDNT.utils as utils
#from NDNT.utils import download_file, ensure_dir
from copy import deepcopy
from NTdatasets.sensory_base import SensoryBase

class DeclanSampleData(SensoryBase):
    """
    DeclanSampleData is a class for handling trial-level data from running project. This takes the 
    most basic stimulus (grating direction (12), and 4 spatial/temporal frequency combinations) and
    spike count on trial level
    """

    def __init__(
            self, filename=None, datadir='', 
            combined_stim=False, 
            nspk_cutoff=500,
            drift_interval=60,
            **kwargs):
        """
        Args:
            filename: duh (required)
            datadir: directory name that data is in: can just lump with filename if want to leave
            combined_stim: whether to combine direction and frequency info into single one-hot category (def: False)
            nspk_cutoff: initial quality criteria, not including neurons with less than this number of spikes
            drift_interval: anchor spacing for drift model
            **kwargs: additional arguments to pass to the parent class
        """
        assert filename is not None, "Must specify filename."

        # call parent constructor -- has base-level variables and lots of dataset functions
        super().__init__(datadir=datadir, filenames=filename, drift_interval=drift_interval, **kwargs)

        # Load data into dataset structure easiest way
        matdat = sio.loadmat( datadir + filename )
        robs_raw = matdat['Robs'].astype(np.float32)
        self.StimDir = matdat['StimDir'].squeeze().astype(np.float32)
        self.StimTF = matdat['StimTF'].squeeze().astype(np.float32)
        self.StimSF = matdat['StimSF'].squeeze().astype(np.float32)
        self.EyeVar = matdat['EyeVar'].squeeze().astype(np.float32)
        self.PupilArea = matdat['PupilArea'].squeeze().astype(np.float32)
        self.RunSpeed = matdat['RunSpeed'].squeeze().astype(np.float32)
        self.SacMag = matdat['SacMag'].squeeze().astype(np.float32)
        self.SacRate = matdat['SacRate'].squeeze().astype(np.float32)
        
        self.NT = len(self.StimSF)
        self.NC = robs_raw.shape[1]

        # Rudimentary unit filtering
        if nspk_cutoff is not None:
            self.kept_cells = np.where(np.sum(robs_raw, axis=0) > nspk_cutoff)[0]
            print("Applying %d-spike cutoff: %d/%d cells remain."%(nspk_cutoff, len(self.kept_cells), self.NC))
            self.NC = len(self.kept_cells)
        else:
            self.kept_cells = np.arange(self.NC)

        self.robs = torch.tensor( robs_raw[:, self.kept_cells], dtype=torch.float32 )
        self.dfs = torch.ones([self.NT, self.NC], dtype=torch.float32)  # currently no datafilters in dataset

        # Assign cross-validation
        Xt = np.arange(2, self.NT, 5, dtype='int64')
        Ut = np.array(list(set(np.arange(self.NT, dtype='int64'))-set(Xt)))

        self.train_inds = Ut
        self.val_inds = Xt

        ##### Additional Stim processing #####
        self.stimDs = torch.tensor( self.construct_onehot_design_matrix(self.StimDir), dtype=torch.float32 )
        NDIR = self.stimDs.shape[1]
        
        binTF = np.log2(self.StimTF).astype(int)
        binSF = (self.StimSF-1).astype(int)
        Fcat = binSF*3 + binTF  # note this give 0 1 4 5
        Fcat[Fcat > 2] += -2  # now categories 0 1 (SF=0, TF=0,1) and 2 3 (SF=1, TF=1,2)
        self.stimFs = torch.tensor( self.construct_onehot_design_matrix(Fcat), dtype=torch.float32 )
        NFS = self.stimFs.shape[1]

        if combined_stim:
            self.stim = torch.einsum('ta,tb->tab', self.stimFs, self.stimDs ).reshape([-1, 48])
            self.stim_dims = [NFS, NDIR, 1, 1]  # put directions in space
        else:
            self.stim = self.stimDs
            self.stim_dims = [1, NDIR, 1, 1]  # put directions in space

        # Make drift matrix
        drift_anchors = np.arange(0, self.NT, drift_interval)
        #self.construct_drift_design_matrix( block_anchors=drift_anchors) 
        self.Xdrift = torch.tensor( self.design_matrix_drift( 
            self.NT, drift_anchors, zero_left=False, zero_right=False, const_left=False, const_right=True, to_plot=False),
            dtype=torch.float32)
    # END DeclanSampleData.__init__()

    def __getitem__(self, idx):
        if len(self.cells_out) == 0:
            out = {'stim': self.stim[idx, :],
                   'stimFs': self.stimFs[idx, :], 
                'robs': self.robs[idx, :],
                'dfs': self.dfs[idx, :]}
            
            if self.speckled:
                out['Mval'] = self.Mval[idx, :]
                out['Mtrn'] = self.Mtrn[idx, :]
        else:
            #assert isinstance(self.cells_out, np.ndarray), 'cells_out must be a numpy array'
            robs_tmp =  self.robs[:, self.cells_out]
            dfs_tmp =  self.dfs[:, self.cells_out]
            out = {'stim': self.stim[idx, :],
                'stimFs': self.stimFs[idx, :], 
                'robs': robs_tmp[idx, :],
                'dfs': dfs_tmp[idx, :]}
            
            if self.speckled:
                M1tmp = self.Mval[:, self.cells_out]
                M2tmp = self.Mtrn[:, self.cells_out]
                out['Mval'] = M1tmp[idx, :]
                out['Mtrn'] = M2tmp[idx, :]
            
        if self.Xdrift is not None:
            out['Xdrift'] = self.Xdrift[idx, :]

        if len(self.covariates) > 0:
            self.append_covariates( out, idx)

        return out
    # END DeclanSampleData.__getitem()