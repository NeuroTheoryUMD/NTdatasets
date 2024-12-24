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
    DeclanSampleData is a class for handling whisker data from the lab of Scott Pluta.
    """

    def __init__(self, filename=None, combined_stim=False, nspk_cutoff=500, drift_interval=60, **kwargs):
        """
        Args:
            filename: duh
            **kwargs: additional arguments to pass to the parent class
        """
        assert filename is not None, "Must specify filename."
        # call parent constructor
        super().__init__(datadir='', filenames=filename, **kwargs)
        #self.expt_name = expt_name

        matdat = sio.loadmat( filename )
        robs_raw = matdat['Robs']
        self.StimDir = matdat['StimDir'].squeeze()
        self.StimTF = matdat['StimTF'].squeeze()
        self.StimSF = matdat['StimSF'].squeeze()
        self.EyeVar = matdat['EyeVar'].squeeze()
        self.PupilArea = matdat['PupilArea'].squeeze()
        self.RunSpeed = matdat['RunSpeed'].squeeze()
        self.SacMag = matdat['SacMag'].squeeze()
        self.SacRate = matdat['SacRate'].squeeze()
        
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
        # Assemble stim

        # Assign cross-validation
        Xt = np.arange(2, self.NT, 5, dtype='int64')
        Ut = np.array(list(set(np.arange(self.NT, dtype='int64'))-set(Xt)))

        self.train_inds = Ut
        self.val_inds = Xt

        ##### Additional Stim processing #####
        self.stimDs = torch.tensor( self.construct_onehot_design_matrix(self.StimDir), dtype=torch.float32 )
        binTF = np.log2(self.StimTF).astype(int)
        binSF = (self.StimSF-1).astype(int)
        Fcat = binSF*3 + binTF  # note this give 0 1 4 5
        Fcat[Fcat > 2] += -2  # now categories 0 1 (SF=0, TF=0,1) and 2 3 (SF=1, TF=1,2)
        self.stimFs = torch.tensor( self.construct_onehot_design_matrix(Fcat), dtype=torch.float32 )
        if combined_stim:
            self.stim = torch.einsum('ta,tb->tab', self.stimFs, self.stimDs ).reshape([-1, 48])
            self.stim_dims = [4, 12, 1, 1]  # put directions in space
        else:
            self.stim = self.stimDs
            self.stim_dims = [1, self.stim.shape[1], 1, 1]  # put directions in space

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

        if self.ACinput is not None:
            out['ACinput'] = self.ACinput[idx, :]

        if len(self.covariates) > 0:
            self.append_covariates( out, idx)

        return out
    # END DeclanSampleData.__getitem()