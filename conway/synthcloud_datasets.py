import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
from ColorDataUtils.simproj_utils import downsample_stim

class SimCloudData(Dataset):
    """
    Data set for simulted cloud data. It is assumed that the data has already been compiled as an HDF5 file.
    """
    def __init__(self,
        file_name,
        device=None,
        block_len=1000,
        down_sample=None,
        num_lags=12,
        cell_idx=None):
        """
        Args:
            file_name: Name of the HDF5 file to be used as a string.
            device: Device in which data lives. (Defalut None)
            block_len: Number of time points in each block. Must be a multiple of the total number of time points. (Defalut 1000)
            down_sample: How much to down sample the stim. If down_sample=2 and stim is of dimension LxL brings down to L/2xL/2. (Default 2)
            num_lags: How many time points to lag by. (Default 12)
            cell_idx: Index of cells to use as list. (Default None)
        """
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device

        
        with h5py.File(file_name, 'r') as f:
            init_stim = f['stim'][:]
            if cell_idx is None:
                self.robs = f['robs'][:]
            else:
                self.robs = f['robs'][:][:,cell_idx]
            file_start_pos = list(f['file_start_pos'][:])

        self.block_len = block_len   # block length
        self.NT = init_stim.shape[0] # number of time points
        assert self.NT%self.block_len == 0, "Number of time points is not divisible by "+str(self.block_len)
        
        if down_sample is not None:
            orig_L = int(np.sqrt(init_stim.shape[1]))
            L = orig_L//down_sample
            self.stim = downsample_stim(init_stim.reshape(self.NT,orig_L,orig_L), 
                                        down_sample).reshape(self.NT,int(L*L))
        else:
            L = int(np.sqrt(init_stim.shape[1]))
            self.stim = init_stim

        self.stim_dims = [1, L, L, 1]
        
        self.NB = self.NT//self.block_len # number of blocks
        self.blocks = np.arange(self.NT, dtype=np.int64).reshape(self.NB,self.block_len) # block indecies
        
        self.train_inds = np.arange((self.NB//5)*4, dtype=np.int64)
        self.val_inds = np.arange((self.NB//5)*4, self.NB, dtype=np.int64)

        self.num_lags = num_lags
        self.dfs = np.ones(self.robs.shape)
        for i in range(len(file_start_pos)):
            j = file_start_pos[i]
            self.dfs[j:j+self.num_lags,:] = 0

    def __len__(self):
        return self.stim.shape[0]

    def __getitem__(self, block_index):
        index = self.blocks[block_index,:].flatten()
        N_blocks = self.blocks[block_index,:].shape[0]
        
        block_dfs = self.dfs[index,...]
        for i in range(N_blocks):
            block_dfs[i*self.block_len:(i*self.block_len)+self.num_lags,:] = 0
        
        data_dict = {}
        data_dict['stim'] = (torch.tensor(self.stim[index,...], dtype=torch.float32).to(self.device)-127.0)/50.0
        data_dict['robs'] = torch.tensor(self.robs[index,...], dtype=torch.float32).to(self.device)
        data_dict['dfs'] = torch.tensor(block_dfs, dtype=torch.float32).to(self.device)
        return data_dict
















