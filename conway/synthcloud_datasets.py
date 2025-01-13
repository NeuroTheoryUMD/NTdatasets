import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
from ColorDataUtils.simproj_utils import downsample_stim
from NTdatasets.sensory_base import SensoryBase

class SimCloudData(Dataset):
    """
    Data set for simulted cloud data. It is assumed that the data has already been compiled as an HDF5 file.
    """
    def __init__(self,
        file_name='data/cloud_data_stim_dim120_robs_sqrad_0.3.hdf5' ,
        block_len=1000,
        down_sample=None,
        num_lags=12,
        cell_type_list=None):
        """
        Args:
            file_name: Name of the HDF5 file to be used as a string.
            block_len: Number of time points in each block. Must be a multiple of the total number of time points. (Defalut 1000)
            down_sample: How much to down sample the stim. If down_sample=2 and stim is of dimension LxL brings down to L/2xL/2. (Default 2)
            num_lags: How many time points to lag by. (Default 12)
            cell_type_list: List of cells to use. All posible cell types are ['X_OFF', 'X_ON', 'V1_Exc_L4', 'V1_Inh_L4', 'V1_Exc_L2/3', 'V1_Inh_L2/3']. Data will be in the order of list.
        """
        all_cell_type_list = ['X_OFF', 'X_ON', 'V1_Exc_L4', 'V1_Inh_L4', 'V1_Exc_L2/3', 'V1_Inh_L2/3']
        
        with h5py.File(file_name, 'r') as f:
            x_pos = f['x_pos'][:]
            cell_key = [str(f['cell_key'][:][i], encoding='utf-8') for i in range(x_pos.shape[0])]

        # Cell key and index
        if cell_type_list is None:
            print("No cell types were chosen. Will use all cells as defalut including LGN.")
            cell_type_list = all_cell_type_list
            cell_idx = [i for i in range(len(cell_key))]
        else:
            cell_idx = []
            aux_cell_key = []
            for cell in cell_type_list:
                assert cell in all_cell_type_list, "This cell type is not in the data set. Please choose from ['X_OFF', 'X_ON', 'V1_Exc_L4', 'V1_Inh_L4', 'V1_Exc_L2/3', 'V1_Inh_L2/3']"
                idx = [i for i, val in enumerate(cell_key) if val == cell]
                cell_idx += idx
                aux_cell_key += [cell]*len(idx)
            cell_key = aux_cell_key
        
        # index for each cell type in data
        self.cell_idx_dict = {}
        for cell in cell_type_list:
            self.cell_idx_dict[cell] = [i for i, val in enumerate(cell_key) if val == cell]
            
        self.NC = len(cell_idx)
        self.cell_type_list = cell_type_list
        self.cell_key = cell_key

        # Load data from HDF5 file
        with h5py.File(file_name, 'r') as f:
            x_pos = f['x_pos'][cell_idx]
            y_pos = f['y_pos'][cell_idx]
            init_stim = f['stim'][:]
            self.robs = f['robs'][:][:,cell_idx]
            file_start_pos = list(f['file_start_pos'][:])

        # Load orientation info
        ori_dict = np.load('data/V1_neuron_orientation_in_deg_and_orientation_selection_sqrad_0.3_GQM.pkl', allow_pickle=True)
        self.thetas = {}
        for cell in self.cell_type_list:
            if cell == 'X_OFF' or cell == 'X_ON':
                continue
            else:
                self.thetas[cell] = ori_dict['thetas'][cell]
                
        self.trial_sample = True
        
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
        self.block_inds = np.arange(self.NT, dtype=np.int64).reshape(self.NB,self.block_len) # block indecies

        self.val_inds = None
        self.train_inds = None
        self.train_blks = np.arange((self.NB//5)*4, dtype=np.int64)
        self.val_blks = np.arange((self.NB//5)*4, self.NB, dtype=np.int64)

        self.num_lags = num_lags
        self.dfs = np.ones(self.robs.shape)
        for i in range(len(file_start_pos)):
            j = file_start_pos[i]
            self.dfs[j:j+self.num_lags,:] = 0

    def __len__(self):
        return self.stim.shape[0]

    def __getitem__(self, index):
        N_blocks = self.block_inds[index,:].shape[0]
        index = self.block_inds[index,:].flatten()
        
        #index = SensoryBase.index_to_array(index, N_blocks)
        #ts = self.block_inds[index[0]]
        #for j in index[1:]:
        #    ts = np.concatenate((ts, self.block_inds[j]), axis=0 )
        #index = ts
        
        block_dfs = self.dfs[index,...]
        for i in range(N_blocks):
            block_dfs[i*self.block_len:(i*self.block_len)+self.num_lags,:] = 0
        
        data_dict = {}
        data_dict['stim'] = (torch.tensor(self.stim[index,...], dtype=torch.float32)-127.0)/50.0
        data_dict['robs'] = torch.tensor(self.robs[index,...], dtype=torch.float32)
        data_dict['dfs'] = torch.tensor(block_dfs, dtype=torch.float32)
        return data_dict
















