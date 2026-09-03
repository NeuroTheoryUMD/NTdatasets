import os
import sys
import numpy as np
import scipy.io as sio
import h5py
import pandas as pd

import torch
from torch.utils.data import Dataset
from copy import deepcopy

sys.path.append('/home/dbutts/Code')
import NDNT.utils as utils
from NTdatasets.sensory_base import SensoryBase
import NTdatasets.LGN.lgndata_tools as lgn_tools
#from stim_lib import StimLibraryH5, scan_all_trials, load_trial


class LGNdataset(SensoryBase):
    """
    Spatiotemporal LGN dataset class
    """

    def __init__(self, datadir, num_lags=12):
        """
        Initializes the LGNdata class.
        """
        super().__init__(
            filenames='', datadir=datadir, device=torch.device('cpu'),
            time_embed=0, num_lags=num_lags, include_MUs=False, 
            drift_interval=None, block_sample=False)

        lgn_tools.set_data_root(datadir)
        self.cell_table = lgn_tools.build_cell_trial_table()
        self.stim_lib = lgn_tools.StimLibraryH5()

        self.dataset = None
    # END LGNdata.__init__
 
    def info(self):
        """
        Instructions on how to use the LGNdata class and its methods.
        """
        print("To assemble a given penetration, use: " \
        "data.assemble_penetration(expt_n, pen_n, stim_type='NAT', contrast='h', frac=1, crop=None, top_corner=None, verbose=True)")
    # END LGNdata.info()

    def find_cells(self, stim_type='NAT', contrast=None):
        """
        Find cells for a given experiment and penetration.
        Args:
            stim_type (str): Stimulus type ('NAT' or 'SBN')
            contrast (str, optional): Contrast level ('h' for high, 'l' for low). If None, all contrasts are returned.
        Returns:
            pd.DataFrame: A DataFrame containing the relevant cell information.
        """
        if contrast is not None:
            print("Note contrast search is not implemented yet. Returning all contrasts.")

        cell_trial_info = lgn_tools.find_cells(modality=stim_type, regime='unique', table=self.cell_table)
        # unique cells
        num_cells = cell_trial_info.drop_duplicates(['experiment_number','penetration_number','cell_number']).shape[0]

        print("Found %d cells over %d trials"%(num_cells, len(cell_trial_info)))
        return cell_trial_info
    # END LGNdata.find_cells()

    def cell_info(self, expt_n, pen_n, cell_n=None):
        """
        Get information about a specific cell.

        Args:
            expt_n (int): Experiment number
            pen_n (int): Penetration number
            cell_n (int, optional): Cell number. If None, information for all cells in the penetration is returned.

        Returns:
            pd.DataFrame: A DataFrame containing the relevant cell information.
        """
        if cell_n is None:
            print('NEED TO FIGURE OUT HOW TO LIST HOW MANY CELLS')
        else:
            return lgn_tools.cell_info(expt_n, pen_n, cell_n, table=self.cell_table)[['trial_key', 'modality', 'regime', 'contrast', 'n_spikes']]
    # END LGNdata.cell_info()

    def assemble_penetration(self, expt_n, pen_n, stim_type='NAT', contrast='h', 
                             frac=1, crop=None, top_corner=None, verbose=True):
        """
        Assemble dataset at frame resolution or frac of. Note that needs to represent stim at full resolution for it to 
        work with LBFGS -- which pushes data through all at once (can't get item-away from full resolution)

        Args:
            expt_n (int): Experiment number
            pen_n (int): Penetration number
            stim_type (str): Stimulus type ('NAT' or 'SBN')
            contrast (str): Contrast level ('h' for high, 'l' for low)
            frac (int): Fraction of frame resolution to use
            crop (int, optional): Size to crop the stimulus to. If None, no cropping is done.
            top_corner (tuple, optional): Top corner coordinates for cropping. If None, center cropping is used.
            verbose (bool): Whether to print information about the dataset assembly

        Returns:
            dict: A dictionary containing the assembled dataset with keys:
                - 'stim': Stimulus data (numpy array)
                - 'robs': Spike counts (numpy array)
                - 'xy': XY classification (numpy array)
                - 'rms_contrast': RMS contrast (float)
        """
                    
        if contrast.lower()[0] == 'h':
            contrast = 'HC'
        elif contrast.lower()[0] == 'l':
            contrast = 'LC'
        else:
            print('Unrecognized contrast. Using high by default')
            contrast = 'HC'
            
        # Find relevant info
        if stim_type.lower()[0] == 'n':
            stim_type = 'NAT'
        elif stim_type.lower()[0] in ['w', 's']:
            stim_type = 'SBN'
            
        pen_rows = lgn_tools.find_cells(
            modality=stim_type, regime='unique', contrast=contrast,
            experiment_number=expt_n, penetration_number=pen_n)

        trial_keys = pen_rows.trial_key.unique()

        if len(trial_keys) == 0:
            print('No unique %s trials found for this penetration'%stim_type)
            return None
        if len(trial_keys) > 1:
            print('WARNING: More than one trial is relevant.', trial_keys)
        print('Penetration:', trial_keys[0])
        trial_data = lgn_tools.load_trial(expt_n, pen_n, trial_keys[0], lib=self.stim_lib)
        num_cells = len(trial_data['spk_times'])
        fts = trial_data['frame_times']
        dt = np.mean(np.diff(fts))
        T = fts[-1]+dt
        if verbose:
            print("%d frames, %d cells"%(len(fts), num_cells))

        # Make bins for Robs
        bin_edges = deepcopy(fts)
        for ii in range(1,frac):
            bin_edges = np.concatenate((bin_edges, deepcopy(fts)+ii*dt/frac), axis=0)
        bin_edges = np.concatenate((bin_edges, [T]))
        if frac > 1:
            bin_edges = np.sort(bin_edges)
        
        robs = np.zeros([len(fts)*frac, num_cells], dtype=np.float32)
        for cc in range(num_cells):
            robs[:,cc] = np.histogram(trial_data['spk_times'][cc], bins=bin_edges)[0]
        
        s = deepcopy(trial_data['stim'])
        if crop is not None:
            if crop == 48:
                crop = None
            else:
                assert crop < 48, "Crop must be smaller dingbat"
                if top_corner is None:
                    top_corner = np.array([(48-crop)//2, (48-crop)//2], dtype=int)
                
                s = s[:, range(top_corner[0], top_corner[0]+crop), :][:,:, range(top_corner[1], top_corner[1]+crop)]

        if frac > 1:
            s = np.repeat(s, frac, axis=0)

        self.dataset = {
            'expt_n': expt_n, 'pen_n': pen_n, 'stim_type': stim_type, 'contrast': contrast,
            'frac': frac, 'crop': crop, 'top_corner': top_corner,
            'stim': s.reshape([len(fts)*frac,-1]), 'robs': robs, 
            'xy': trial_data['xy_class'], 'rms_contrast': trial_data['contrast_rms'], 
            'names': trial_data['unit_names']}
        return self.dataset
        # END LGNdata.assemble_penetration()


    def make_datasets(self, expt_n, pen_n, stim_type='NAT', contrast='h', cell_list=None,
                      frac=1, crop=None, top_corner=None, spk_hist_dict=None,
                      trn_inds=None, val_inds=None, 
                      device=None, verbose=True):
        """
        Build a dataset for a specific experiment and penetration.
        spk_hist_dict would define spike history term to add with following arguments: 
            -- num_lags (required)
            -- dt (optional, default 1)
            -- doubling_time (optional, default None)

        Args:
            expt_n (int): Experiment number
            pen_n (int): Penetration number
            stim_type (str): Stimulus type ('NAT' or 'SBN')
            contrast (str): Contrast level ('h' for high, 'l' for low)
            cell_list (list, optional): List of cell indices to include. If None, all cells are included.
            frac (int): Fraction of frame resolution to use
            crop (int, optional): Size to crop the stimulus to. If None, stimulus at native size (48x48)
            top_corner (tuple, optional): Top corner coordinates for cropping. If None, center cropping is used.
            spk_hist_dict (dict, optional): Dictionary defining spike history term with keys: 'num_lags', 'dt', 'doubling_time'.
            trn_inds (array-like, optional): Indices for training data. If None, they will be generated automatically.
            val_inds (array-like, optional): Indices for validation data. If None, they will be generated automatically.
            device (torch.device, optional): Device for data inside dicts. Defaults to CPU if None.
            verbose (bool): print extra info about building process
        """
        if device is None:
            device = torch.device('cpu') # default is to store on CPU

        expt_dict = self.assemble_penetration(
            expt_n, pen_n, stim_type=stim_type, contrast=contrast, 
            frac=frac, crop=crop, top_corner=top_corner, verbose=verbose)

        if expt_dict is None:
            return None, None

        #Nframes = expt_dict['stim'].shape[0]
        NT, NC = expt_dict['robs'].shape
        #assert Nframes*frac == NT, "Mismatch in number of frames and robs"
        if cell_list is not None:
            if utils.is_int(cell_list):
                cell_list = [cell_list]
            if isinstance(cell_list, list):
                cell_list = np.array(cell_list, dtype=int)
            assert np.all(cell_list < NC), "Cell list contains invalid cell indices"
            expt_dict['robs'] = expt_dict['robs'][:, cell_list]
            #print(cell_list, cell_list.shape, expt_dict['xy'])
            #expt_dict['xy'] = expt_dict['xy'][cell_list]
            expt_dict['xy'] = [expt_dict['xy'][i] for i in cell_list]
            NC = len(cell_list)
        else:
            cell_list = np.arange(NC)
        if verbose:
            for cc in range(NC):
                print("  Cell %d (%s): %d spikes"%(cell_list[cc], expt_dict['xy'][cc], np.sum(expt_dict['robs'][:, cc])))

        buffer_size = self.num_lags*frac
        dfs = np.ones([NT, NC], dtype=np.float32)

        if trn_inds is not None:
            print("Not fully set up for this -- need to make corresponding data-filters")
            self.trn_inds = deepcopy(trn_inds)
            if val_inds is not None:
                self.val_inds = deepcopy(val_inds)
            else: 
                print("No val_inds provided -- not set up yet to handle")
        else:
            # break into 3 chunks and take middle chunk of each for crossl-validation
            chunk_size = NT//3
            chunk_starts = np.arange(0, NT, chunk_size)
            val_size = int(np.round(chunk_size/5))
            trn_inds = []
            val_inds = []

            for ii in range(3):
                trn_inds.append(np.arange(chunk_starts[ii], chunk_starts[ii]+2*val_size))
                val_inds.append(np.arange(chunk_starts[ii]+2*val_size, chunk_starts[ii]+3*val_size))
                trn_inds.append(np.arange(chunk_starts[ii]+3*val_size, chunk_starts[ii+1] if ii < 2 else NT))
                dfs[chunk_starts[ii]+np.arange(buffer_size), :] = 0
                dfs[chunk_starts[ii]+2*val_size+np.arange(buffer_size), :] = 0
                dfs[chunk_starts[ii]+3*val_size+np.arange(buffer_size), :] = 0

        trn_inds = np.concatenate(trn_inds)
        val_inds = np.concatenate(val_inds)
        self.dataset['trn_inds'] = trn_inds
        self.dataset['val_inds'] = val_inds

        # Put info about particular dataset into general dataset object and then return generic dataset 
        ds_trn = {
            'stim': torch.tensor(expt_dict['stim'][trn_inds, :], dtype=torch.float32, device=device),
            'robs': torch.tensor(expt_dict['robs'][trn_inds, :], dtype=torch.float32, device=device),
            'dfs': torch.tensor(dfs[trn_inds, :], dtype=torch.float32, device=device)}

        ds_val = {
            'stim': torch.tensor(expt_dict['stim'][val_inds, :], dtype=torch.float32, device=device),
            'robs': torch.tensor(expt_dict['robs'][val_inds, :], dtype=torch.float32, device=device),
            'dfs': torch.tensor(dfs[val_inds, :], dtype=torch.float32, device=device)}

        # Make spike-history term if relevant
        if spk_hist_dict is not None:
            if not isinstance(spk_hist_dict, dict):
                print("spk_hist_dict must be a dictionary with keys: nlags (req), dt (default 1), doubling_time (def None)")
                print("...Not included until you get it right")
            else:
                Xspike_history = self.generate_spike_history(robs=expt_dict['robs'], **spk_hist_dict)
                print( "  Spike-history has %d temporal dims"%(Xspike_history.shape[1]))
                ds_trn['Xspk_hist'] = torch.tensor(Xspike_history.reshape([NT, -1]), dtype=torch.float32, device=device)[trn_inds, :]
                ds_val['Xspk_hist'] = torch.tensor(Xspike_history.reshape([NT, -1]), dtype=torch.float32, device=device)[val_inds, :]

        # OLD GENERIC DATASET CONSTRUCTION -- now just returning dicts with torch tensors
        # from NTdatasets.generic import GenericDataset
        #ds_trn = GenericDataset({
        #    'stim': torch.tensor(expt_dict['stim'][trn_inds, :], dtype=torch.float32),
        #    'robs': torch.tensor(expt_dict['robs'][trn_inds, :], dtype=torch.float32),
        #    'dfs': torch.tensor(dfs[trn_inds, :], dtype=torch.float32),
        #    'Xspk_hist': torch.tensor(Xspike_history.reshape([NT, -1])[trn_inds, :], dtype=torch.float32)
        #    }, device=device)

        #ds_val = GenericDataset({
        #    'stim': torch.tensor(expt_dict['stim'][val_inds, :], dtype=torch.float32),
        #    'robs': torch.tensor(expt_dict['robs'][val_inds, :], dtype=torch.float32),
        #    'dfs': torch.tensor(dfs[val_inds, :], dtype=torch.float32),
        #    'Xspk_hist': torch.tensor(Xspike_history.reshape([NT, -1])[val_inds, :], dtype=torch.float32)
        #    }, device=device)

        #print("%d cells set up from penetration %d of experiment %d"%(NC, pen_n, expt_n)) 
        return ds_trn, ds_val
    # END LGNdata.make_datasets()

    def quick_stas( self, lag=None, to_plot=True ):
        """
        Quick STA for self.dataset 
        """
        if self.dataset is None:
            print("No dataset has been built -- need to run assemble_penetration first")
            return None

        if lag is None:
            lag = 1*self.dataset.frac  # one lag in is pretty good, somehow

        L = int(np.sqrt(self.dataset['stim'].shape[1]))
        NC = self.dataset['robs'].shape[1]

        # Cross-correlation of stimulus and response
        if lag == 0:
            stas = (self.dataset['stim'].T@self.dataset['robs']).reshape([L, L, NC]) / self.dataset['robs'].sum(axis=0)[None, None, :]
        else:
            stas = (self.dataset['stim'][:-lag, :].T@self.dataset['robs'][lag:, :]).reshape([L, L, NC]) / self.dataset['robs'][lag:, :].sum(axis=0)[None, None, :]

        num_rows = int(np.ceil(NC/6))        
        if to_plot:
            import matplotlib.pyplot as plt
            utils.ss(num_rows, 6, rh=2.8)
            for cc in range(NC):
                plt.subplot(num_rows, 6, cc+1)
                utils.imagesc(stas[:, :, cc])
                plt.title('Cell %d (max: %.2f)' % (cc, abs(stas[:, :, cc]).max()))
            plt.show()

        return stas
    # END LGNdata.quick_sta()

    def trial_sequence(self, expt_n, pen_n, to_display=True ):
        """
        Claude-authored:
        Trial types for one penetration, alphabetical by trial_key -- used
        as a stand-in for real recording order (the letter in trial_key
        usually reflects recording order; not guaranteed -- e.g. Expt3's d13
        has a ~16hr real gap between letters that look adjacent, see
        CLAUDE.md). Reads modality/regime/code/contrast straight from
        cell_table -- no pickle or raw-file access. Expects a valid_only=True
        table (the default) -- this is about usable data you'd load.
        """
        seq = self.cell_table[
            (self.cell_table.experiment_number == expt_n)
            & (self.cell_table.penetration_number == pen_n)].drop_duplicates('trial_key')#.sort_values('trial_key')

        if to_display:
            print(f"{'#':>2}  {'trial_key':16s} {'modality':8s} {'contrast':8s} {'n_frames':>9s} {'n_reps':>7s}  {'code'} ")
            for i, r in enumerate(seq.itertuples()):
                num_rep = 1
                if r.n_reps == r.n_reps:  # cheap NaN check
                    num_rep = int(r.n_reps)
                n_frames = '' if r.n_frames != r.n_frames else f"{int(r.n_frames)//num_rep}"  # NaN check without importing pandas
                n_reps = '' if r.n_reps != r.n_reps else f"{num_rep}"
                print(f"{i:2d}  {r.trial_key:16s} {str(r.modality):8s} {str(r.contrast):8s} {n_frames:>9s} {n_reps:>7s}  {str(r.code):>4s}")
                #print(f"{'#':>2}  {'trial_key':16s} {'modality':8s} {'contrast':8s} {'type':8s} {'code'}")
                #for i, r in enumerate(seq.itertuples()):
                #    print(f"{i:2d}  {r.trial_key:16s} {str(r.modality):8s} {str(r.contrast):8s} {r.regime:8s} {r.code}")
        else:
            return seq[['trial_key', 'modality', 'type', 'code', 'contrast']].reset_index(drop=True)
    # END LGNdata.trial_sequence()

    def __getitem__(self, index):
        
        idx = self.index_to_array(index, len(self))

        # Translate into blocks if using temporally contigous blocks
        if len(self.block_inds) == 0:
            if self.time_embed:
                stim = self.Xstim[idx, :]  
            else:
                stim = self.stim[idx, :]
            robs = self.robs[idx, :]
            dfs = self.dfs[idx, :]
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
