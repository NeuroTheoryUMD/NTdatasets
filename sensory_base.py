import os
import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import Dataset
import NDNT.utils as utils
#from NDNT.utils import download_file, ensure_dir
from copy import deepcopy
import h5py

class SensoryBase(Dataset):
    """Parent class meant to hold standard variables and functions used by general sensory datasets
    
    General consistent formatting:
    -- self.robs, dfs, and any design matrices are generated as torch vectors on device
    -- stimuli are imported separately as dataset-specific numpy arrays, and but then prepared into 
        self.stim (tensor) by a function self.prepare_stim, which must be overloaded
    -- self.stim_dims gives the dimension of self.stim in 4-dimensional format
    -- all tensors are stored on default device (cpu)

    General book-keeping variables
    -- self.block_inds is empty but must be filled in by specific datasets
    """

    def __init__(self,
        filenames, # this could be single filename or list of filenames, to be processed in specific way
        datadir, 
        # Stim setup
        block_sample=False,
        num_lags=10,
        time_embed=0,  # 0 is no time embedding, 1 is time_embedding with get_item, 2 is pre-time_embedded
        #maxT = None,
        # other
        include_MUs = False,
        drift_interval = None,
        device=torch.device('cpu')
        ):
        """
        Constructor options
        
        Args:
            filenames: the filenames to use
            datadir: the data directory
            block_sample: whether to sample trials
            num_lags: the number of lags presumably used by models (stored but not used in SensoryBase)
            time_embed: the time embedding to use
            include_MUs: whether to include MUs
            drift_interval: the drift interval to use
            device: the device to use
        """
        self.datadir = datadir
        self.filenames = filenames
        self.device = device
        
        self.block_sample = block_sample
        self.num_lags = num_lags
        self.stim_dims = None
        self.time_embed = time_embed
        self.preload = True
        self.drift_interval = drift_interval

        # Assign standard variables
        self.num_units, self.num_SUs, self.num_MUs = [], [], []
        self.SUs = []
        self.NC = 0    
        self.block_inds = []
        self.block_filemapping = []
        self.include_MUs = include_MUs
        self.SUinds = []
        self.MUinds = []
        self.cells_out = []  # can be list to output specific cells in get_item
        self.robs_out = None
        self.dfs_out = None

        self.avRs = None

        # Set up to store default train_, val_, test_inds
        self.test_inds = None
        self.val_inds = None
        self.train_inds = None
        self.test_blks = None
        self.val_blks = None
        self.train_blks = None
        self.used_inds = []
        self.speckled = False
        self.Mtrn, self.Mval = None, None  # Data-filter masks for speckled XV
        self.Mtrn_out, self.Mval_out = None, None  # Data-filter masks for speckled XV
        self.Xdrift = None
        
        # Basic default memory things
        self.stim = []
        self.dfs = []
        self.robs = []
        self.NT = 0
    
        # Additional covariate list
        self.covariates = {}
        self.cov_dims = {}
        # General file i/o -- this is not general, so taking out
        #self.fhandles = [h5py.File(os.path.join(datadir, sess + '.mat'), 'r') for sess in self.sess_list]
            
    # END SensoryBase.__init__

    def add_covariate( self, cov_name=None, cov=None, dtype=torch.float32 ):
        """
        Adds a covariate to the dataset

        Args:
            cov_name: name of the covariate
            cov: the covariate itself

        Returns:
            None
        """
        assert cov_name is not None, "Need cov_name"
        assert cov is not None, "Missing cov"
        if len(cov.shape) == 1:
            cov = cov[:, None]
        if len(cov.shape) > 2:
            dims = cov.shape[1:]
            if len(dims) < 4:
                dims = np.concatenate( (dims, np.ones(4-len(dims))), axis=0 )
            cov = cov.reshape([-1, np.prod(dims)])
        else:
            dims = [1, cov.shape[1], 1, 1]
        NT = cov.shape[0]
        assert self.NT == NT, "Wrong number of time points"

        if cov_name in self.covariates:
            print('  Discarding old', cov_name)
            del self.covariates[cov_name]

        self.cov_dims[cov_name] = dims
        if isinstance(cov, torch.Tensor):
            self.covariates[cov_name] = deepcopy(cov)
        else:
            self.covariates[cov_name] = torch.tensor(cov, dtype=dtype)
    # END SensoryBase.add_covariate()

    def append_covariates( self, out, idx ):
        """
        Complements __get_item__ to add covariates to existing dictionary
        
        Args:
            out: the dictionary to append to
            idx: the index to append at

        Returns:
            None
        """
        for cov_name, cov in self.covariates.items():
            out[cov_name] = cov[idx, :]
        # Return out, or not?

    def prepare_stim( self ):
        """
        This function is meant to be overloaded by child classes

        Returns:
            None
        """
        print('Default prepare stimulus method.')

    def set_cells( self, cell_list=None, verbose=True):
        """
        Set outputs to potentially limit robs/dfs to certain cells 
        This sets cells_out but also constructs efficient data structures
        
        Args:
            cell_list: list of cells to output
            verbose: whether to print out the number of cells

        Returns:
            None
        """
        if cell_list is None:
            # Then reset to full list
            self.cells_out = []
            self.robs_out = None
            self.dfs_out = None
            self.Mtrn_out = None
            self.Mval_out = None
            if verbose:
                print("  Reset cells_out to full dataset (%d cells)."%self.NC )
        else:
            if not isinstance(cell_list, list):
                if utils.is_int(cell_list):
                    cell_list = [cell_list]
                else:
                    cell_list = list(cell_list)
            assert np.max(np.array(cell_list)) < self.NC, "ERROR: cell_list too high."
            if verbose and len(cell_list)>1:
                print("Output set to %d cells"%len(cell_list))
            self.cells_out = cell_list
            self.robs_out = deepcopy(self.robs[:, cell_list])
            self.dfs_out = deepcopy(self.dfs[:, cell_list])
            if self.Mtrn is not None:
                self.Mtrn_out = deepcopy(self.Mtrn[:, cell_list])
                self.Mval_out = deepcopy(self.Mval[:, cell_list])
    # END SensoryBase.set_cells()

    def time_embedding( self, stim=None, nlags=None, verbose=True ):
        """
        Assume all stim dimensions are flattened into single dimension. 
        Will only act on self.stim if 'stim' argument is left None
        
        Args:
            stim: the stimulus to time-embed
            nlags: the number of lags to use
            verbose: whether to print out the time embedding process

        Returns:
            tmp_stim: the time-embedded stimulus
        """

        if nlags is None:
            nlags = self.num_lags
        if stim is None:
            assert self.stim_dims is not None, "Need to assemble stim before time-embedding."
            tmp_stim = deepcopy(self.stim)
            if self.stim_dims[3] == 1:  # should only time-embed stim by default, but not all the time
                self.stim_dims[3] = nlags
        else:
            if isinstance(stim, np.ndarray):
                tmp_stim = torch.tensor( stim, dtype=torch.float32)
            else:
                tmp_stim = deepcopy(stim)
 
        if verbose:
            print( "  Time embedding..." )
        NT = stim.shape[0]
        original_dims = None
        if len(tmp_stim.shape) != 2:
            original_dims = tmp_stim.shape
            if verbose:
                print( "Time embed: flattening stimulus from", original_dims)
        tmp_stim = tmp_stim.reshape([NT, -1])  # automatically generates 2-dimensional stim

        #assert self.NT == NT, "TIME EMBEDDING: stim length mismatch"

        # Actual time-embedding itself
        tmp_stim = tmp_stim[np.arange(NT)[:,None]-np.arange(nlags), :]
        tmp_stim = tmp_stim.permute((0,2,1)).reshape([NT, -1])
        if verbose:
            print( "  Done.")
        return tmp_stim
    # END SensoryBase.time_embedding()

    def construct_drift_design_matrix( self, block_anchors=None, zero_right=False):
        """
        Note this requires self.block_inds, and either uses self.drift_interval or block_anchors
        Option to anchor to zero on the right size, and otherwise constant after last tent anchor

        Args:
            block_anchors: the block anchors to use
            zero_right: whether to anchor to zero on the right (Default: False)

        Returns:
            None
        """

        assert self.block_inds is not None, "Need block_inds defined as an internal variable"

        if block_anchors is None:
            NBL = len(self.block_inds)
            if self.drift_interval is None:
                self.Xdrift = None
                return
            block_anchors = np.arange(0, NBL, self.drift_interval)

        Nanchors = len(block_anchors)
        anchors = np.zeros(Nanchors, dtype=np.int64)
        for bb in range(Nanchors):
            anchors[bb] = self.block_inds[block_anchors[bb]][0]
        
        self.anchors = anchors
        self.Xdrift = torch.tensor( 
            self.design_matrix_drift(self.NT, anchors, zero_left=False,
                                     zero_right=zero_right, const_right=True),
            dtype=torch.float32)
    # END SenspryBase.construct_drift_design_matrix()

    def trial_psths( self, trials=None, R=None, trial_size=None, ignore_dfs=True, verbose=False ):
        """
        Computes average firing rate of cells_out at bin-resolution, averaged across trials
        given in block_inds
        
        Args:
            trials: the trials to compute the PSTHs for
            R: the firing rates to use
            trial_size: the size of the trials
            ignore_dfs: whether or not dfs should be ignored when computing PSTH (default True)
            verbose: whether to print out the trial sizes

        Returns:
            psths: the PSTHs
        """

        Ntr = len(self.block_inds)
        assert Ntr > 0, "Cannot compute PSTHs without block_inds established in dataset."

        if len(self.cells_out) > 0:
            ccs = self.cells_out
        else:
            ccs = np.arange(self.NC)

        if R is None:  #then use [internal] Robs
            R = deepcopy( self.robs[:, ccs].detach().numpy() )  
        if len(R.shape) == 1:
            R = R[:, None]         
        num_psths = R.shape[1]  # otherwise use existing input

        if (R.shape[1] == len(ccs)) and not ignore_dfs:
            dfs = self.dfs[:, ccs].detach().numpy()
        else:
            if verbose:
                print('  Ignoring dfs.')
            dfs = np.ones([self.dfs.shape[0], R.shape[1]])

        # Compute median trial size
        if trial_size is None:
            # select median trial size, but look at all
            tr_sizes = np.zeros(len(self.block_inds))
            for bb in range(Ntr):
                tr_sizes[bb] = len(self.block_inds[bb])
            T = int(np.median(tr_sizes))
            if verbose:
                print("  Trial lengths: Min %d, Max %d, Median %d. Selecting median."%(np.min(tr_sizes), np.max(tr_sizes), T))

        if trials is None:
            trials = np.arange(Ntr)

        psths = np.zeros([T, num_psths])
        df_count = np.zeros([T, num_psths])

        if len(trials) > 0:
            for ii in trials:
                if len(self.block_inds[ii]) < T:
                    trT = len(self.block_inds[ii])
                else: 
                    trT = T
                psths[:trT, :] += R[self.block_inds[ii][:trT], :] * dfs[self.block_inds[ii][:trT], :]
                df_count[:trT, :] += dfs[self.block_inds[ii][:trT], :]
            
            psths = np.divide( psths, np.maximum(df_count, 1.0) )

        return psths
    # END SensoryBase.calculate_psths()

    def construct_LVtents( self, tent_spacing=12 ):
        """
        Constructs tent-basis-style trial-based tent function
        
        Args: 
            tent_spacing: the spacing of the tent functions

        Returns:
            XLV: the design matrix
            LVdims: the dimensions of the design matrix
        """
        
        if self.block_inds is not None:
            # Compute minimum and maximum trial size
            Ntr = len(self.block_inds)
            min_trial_size, max_trial_size = len(self.block_inds[0]), 0
            for ii in range(Ntr):
                if len(self.block_inds[ii]) < min_trial_size:
                    min_trial_size = len(self.block_inds[ii])
                if len(self.block_inds[ii]) > max_trial_size:
                    max_trial_size = len(self.block_inds[ii])
                    
            if tent_spacing > min_trial_size:
                print('Using one LV per trial')
                XLV = np.zeros((self.NT, Ntr), dtype=np.float32)
                for tr in range(Ntr):
                    XLV[self.block_inds[tr], tr] = 1.0
                LVdims = [1, Ntr]
            else:
                # automatically wont have any anchors past min_trial_size
                anchors = np.arange(0, min_trial_size, tent_spacing) 
                # Generate master tent_basis
                trial_tents = self.design_matrix_drift(
                    max_trial_size, anchors, zero_left=False, zero_right=True, const_right=False)
                num_tents_tr = trial_tents.shape[1]
                num_tents = num_tents_tr * Ntr
                LVdims = [Ntr, num_tents_tr] 
                XLV = np.zeros((self.NT, num_tents), dtype=np.float32)
                for tr in range(Ntr):
                    L = len(self.block_inds[tr])
                    vslice = np.zeros([L, num_tents], dtype=np.float32)
                    vslice[:, tr*num_tents_tr+np.arange(num_tents_tr)] = trial_tents[:L, :]
                    XLV[self.block_inds[tr], :] = deepcopy(vslice)
        else:
            #print('Trial-less LV setup not implemented yet, but should be easy.'
            anchors = np.arange(0, self.NT, tent_spacing) 
            XLV = self.design_matrix_drift(
                max_trial_size, anchors, zero_left=False, zero_right=True, const_right=False)
            num_tents = XLV.shape[1]
            LVdims = [1, num_tents]
        return XLV, LVdims  # numpy array
    # END SensoryBase.construct_LVtents()
    
    def setup_trial_LVs( self ):
        """
        Usage: XLV = dataset.setup_trial_LVs()
        
        This makes design matrix as input for LVlayer (indexed LV for each trial) and outputs X, LVdims 
        to actually figure out which LVs correspond to which trial, once it is done

        Returns:
            X: the design matrix
        """

        num_trials = len(self.block_inds)

        # LVs indexed 0-num_trials
        X = np.zeros([self.NT, 1], dtype=np.float32)
        for ii in range(num_trials):
            X[self.block_inds[ii]] = ii

        return X
    # END SensoryBase.setup_trial_LVs()

    def setup_LVLayer_input( self, tent_spacing=10, trsize=None ):
        """
        Usage: X, filter_dims = data.setup_LVLayer_input( tent_spacing=10, trsize=None)

        Sets up tent-basis-input to LVLayer (part of NDNT code)
        This preserves all data by using multiple effective trials for long trials. 

        Args:
            tent_spacing: the spacing of the tent functions
            trsize: the size of the trials
        """
        # Determine trial_size
        Ntr = len(self.block_inds)
        trsizes=[]
        for ii in range(Ntr):
            trsizes.append(len(self.block_inds[ii]))
        if trsize is None:
            L0 = np.median(trsizes)

        X = np.ones([self.NT, 3])
        ramp_down = 1.0-(np.arange(tent_spacing))/tent_spacing
        NW0 = int(np.floor(L0/tent_spacing))+1  # how many LVs in standard trial
        print( "Trial size = %d, %d LV indices per trial"%(L0, NW0) )

        Ntr_eff = 0
        LVindex_count = 0
        for ii in range(Ntr):
            ts = self.block_inds[ii]
            L = len(ts)
            NLVts = int(np.floor(L/tent_spacing)+1)  # this gets one on edge, given partials are pegged at 1
            assigned_LVs_inds = np.arange(NLVts) + LVindex_count
            
            # Map these LVs 
            X[ts, 0] = np.repeat(assigned_LVs_inds[:-1], tent_spacing)[:L]
            X[ts, 1] = np.repeat(assigned_LVs_inds[1:], tent_spacing)[:L]
            X[ts, 2] = np.tile(ramp_down, int(np.floor(L/tent_spacing)))[:L]  # this will leave the last bit ones
            
            # Need to do multiples of trial length so that smoothing works
            num_trial_blocks = int(np.ceil(L/L0))
            LVindex_count += num_trial_blocks * NW0
            Ntr_eff += num_trial_blocks
            
        filter_dims = [Ntr_eff, NW0]
        print("%d time points, %d LV indices"%(self.NT, LVindex_count))
        return X, filter_dims
        # END SensoryBase.setup_LVLayer_input()

    @staticmethod
    def design_matrix_drift( NT, anchors, zero_left=True, zero_right=False, const_left=False, const_right=False, to_plot=False):
        """
        Produce a design matrix based on continuous data (s) and anchor points for a tent_basis.
        Here s is a continuous variable (e.g., a stimulus) that is function of time -- single dimension --
        and this will generate apply a tent basis set to s with a basis variable for each anchor point. 
        The end anchor points will be one-sided, but these can be dropped by changing "zero_left" and/or
        "zero_right" into "True".

        Args: 
            NT: length of design matrix
            anchors: list or array of anchor points for tent-basis set
            zero_left, zero_right: boolean whether to drop the edge bases (default for both is False)
            const_left, const_right: boolean whether to make constant basis on left/right (default for both is False)
            to_plot: whether to plot the design matrix
        
        Returns:
            X: design matrix that will be NT x the number of anchors left after zeroing out left and right
        """
        anchors = list(anchors)
        if anchors[0] > 0:
            if not const_left:
                anchors = [0] + anchors
        #if anchors[-1] < NT:
        #    anchors = anchors + [NT]
        NA = len(anchors)

        X = np.zeros([NT, NA])
        for aa in range(NA):
            if aa > 0:
                dx = anchors[aa]-anchors[aa-1]
                X[range(anchors[aa-1], anchors[aa]), aa] = np.arange(dx)/dx
            if aa < NA-1:
                dx = anchors[aa+1]-anchors[aa]
                X[range(anchors[aa], anchors[aa+1]), aa] = 1-np.arange(dx)/dx

        if zero_left:
            X = X[:, 1:]
        elif const_left:  # makes constant from first anchor back to origin -- wont work without zero-left
            X[range(anchors[0]), 0] = 1.0

        if const_right:
            X[range(anchors[-1], NT), -1] = 1.0

        if zero_right:
            X = X[:, :-1]

        if to_plot:
            import matplotlib.pyplot as plt
            plt.imshow(X.T, aspect='auto', interpolation='none')
            plt.show()

        return X
    
    @staticmethod
    def construct_onehot_design_matrix( stim=None, return_categories=False ):
        """
        Construct one-hot design matrix from stimulus.

        The stimulus should be numpy -- not meant to be used with torch currently.
        
        Args:
            stim: the stimulus to construct one-hot design matrix from
            return_categories: whether to return the categories

        Returns:
            OHmatrix: the one-hot design matrix
        """
        assert stim is not None, "Must pass in stimulus"
        assert len(stim.shape) < 3, "Stimulus must be one-dimensional"
        assert isinstance( stim, np.ndarray ), "stim must be a numpy array"

        category_list = np.unique(stim)
        NSTIM = len(category_list)
        assert NSTIM < 50, "Must have less than 50 classifications in one-hot: something wrong?"
        OHmatrix = np.zeros([stim.shape[0], NSTIM], dtype=np.float32)
        for ss in range(NSTIM):
            OHmatrix[stim == category_list[ss], ss] = 1.0
        
        if return_categories:
            return OHmatrix, category_list
        else:
            return OHmatrix
    # END staticmethod.construct_onehot_design_matrix()

    def avrates( self, inds=None ):
        """
        Calculates average firing probability across specified inds (or whole dataset)
        -- Note will respect datafilters
        -- will return precalc value to save time if already stored

        Args:
            inds: the indices to calculate the average firing probability across

        Returns:
            avRs: the average firing probability
        """
        if inds is None:
            inds = range(self.NT)
        if len(self.cells_out) == 0:
            cells = np.arange(self.NC)
        else:
            cells = self.cells_out

        if len(inds) == self.NT:
            # then calculate across whole dataset
            if self.avRs is not None:
                # then precalculated and do not need to do
                return self.avRs[cells]

        # Otherwise calculate across all data
        if self.preload:
            Reff = np.sum(self.dfs * self.robs, axis=0)
            Teff = np.maximum(np.sum(self.dfs, axis=0), 1)
            return (Reff/Teff)[cells]
        else:
            print('Still need to implement avRs without preloading')
            return None
    # END SensoryBase.avrates()

    def crossval_setup(self, folds=5, random_gen=False, test_set=False, verbose=False, which_fold=None):
        """
        This sets the cross-validation indices up We can add featuers here. Many ways to do this
        but will stick to some standard for now. It sets the internal indices, which can be read out
        directly or with helper functions. Perhaps helper_functions is the best way....
        
        Args:
            folds: the number of folds to use
            random_gen: whether to pick random fixations for validation or uniformly distributed
            test_set: whether to set aside first an n-fold test set, and then within the rest n-fold train/val sets
            verbose: whether to print out the number of fixations in each set
            which_fold: if a different fold other than the 'middle' fold should be chosen. Note in the case of
                a test_set, it specifies the test-set fold, and val_set is not moveable (in this code)

        Returns:
            None: sets internal variables test_inds, train_inds, val_inds
        """
        assert self.used_inds is not None, "Must first specify valid_indices before setting up cross-validation."

        # Reflect block structure
        Nblks = len(self.block_inds)
        val_blk1, tr_blk1 = self.fold_sample(Nblks, folds, random_gen=random_gen, which_fold=which_fold)

        if test_set:
            self.test_blks = val_blk1
            val_blk2, tr_blk2 = self.fold_sample(len(tr_blk1), folds, random_gen=random_gen)
            self.val_blks = tr_blk1[val_blk2]
            self.train_blks = tr_blk1[tr_blk2]
        else:
            self.val_blks = val_blk1
            self.train_blks = tr_blk1
            self.test_blks = []

        if verbose:
            print("Partitioned %d blocks total: tr %d, val %d, te %d"
                %(len(self.test_blks)+len(self.train_blks)+len(self.val_blks),len(self.train_blks), len(self.val_blks), len(self.test_blks)))  

        # Now pull indices from each saccade 
        tr_inds, te_inds, val_inds = [], [], []
        for nn in self.train_blks:
            tr_inds += list(deepcopy(self.block_inds[nn]))
        for nn in self.val_blks:
            val_inds += list(deepcopy(self.block_inds[nn]))
        for nn in self.test_blks:
            te_inds += list(deepcopy(self.block_inds[nn]))

        if verbose:
            print( "Pre-valid data indices: tr %d, val %d, te %d" %(len(tr_inds), len(val_inds), len(te_inds)) )

        # Finally intersect with used_inds
        if len(self.used_inds) > 0:
            self.train_inds = np.array(list(set(tr_inds) & set(self.used_inds)))
            self.val_inds = np.array(list(set(val_inds) & set(self.used_inds)))
            self.test_inds = np.array(list(set(te_inds) & set(self.used_inds)))

            if verbose:
                print( "Valid data indices: tr %d, val %d, te %d" %(len(self.train_inds), len(self.val_inds), len(self.test_inds)) )
        else:
            self.train_inds = tr_inds
            self.val_inds = val_inds
            self.test_inds = te_inds
            
        # make the inds and blks into numpy arrays
        self.train_inds = np.array(self.train_inds)
        self.val_inds = np.array(self.val_inds)
        self.test_inds = np.array(self.test_inds)
        self.train_blks = np.array(self.train_blks)
        self.val_blks = np.array(self.val_blks)
        self.test_blks = np.array(self.test_blks)
        
    # END SensoryBase.crossval_setup

    def fold_sample( self, num_items, folds, random_gen=False, which_fold=None):
        """
        This really should be a general method not associated with self
        
        Args:
            num_items: the number of items to sample
            folds: the number of folds to use
            random_gen: whether to pick random fixations for validation or uniformly distributed
            which_fold: which fold to use

        Returns:
            val_items: the validation items
            rem_items: the remaining items
        """
        if random_gen:
            num_val = int(num_items/folds)
            tmp_seq = np.random.permutation(num_items)
            val_items = np.sort(tmp_seq[:num_val])
            rem_items = np.sort(tmp_seq[num_val:])
        else:
            if which_fold is None:
                offset = int(folds//2)
            else:
                offset = which_fold%folds
            val_items = np.arange(offset, num_items, folds, dtype=np.int64)
            rem_items = np.delete(np.arange(num_items, dtype=np.int64), val_items)
        return val_items, rem_items

    def speckledXV_setup( self, folds=5, random_gen=False ):
        """
        Produce data-filter masks for training and XV speckles
        Will be produced for whole dataset, and must be reduced if cells_out used

        Args:
            folds: the number of folds to use
            random_gen: whether to pick random fixations for validation or uniformly distributed

        Returns:
            None
        """
        if len(self.block_inds) > 0: # will use time points rather than trials if no trial structure in dataset
            Ntr = len(self.block_inds)
        else:
            Ntr = self.NT
            print('  Warning: speckled is on time-point basis')
        
        # Choose trials to leave out for each unit
        self.Mval = torch.zeros(self.dfs.shape, dtype=torch.float32)
        self.Mtrn = torch.ones(self.dfs.shape, dtype=torch.float32)
        for cc in range(self.NC):
            ival,_ = self.fold_sample( 
                Ntr, folds=folds, random_gen=random_gen, which_fold=cc%folds)
            if len(self.block_inds) == 0:
                self.Mval[ival, cc] = 1.0
                self.Mtrn[ival, cc] = 0.0
            else:
                for tr in ival:
                    self.Mval[self.block_inds[tr], cc] = 1.0
                    self.Mtrn[self.block_inds[tr], cc] = 0.0

        if len(self.cells_out) > 0:
            self.Mtrn_out = deepcopy(self.Mtrn[:, self.cells_out])
            self.Mval_out = deepcopy(self.Mval[:, self.cells_out])
    # END SensoryBase.speckledXV_setup
    
    def set_speckledXV(self, val=True, folds=5, random_gen=False):
        """
        Set up speckled cross-validation with data-filter masks

        Args:
            val: whether to set up speckled cross-validation
            folds: the number of folds to use
            random_gen: whether to pick random fixations for validation or uniformly distributed

        Returns:
            None
        """
        self.speckled = val
        if val:
            if self.Mval is None:
                self.speckledXV_setup(folds=folds, random_gen=random_gen) 
            if len(self.cells_out) > 0:
                self.Mval_out = self.Mval[:, self.cells_out]
                self.Mtrn_out = self.Mtrn[:, self.cells_out]
            else:
                self.Mval_out = None
                self.Mtrn_out = None
    # END SensoryBase.set_speckledXV

    def make_data_dicts(self, device=None, all=False):
        """
        Usage: train_ds, val_ds = dataset.make_data_dicts( device=None, all=False )
        
        Produce generic datasets on device of choice
        device defaults to cuda-0
        If <all> is True, then will make single dataset without XVal

        Args:
            device: the device to put the datasets on
            all: whether to use all data
        
        Returns:
            train_ds: the training dataset
            val_ds: the validation dataset
        """
        from NTdatasets.generic import GenericDataset

        if device is None:
            print('  Datasets will be on cuda:0')
            device = torch.device('cuda:0')

        if self.speckled or all:
            trn_inds = range(len(self))
            val_inds = range(len(self))
        else:
            assert not self.block_sample, "trial-sample will not work with generic datasets"
            trn_inds = self.train_inds
            val_inds = self.val_inds

        ks = self[:2].keys()
        dictTRN, dictVAL = {}, {}
        for k in ks:
            if not (k in ['dfs', 'Mtrn', 'Mval']):
                dictTRN[k] = deepcopy(self[trn_inds][k])
                dictVAL[k] = deepcopy(self[val_inds][k])

        if all:
            dictTRN['dfs'] = self[:]['dfs']
        else:
            if self.speckled:
                assert ('Mtrn' in ks), "speckled not set on dataset"
                dictTRN['dfs'] = self[:]['dfs'] * self[:]['Mtrn']
                dictVAL['dfs'] = self[:]['dfs'] * self[:]['Mval']
            else:
                dictTRN['dfs'] = self[trn_inds]['dfs']
                dictVAL['dfs'] = self[val_inds]['dfs']

        train_ds = GenericDataset( dictTRN, device=device )
        if all:
            return train_ds

        val_ds = GenericDataset( dictVAL, device=device )
        return train_ds, val_ds
    # END SensoryBase.make_data_dicts()

    def get_max_samples(self, gpu_n=0, history_size=1, nquad=0, num_cells=None, buffer=1.2):
        """
        get the maximum number of samples that fit in memory -- for GLM/GQM x LBFGS

        Args:
            gpu_n: the gpu number to use
            history_size: the history size
            nquad: the number of quadrature points
            num_cells: the number of cells to use
            buffer: the buffer to use

        Returns:
            maxsamples: the maximum number of samples that fit in memory
        """
        if gpu_n == 0:
            device = torch.device('cuda:0')
        else:
            device = torch.device('cuda:1')

        if num_cells is None:
            num_cells = self.NC
        
        t = torch.cuda.get_device_properties(device).total_memory
        r = torch.cuda.memory_reserved(device)
        a = torch.cuda.memory_allocated(device)
        free = t - (a+r)

        data = self[0]
        mempersample = data['stim'].element_size() * data['stim'].nelement() + 2*data['robs'].element_size() * data['robs'].nelement()
    
        mempercell = mempersample * (nquad+1) * (history_size + 1)
        buffer_bytes = buffer*1024**3

        maxsamples = int(free - mempercell*num_cells - buffer_bytes) // mempersample
        print("# samples that can fit on device: {}".format(maxsamples))
        return maxsamples
    # END .get_max_samples

    def assemble_stimulus( self, **kwargs ):
        """
        This function is meant to be overloaded by child classes

        Returns:
            None
        """
        print("SensoryBase: assemble_stimulus not implemented in class child.")
        return

    def __getitem__(self, idx):
        return {}

    def __len__(self):
        return self.robs.shape[0]

    @staticmethod
    def is_int( val ):
        """
        Returns a boolean as to whether val is one of many types of integers

        Returns:
            True if val is an integer, False otherwise
        """
        if isinstance(val, int) or \
            isinstance(val, np.int32) or isinstance(val, np.int64) or \
            (isinstance(val, np.ndarray) and (len(val.shape) == 0)):
            return True
        else:
            return False

    @staticmethod
    def index_to_array( index, max_val ):
        """
        This converts any for index to dataset, including slices, and plain ints, into numpy array

        Args:
            index: the index to convert
            max_val: the maximum value to use

        Returns:
            index: the converted index
        """
        if isinstance(index, slice):
            start = index.start
            stop = index.stop
            step = index.step
            if start is None:
                start = 0
            if stop is None:
                stop = max_val
            if step is None:
                step = 1
            return np.arange(start,stop, step)
        elif SensoryBase.is_int(index):
            return [index]
        elif isinstance(index, list):
            return np.array(index, dtype=np.int64)
        return index
