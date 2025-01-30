
#from inspect import BlockFinder
import os
import numpy as np
#import scipy.io as sio

import torch
#from torch.utils.data import Dataset
import NDNT.utils as utils
#from NDNT.utils import download_file, ensure_dir
from copy import deepcopy
import h5py
from NTdatasets.sensory_base import SensoryBase
from NTdatasets.generic import GenericDataset


class MultiData( MultiClouds ):
    """
    Shell built on top of multi-cloud that adds automatic (and manual) data updates
    """

    def __init__(self,
        expt_list,
        datadir, 
        num_lags=10, 
        include_MUs=False,
        drift_interval=None,
        #trial_sample=True,
        luminance_only=True,
        LMS=False,
        binocular=False, # whether to include separate filters for each eye
        eye_config=3,  # 0 = all, 1, 2, and 3 are options (3 = binocular)
        eye_contiguous=True, # whether to only use eye_config data that is contiguous 
        #cell_lists = None,
        test_set = True, # whether to include a test-set in cross-validation
        device=torch.device('cpu')):

        super().__init__(
                filenames=expt_list, datadir=datadir, num_lags=num_lags,
                include_MUs=include_MUs, drift_interval=drift_interval,
                trial_sample=True, luminance_only=luminance_only, LMS=LMS,
                binocular=binocular, eye_config=eye_config, eye_contiguous=eye_contiguous, # whether to only use eye_config data that is contiguous 
                cell_lists = None, test_set=test_set, device=device )
        
        # Build automatic file-read given expt_list        
    # END MultiData.__init__()            
            

class MultiClouds(SensoryBase):
    """
    -- can load batches from multiple datasets
    -- hdf5 files must have the following information:
        Robs
        RobsMU
        stim: 4-d stimulus: time x nx x ny x color
        block_inds: start and stop of 'trials' (perhaps fixations for now)
        other things: saccades? or should that be in trials? 

    Constructor will take eye position, which for now is an input from data
    generated in the session (not on disk). It should have the length size 
    of the total number of fixations x1.

    Input arguments (details):
        stim_crop = None, should be of form [x1, x2, y1, y2] where each number is the 
            extreme point to be include as an index, e.g. range(x1, x2+1), ... 
    """

    def __init__(self,
        filenames,
        datadir, 
        num_lags=10, 
        include_MUs=False,
        drift_interval=None,
        trial_sample=True,
        luminance_only=True,
        LMS=False,
        binocular=False, # whether to include separate filters for each eye
        eye_config=3,  # 0 = all, 1, 2, and 3 are options (3 = binocular)
        eye_contiguous=True, # whether to only use eye_config data that is contiguous 
        cell_lists = None,
        test_set = True, # whether to include a test-set in cross-validation
        device=torch.device('cpu')):
        """
        Constructor options
        
        Args:
            filenames (list): list of strings of the filenames to be loaded
            datadir (str): directory where the data is stored
            num_lags (int): number of lags to include in the stimulus
            include_MUs (bool): whether to include multi-units in the dataset
            drift_interval (int): number of blocks to include in the drift term
            trial_sample (bool): whether to sample trials for train/val/test
            luminance_only (bool): whether to only include luminance in the stimulus
            binocular (bool): whether to include separate filters for each eye
            eye_config (int): which eye configuration to use (0=all, 1=left, 2=right, 3=binocular)
            eye_contiguous (bool): whether to only use contiguous eye configurations
            cell_lists (list): list of lists of cell indices to include in the dataset
            device (torch.device): device to store the data on
        """

        super().__init__(
            filenames=filenames, datadir=datadir, device=device,
            time_embed=0, num_lags=num_lags, include_MUs=include_MUs, 
            drift_interval=drift_interval, trial_sample=trial_sample)

        # Done in parent constructor
        #self.datadir = datadir
        #self.filenames = filenames
        #self.device = device
        #self.num_lags = 10  # default: to be set later
        #if time_embed == 2:
        #    assert preload, "Cannot pre-time-embed without preloading."
        self.time_embed = None  # not set
        #self.preload = preload

        self.Nexpts = len(filenames)

        # Stim-specific
        self.eye_config = eye_config
        self.eye_contiguous = eye_contiguous
        self.binocular = binocular
        self.luminance_only = luminance_only
        self.includeMUs = include_MUs
        self.generate_Xfix = False
        self.output_separate_eye_stim = False
        self.expt_stims = [None]*self.Nexpts
        self.L = None
        self.LMS = LMS
        
        self.start_t = 0
        self.drift_interval = drift_interval

        # Set up to store default train_, val_, test_inds -- in SensoryBase
        #self.test_inds = None
        #self.val_inds = None
        #self.train_inds = None
        #self.used_inds = []

        # Data to construct and store in memory -- some in SensoryBase
        #self.stim = []
        #self.dfs = []
        #self.robs = []
        self.used_inds = []
        self.NT = 0

        # build index map -- exclude variables already set in sensory-base
        #self.num_blks = np.zeros(len(filenames), dtype=int)
        #self.file_index = [] # which file the block corresponds to
        #self.stim_dims = None

        #self.unit_ids = []
        #self.num_units, self.num_SUs, self.num_MUs = [], [], []
        #self.SUs = []
        #self.NC = 0    
        #self.block_inds = []
        #self.block_filemapping = []
        #self.include_MUs = include_MUs
        #self.SUinds = []
        #self.MUinds = []
        #self.cells_out = []  # can be list to output specific cells in get_item
        #self.avRs = None

        self.fhandles = [h5py.File(os.path.join(datadir, sess + '.mat'), 'r') for sess in self.filenames]
        self.file_info = [None] * self.Nexpts
        self.exptNT = np.zeros(self.Nexpts, dtype=np.int64)  # used to be fileNT
        self.exptNBLK = np.zeros(self.Nexpts, dtype=np.int64) # used to be fileNBLK
        self.exptNC = np.zeros(self.Nexpts, dtype=np.int64) # used to be 
        self.exptNA = np.zeros(self.Nexpts, dtype=np.int64) # for the drift term -- used to be fileNA
        self.expt_tstart = np.zeros(self.Nexpts, dtype=np.int64) # used to be file_tstart
        self.expt_blkstart = np.zeros(self.Nexpts, dtype=np.int64) # used to be file_blkstart

        # The following are specific to the cells and tranges selected for the particular instance of the dataset
        # the file_info has the full information about experimental data (from the files themselves)
        self.tranges = [None] * self.Nexpts
        self.cranges = [None] * self.Nexpts
        self.block_inds = []
        tcount, blkcount = 0, 0

        # Structure of file on time/trial level 
        for ee in range(self.Nexpts):
            # This is a catalog of what is in the files themselves (without filtering for this data)
            self.file_info[ee] = self.read_file_info(ee, filenames[ee])

            # Set initial crange (all cells in absence of clists) 
            if cell_lists is None:
                self.cranges[ee] = np.arange(self.file_info[ee]['NC'], dtype=np.int64)
            elif len(cell_lists[ee]) > 0:
                if isinstance( cell_lists[ee], list):
                    cell_lists[ee] = np.array(cell_lists[ee], dtype=np.int64)
                self.cranges[ee] = deepcopy(cell_lists[ee]).astype(np.int64)
            self.exptNC[ee] = len(self.cranges[ee])

            # Parse timing and block information given eye_config
            if self.eye_config == 0:
                self.tranges[ee] = np.arange(self.file_info[ee]['NT'])
            else:
                self.tranges[ee] = np.where(self.file_info[ee]['LRpresent'] == self.eye_config)[0]
                # Check for contiguous option (throw away disjoint eye config)
                if self.eye_contiguous & (self.eye_config > 0):
                    tbreaks = np.where(np.diff(self.tranges[ee]) > 1)[0]
                    if len(tbreaks) > 0:
                        print("  Disjoint data exists with this eye_config -- trunctating to first section.")
                        self.tranges[ee] = self.tranges[ee][range(tbreaks[0]+1)]

            #if self.include_MUs:
            #    self.fileNC[ff] = self.file_info[ff]['NSUs'] + self.file_info[ff]['NMUs']
            #else:
            #    self.fileNC[ff] = self.file_info[ff]['NSUs']

            # Consolidate valid t-ranges based on binocular choice

            # Make one long block-list -- this is now down in assemble_robs()
            #NBLK = self.file_info[ff]['trial_info'].shape[0]
            #for bb in range(NBLK):
            #    self.block_inds.append( 
            #        tcount + np.arange(self.file_info[ff]['trial_info'][bb, 0], self.file_info[ff]['trial_info'][bb, 1]) )
            #self.fileNBLK[ff] = self.file_info[ff]['trial_info'].shape[0]
            #self.file_blkstart[ff] = blkcount
            #blkcount += NBLK

            self.exptNT[ee] = len(self.tranges[ee])
            self.expt_tstart[ee] = tcount
            tcount += self.exptNT[ee]

        # Assemble robs and dfs given current information
        self.NT = tcount
        self.NC = np.sum(self.exptNC)

        # Prune cranges and assemble robs (and blocks etc)
        if cell_lists is not None:
            self.modify_included_cells(cell_lists)
            # this will automatically assemble_robs at the end
        else:
            self.assemble_robs()

        # MAKE DRIFT TERM BASED ON tranges determined by eye_config and nothing else
        if self.drift_interval is None:
            self.Xdrift = None
        else:
            Xdrift_expts = []
            Nanchors_tot = 0
            for ff in range(self.Nexpts):
                # Want to go across the whole experiment relevant given the eye configuration
                Nanchors = np.ceil(self.exptNBLK[ff]/self.drift_interval).astype(int)
                anchors = np.zeros(Nanchors, dtype=np.int64)
                for bb in range(Nanchors):
                    anchors[bb] = self.block_inds[self.drift_interval*bb][0]
                #self.Xdrift = utils.design_matrix_drift( self.NT, anchors, zero_left=False, const_right=True)
                Xdrift_expts.append(self.design_matrix_drift( self.exptNT[ff], anchors, zero_left=False, const_right=True))
                self.exptNA[ff] = Nanchors # store the number of anchors for each file
                Nanchors_tot += Nanchors

            # Assemble whole drift matrix
            self.Xdrift = np.zeros( [self.NT, Nanchors_tot], dtype=np.float32 )
            anchor_count = 0
            for ff in range(self.Nexpts):
                tslice = np.zeros( [self.exptNT[ff], Nanchors_tot], dtype=np.float32 )
                tslice[:, anchor_count+np.arange(Xdrift_expts[ff].shape[1])] = Xdrift_expts[ff]
                self.Xdrift[self.expt_tstart[ff]+np.arange(self.exptNT[ff]), :] = deepcopy(tslice)
                anchor_count += Xdrift_expts[ff].shape[1]

        print( "  MULTIDATASET %d expts: %d total time steps, %d units"%(self.Nexpts, self.NT, self.NC) )
 
        # Assemble current list of fixations
        self.sacc_inds = [None]*self.Nexpts
        self.stim_shifts = [None]*self.Nexpts
        to_assemble = True
        for ff in range(self.Nexpts):
            sacc_inds = np.array(self.fhandles[ff]['sacc_inds'], dtype=np.int64)
            if len(sacc_inds) < 2: # assume its no good
                sacc_inds = None
                to_assemble = False
            else:
                if len(sacc_inds.shape) > 1:
                    sacc_inds[:, 0] += -1  # convert to python so range works
                else:
                    sacc_inds = None
                    to_assemble = False

            self.sacc_inds[ff] = deepcopy(sacc_inds)

        #if to_assemble:
        #    self.assemble_saccade_inds()

        ### Set up train, val and test inds and blks
        self.crossval_setup(test_set=test_set)
    # END MultiClouds.__init__

    def read_file_info( self, file_n, filename ):
        """
        Initial processing of each file to pull out salient info needed to put data together for
        multiexperiment dataset: particularly for building stim and responses, and trial-indexing
        
        Args:
            file_n (int): index of the file
            filename (str): name of the file

        Returns:
            dict: dictionary of file information
        """

        f = self.fhandles[file_n]
        NT, NSUs = f['Robs'].shape
        # Check for valid RobsMU
        if len(f['RobsMU'].shape) > 1:
            NMUs = f['RobsMU'].shape[1]
        else: 
            NMUs = 0

        # Unit information
        if self.includeMUs:
            NC = NSUs + NMUs
        channel_map = np.array(f['Robs_probe_ID'], dtype=np.int64)[0, :]
        channel_ratings = np.array(f['Robs_rating']).squeeze()
        if (NMUs > 0) & self.includeMUs:
            channel_map = np.concatenate( 
                (channel_map, np.array(f['RobsMU_probe_ID'], dtype=np.int64).squeeze()), axis=0)
            channel_ratings = np.concatenate( 
                (channel_ratings, np.array(f['RobsMU_rating'], dtype=np.int64).squeeze()), axis=0)

        # Block information
        blk_inds = np.array(f['block_inds'], dtype=np.int64)
        blk_inds[:, 0] += -1  # convert to python so range works
        # Check to make sure not inverted blk_inds (older version of data)
        if blk_inds.shape[0] == 2:
            print('WARNING: blk_inds is stored old-style: transposing')
            blk_inds = blk_inds.T
        NBLK = blk_inds.shape[0]

        # Stim information
        fix_loc = np.array(f['fix_location'], dtype=np.int64).squeeze()
        fix_size = np.array(f['fix_size'], dtype=np.int64).squeeze()
        stim_scale = np.array(f['stimscale'], dtype=np.int64).squeeze()
        stim_locsLP = np.array(f['stim_location'], dtype=np.int64),

        if len(stim_locsLP) == 1: # then list of arrays for some reason
            print('  FILE_INFO: stim_locsLP list again -- ok but output check')
            stim_locsLP = stim_locsLP[0]
        stim_locsET = np.array(f['ETstim_location'], dtype=np.int64)

        if 'stim_location_deltas' in f:
            stim_location_deltas = np.array(f['stim_location_deltas'], dtype=np.int64).T
        else:
            stim_location_deltas = None
        if 'cloud_binary' in f:
            cloud_binary = np.array(f['cloud_binary'], dtype=np.int64)
            # 0 = normal clouds, 1 = binary clouds, 2 = contrast-matched clouds
            #cloud_area = np.array(f['cloud_area'], dtype=np.int64)
            cloud_scale = np.array(f['cloud_scale'], dtype=np.int64)
            cloud_info = np.concatenate( (cloud_binary, cloud_scale), axis=1 )
        else:
            cloud_info = None

        blockIDs = np.array(f['blockID'], dtype=np.int64).squeeze()  # what is this?
        #self.ETtrace = np.array(f['ETtrace'], dtype=np.float32)
        #self.ETtraceHR = np.array(f['ETtrace_raw'], dtype=np.float32)

        # Binocular information
        Lpresent = np.array(f['useLeye'], dtype=int)[:,0]
        Rpresent = np.array(f['useReye'], dtype=int)[:,0]
        LRpresent = Lpresent + 2*Rpresent

        valid_inds = np.array(f['valid_data'], dtype=np.int64)-1  #range(self.NT)  # default -- to be changed at end of init
        # Make valid data here
        valid_data = np.zeros(NT, dtype=np.uint8)
        valid_data[valid_inds] = 1


        # self.tranges set initially by correct eye config 
        #self.tranges[file_n] = deepcopy(tmap)  

            # Remap valid_inds to smaller trange -- does not use val_track: just temp variable
            #val_track = np.zeros(NT, dtype=np.int64)
            #val_track[valid_inds] = 1
            #valid_inds = np.where(val_track[tmap] == 1)[0]

            #NT = len(tmap)
 
            # Remap block_inds to reduced map
            #block_inds, bmap = self.parse_trial_times_expt( file_n, tmap )  # might want to do dynamically
            #bmap = []
            #tcount = 0
            #block_inds = []
            #for bb in range(NBLK):
            #    if blk_inds[bb,0] in tmap:
            #        bmap.append(bb)
            #        NTblk = blk_inds[bb,1]-blk_inds[bb,0]
            #        block_inds.append([tcount, tcount+NTblk])
            #        tcount += NTblk
            #block_inds = np.array(block_inds, dtype=np.int64)

            # probably have to modify blockIDs -- not done yet

        return {
            'filename': filename,
            'NT': NT,
            'NC': NC,
            #'tmap': tmap, 
            #'trial_info': block_inds,  # begin/end of each trial in tmap -- might want to generate dynamically
            'blk_inds': blk_inds,  # begin/end of each trial in absolute trial time
            'LRpresent': LRpresent,
            #'valid_inds': valid_inds.squeeze(),
            'valid_data': valid_data,
            'blockIDs': blockIDs, # number corresponding to see that identifies stim in trial
            'NSUs': NSUs,
            'NMUs': NMUs,
            'channel_map': channel_map,
            'channel_ratings': channel_ratings,
            'fix_loc': fix_loc,
            'fix_size': fix_size,
            'stim_scale': stim_scale,
            'stim_locsLP': stim_locsLP,
            'stim_locsET': stim_locsET,
            'stim_location_deltas': stim_location_deltas,
            'cloud_info': cloud_info}
    # END MultiClouds.read_file_info()  

    def parse_trial_times_expt( self, expt_n, trange ):
        """
        Makes block_inds for the trange within a given experiment, indexed to the beginning of the expt
        
        Args: 
            expt_n (int): which experiment
            trange (array): times in the experiment to include

        Returns:
            e_block_inds: a list of the indices associated with each trial
            bmap: list of trials that were included within the experiment
        """

        blk_inds = self.file_info[expt_n]['blk_inds']  # begin/end of trials in absolute expt time
        bmap = []  # which overall trials are included 
        tcount = 0
        e_block_inds = []
        for bb in range(blk_inds.shape[0]):
            if blk_inds[bb,0] in trange:
                bmap.append(bb)
                NTblk = blk_inds[bb,1]-blk_inds[bb,0]
                e_block_inds.append([tcount, tcount+NTblk])
                tcount += NTblk
        e_block_inds = np.array(e_block_inds, dtype=np.int64)   # num_included_trials x 2
        return e_block_inds, bmap
    # END MultiClouds.parse_trial_times_expt()

    def modify_included_cells(self, clists, expt_n=None, reset_cell_lists=False):
        """
        Modify the included cells in the dataset

        Args:
            clists (list): list of lists of cell indices to include in the dataset
            expt_n (int): index of the experiment to modify
            reset_cell_lists (Boolean): whether to reset DFs to default if not included in list
        Returns:
            None
        """

        if expt_n is None:
            expts = np.arange(self.Nexpts)
            assert len(clists) == self.Nexpts, "Number of cell_lists must match number of experiments."
        else:
            expts = [expt_n]
            clists_tmp = [[]]*self.Nexpts
            clists_tmp[expt_n] = deepcopy(clists)
            clists = clists_tmp

        for ff in expts:
            if len(clists[ff]) > 0:  # then modify this experiment
                assert np.max(clists[ff]) < (self.file_info[ff]['NSUs'] + self.file_info[ff]['NMUs']), "clists too large"
                self.cranges[ff] = deepcopy(clists[ff])
            else: # don't modify experiment: use default (full) cranges
                if reset_cell_lists:
                    if self.cranges[ff] is None:
                        if self.includeMUs:
                            self.cranges[ff] = np.arange(self.file_info[ff]['NSUs']+self.file_info[ff]['NMUs'])
                        else:
                            self.cranges[ff] = np.arange(self.file_info[ff]['NSUs'])

            self.exptNC[ff] = len(self.cranges[ff])
        self.NC = np.sum(self.exptNC)

        self.assemble_robs()
    # END MultiClouds.modify_included_cells()

    def modify_expt_time_range(self, trange, expt_n=0, absolute_time_scale=True):
        """
        there is an existing trange -- trange assumed to be absolute_time_scale, but could be mod
        #self.tranges[ff] = self.file_info[ff]['tmap']
        # will have to change self.tranges, re-assemble_robs, and rebuild stim (if built already)
        # will also have to go through and trim out irrelevant trials, and renumber -- how do this?
        """
        if absolute_time_scale:
            self.tranges[expt_n] = np.intersect1d( trange, self.tranges[expt_n])
        else:
            self.tranges[expt_n] = self.tranges[expt_n][trange]
        
        # Recalculate NT and expt_tstarts and exptNT
        self.NT = 0
        for ee in range(self.Nexpts):
            self.expt_tstart[ee] = self.NT
            self.exptNT[ee] = len(self.tranges[ee])
            self.NT += self.exptNT[ee]

        self.assemble_robs()
        print('  Redoing cross-validation indices')
        self.crossval_setup()
    # END MultiClouds.modify_expt_time_range()

    def generate_array_cell_list(self, expt_n=0, which_array=0):
        """
        Formula for generating cell list given channel maps and basic eligibility
        
        Args:
            expt_n (int): index of the experiment
            which_array (int or str): which array to generate the cell list for

        Returns:
            np.array: array of cell indices
        """
        #expt_val = np.where( np.sum( self.robs[:, np.arange(cstart, cstart+self.fileNC[expt_n])], axis=0 ) > 10)[0]

        # Should only do this if have not already reduced channel list
        NC = self.file_info[expt_n]['NSUs']
        if self.includeMUs:
            NC += self.file_info[expt_n]['NMUs']
        assert len(self.cranges[expt_n]) == NC, "Can only generate cell_array if using currently using full cell_list"

        if which_array in [1, 'u', 'U', 'utah']:
            array_cells = np.where(self.file_info[expt_n]['channel_map'] >= 32+128)[0]
        elif which_array in [0, 'l', 'L', 'lam', 'laminar']:
            array_cells = np.where(self.file_info[expt_n]['channel_map'] < 32)[0]
        else: # assume Nform
            array_cells = np.where((self.file_info[expt_n]['channel_map'] >= 32) & (self.file_info[expt_n]['channel_map'] < 32+128))[0]

        cstart = 0
        for ff in range(expt_n):
            cstart += self.exptNC[ff]  # not vetted
        nspks = np.sum(self.robs.astype(np.float32)*self.dfs.astype(np.float32), axis=0)[cstart+array_cells]

        val_array = array_cells[nspks > 20]
        return val_array
    # END MultiClouds.generate_array_cell_list()

    def assemble_robs(self, reset_dfs=False):
        """
        Takes current information (robs and dfs) to make robs and dfs (full version).
        This uses the info in self.tranges() and squares with the file_info.
        It will also re-generate block_inds
        ** Note this can be replaced by using the spike times explicitly
        
        Returns:
            None
        """

        self.robs = np.zeros( [self.NT, self.NC], dtype=np.uint8 )
        self.dfs = np.zeros( [self.NT, self.NC], dtype=np.uint8 )
        self.block_inds = []

        tcount, ccount = 0, 0
        blkcount = 0
        for ff in range(self.Nexpts):
            #NTexpt = self.file_info[ff]['NT']
            NTexpt = len(self.tranges[ff])
            #NSUs = self.file_info[ff]['NSUs']
            #su_list = self.cranges[ff][self.cranges[ff] < NSUs]
            valid_data = self.file_info[ff]['valid_data']

            # Assume cranges are correct
             
            # Classify cell-lists in terms of SUs and MUs            
            R_tslice = np.zeros( [NTexpt, self.NC], dtype=np.int64 )
            df_tslice = np.zeros( [NTexpt, self.NC], dtype=np.uint8 )

            tslice = np.array(self.fhandles[ff]['Robs'], dtype=np.int64)[self.tranges[ff], :]
            if self.include_MUs:
                tslice = np.concatenate( 
                    (tslice, np.array(self.fhandles[ff]['RobsMU'], dtype=np.int64)[self.tranges[ff], :]),
                    axis=1)
            #R_tslice[:, ccount+np.arange(len(su_list))] = deepcopy(tslice[:, su_list])
            R_tslice[:, ccount+np.arange(self.exptNC[ff])] = deepcopy(tslice[:, self.cranges[ff]])
            
            tslice = np.array(self.fhandles[ff]['datafilts'], dtype=np.uint8)[self.tranges[ff], :]
            if self.include_MUs:
                tslice = np.concatenate( 
                    (tslice, np.array(self.fhandles[ff]['datafiltsMU'], dtype=np.uint8)[self.tranges[ff], :]),
                    axis=1)
            # Also take valid_data into account
            #df_tslice[:, ccount+np.arange(len(su_list))] = deepcopy(tslice[:, su_list]) * valid_data[self.tranges[ff], None]
            df_tslice[:, ccount+np.arange(self.exptNC[ff])] = deepcopy(tslice[:, self.cranges[ff]]) * valid_data[self.tranges[ff], None]
            ccount += self.exptNC[ff] #len(su_list)

            #if self.include_MUs:
            #    NMUs = self.file_info[ff]['NMUs']
            #    mu_list = self.cranges[ff][self.cranges[ff] >= NSUs]-NSUs
            #    if reset_dfs:
            #        tslice = np.array(self.fhandles[ff]['RobsMU'], dtype=np.int64)[self.tranges[ff], :]
            #    else: 
            #        tslice = 
            #    R_tslice[:, ccount+np.arange(len(mu_list))] = deepcopy(tslice[:, mu_list])
            #    tslice = np.array(self.fhandles[ff]['datafiltsMU'], dtype=np.int64)[self.tranges[ff], :]
            #    df_tslice[:, ccount+np.arange(len(mu_list))] = deepcopy(tslice[:, mu_list]) * valid_data[self.tranges[ff], None]
            #    ccount += len(mu_list)

            # Check that clipping to uint8 wont screw up any robs
            robs_ceil = np.where(R_tslice > 255)
            if len(robs_ceil[0]) > 0:
                print( "Neurons in expt %d have single-bin spike counts above 255:"%ff, robs_ceil[1] )
                # Currently do nothing -- this willbe modded: but if problems should probably make dfs = 0 there

            # Write tslice into 
            self.robs[tcount+np.arange(NTexpt), :] = deepcopy( R_tslice.astype(np.uint8) )
            self.dfs[tcount+np.arange(NTexpt), :] = deepcopy( df_tslice )
            #self.robs[tcount+np.arange(NTslice), :] = deepcopy( R_tslice[self.tranges[ff], :].astype(np.uint8) )
            #self.dfs[tcount+np.arange(NTslice), :] = deepcopy( df_tslice[self.tranges[ff], :] )

            # Make block_inds
            e_block_inds, _ = self.parse_trial_times_expt(ff, self.tranges[ff] )
            NBLK = e_block_inds.shape[0]
            for bb in range(NBLK):
                self.block_inds.append( 
                    tcount + np.arange(e_block_inds[bb, 0], e_block_inds[bb, 1]) )
            self.exptNBLK[ff] = NBLK
            self.expt_blkstart[ff] = blkcount
            
            blkcount += NBLK
            tcount += NTexpt

        self.NT = tcount
        self.trialfilter_dfs()
    # END MultiClouds.assemble_robs()

    def list_expts( self ):
        """
        Show filenames with experiment number
        """
        for ff in range(self.Nexpts):
            print('  %2d  %s'%(ff, self.filenames[ff]) )
    # END MultiClouds.list_expts()
        
    def updateDF( self, dfs=None, expt_n=0, reduce_cells=False ):
        """
        Import updated DF for given experiment, as numbered (can see with 'list_expts')
        Will check for neurons with no spikes and reduce robs and dataset if reduce_cells=True
        
        Args:
            dfs (np.array): array of new dfs
            expt_n (int): index of the experiment, default=0
            reduce_cells (bool): whether to reduce the number of cells

        Returns:
            None
        """

        assert dfs is not None, "updateDF: forgot dfs, dingbat"
        assert expt_n < self.Nexpts, "updateDF: expt_n too large: not that many experiments"

        # if eye_config, then want to replace whole DFs, or relevant DFs 
        #if dfs.shape[0] != self.file_info[expt_n]['NT']:
        if dfs.shape[0] > len(self.tranges[expt_n]):
            # Assume need to use trange
            dfs = dfs[self.tranges[expt_n], :]
        #if dfs.shape[1] > len(self.cranges[expt_n]):
        #    dfs = dfs[:, self.cranges[expt_n]]

        assert dfs.shape[0] == len(self.tranges[expt_n]) #self.file_info[expt_n]['NT'], "DF file mismatch: wrong length"
        dfs = dfs[:, self.cranges[expt_n]]

        # Replace dfs with updated
        trange = self.expt_tstart[expt_n] + np.arange(dfs.shape[0])
        df_tslice = deepcopy( self.dfs[trange, :] )
        #crange = self.cranges[expt_n]
        cindx = np.arange(len(self.cranges[expt_n]))
        if expt_n > 0:
            #crange += np.sum(self.exptNC[:expt_n])
            cindx += np.sum(self.exptNC[:expt_n])
        #df_tslice[:, crange] = dfs.astype(np.uint8)
        df_tslice[:, cindx] = dfs.astype(np.uint8)
        self.dfs[trange, :] = deepcopy(df_tslice)

        if reduce_cells:
            keep_cells = np.where(np.sum(dfs, axis=0) > 0)[0]
            if len(keep_cells) < self.exptNC[expt_n]:
                # note this list is already assuming previous cranges
                print( '  updateDF: eliminating %d out of %d cells'%(self.exptNC[expt_n]-len(keep_cells), self.exptNC[expt_n]) )
                self.modify_included_cells(self.cranges[expt_n][keep_cells], expt_n=expt_n)
        
        self.trialfilter_dfs()
    # END MultiClouds.updateDF()

    def trialfilter_dfs( self ):
        """
        Zeros out dfs at the beginning of trials up to num_lags
        
        Returns:
            None
        """
        if self.num_lags > 0:
            for bb in range(len(self.block_inds)):
                self.dfs[self.block_inds[bb][0] + np.arange(self.num_lags), :] = 0

    def assemble_saccade_inds( self ):
        """
        Assemble saccade indices for all experiments

        Returns:
            None
        """
        print('Currently not implemented -- needs to have microsaccades labeled well with time first')
    # END MultiClouds.assemble_saccade_inds()

    def is_fixpoint_present( self, boxlim, expt_n ):
        """
        Return if any of fixation point is within the box given by top-left to bottom-right corner
        
        Args:
            boxlim (list): list of four numbers (top-left, bot-right)
            expt_n (int): index of the experiment

        Returns:
            bool: whether the fixation point is present
        """
        fix_loc = self.file_info[expt_n]['fix_loc']
        if fix_loc is None:
            return False
        fix_size = self.file_info[expt_n]['fix_size']
        fix_present = True
        #if self.file_info[expt_n]['stim_loc'].shape[1] == 1:
        # if there is multiple windows, needs to be manual: so this is the automatic check:
        for dd in range(2):
            if (fix_loc[dd]-boxlim[dd] <= -fix_size):
                fix_present = False
            if (fix_loc[dd]-boxlim[dd+2] > fix_size):
                fix_present = False
        return fix_present
    # END .is_fixpoint_present

    def build_stim(
            self, expt_n=None,
            which_stim=None, top_corner=None, L=None,  # position of stim
            time_embed=0, 
            eyepos=None, BUF=20, # eye position (not shifts)
            stim_crop=None,
            LMS=False,
            fixdot=0 ):
        """
        This assembles a stimulus from the raw numpy-stored stimuli into self.stim
        which_stim: determines what stimulus is assembled from 'ET'=0, 'lam'=1, None
            If none, will need top_corner present: can specify with four numbers (top-left, bot-right)
            or just top_corner and L
        which is torch.tensor on default device
        stim_wrap: only works if using 'which_stim', and will be [hwrap, vwrap]
        
        Args:
            expt_n (int): index of the experiment
            which_stim (int or str): which stimulus to use
            top_corner (list): top corner of the stimulus
            L (int): size of the stimulus
            time_embed (int): time embedding
            eyepos (np.array): eye position
            BUF (int): buffer for eye position
            stim_crop (list): crop the stimulus
            LMS (bool): whether to use LMS
            fixdot (int): fixation point

        Returns:
            None
        """

        #eyepos = shifts   # shifts that is passed in is actually the eye position

        assert expt_n is not None, "BUILD_STIM: must specify expt_n"
        if self.file_info[expt_n]['stim_location_deltas'] is not None:
            assert np.sum(abs(self.file_info[expt_n]['stim_location_deltas'])) == 0, "BUILD_STIM: There are stim-deltas but not implemented yet."
        # Delete existing stim and clear cache to prevent memory issues on GPU
        if eyepos is not None:  # make sure empty list is same as None
            if len(eyepos) == 0:
                eyepos = None

        need2crop = False
        if LMS:
            assert not self.luminance_only, "Cannot convert color spaces if luminance-only."
        self.LMS = LMS

        locsET = self.file_info[expt_n]['stim_locsET']
        locsLP = self.file_info[expt_n]['stim_locsLP']
        
        if which_stim is not None:
            assert L is None, "CONSTRUCT_STIMULUS: cannot specify L if using which_stim (i.e. prepackaged stim)"
            if not isinstance(which_stim, int):
                if which_stim in ['ET', 'et', 'stimET']:
                    which_stim=0
                else:
                    which_stim=1
            if which_stim == 0:
                print( "Stim #%d: using ET stimulus"%expt_n )
                stim_tmp = np.array(self.fhandles[expt_n]['stimET'], dtype=np.int8)
                #self.stim_pos = self.stim_locationET[:,0]
                stim_pos = locsET[:,0]  # take first one -- should use other approach if many
            else:
                print( "Stim #%d: using laminar probe stimulus"%expt_n )
                stim_tmp = np.array(self.fhandles[expt_n]['stim'], dtype=np.int8)
                stim_pos = locsLP[:,0]  # take first one -- should use other approach if many
                #self.stim_pos = self.stim_location[:, 0]

            if self.luminance_only:
                newstim = stim_tmp[self.tranges[expt_n], 0, ...][:, None, ...]
            else:
                newstim = stim_tmp[self.tranges[expt_n], ...]
            
            L = newstim.shape[2]
            # Forget binocular for now
            if self.binocular:
                print('currently not implemented')

        else:
            # Assemble from combination of ET and laminer probe (NP) stimulus
            assert top_corner is not None, "Need top corner if which_stim unspecified"
            assert stim_crop is None, "Cannot specify stim_crop and top_corner"

            # Determine stim location
            if len(top_corner) == 4:
                stim_pos = top_corner
                #L = self.stim_pos[2]-self.stim_pos[0]
                assert stim_pos[3]-stim_pos[1] == stim_pos[2]-stim_pos[0], "Stim must be square (for now)"
                if L is not None:
                    assert L == stim_pos[3]-stim_pos[1], "L does not match specified stim size"
                else:
                    L = stim_pos[3]-stim_pos[1]
            else:
                if L is None:
                    L = self.L
                assert L is not None, "Need to specify stimulus size"
                if self.L is not None:
                    assert L == self.L, "BUILD_STIM: size of stimuli much match. L=%d"%self.L 

                stim_pos = [top_corner[0], top_corner[1], top_corner[0]+L, top_corner[1]+L]

            if eyepos is not None:
                need2crop = True
                # Extend stim window by BUFF-per-side in each direction
                stim_pos = [
                    stim_pos[0]-BUF,
                    stim_pos[1]-BUF,
                    stim_pos[2]+BUF,
                    stim_pos[3]+BUF]
                print( "  Stim expansion for shift:", stim_pos)
                L += 2*BUF

            if self.luminance_only:
                num_clr = 1
            else:
                num_clr = 3

            # Read in stimuli
            if len(self.fhandles[expt_n]['stimET'].shape) == 1:
                stimET_base = None
            else:
                stimET_base = np.array(self.fhandles[expt_n]['stimET'][self.tranges[expt_n], ...], dtype=np.int8)
            
            locsET = self.file_info[expt_n]['stim_locsET']
            stimLP_base = np.array(self.fhandles[expt_n]['stim'][self.tranges[expt_n], ...], dtype=np.int8)
            locsLP = self.file_info[expt_n]['stim_locsLP']
            fhandle = self.fhandles[expt_n]
            sz = fhandle['stim'].shape
            inds = np.arange(sz[0], dtype=np.int64)

            #stimET_base = np.array(self.fhandles[expt_n]['stimET'][self.tranges[expt_n], ...], dtype=np.int8)
            #stimLP_base = np.array(self.fhandles[expt_n]['stim'][self.tranges[expt_n], ...], dtype=np.int8)
            #locsET = self.file_info[expt_n]['stim_locsET']
            #locsLP = self.file_info[expt_n]['stim_locsLP']
            if self.binocular:
                num_clr *= 2  # make binocular be like just 2x more colors (channel dim)
                #Lpresent = np.array(fhandle['useLeye'], dtype=int)[:,0]
                #Rpresent = np.array(fhandle['useReye'], dtype=int)[:,0]
                #LRpresent = Lpresent + 2*Rpresent
                LRpresent = self.file_info[expt_n]['LRpresent'][self.tranges[expt_n]]
                Leye = inds[LRpresent[inds] != 2]
                Reye = inds[LRpresent[inds] != 1]
                self.binocular_gain = torch.zeros( [len(LRpresent), 2], dtype=torch.float32 )
                self.binocular_gain[LRpresent == 1, 0] = 1.0
                self.binocular_gain[LRpresent == 2, 1] = 1.0
                if self.luminance_only: 
                    #empty_stimET=np.zeros((np.shape(stimET)[0], 2, np.shape(stimET)[2], np.shape(stimET)[3]))
                    stimLP = np.zeros((np.shape(stimLP_base)[0], 2, np.shape(stimLP_base)[2], np.shape(stimLP_base)[3]))
                    stimLP[Leye, 0, ...] = np.array(fhandle['stim'], dtype=np.float32)[Leye, 0, ...]
                    stimLP[Reye, 1, ...] = np.array(fhandle['stim'], dtype=np.float32)[Reye, 0, ...]
                    if stimET_base is not None:
                        stimET = np.zeros((np.shape(stimET_base)[0], 2, np.shape(stimET_base)[2], np.shape(stimET_base)[3]))
                        stimET[Leye, 0, ...] = np.array(fhandle['stimET'], dtype=np.float32)[Leye, 0, ...]
                        stimET[Reye, 1, ...] = np.array(fhandle['stimET'], dtype=np.float32)[Reye, 0, ...]
                else:
                    stimLP = np.zeros((np.shape(stimLP_base)[0], 6, np.shape(stimLP_base)[2], np.shape(stimLP_base)[3]))
                    stimLP[Leye, 0:3, ...] = np.array(fhandle['stim'], dtype=np.float32)[Leye, ...]
                    stimLP[Reye, 3:6, ...] = np.array(fhandle['stim'], dtype=np.float32)[Reye, ...]
                    if stimET_base is not None:
                        stimET = np.zeros((np.shape(stimET_base)[0], 6, np.shape(stimET_base)[2], np.shape(stimET_base)[3]))
                        stimET[Leye, 0:3, ...] = np.array(fhandle['stimET'], dtype=np.float32)[Leye, ...]
                        stimET[Reye, 3:6, ...] = np.array(fhandle['stimET'], dtype=np.float32)[Reye, ...]
            else:
                stimET = stimET_base
                stimLP = stimLP_base
                if self.luminance_only:
                    stimLP = stimLP[:, 0, ...][:, None, ...]
                if stimET_base is not None:
                    # stimET=stimET_base
                    if self.luminance_only:
                        if stimET_base is not None: 
                            stimET = stimET[:, 0, ...][:, None, ...]  # maintain 2nd dim (length 1)
         
            NT = self.exptNT[expt_n]
            newstim = np.zeros( [NT, num_clr, L, L], dtype=np.int8 )
            for ii in range(locsLP.shape[1]):
                OVLP = self.rectangle_overlap_ranges(stim_pos, locsLP[:, ii])
                if OVLP is not None:
                    print( "  Writing lam stim %d: overlap %d, %d"%(ii, len(OVLP['targetX']), len(OVLP['targetY'])))
                    strip = deepcopy(newstim[:, :, OVLP['targetX'], :]) #np.zeros([self.NT, num_clr, len(OVLP['targetX']), L])
                    strip[:, :, :, OVLP['targetY']] = deepcopy((stimLP[:, :, OVLP['readX'], :][:, :, :, OVLP['readY']]))
                    newstim[:, :, OVLP['targetX'], :] = deepcopy(strip)
                
            if stimET is not None:    
                for ii in range(locsET.shape[1]):
                    OVLP = self.rectangle_overlap_ranges(stim_pos, locsET[:, ii])
                    if OVLP is not None:
                        print( "  Writing ETstim %d: overlap %d, %d"%(ii, len(OVLP['targetX']), len(OVLP['targetY'])))
                        strip = deepcopy(newstim[:, :, OVLP['targetX'], :])
                        strip[:, :, :, OVLP['targetY']] = deepcopy((stimET[:, :, OVLP['readX'], :][:, :, :, OVLP['readY']]))
                        newstim[:, :, OVLP['targetX'], :] = deepcopy(strip) 

            #stim = torch.tensor( newstim, dtype=torch.float32, device=self.device )
        # Note stim stored in numpy is being represented as full 3-d + 1 tensor (time, channels, NX, NY)

        # Insert fixation point
        if (fixdot is not None) and self.is_fixpoint_present( stim_pos, expt_n ):
            fixranges = [None, None]
            fix_loc = self.file_info[expt_n]['fix_loc']
            fix_size = self.file_info[expt_n]['fix_size']
            for dd in range(2):
                fixranges[dd] = np.arange(
                    np.maximum(fix_loc[dd]-fix_size-stim_pos[dd], 0),
                    np.minimum(fix_loc[dd]+fix_size+1, stim_pos[dd+2])-stim_pos[dd] 
                    ).astype(np.int8)
            # Write the correct value to stim
            assert fixdot == 0, "Haven't yet put in other fixdot settings than zero" 
            print('  Adding fixation point')
            for xx in fixranges[0]:
                newstim[:, :, xx, fixranges[1]] = 0

        if eyepos is not None:
            # Would want to shift by input eye positions if input here
            #print('eye-position shifting not implemented yet')
            print('  Shifting stim...')
            if len(eyepos) > newstim.shape[0]:
                eyepos = eyepos[self.tranges[expt_n]]
            if len(eyepos) < newstim.shape[0] and self.binocular:
                eyepos_padded=np.zeros((newstim.shape[0], 2))
                #Lpresent = np.array(fhandle['useLeye'], dtype=int)[:,0]
                #Rpresent = np.array(fhandle['useReye'], dtype=int)[:,0]
                #LRpresent = Lpresent + 2*Rpresent
                LRpresent = self.file_info[expt_n]['LRpresent'][self.tranges[expt_n]]

                bin_inds=inds[np.where(LRpresent[inds]==3)]
                if len(bin_inds)!=len(eyepos):
                    eyepos_padded[bin_inds[0]:bin_inds[0]+len(eyepos)]=eyepos #Off by one errors, probably. 
                else:
                    eyepos_padded[bin_inds]=eyepos
                eyepos=eyepos_padded 
            if num_clr == 6:                 
                newstim = self.shift_stim( newstim, eyepos, batch_size=10)
            else:
                newstim = self.shift_stim(newstim, eyepos) 

        # Reduce size back to original If expanded to handle shifts
        if need2crop:
            #assert self.stim_crop is None, "Cannot crop stim at same time as shifting"
            newstim = self.crop_stim( newstim, [BUF, L-BUF-1, BUF, L-BUF-1] )  # move back to original size
            L = L-2*BUF
        else:
            # if used which_stim and added crop
            if stim_crop is not None:
                newstim = self.crop_stim( newstim, stim_crop )
            L = newstim.shape[2]

        # Verify L matches current stim's L
        if self.L is not None:
            assert L == self.L, "BUILD_STIM: L mismatch (which_stim)"
        self.L = L

        if self.time_embed is None:
            self.time_embed = time_embed
        else:
            assert self.time_embed == time_embed, "time_embed setting must match"
            
        if time_embed > 0:
            #self.stim_dims[3] = num_lags  # this is set by time-embedding
            if time_embed == 2:
                newstim = self.time_embedding( newstim, nlags=self.num_lags )
        # now stimulus is represented as full 4-d + 1 tensor (time, channels, NX, NY, num_lags)

        # Translate to cone-isolating stimmulus if DKL is false
        if LMS:
            # Make LMS conversion matrix
            DKL2LMS = np.array([[0.7586, 0.6515, 0], [0.5536, -0.8328, 0], [0.5000, 0, 0.8660]], dtype=np.float32)
            print( '  Converting to LMS space')
            newstim = np.einsum( 'axcd,bx->abcd', newstim, DKL2LMS)

        # Flatten stim 
        self.expt_stims[expt_n] = deepcopy(newstim.reshape([self.exptNT[expt_n], -1]))
        print( "  Done: expt", expt_n )
    # END MultiClouds.build_stim()

    def assemble_stim( self ):
        """
        Assemble stimulus from all experiments into self.stim

        Returns:
            None
        """

        self.stim_dims = [3, self.L, self.L, 1]
        if self.luminance_only:
            self.stim_dims[0] = 1
        if self.binocular:
            self.stim_dims[0]=2*self.stim_dims[0]
        if self.time_embed:
            self.stim_dims[3] = self.num_lags
        num_dims = np.prod(self.stim_dims)
        self.stim = np.zeros( [self.NT, num_dims], dtype=np.int8 )
        for ff in range(self.Nexpts):
            assert self.expt_stims[ff] is not None, 'ASSEMBLE_STIM: stim %d is not yet built.'%ff  
            trange = range(self.expt_tstart[ff], self.expt_tstart[ff]+self.exptNT[ff])
            self.stim[trange, :] = self.expt_stims[ff]
        print( "Stimulus assembly complete")
    # END MultiClouds.assemble_stimulus()

    def time_embedding( self, stim=None, nlags=None ):
        """
        Note this overloads SensoryBase because reshapes in full dimensions to handle folded_lags
        
        Args:
            stim (np.array): stimulus to time-embed
            nlags (int): number of lags

        Returns:
            np.array: time-embedded stimulus
        """
        assert self.stim_dims is not None, "Need to assemble stim before time-embedding."

        if nlags is None:
            nlags = self.num_lags
        #if self.stim_dims[3] == 1:
        #    self.stim_dims[3] = nlags
        #if stim is None:
        #    tmp_stim = deepcopy(self.stim)
    
        NT = stim.shape[0]
        print("  Time embedding...")
        #if len(tmp_stim.shape) == 2:
        #    print( "Time embed: reshaping stimulus ->", self.stim_dims)
        #    tmp_stim = tmp_stim.reshape([NT] + self.stim_dims)

        #assert self.NT == NT, "TIME EMBEDDING: stim length mismatch"

        #tmp_stim = tmp_stim[np.arange(NT)[:,None]-np.arange(nlags), :, :, :]
        tmp_stim = tmp_stim[np.arange(NT)[:,None]-np.arange(nlags), :]
        #if self.folded_lags:
        #    #tmp_stim = np.transpose( tmp_stim, axes=[0,2,1,3,4] ) 
        #    tmp_stim = torch.permute( tmp_stim, (0,2,1,3,4) ) 
        #    print("Folded lags: stim-dim = ", self.stim.shape)
        #else:
        #    #tmp_stim = np.transpose( tmp_stim, axes=[0,2,3,4,1] )
        #    tmp_stim = torch.permute( tmp_stim, (0,2,3,4,1) )
        tmp_stim = torch.permute( tmp_stim, (0,2,1) )
        return tmp_stim
    # END .time_embedding()

    @staticmethod
    def rectangle_overlap_ranges( A, B ):
        """
        Figures out ranges to write relevant overlap of B onto A
        All info is of form [x0, y0, x1, y1]
        
        Args:
            A (list): first rectangle
            B (list): second rectangle

        Returns:
            dict: dictionary of ranges
        """

        C, D = np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.int64)
        for ii in range(2):
            if A[ii] >= B[ii]: 
                C[ii] = 0
                D[ii] = A[ii]-B[ii]
                if A[2+ii] <= B[2+ii]:
                    C[2+ii] = A[2+ii]-A[ii]
                    D[2+ii] = A[2+ii]-B[ii] 
                else:
                    C[2+ii] = B[2+ii]-A[ii]
                    D[2+ii] = B[2+ii]-B[ii]
            else:
                C[ii] = B[ii]-A[ii]
                D[ii] = 0
                if A[2+ii] <= B[2+ii]:
                    C[2+ii] = A[2+ii]-A[ii]
                    D[2+ii] = A[2+ii]-B[ii]
                else:
                    C[2+ii] = B[2+ii]-A[ii]
                    D[2+ii] = B[2+ii]-B[ii]

        if (C[2]<=C[0]) | (C[3]<=C[1]):
            return None  
        ranges = {
            'targetX': np.arange(C[0], C[2]),
            'targetY': np.arange(C[1], C[3]),
            'readX': np.arange(D[0], D[2]),
            'readY': np.arange(D[1], D[3])}
        return ranges
    # END STATIC.rectangle_overlap_ranges

    def crop_stim( self, stim0, stim_crop=None ):
        """
        Crop existing (torch) stimulus and change relevant variables [x1, x2, y1, y2]
        
        Args:
            stim0 (np.array): stimulus to crop
            stim_crop (list): crop the stimulus

        Returns:
            np.array: cropped stimulus
        """

        assert len(stim_crop) == 4, "stim_crop must be of form: [x1, x2, y1, y2]"
        assert len(stim0.shape) >= 4, "STIM_CROP: Will only work for unflattened stimulus"
        
        #stim_crop = np.array(stim_crop, dtype=np.int64) # make sure array
        xs = np.arange(stim_crop[0], stim_crop[1]+1)
        ys = np.arange(stim_crop[2], stim_crop[3]+1)
        if len(stim0.shape) == 4:
            newstim = stim0[:, :, :, ys][:, :, xs, :]
        else:  # then lagged -- need extra dim
            newstim = stim0[:, :, :, ys, :][:, :, xs, :, :]

        print("  CROP: New stim size: %d x %d"%(len(xs), len(ys)))
        return newstim
    # END MultiClouds.crop_stim()

#    def augment_dfs( self, new_dfs, cells=None ):
#        """Replaces data-filter for given cells. note that new_df should be np.ndarray"""
#        
#        NTdf, NCdf = new_dfs.shape 
#        if cells is None:
#            assert NCdf == self.dfs.shape[1], "new DF is wrong shape to replace DF for all cells."
#            cells = range(self.dfs.shape[1])
#        if self.NT < NTdf:
#            self.dfs[:, cells] *= torch.tensor(new_dfs[:self.NT, :], dtype=torch.float32)
#        else:
#            if self.NT > NTdf:
#                # Assume dfs are 0 after new valid region
#                print("Truncating valid region to new datafilter length", NTdf)
#                new_dfs = np.concatenate( 
#                    (new_dfs, np.zeros([self.NT-NTdf, len(cells)], dtype=np.float32)), 
#                    axis=0)
#            self.dfs[:, cells] *= torch.tensor(new_dfs, dtype=torch.float32)
        # END ColorClouds.augment_dfs()

    def draw_stim_locations( self, expt_n=0, top_corner=None, L=60, row_height=5.0 ):
        """
        Draw stimulus locations for given experiment

        Args:
            expt_n (int): index of the experiment
            top_corner (list): top corner of the stimulus
            L (int): size of the stimulus
            row_height (float): height of the row

        Returns:
            None
        """

        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        lamlocs = self.file_info[expt_n]['stim_locsLP']
        ETlocs = self.file_info[expt_n]['stim_locsET']
        fixloc = self.file_info[expt_n]['fix_loc']
        fixsize = self.file_info[expt_n]['fix_size']

        BUF = 10
        if L is None:
            L = lamlocs[2,0]-lamlocs[0,0]

        fig, ax = plt.subplots()
        fig.set_size_inches(row_height, row_height)
        nET = ETlocs.shape[1]
        nLAM = lamlocs.shape[1]
        x0 = np.minimum( np.min(lamlocs[0,:]), np.min(ETlocs[0,:]) )
        x1 = np.maximum( np.max(lamlocs[2,:]), np.max(ETlocs[2,:]) )
        y0 = np.minimum( np.min(lamlocs[1,:]), np.min(ETlocs[1,:]) )
        y1 = np.maximum( np.max(lamlocs[3,:]), np.max(ETlocs[3,:]) )
        #print(x0,x1,y0,y1)
        for ii in range(nLAM):
            ax.add_patch(
                Rectangle((lamlocs[0, ii], lamlocs[1, ii]), 60, 60, 
                edgecolor='red', facecolor='none', linewidth=1.5))
        clrs = ['blue', 'green', 'purple']
        for ii in range(nET):
            ax.add_patch(Rectangle((ETlocs[0,ii], ETlocs[1,ii]), 60, 60, 
                                edgecolor=clrs[ii], facecolor='none', linewidth=1))
        if fixloc is not None:
            ax.add_patch(Rectangle((fixloc[0]-fixsize-1, fixloc[1]-fixsize-1), fixsize*2+1, fixsize*2+1, 
                                facecolor='cyan', linewidth=0))
        if top_corner is not None:
            ax.add_patch(Rectangle(top_corner, L, L, 
                                edgecolor='orange', facecolor='none', linewidth=1, linestyle='dashed'))
            
        ax.set_aspect('equal', adjustable='box')
        plt.xlim([x0-BUF,x1+BUF])
        plt.ylim([y0-BUF,y1+BUF])
        plt.show()
    # END .draw_stim_locations()
    
    def restrict2good_fixations( self, ETmetrics, expt_n=None, thresh=0.9 ):
        """
        Modifies data_filters with valid fixation data. For now, would need 
        to re-run assemble_robs. ETmetrics can be for all experiments (if list), or just one

        Args: 
            ETmetrics: array that is length of trange[ee]. This metric that is best correlated 
                with fixation quality and has a range in the 0.8-1.0 (where 0.8 is best). Make this
                a list of ETmetrics (one item for each expt) if passing in multiple.
            expt_n: which experiment to apply metrics to. Will assume all if ETmetrics is a list, or
                expt 0 if ETmetrics is just a single value
            thresh: metric threshold (between 0.8-1, default=0.9), with lower corresponding to better qual

        Returns:
            None, but modifies self.dfs accordingly
        """

        # First make sure that getting a list of experiments with matching list of ETmetrics
        if isinstance(ETmetrics, list):
            assert expt_n is None, "Cannot use ETmetrics as list if not doing all experiments"
            assert len(ETmetrics) == self.Nexpts, "ETmetrics list is wrong length"
            expt_n = np.arange(self.Nexpts)
        else:
            ETmetrics = [ETmetrics]
            if self.Nexpts == 1:
                expt_n = [0]
            else:
                assert expt_n is not None, "Need to specify which expt_n"
                if not isinstance(expt_n, list):
                    expt_n = [expt_n]

        for ii in range(len(expt_n)):
            ee = expt_n[ii]
            assert len(ETmetrics[ii]) == self.exptNT[ee], "ETmetrics is wrong temporal length"
            trange = np.sum(self.exptNT[:ee]) + np.arange(self.exptNT[ee])
            crange = np.sum(self.exptNC[:ee]) + np.arange(self.exptNC[ee])
            df_tslice = deepcopy(self.dfs[trange,:])  # this is all cells from all experiments
        
            val_fix = np.zeros([len(trange),1], dtype=np.uint8)
            val_fix[np.where(ETmetrics[ii] <= thresh)[0]] = 1

            print( "  Expt #%d: %0.1f percent of data remaining."%(ee, 100*np.sum(val_fix)/len(val_fix)) )
            # Make so that not including small intervals
            val_transitions = np.where(np.diff(val_fix.squeeze()) > 0)[0]
            for jj in range(len(val_transitions)-1):
                if val_transitions[jj+1]-val_transitions[jj] < 4:
                    if val_fix[val_transitions[jj]+1] > 0:
                        val_fix[val_transitions[jj]:(val_transitions[jj+1]+1)] = 0

            df_tslice[:, crange] *= val_fix
            self.dfs[trange, :] = deepcopy(df_tslice)
    # END MultiClouds.valid_fixations()

    def avrates( self, inds=None ):
        """
        Calculates average firing probability across specified inds (or whole dataset)
        -- Note will respect datafilters
        -- will return precalc value to save time if already stored

        Args:
            inds (list): indices to calculate across

        Returns:
            np.array: average firing rates
        """
        if inds is None:
            inds = range(self.NT)
        if len(inds) == self.NT:
            # then calculate across whole dataset
            if self.avRs is not None:
                # then precalculated and do not need to do
                return self.avRs

        # Otherwise calculate across all data
        if self.preload:
            Reff = (self.dfs * self.robs).sum(dim=0).cpu()
            Teff = self.dfs.sum(dim=0).clamp(min=1e-6).cpu()
            return (Reff/Teff).detach().numpy()
        else:
            print('Still need to implement avRs without preloading')
            return None
    # END .avrates()

    #def set_cells(self, **kwargs):
    #    raise( Exception('set_cells does not work with MultiClouds.') )
    # USE SENSORY-BASE -- should be set up now....

    def shift_stim_oldstyle(
        self, stim=None, pos_shifts=None, metrics=None, metric_threshold=1, ts_thresh=8, fix_n=None,
        shift_times=None ):
        """
        Shift stimulus given standard shifting input (TBD)
        use 'shift-times' if given shifts correspond to range of times
        
        Args:
            stim (np.array): stimulus to shift
            pos_shifts (np.array): shifts to apply
            metrics (np.array): metrics to apply
            metric_threshold (float): metric threshold
            ts_thresh (int): time threshold
            fix_n (np.array): fixations
            shift_times (np.array): times to shift

        Returns:
            np.array: shifted stimulus
        """
        NX = self.stim_dims[1]
        nlags = self.stim_dims[3]

        # Check if has been time-lagged yet
        if stim is None:
            stim0 = self.stim
        else:
            stim0 = stim

        already_lagged=False
        if len(stim0.shape) == 2:
            [NT, ND] = stim0.shape
            if ND > self.stim_dims[0]*NX*NX:  # then lags in stim
                re_stim = deepcopy(stim0).reshape([-1] + self.stim_dims)[..., 0]
                already_lagged = True
            else:
                re_stim = deepcopy(stim0).reshape([-1] + self.stim_dims[:3])
        else:
            assert len(stim0) > 2, "Stim be already shaped according shape, but seems not"
            re_stim = deepcopy(stim)

        if fix_n is None:
            fix_n = np.array(self.fix_n, dtype=np.int64)

        # Apply shift-times (subset)
        if shift_times is not None:
            fix_n = fix_n[shift_times]
            # Find minimum fix_n and make = 1
            min_fix_n = np.min(fix_n[fix_n > 0])
            #print('min_fix_n', min_fix_n, 'adjust', 1-min_fix_n)
            fix_n[fix_n > 0] += 1-min_fix_n
            re_stim = re_stim[shift_times, ...]
            #print('max fix', np.max(fix_n), fix_n.shape)
        
        NF = np.max(fix_n)
        NTtmp = re_stim.shape[0]
        nclr = self.stim_dims[0]
        #sh0 = -(pos_shifts[:,0]-self.dims[1]//2)
        #sh1 = -(pos_shifts[:,1]-self.dims[2]//2)
        sh0 = pos_shifts[:, 0]  # this should be in units of pixels relative to 0
        sh1 = pos_shifts[:, 1]

        sp_stim = deepcopy(re_stim)
        if metrics is not None:
            val_fix = metrics > metric_threshold
            print("  Applied metric threshold: %d/%d"%(np.sum(val_fix), NF))
        else:
            val_fix = np.array([True]*NF)

        for ff in range(NF):
            ts = np.where(fix_n == ff+1)[0]
            #print(ff, len(ts), ts[0], ts[-1])

            if (abs(sh0[ff])+abs(sh1[ff]) > 0) & (len(ts) > ts_thresh) & val_fix[ff] & (ts[-1] < self.NT):
                # FIRST SP DIM shift
                sh = int(sh0[ff])
                stim_seg = re_stim[ts, ...]
                if sh > 0:
                    stim_tmp = torch.zeros([len(ts), nclr, NX, NX], dtype=torch.float32)
                    stim_tmp[:, :,sh:, :] = deepcopy(stim_seg[:, :, :(-sh), :])
                elif sh < 0:
                    stim_tmp = torch.zeros([len(ts), nclr, NX, NX], dtype=torch.float32)
                    stim_tmp[:, :, :sh, :] = deepcopy(stim_seg[:, :, (-sh):, :])
                else:
                    stim_tmp = deepcopy(stim_seg)

                # SECOND SP DIM shift
                sh = int(sh1[ff])
                if sh > 0:
                    stim_tmp2 = torch.zeros([len(ts), nclr, NX,NX], dtype=torch.float32)
                    stim_tmp2[... , sh:] = deepcopy(stim_tmp[..., :(-sh)])
                elif sh < 0:
                    stim_tmp2 = torch.zeros([len(ts), nclr, NX,NX], dtype=torch.float32)
                    stim_tmp2[..., :sh] = deepcopy(stim_tmp[..., (-sh):])
                else:
                    stim_tmp2 = deepcopy(stim_tmp)
                    
                sp_stim[ts, ... ] = deepcopy(stim_tmp2)

        if already_lagged:
            # Time-embed
            idx = np.arange(NTtmp)
            laggedstim = sp_stim[np.arange(NTtmp)[:,None]-np.arange(nlags), ...]
            return np.transpose( laggedstim, axes=[0,2,3,4,1] )
        else:
            return sp_stim
    # END ColorCloud.shift_stim -- note outputs stim rather than overwrites

    def process_fixations(self, sacc_in=None, expt_n=0, 
                          sacc_metrics=None, thresh=None, dur_thresh=8,
                          modify_dfs=False):
        """
        Generates fix_n based on existing trial structure and imported fixations. Will only work for specified
        experiment (expt_n variable) and assume timings are based on the beginning of that experiment.
        Output: fix_n

        Default: modify_dfs=True will zero out data where there is no assigned fixation
        
        Note that this will assume that saccades correspond to relevant trange (e.g., binocular section) rather
        than the whole experiment.
        
        Can also use metric criteria to only use fraction of total saccades: sacc_metrics can be amplitude, and 
        sacc_thresh be inclusion criteria (must be greater than sacc_thresh)

        Args:
            sacc_in (np.array): saccade times
            expt_n (int): index of the experiment
            sacc_metrics (np.array): saccade metrics
            thresh (float): threshold for saccade metrics
            dur_thresh (int): duration threshold
            modify_dfs (bool): modify dfs

        Returns:
            np.array: fixation numbers
        """

        assert sacc_in is not None, "Need to enter saccade times"
        NT = self.exptNT[expt_n]
        assert np.max(sacc_in) < NT, "sacc list is too long"
        if sacc_metrics is None:
            sac_ts = sacc_in
        else:
            assert len(sacc_metrics) == len(sacc_in), "Saccade metrics must match length of sacc_in"
            sac_ts = sacc_in[np.where(sacc_metrics >= thresh)[0]]
        sacc_rate = len(sac_ts)/(NT/60.0)
        print( "  Generating fix_n for expt %d with %d valid saccades (%0.2f Hz)"%(expt_n, len(sac_ts), sacc_rate) )

        fix_n = np.zeros(NT, dtype=np.int64) 
        fix_count = 0
        for ii in range(len(self.block_inds)):
            #if self.block_inds[ii][0] < self.NT:
            # note this will be the inds in each file -- file offset must be added for mult files
            #self.block_inds.append(np.arange( blk_inds[ii,0], blk_inds[ii,1], dtype=np.int64))

            # Parse fixation numbers within block
            rel_saccs = np.where((sac_ts > self.block_inds[ii][0]+6) & (sac_ts < self.block_inds[ii][-1]-5))[0]

            tfix = self.block_inds[ii][0]  # Beginning of fixation by definition
            for mm in range(len(rel_saccs)):
                if sac_ts[rel_saccs[mm]]-tfix >= dur_thresh:
                    fix_count += 1
                    # Range goes to beginning of next fixation (note no gap)
                    fix_n[ range(tfix, sac_ts[rel_saccs[mm]]) ] = fix_count
                tfix = sac_ts[rel_saccs[mm]]
            # Put in last (or only) fixation number
            if tfix < self.block_inds[ii][-1]:
                fix_count += 1
                fix_n[ range(tfix, self.block_inds[ii][-1]) ] = fix_count

        print("  Created %d fixations"%fix_count)

        if modify_dfs:
            invalid_fix_n = np.where(fix_n == 0)[0]
            self.dfs[invalid_fix_n, :] = 0
            
        return fix_n.astype(np.int64)
    # END: MultiClouds.process_fixations()

    @staticmethod 
    def shift_stim( stim, eyepos, input_dims=None, batch_size=5000):
        """
        Shift stimulus based on eye position

        Args:
            stim (np.array): stimulus to shift
            eyepos (np.array): eye position
            input_dims (list): input dimensions
            batch_size (int): batch size

        Returns:
            np.array: shifted stimulus
        """
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        if len(stim.shape) == 2:
            assert input_dims is not None, "  SHIFT_STIM: stim must be unflatteded or input_dims passed in"
            stim = stim.reshape([-1]+input_dims[:3])
            to_flatten=True
        else:
            to_flatten=False

        # Determine scaling for translation
        Lscale = 2.0/stim.shape[2]

        stim_data = {
            'ts': torch.tensor(np.arange(stim.shape[0]), dtype=torch.int64),  # keep track of indices for remapping
            'stim': torch.tensor(stim, dtype=torch.float32),
            'eyepos': torch.tensor(eyepos, dtype=torch.float32)*Lscale }

        ds = GenericDataset(stim_data, device=None)
        #ds.covariates['stim'] = ds.covariates['stim'].flatten(start_dim=1)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count()//2)

        stimSH = deepcopy(stim_data['stim'])
        for batch in tqdm(dl):
            stimSH[batch['ts'][:,0], :] = MultiClouds.shift_im(
                batch['stim'], batch['eyepos'][:,[1,0]], 
                False, mode='nearest')
        
        if to_flatten:
            stimSH = stimSH.flatten(start_dim=1)
        return stimSH.detach().numpy().astype(np.int8)
    
    @staticmethod
    def shift_im( stim, shift, affine=False, mode='nearest', batch_size=None):
        """
        Primary function for shifting the intput stimulus
        Args:
            stim: [Batch x channels x height x width] (use Fold2d to fold lags if necessary)
            shift: [Batch x 2] or [Batch x 4] if translation only or affine
            affine: [Boolean] set to True if using affine transformation
            mode: [str] 'bilinear' (default) or 'nearest'
            batch_size: [int] if None, will use all data at once
            NOTE: mode must be bilinear during fitting otherwise the gradients don't propogate well

        Returns:
            torch.tensor: shifted stimulus
        """
        import torch.nn.functional as F

        batch_size = stim.shape[0]

        # build affine transformation matrix size = [batch x 2 x 3]
        affine_trans = torch.zeros((batch_size, 2, 3) , dtype=stim.dtype, device=stim.device)
        
        if affine:
            # fill in rotation and scaling
            costheta = torch.cos(shift[:,2].clamp(-.5, .5))
            sintheta = torch.sin(shift[:,2].clamp(-.5, .5))
            
            scale = shift[:,3]**2 + 1.0

            affine_trans[:,0,0] = costheta*scale
            affine_trans[:,0,1] = -sintheta*scale
            affine_trans[:,1,0] = sintheta*scale
            affine_trans[:,1,1] = costheta*scale

        else:
            # no rotation or scaling
            affine_trans[:,0,0] = 1
            affine_trans[:,0,1] = 0
            affine_trans[:,1,0] = 0
            affine_trans[:,1,1] = 1

        # translation
        affine_trans[:,0,2] = shift[:,0]
        affine_trans[:,1,2] = shift[:,1]

        grid = F.affine_grid(affine_trans, stim.shape, align_corners=False)

        return F.grid_sample(stim, grid, mode=mode, align_corners=False)
    # END shift_im

    def get_max_samples(self, gpu_n=0, history_size=1, nquad=0, num_cells=None, buffer=1.2):
        """
        get the maximum number of samples that fit in memory -- for GLM/GQM x LBFGS

        Args:
            gpu_n (int): GPU number
            history_size (int): history size
            nquad (int): number of quadrature points
            num_cells (int): number of cells
            buffer (float): buffer size

        Returns:
            int: maximum number of samples
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
        mempersample = data[self.stimname].element_size() * data[self.stimname].nelement() + 2*data['robs'].element_size() * data['robs'].nelement()
    
        mempercell = mempersample * (nquad+1) * (history_size + 1)
        buffer_bytes = buffer*1024**3

        maxsamples = int(free - mempercell*num_cells - buffer_bytes) // mempersample
        print("# samples that can fit on device: {}".format(maxsamples))
        return maxsamples
    # END .get_max_samples

        ### THIS IS NOT USED ANY MORE BUT STILL HAS SOME GOOD CODE INSIDE ###
    def preload_numpy(self):
        """
        Note this loads stimulus but does not time-embed
        
        Returns:
            None
        """

        NT = self.NT
        ''' 
        Pre-allocate memory for data
        '''
        self.stimLP = np.zeros( [NT] + self.dims[:3], dtype=np.float32)
        if 'stimET' in self.fhandles[0]:
            # Check to see if 1-d or 2-d eye tracking
            tmp = np.array(self.fhandles[0]['stimET'], dtype=np.float32)[:100, ...]
            if len(tmp.shape) < 4:
                print("  ET stimulus is 1-d bars.")
                self.stimET = np.zeros( [NT, tmp.shape[1]], dtype=np.float32)
            else:
                self.stimET = np.zeros( [NT] + self.dims[:3], dtype=np.float32)
        else:
            self.stimET = None
            print('Missing ETstimulus')

        self.robs = np.zeros( [NT, self.NC], dtype=np.float32)
        self.dfs = np.ones( [NT, self.NC], dtype=np.float32)
        #self.eyepos = np.zeros([NT, 2], dtype=np.float32)
        #self.frame_times = np.zeros([NT,1], dtype=np.float32)

        t_counter = 0
        unit_counter = 0
        for ee in range(len(self.fhandles)):
            
            fhandle = self.fhandles[ee]
            sz = fhandle['stim'].shape
            inds = np.arange(t_counter, t_counter+sz[0], dtype=np.int64)
            #inds = self.stim_indices[expt][stim]['inds']
            #self.stim[inds, ...] = np.transpose( np.array(fhandle[self.stimname], dtype=np.float32), axes=[0,3,1,2])
            if self.binocular:
                Leye = inds[self.LRpresent[inds] != 2]
                Reye = inds[self.LRpresent[inds] != 1]
                #Leye = inds[np.where(self.LRpresent[inds] != 2)[0]]
                #Reye = inds[np.where(self.LRpresent[inds] != 1)[0]]
                #print(len(Leye), len(Reye))
                if self.luminance_only:
                    self.stimLP[Leye, 0, ...] = np.array(fhandle['stim'], dtype=np.float32)[Leye, 0, ...]
                    self.stimLP[Reye, 1, ...] = np.array(fhandle['stim'], dtype=np.float32)[Reye, 0, ...]
                    if self.stimET is not None:
                        self.stimET[Leye, 0, ...] = np.array(fhandle['stimET'], dtype=np.float32)[Leye, 0, ...]
                        self.stimET[Reye, 1, ...] = np.array(fhandle['stimET'], dtype=np.float32)[Reye, 0, ...]
                else:
                    stim_tmp = deepcopy(self.stimLP[:, :3, ...])
                    stim_tmp[Leye, ...] = np.array(fhandle['stim'], dtype=np.float32)[Leye, ...]
                    self.stimLP[:, np.arange(3), ...] = deepcopy(stim_tmp) # this might not be necessary
                    stim_tmp = deepcopy(self.stimLP[:, 3:, ...])
                    stim_tmp[Reye, ...] = np.array(fhandle['stim'], dtype=np.float32)[Reye, ...]
                    self.stimLP[:, np.arange(3,6), ...] = deepcopy(stim_tmp) # this might not be necessary

                    if self.stimET is not None:
                        stim_tmp = deepcopy(self.stimET[:, :3, ...])
                        stim_tmp[Leye, ...] = np.array(fhandle['stimET'], dtype=np.float32)[Leye, ...]
                        self.stimET[:, np.arange(3), ...] = deepcopy(stim_tmp) # this might not be necessary
                        stim_tmp = deepcopy(self.stimET[:, 3:, ...])
                        stim_tmp[Reye, ...] = np.array(fhandle['stimET'], dtype=np.float32)[Reye, ...]
                        self.stimET[:, np.arange(3,6), ...] = deepcopy(stim_tmp) # this might not be necessary
                        #self.stimET[Leye, np.arange(3), ...] = np.array(fhandle['stimET'], dtype=np.float32)[Leye, ...]
                        #self.stimET[Reye, np.arange(3,6), ...] = np.array(fhandle['stimET'], dtype=np.float32)[Reye, ...]
                        
            else:  # Monocular
                if self.luminance_only:
                    self.stimLP[inds, 0, ...] = np.array(fhandle['stim'], dtype=np.float32)[:, 0, ...]
                    if self.stimET is not None:
                        #print(np.array(fhandle['stimET'], dtype=np.float32).shape, self.stimET[inds, 0, ...].shape)
                        self.stimET[inds, 0, ...] = np.array(fhandle['stimET'], dtype=np.float32)[:, 0, ...]
                else:
                    self.stimLP[inds, ...] = np.array(fhandle['stim'], dtype=np.float32)
                    if self.stimET is not None:
                        self.stimET[inds, ...] = np.array(fhandle['stimET'], dtype=np.float32)

            t_counter += sz[0]
            unit_counter += self.num_units[ee]

        # Adjust stimulus since stored as ints
        if np.std(self.stimLP) > 5: 
            if np.mean(self.stimLP) > 50:
                self.stimLP += -127
                if self.stimET is not None:
                    self.stimET += -127
            self.stimLP *= 1/128
            if self.stimET is not None:
                self.stimET *= 1/128
            print( "Adjusting stimulus read from disk: mean | std = %0.3f | %0.3f"%(np.mean(self.stimLP), np.std(self.stimLP)))
    # END .preload_numpy()

    def __getitem__(self, idx):

        assert self.stim is not None, "Have to specify stimulus before pulling data."
        #if isinstance(idx, np.ndarray):
        #    idx = list(idx)
        # Convert trials to indices if trial-sample
        if self.trial_sample:
            idx = self.index_to_array(idx, len(self.block_inds))
            ts = self.block_inds[idx[0]]
            for ii in idx[1:]:
                ts = np.concatenate( (ts, self.block_inds[ii]), axis=0 )
            idx = ts
        else:
            idx = self.index_to_array(idx, self.NT)

        if self.time_embed == 1:
            print("get_item time embedding not implemented yet")
                
        if len(self.cells_out) == 0:
            out = {'stim': torch.tensor(self.stim[idx, :], dtype=torch.float32, device=self.device)/128,
                   'robs': torch.tensor(self.robs[idx, :], dtype=torch.float32, device=self.device),
                   'dfs': torch.tensor(self.dfs[idx, :], dtype=torch.float32, device=self.device)}
        else:
            if self.robs_out is not None:
                robs_tmp = self.robs_out
                dfs_tmp = self.dfs_out
            else:
                assert isinstance(self.cells_out, list), 'cells_out must be a list'
                robs_tmp =  self.robs[:, self.cells_out]
                dfs_tmp =  self.dfs[:, self.cells_out]

            out = {'stim': torch.tensor(self.stim[idx, :], dtype=torch.float32, device=self.device)/128,
                   'robs': torch.tensor(robs_tmp[idx, :], dtype=torch.float32, device=self.device),
                   'dfs': torch.tensor(dfs_tmp[idx, :], dtype=torch.float32, device=self.device)}
            # FIXME: we are not using fix_n at the moment
            # if len(self.fix_n) > 0:
            #     out['fix_n'] = self.fix_n[idx]

        if self.speckled:
            if self.Mtrn_out is None:
                M1tmp = self.Mval[:, self.cells_out]
                M2tmp = self.Mtrn[:, self.cells_out]
                out['Mval'] = M1tmp[idx, :]
                out['Mtrn'] = M2tmp[idx, :]
            else:
                out['Mval'] = self.Mtrn_out[idx, :]
                out['Mtrn'] = self.Mtrn_out[idx, :]

        if self.binocular and self.output_separate_eye_stim:
            # Overwrite left stim with left eye only
            tmp_dims = out['stim'].shape[-1]//2
            stim_tmp = self.stim[idx, :].reshape([-1, 2, tmp_dims])
            out['stim'] = stim_tmp[:, 0, :]
            out['stimR'] = stim_tmp[:, 1, :]            

        # Addition whether-or-not preloaded
        if self.Xdrift is not None:
            out['Xdrift'] = torch.tensor(self.Xdrift[idx, :], dtype=torch.float32, device=self.device)
        if self.binocular:
            out['binocular'] = self.binocular_gain[idx, :]
            
        if len(self.covariates) > 0:
            self.append_covariates( out, idx)
     
        return out
    # END: MultiCloud.__get_item__

    #@property
    #def NT(self):
    #    return len(self.used_inds)

    def __len__(self):
        return self.robs.shape[0]