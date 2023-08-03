
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
        binocular=False, # whether to include separate filters for each eye
        eye_config=3,  # 0 = all, 1, 2, and 3 are options (3 = binocular)
        eye_contiguous=True, # whether to only use eye_config data that is contiguous 
        cell_lists = None,
        device=torch.device('cpu')):
        """Constructor options"""

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
        self.fileNT = np.zeros(self.Nexpts, dtype=np.int64)
        self.fileNBLK = np.zeros(self.Nexpts, dtype=np.int64)
        self.fileNC = np.zeros(self.Nexpts, dtype=np.int64)
        self.file_tstart = np.zeros(self.Nexpts, dtype=np.int64)
        self.file_blkstart = np.zeros(self.Nexpts, dtype=np.int64)

        tcount, blkcount = 0, 0
        self.tranges = [None] * self.Nexpts
        self.cranges = [None] * self.Nexpts
        self.block_inds = []

        # Structure of file on time/trial level
 
        for ff in range(self.Nexpts):
            # get hdf5 file handles
            self.file_info[ff] = self.read_file_info(ff, filenames[ff])
            if self.include_MUs:
                self.fileNC[ff] = self.file_info[ff]['NSUs'] + self.file_info[ff]['NMUs']
            else:
                self.fileNC[ff] = self.file_info[ff]['NSUs']

            # Consolidate valid t-ranges based on binocular choice
            self.tranges[ff] = self.file_info[ff]['tmap']
            self.cranges[ff] = np.arange(self.fileNC[ff], dtype=np.int64)
            # Make one long block-list
            NBLK = self.file_info[ff]['trial_info'].shape[0]
            for bb in range(NBLK):
                self.block_inds.append( 
                    tcount + np.arange(self.file_info[ff]['trial_info'][bb, 0], self.file_info[ff]['trial_info'][bb, 1]) )
            self.fileNBLK[ff] = self.file_info[ff]['trial_info'].shape[0]
            self.file_blkstart[ff] = blkcount
            blkcount += NBLK

            self.fileNT[ff] = self.file_info[ff]['NT']
            self.file_tstart[ff] = tcount
            tcount += self.file_info[ff]['NT']

        # Assemble robs and dfs given current information
        self.NT = tcount
        self.NC = np.sum(self.fileNC)
        print( "%d total time steps, %d units"%(self.NT, self.NC) )
        if cell_lists is not None:
            self.modify_included_cells(cell_lists)
            # this will automatically assemble_robs at the end
        else:
            self.assemble_robs()

        # Determine total experiment time and number of cells
 
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

        ### Construct drift term if relevant
        if self.drift_interval is None:
            self.Xdrift = None
        else:
            Nanchors_tot = 0
            # Make drift for each dataset
            Xdrift_expts = []
            for ff in range(self.Nexpts):
                NBL = self.fileNBLK[ff]
                Nanchors = np.ceil(NBL/self.drift_interval).astype(int)
                anchors = np.zeros(Nanchors, dtype=np.int64)
                for bb in range(Nanchors):
                    anchors[bb] = self.block_inds[self.drift_interval*bb][0]
                #self.Xdrift = utils.design_matrix_drift( self.NT, anchors, zero_left=False, const_right=True)
                Xdrift_expts.append(self.design_matrix_drift( self.fileNT[ff], anchors, zero_left=False, const_right=True))
                Nanchors_tot += Nanchors

            # Assemble whole drift matrix
            self.Xdrift = np.zeros( [self.NT, Nanchors_tot], dtype=np.float32 )
            anchor_count = 0
            for ff in range(self.Nexpts):
                tslice = np.zeros( [self.fileNT[ff], Nanchors_tot], dtype=np.float32 )
                tslice[:, anchor_count+np.arange(Xdrift_expts[ff].shape[1])] = Xdrift_expts[ff]
                self.Xdrift[self.file_tstart[ff]+np.arange(self.fileNT[ff]), :] = deepcopy(tslice)
                anchor_count += Xdrift_expts[ff].shape[1]

            # Determine overall valid-inds and cross-validation indices
            self.val_blks, self.train_blks = [], []
            folds = 5
            random_gen = False
            for ff in range(self.Nexpts):
                Nblks = self.file_info[ff]['trial_info'].shape[0]
                val_blk_e, tr_blk_e = self.fold_sample(Nblks, folds, random_gen=random_gen)
                if ff == 0:
                    self.val_blks = deepcopy(val_blk_e) + self.file_blkstart[ff]
                    self.train_blks = deepcopy(tr_blk_e) + self.file_blkstart[ff]
                else:
                    self.val_blks = np.concatenate( (self.val_blks, deepcopy(val_blk_e)+self.file_blkstart[ff]), axis=0)
                    self.train_blks = np.concatenate( (self.train_blks, deepcopy(tr_blk_e)+self.file_blkstart[ff]), axis=0 )

            self.val_inds, self.train_inds = [], []
            for bb in self.val_blks:
                if len(self.val_inds) == 0:
                    self.val_inds = deepcopy(self.block_inds[bb])
                else:
                    self.val_inds = np.concatenate( (self.val_inds, self.block_inds[bb]), axis=0 )
            for bb in self.train_blks:
                self.train_inds = np.concatenate( (self.train_inds, self.block_inds[bb]), axis=0 )
    # END MultiClouds.__init__

    def read_file_info( self, file_n, filename ):
        """Initial processing of each file to pull out salient info for building stim and responses"""

        f = self.fhandles[file_n]
        NT, NSUs = f['Robs'].shape
        # Check for valid RobsMU
        if len(f['RobsMU'].shape) > 1:
            NMUs = f['RobsMU'].shape[1]
        else: 
            NMUs = 0

        # Unit information
        channel_map = np.array(f['Robs_probe_ID'], dtype=np.int64)[0, :]
        channel_ratings = np.array(f['Robs_rating'])[0, :]
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
        blockIDs = np.array(f['blockID'], dtype=np.int64).squeeze()  # what is this?
        #self.ETtrace = np.array(f['ETtrace'], dtype=np.float32)
        #self.ETtraceHR = np.array(f['ETtrace_raw'], dtype=np.float32)

        # Binocular information
        Lpresent = np.array(f['useLeye'], dtype=int)[:,0]
        Rpresent = np.array(f['useReye'], dtype=int)[:,0]
        LRpresent = Lpresent + 2*Rpresent

        valid_inds = np.array(f['valid_data'], dtype=np.int64)-1  #range(self.NT)  # default -- to be changed at end of init

        # Parse timing and block information given eye_config
        if self.eye_config == 0:
            tmap = np.arange(NT)
            bmap = np.arange(NBLK)
            block_inds = deepcopy(blk_inds)
        else:
            tmap = np.where(LRpresent == self.eye_config)[0]
            # Check for contiguous option (throw away disjoint eye config)
            if self.eye_contiguous & (self.eye_config > 0):
                tbreaks = np.where(np.diff(tmap) > 1)[0]
                if len(tbreaks) > 0:
                    print("  Disjoint data exists with this eye_config -- trunctating to first section.")
                    tmap = tmap[range(tbreaks[0]+1)]

            # Remap valid_inds to smaller trage
            val_track = np.zeros(NT, dtype=np.int64)
            val_track[valid_inds] = 1
            valid_inds = np.where(val_track[tmap] == 1)[0]

            NT = len(tmap)
 
            # Remap block_inds to reduced map
            bmap = []
            tcount = 0
            block_inds = []
            for bb in range(NBLK):
                if blk_inds[bb,0] in tmap:
                    bmap.append(bb)
                    NTblk = blk_inds[bb,1]-blk_inds[bb,0]
                    block_inds.append([tcount, tcount+NTblk])
                    tcount += NTblk
            block_inds = np.array(block_inds, dtype=np.int64)
            # might have to modify blockIDs -- not done yet

        return {
            'filename': filename,
            'NT': NT,
            'tmap': tmap, 
            'trial_info': block_inds, 
            'LRpresent': LRpresent,
            'valid_inds': valid_inds.squeeze(),
            'blockIDs': blockIDs, # What is this again?  
            'NSUs': NSUs,
            'NMUs': NMUs,
            'channel_map': channel_map,
            'channel_ratings': channel_ratings,
            'fix_loc': fix_loc,
            'fix_size': fix_size,
            'stim_scale': stim_scale,
            'stim_locsLP': stim_locsLP,
            'stim_locsET': stim_locsET}
    # END MultiClouds.read_file_info()  

    def modify_included_cells(self, clists, expt_n=None):
        if expt_n is None:
            expts = np.arange(self.Nexpts)
            assert len(clists) == self.Nexpts, "Number of cell_lists must match number of experiments."
        else:
            expts = [expt_n]
            
        for ff in expts:
            if len(clists[ff]) > 0:
                assert np.max(clists[ff]) < (self.file_info[ff]['NSUs'] + self.file_info[ff]['NMUs']), "clists too large"
                self.cranges[ff] = deepcopy(clists[ff])
            else:
                if self.includeMUs:
                    self.cranges[ff] = np.arange(self.file_info[ff]['NSUs']+self.file_info[ff]['NMUs'])
                else:
                    self.cranges[ff] = np.arange(self.file_info[ff]['NSUs'])

            self.fileNC[ff] = len(self.cranges[ff])
        self.NC = np.sum(self.fileNC)

        self.assemble_robs()
    # END MultiClouds.modify_included_cells()

    def generate_array_cell_list(self, expt_n=0, which_array=0):
        """Formula for generating cell list given channel maps and basic eligibility"""
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
            cstart += self.fileNC[ff]
        nspks = np.sum(self.robs.astype(np.float32)*self.dfs.astype(np.float32), axis=0)[cstart+array_cells]

        val_array = array_cells[nspks > 20]
        return val_array
    # END MultiClouds.generate_array_cell_list()

    def assemble_robs(self):
        """Takes current information (robs and dfs) to make robs and dfs (full version)
        Note this can be replaced by using the spike times explicitly"""

        self.robs = np.zeros( [self.NT, self.NC], dtype=np.uint8 )
        self.dfs = np.zeros( [self.NT, self.NC], dtype=np.uint8 )

        tcount, ccount = 0, 0
        for ff in range(self.Nexpts):
            NTexpt = self.file_info[ff]['NT']
            NSUs= self.file_info[ff]['NSUs']
            # Classify cell-lists in terms of SUs and MUc
            su_list = self.cranges[ff][self.cranges[ff] < NSUs]
            R_tslice = np.zeros( [NTexpt, self.NC], dtype=np.int64 )
            df_tslice = np.zeros( [NTexpt, self.NC], dtype=np.uint8 )

            tslice = np.array(self.fhandles[ff]['Robs'], dtype=np.int64)[self.tranges[ff], :]
            R_tslice[:, ccount+np.arange(len(su_list))] = deepcopy(tslice[:, su_list])
            tslice = np.array(self.fhandles[ff]['datafilts'], dtype=np.uint8)[self.tranges[ff], :]
            df_tslice[:, ccount+np.arange(len(su_list))] = deepcopy(tslice[:, su_list])
            ccount += len(su_list)

            if self.include_MUs:
                NMUs= self.file_info[ff]['NMUs']
                mu_list = self.cranges[ff][self.cranges[ff] >= NSUs]-NSUs
                tslice = np.array(self.fhandles[ff]['RobsMU'], dtype=np.int64)[self.tranges[ff], :]
                R_tslice[:, ccount+np.arange(len(mu_list))] = deepcopy(tslice[:, mu_list])
                tslice = np.array(self.fhandles[ff]['datafiltsMU'], dtype=np.int64)[self.tranges[ff], :]
                df_tslice[:, ccount+np.arange(len(mu_list))] = deepcopy(tslice[:, mu_list])
                ccount += len(mu_list)

            # Check that clipping to uint8 wont screw up any robs
            robs_ceil = np.where(R_tslice > 255)
            if len(robs_ceil[0]) > 0:
                print( "Neurons in expt %d have single-bin spike counts above 255:"%ff, robs_ceil[1] )
                # Currently do nothing -- this willbe modded: but if problems should probably make dfs = 0 there

            # Write tslice into 
            self.robs[tcount+np.arange(NTexpt), :] = deepcopy( R_tslice.astype(np.uint8) )
            self.dfs[tcount+np.arange(NTexpt), :] = deepcopy( df_tslice )

            tcount += NTexpt

    # END MultiClouds.assemble_robs()

    def list_expts( self ):
        """Show filenames with experiment number"""
        for ff in range(self.Nexpts):
            print('  %2d  %s'%(ff, self.filenames[ff]) )
        
    def updateDF( self, expt_n, dfs, reduce_cells=False ):
        """Import updated DF for given experiment, as numbered (can see with 'list_expts')
        Will check for neurons with no robs and reduce robs and dataset if reduce_cells=True"""

        assert expt_n < self.Nexpts, "updateDF: expt_n too large: not that many experiments"

        # if eye_config, then want to replace whole DFs, or relevant DFs 
        if dfs.shape[0] != self.file_info[expt_n]['NT']:
            # Assume need to use trange
            dfs = dfs[self.tranges[expt_n], :]
        assert dfs.shape[0] == self.file_info[expt_n]['NT'], "DF file mismatch: wrong length"
        dfs = dfs[:, self.cranges[expt_n]]

        # Replace dfs with updated
        trange = self.file_tstart[expt_n] + np.arange(dfs.shape[0])
        df_tslice = deepcopy( self.dfs[trange, :] )
        crange = np.arange(self.fileNC[expt_n])
        if expt_n > 0:
            crange += np.sum(self.fileNC[:expt_n])
        df_tslice[:, crange] = dfs.astype(np.uint8)
        self.dfs[trange, :] = deepcopy(df_tslice)

        if reduce_cells:
            elim_cells = np.where(np.sum(dfs, axis=0) == 0)[0]
            if elim_cells > 0:
                print('  reduce_cells not implemented yet')
                print( ' Cells to reduce:', elim_cells )
    # END MultiClouds.updateDF()

    def assemble_saccade_inds( self ):
        print('Currently not implemented -- needs to have microsaccades labeled well with time first')

    def is_fixpoint_present( self, boxlim, expt_n ):
        """Return if any of fixation point is within the box given by top-left to bottom-right corner"""
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
            shifts=None, BUF=20, # shift buffer
            stim_crop=None,
            fixdot=0 ):
        """This assembles a stimulus from the raw numpy-stored stimuli into self.stim
        which_stim: determines what stimulus is assembled from 'ET'=0, 'lam'=1, None
            If none, will need top_corner present: can specify with four numbers (top-left, bot-right)
            or just top_corner and L
        which is torch.tensor on default device
        stim_wrap: only works if using 'which_stim', and will be [hwrap, vwrap]"""

        assert expt_n is not None, "CONSTRUCT_STIMULUS: must specify expt_n"
        # Delete existing stim and clear cache to prevent memory issues on GPU

        need2crop = False

        if which_stim is not None:
            assert L is None, "CONSTRUCT_STIMULUS: cannot specify L if using which_stim (i.e. prepackaged stim)"
            if not isinstance(which_stim, int):
                if which_stim in ['ET', 'et', 'stimET']:
                    which_stim=0
                else:
                    which_stim=1
            if which_stim == 0:
                print( "Stim #%d: using ET stimulus", expt_n )
                stim_tmp = np.array(self.fhandles[expt_n]['stimET'], dtype=np.int8)
                #self.stim_pos = self.stim_locationET[:,0]
            else:
                print( "Stim #%d: using laminar probe stimulus", expt_n )
                stim_tmp = np.array(self.fhandles[expt_n]['stim'], dtype=np.int8)
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

            if shifts is not None:
                need2crop = True
                # Extend stim window by BUFF-per-side in each direction
                stim_pos = [
                    stim_pos[0]-BUF,
                    stim_pos[1]-BUF,
                    stim_pos[2]+BUF,
                    stim_pos[3]+BUF]
                print( "  Stim expansion for shift:", stim_pos)
                L += 2*BUF

            # Read in stimuli
            stimET = np.array(self.fhandles[expt_n]['stimET'][self.tranges[expt_n], ...], dtype=np.int8)
            stimLP = np.array(self.fhandles[expt_n]['stim'][self.tranges[expt_n], ...], dtype=np.int8)
            locsET = self.file_info[expt_n]['stim_locsET']
            locsLP = self.file_info[expt_n]['stim_locsLP']

            if self.luminance_only:
                stimET = stimET[:, 0, ...][:, None, ...]  # maintain 2nd dim (length 1)
                stimLP = stimLP[:, 0, ...][:, None, ...]
                num_clr = 1
            else:
                num_clr = 3

            NT = self.fileNT[expt_n]
            newstim = np.zeros( [NT, num_clr, L, L], dtype=np.int8 )
            for ii in range(locsLP.shape[1]):
                OVLP = self.rectangle_overlap_ranges(stim_pos, locsLP[:, ii])
                if OVLP is not None:
                    print( "  Writing lam stim %d: overlap %d, %d"%(ii, len(OVLP['targetX']), len(OVLP['targetY'])))
                    strip = deepcopy(newstim[:, :, OVLP['targetX'], :]) #np.zeros([self.NT, num_clr, len(OVLP['targetX']), L])
                    strip[:, :, :, OVLP['targetY']] = deepcopy((stimLP[:, :, OVLP['readX'], :][:, :, :, OVLP['readY']]))
                    newstim[:, :, OVLP['targetX'], :] = deepcopy(strip)
                
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

        if shifts is not None:
            # Would want to shift by input eye positions if input here
            #print('eye-position shifting not implemented yet')
            print('  Shifting stim...')
            if len(shifts) > newstim.shape[0]:
                shifts = shifts[self.tranges[expt_n]]
            newstim = self.shift_stim( newstim, shifts )

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

        # Flatten stim 
        self.expt_stims[expt_n] = deepcopy(newstim.reshape([self.fileNT[expt_n], -1]))
        print( "  Done: expt", expt_n )
    # END MultiClouds.build_stim()

    def assemble_stim( self ):

        self.stim_dims = [3, self.L, self.L, 1]
        if self.luminance_only:
            self.stim_dims[0] = 1
        if self.time_embed:
            self.stim_dims[3] = self.num_lags
        num_dims = np.prod(self.stim_dims)
        self.stim = np.zeros( [self.NT, num_dims], dtype=np.int8 )
        for ff in range(self.Nexpts):
            assert self.expt_stims[ff] is not None, 'ASSEMBLE_STIM: stim %d is not yet built.'%ff  
            trange = range(self.file_tstart[ff], self.file_tstart[ff]+self.fileNT[ff])
            self.stim[trange, :] = self.expt_stims[ff]
        print( "Stimulus assembly complete")
    # END MultiClouds.assemble_stimulus()

    def time_embedding( self, stim=None, nlags=None ):
        """Note this overloads SensoryBase because reshapes in full dimensions to handle folded_lags"""
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
        """Figures out ranges to write relevant overlap of B onto A
        All info is of form [x0, y0, x1, y1]"""
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
        """Crop existing (torch) stimulus and change relevant variables [x1, x2, y1, y2]"""

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

#    def process_fixations( self, sacc_in=None ):
#        """Processes fixation informatiom from dataset, but also allows new saccade detection
#        to be input and put in the right format within the dataset (main use)"""
#        if sacc_in is None:
#            sacc_in = self.sacc_inds[:, 0]
#        else:
#            print( "  Redoing fix_n with saccade inputs: %d saccades"%len(sacc_in) )
#            if self.start_t > 0:
#                print( "  -> Adjusting timing for non-zero start time in this dataset.")
#            sacc_in = sacc_in - self.start_t
#            sacc_in = sacc_in[sacc_in > 0]
#
#        fix_n = np.zeros(self.NT, dtype=np.int64) 
#        fix_count = 0
#        for ii in range(len(self.block_inds)):
#            # Parse fixation numbers within block
#            rel_saccs = np.where((sacc_in > self.block_inds[ii][0]+6) & (sacc_in < self.block_inds[ii][-1]-5))[0]
#
#            tfix = self.block_inds[ii][0]  # Beginning of fixation by definition
#            for mm in range(len(rel_saccs)):
#                fix_count += 1
#                # Range goes to beginning of next fixation (note no gap)
#                fix_n[ range(tfix, sacc_in[rel_saccs[mm]]) ] = fix_count
#                tfix = sacc_in[rel_saccs[mm]]
#            # Put in last (or only) fixation number
#            if tfix < self.block_inds[ii][-1]:
#                fix_count += 1
#                fix_n[ range(tfix, self.block_inds[ii][-1]) ] = fix_count
#
#        # Determine whether to be numpy or tensor
#        if isinstance(self.robs, torch.Tensor):
#            self.fix_n = torch.tensor(fix_n, dtype=torch.int64, device=self.robs.device)
#        else:
#            self.fix_n = fix_n
    # END: ColorClouds.process_fixations()

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
    
    def avrates( self, inds=None ):
        """
        Calculates average firing probability across specified inds (or whole dataset)
        -- Note will respect datafilters
        -- will return precalc value to save time if already stored
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

    @staticmethod 
    def shift_stim( stim, shifts, input_dims=None, batch_size=5000):

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
            'eyepos': torch.tensor(shifts, dtype=torch.float32)*Lscale }

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
        '''
        Primary function for shifting the intput stimulus
        Inputs:
            stim [Batch x channels x height x width] (use Fold2d to fold lags if necessary)
            shift [Batch x 2] or [Batch x 4] if translation only or affine
            affine [Boolean] set to True if using affine transformation
            mode [str] 'bilinear' (default) or 'nearest'
            NOTE: mode must be bilinear during fitting otherwise the gradients don't propogate well
        '''
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

        Inputs:
            dataset: the dataset to get the samples from
            device: the device to put the samples on
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
        """Note this loads stimulus but does not time-embed"""

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

            out = {'stim': self.stim[idx, :],
                'robs': robs_tmp[idx, :],
                'dfs': dfs_tmp[idx, :]}
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