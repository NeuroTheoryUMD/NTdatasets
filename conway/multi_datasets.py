
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
        time_embed=None,  # 0 is no time embedding, 1 is time_embedding with get_item, 2 is pre-time_embedded
        num_lags=10, 
        include_MUs=False,
        drift_interval=None,
        trial_sample=True,
        luminance_only=True,
        binocular = False, # whether to include separate filters for each eye
        eye_config = 2,  # 0 = all, 1, -1, and 2 are options (2 = binocular)
        cell_lists = None,
        # device=torch.device('cpu'),
        # Dataset-specitic inputs
        # Stim setup -- if dont want to assemble stimulus: specify all things here for default options
        #which_stim=None,  # 'et' or 0, or 1 for lam, but default assemble later
        #stim_crop=None,  # should be list/array of 4 numbers representing inds of edges
        #ignore_saccades=True,
        #folded_lags=False, 
        maxT = None):
        """Constructor options"""

        super().__init__(
            filenames=filenames, datadir=datadir, 
            time_embed=time_embed, num_lags=num_lags, include_MUs=include_MUs, 
            drift_interval=drift_interval, trial_sample=trial_sample)

        # Done in parent constructor
        #self.datadir = datadir
        #self.filenames = filenames
        #self.device = device
        #self.num_lags = 10  # default: to be set later
        #if time_embed == 2:
        #    assert preload, "Cannot pre-time-embed without preloading."
        #self.time_embed = time_embed
        #self.preload = preload

        self.Nexpts = len(filenames)

        # Stim-specific
        self.eye_config = eye_config
        self.binocular = binocular
        self.luminance_only = luminance_only
        self.includeMUs = include_MUs
        self.generate_Xfix = False
        self.output_separate_eye_stim = False
        self.expt_stims = [None]*self.Nexpts

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
        self.data_threshold = 6  # how many valid time points required to include saccade?
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
            self.fileNC[ff] = self.file_info[ff]['NC']

            # Consolidate valid t-ranges based on binocular choice
            self.tranges[ff] = self.NTfile[ff]['tmap']
            self.cranges[ff] = np.arange(self.fileNC[ff], dtype=np.int64)

            # Make one long block-list
            NBLK = self.file_info[ff]['block_inds'].shape[1]
            for bb in range(NBLK):
                self.block_inds.append( tcount + np.arange(self.file_info[ff]['block_inds'][bb, :]) )
            self.fileNBLK[ff] = self.file_info[ff]['NT']
            self.file_blkstart[ff] = blkcount
            blkcount += NBLK

            self.fileNT[ff] = self.file_info[ff]['NT']
            self.file_tstart[ff] = tcount
            tcount += self.file_info[ff]['NT']

        # Assemble robs and dfs given current information
        if cell_lists is None:
            self.modify_included_cells(cell_lists)
            # this will automatically assemble_robs at the end
        else:
            self.assemble_robs()

        # Determine total experiment time and number of cells
        self.NT = tcount
        self.NC = np.sum(self.fileNC)
        print( "%d total time steps, %d units"%(self.NT, self.NC) )

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

        if to_assemble:
            self.assemble_saccade_inds()

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
                Xdrift_expts.append(self.design_matrix_drift( self.NT, anchors, zero_left=False, const_right=True))
                Nanchors_tot += Nanchors

            # Assemble whole drift matrix
            self.Xdrift = np.zeros( [self.NT, Nanchors_tot], dtype=np.float32 )
            anchor_count, tcount = 0, 0
            for ff in range(self.Nexpts):
                tslice = np.zeros( np.zeros( [self.file_info[ff]['NT'], Nanchors_tot], dtype=np.float32 ) )
                tslice[:, anchor_count+np.arange(Xdrift_expts[ff].shape[1])] = Xdrift_expts[ff]
                self.Xdrift[tcount+np.arange(self.file_info[ff]['NT']), :] = deepcopy(tslice)

            # Determine overall valid-inds and cross-validation indices
            self.create_val_indices()
            # how divide up train and test inds -- definitely by blocks
            self.val_blks, self.train_blocks = [], []
            folds = 5
            random_gen = False
            for ff in range(self.Nexpts):
                Nblks = self.file_info['trial_info'].shape[1]
                val_blk_e, tr_blk_e = self.fold_sample(Nblks, folds, random_gen=random_gen)
                self.val_blks.append(deepcopy(val_blk_e)+self.blkstart)
                self.train_blks.append(deepcopy(tr_blk_e)+self.blkstart)

            self.val_inds, self.train_inds = [], []
            for bb in self.val_blks:
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
        NBLK = blk_inds.shape[1]

        # Stim information
        fix_loc = np.array(f['fix_location'], dtype=np.in64).squeeze()
        fix_size = np.array(f['fix_size'], dtype=np.in64).squeeze()
        stim_scale = np.array(f['stimscale'], dtype=np.in64).squeeze()
        stim_locsLP = np.array(f['stim_location'], dtype=np.in64),
        stim_locsET = np.array(f['ETstim_location'], dtype=np.in64)
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
                if LRpresent[blk_inds[bb,0]] == self.eye_config:
                    bmap.append(bb)
                    NTblk = blk_inds[bb,1]-blk_inds[bb,0] + 1
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
                assert np.max(clists[ff]) < self.file_info[ff]['NC']
                self.cranges[ff] = deepcopy(clists[ff])
            else:
                self.cranges[ff] = deepcopy(np.arange(self.file_info[ff]['NC']))
            self.fileNC[ff] = len(self.cranges[ff])
        self.NC = np.sum(self.fileNC)

        self.assemble_robs()
    # END MultiClouds.modify_included_cells()

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
            su_list = self.cell_lists[ff][self.cell_lists[ff] < NSUs]
            R_tslice = np.zeros( [NTexpt, self.NC], dtype=np.int64 )
            df_tslice = np.zeros( [NTexpt, self.NC], dtype=np.uint8 )

            R_tslice[:, ccount+np.arange(NSUs)] = np.array(self.fhandles[ff]['Robs'], dtype=np.int64)[:, su_list]
            df_tslice[:, ccount+np.arange(NSUs)] = np.array(self.fhandles[ff]['datafilts'], dtype=np.uint8)[:, su_list]
            ccount += len(su_list)

            if self.include_MUs:
                NMUs= self.file_info[ff]['NMUs']
                mu_list = self.cell_lists[ff][self.cell_lists[ff] >= NSUs]-NSUs
                R_tslice[:, ccount+np.arange(NMUs)] = np.array(self.fhandles[ff]['RobsMU'], dtype=np.int64)[:, mu_list]
                df_tslice[:, ccount+np.arange(NMUs)] = np.array(self.fhandles[ff]['datafiltsMU'], dtype=np.uint8)[:, mu_list]
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
        
    def updateDF( self, expt_n, dfs, reduce_cells=True ):
        """Import updated DF for given experiment, as numbered (can see with 'list_expts')
        Will check for neurons with no robs and reduce robs and dataset if reduce_cells=True"""

        assert expt_n < self.Nexpts, "updateDF: expt_n too large: not that many experiments"

        # if eye_config, then want to replace whole DFs, or relevant DFs 
        if dfs.shape[0] != self.file_info[expt_n]['NT']:
            # Assume need to use trange
            dfs = dfs[self.tranges[expt_n], :]
        assert dfs.shape[0] == self.file_info[expt_n]['NT'], "DF file mismatch: wrong length"

        # Replace dfs with updated
        trange = self.file_tstart[expt_n] + np.arange(dfs.shape[0])
        df_tslice = deepcopy( self.dfs[trange, :] )
        crange = np.arange(self.fileNC[expt_n])
        if expt_n > 0:
            crange += np.sum(self.fileNC[:expt_n])
        df_tslice[:, crange] = dfs.astype(np.unit8)
        self.dfs[trange, :] = deepcopy(df_tslice)

        if reduce_cells:
            elim_cells = np.where(np.sum(dfs, axis=0) == 0)[0]
            if elim_cells > 0:
                print('  reduce_cells not implemented yet')
                print( ' Cells to reduce:', elim_cells )
    # END MultiClouds.updateDF()

    def assemble_saccade_inds( self ):
        print('Currently not implemented -- needs to have microsaccades labeled well with time first')

    def is_fixpoint_present( self, boxlim ):
        """Return if any of fixation point is within the box given by top-left to bottom-right corner"""
        if self.fix_location is None:
            return False
        fix_present = True
        if self.stim_location.shape[1] == 1:
            # if there is multiple windows, needs to be manual: so this is the automatic check:
            for dd in range(2):
                if (self.fix_location[dd]-boxlim[dd] <= -self.fix_size):
                    fix_present = False
                if (self.fix_location[dd]-boxlim[dd+2] > self.fix_size):
                    fix_present = False
        return fix_present
    # END .is_fixpoint_present

    def construct_stimulus(
            self, expt_n=None,
            which_stim=None, top_corner=None, L=None,  # position of stim
            time_embed=0, num_lags=10,
            shifts=None, BUF=20, # shift buffer
            shift_times=None, # So that can put partial-shifts over time range in stimulus
            fixdot=0 ):
        """This assembles a stimulus from the raw numpy-stored stimuli into self.stim
        which_stim: determines what stimulus is assembled from 'ET'=0, 'lam'=1, None
            If none, will need top_corner present: can specify with four numbers (top-left, bot-right)
            or just top_corner and L
        which is torch.tensor on default device
        stim_wrap: only works if using 'which_stim', and will be [hwrap, vwrap]"""

        assert expt_n is not None, "CONSTRUCT_STIMULUS: must specify expt_n"
        # Delete existing stim and clear cache to prevent memory issues on GPU
        if self.stims[expt_n] is not None:
            del self.stims[expt_n]
            self.stim = None
            torch.cuda.empty_cache()

        need2crop = False

        if which_stim is not None:
            assert L is None, "CONSTRUCT_STIMULUS: cannot specify L if using which_stim (i.e. prepackaged stim)"
            L = self.dims[1]
            if not isinstance(which_stim, int):
                if which_stim in ['ET', 'et', 'stimET']:
                    which_stim=0
                else:
                    which_stim=1
            if which_stim == 0:
                print( "Stim #%d: using ET stimulus", expt_n )
                stim_tmp = np.array(self.fhandles[expt_n]['stimET'], dtype=np.uint8)
                #self.stim_pos = self.stim_locationET[:,0]
            else:
                print( "Stim #%d: using laminar probe stimulus", expt_n )
                stim_tmp = np.array(self.fhandles[expt_n]['stim'], dtype=np.uint8)
                #self.stim_pos = self.stim_location[:, 0]

            if self.luminance_only:
                stim_tmp = stim_tmp[self.tranges[expt_n], [0], ...]
            else:
                stim_tmp = stim_tmp[self.tranges[expt_n], ...]

            # Forget binocular for now
            if self.binocular:
                print('currently not implemented')

        else:
            assert top_corner is not None, "Need top corner if which_stim unspecified"
            # Assemble from combination of ET and laminer probe (NP) stimulus
            if len(top_corner) == 4:
                stim_pos = top_corner
                #L = self.stim_pos[2]-self.stim_pos[0]
                assert stim_pos[3]-stim_pos[1] == stim_pos[2]-stim_pos[0], "Stim must be square (for now)"
            else:
                if L is None:
                    L = self.dims[1]
                self.stim_pos = [top_corner[0], top_corner[1], top_corner[0]+L, top_corner[1]+L]

            if shifts is not None:
                need2crop = True
                # Modify stim window by 20-per-side
                self.stim_pos = [
                    stim_pos[0]-BUF,
                    stim_pos[1]-BUF,
                    stim_pos[2]+BUF,
                    stim_pos[3]+BUF]
                print( "  Stim expansion for shift:", stim_pos)

            L = stim_pos[2]-stim_pos[0]
            assert stim_pos[3]-stim_pos[1] == L, "Stimulus not square"

            stimET = np.array(self.fhandles[expt_n]['stimET'], dtype=np.uint8)
            stimLP = np.array(self.fhandles[expt_n]['stim'], dtype=np.uint8)
            if self.luminance_only:
                stimET = stimET[self.tranges[expt_n], [0], ...]
                stimLP = stimLP[self.tranges[expt_n], [0], ...]
                num_clr = 1
            else:
                stimET = stimET[self.tranges[expt_n], ...]
                stimLP = stimLP[self.tranges[expt_n], ...]
                num_clr = 3

            NT = self.fileNT[expt_n]
            newstim = np.zeros( [NT, num_clr, L, L] )
            for ii in range(self.stim_location.shape[1]):
                OVLP = self.rectangle_overlap_ranges(stim_pos, self.stim_location[:, ii])
                if OVLP is not None:
                    print( "  Writing lam stim %d: overlap %d, %d"%(ii, len(OVLP['targetX']), len(OVLP['targetY'])))
                    strip = deepcopy(newstim[:, :, OVLP['targetX'], :]) #np.zeros([self.NT, num_clr, len(OVLP['targetX']), L])
                    strip[:, :, :, OVLP['targetY']] = deepcopy((stimLP[:, :, OVLP['readX'], :][:, :, :, OVLP['readY']]))
                    newstim[:, :, OVLP['targetX'], :] = deepcopy(strip)
                
            for ii in range(self.stim_locationET.shape[1]):
                OVLP = self.rectangle_overlap_ranges(stim_pos, self.stim_locationET[:,ii])
                if OVLP is not None:
                    print( "  Writing ETstim %d: overlap %d, %d"%(ii, len(OVLP['targetX']), len(OVLP['targetY'])))
                    strip = deepcopy(newstim[:, :, OVLP['targetX'], :])
                    strip[:, :, :, OVLP['targetY']] = deepcopy((stimET[:, :, OVLP['readX'], :][:, :, :, OVLP['readY']]))
                    newstim[:, :, OVLP['targetX'], :] = deepcopy(strip) 

            stim = torch.tensor( newstim, dtype=torch.float32, device=self.device )

        #print('Stim shape', self.stim.shape)
        # Note stim stored in numpy is being represented as full 3-d + 1 tensor (time, channels, NX, NY)
        self.stim_dims = [self.dims[0], L, L, 1]

        # Insert fixation point
        if (fixdot is not None) and self.is_fixpoint_present( self.stim_pos ):
            fixranges = [None, None]
            for dd in range(2):
                fixranges[dd] = np.arange(
                    np.maximum(self.fix_location[dd]-self.fix_size-stim_pos[dd], 0),
                    np.minimum(self.fix_location[dd]+self.fix_size+1, stim_pos[dd+2])-stim_pos[dd] 
                    ).astype(int)
            # Write the correct value to stim
            #print(fixranges)
            assert fixdot == 0, "Haven't yet put in other fixdot settings than zero" 
            #strip = deepcopy(self.stim[:, :, fixranges[0], :])
            #strip[:, :, :, fixranges[1]] = 0
            #self.stim[:, :, fixranges[0], :] = deepcopy(strip) 
            print('  Adding fixation point')
            for xx in fixranges[0]:
                self.stim[:, :, xx, fixranges[1]] = 0

        self.stim_shifts = shifts
        if self.stim_shifts is not None:
            # Would want to shift by input eye positions if input here
            #print('eye-position shifting not implemented yet')
            print('  Shifting stim...')
            if shift_times is None:
                self.stim = self.shift_stim( shifts, shift_times=shift_times, already_lagged=False )
            else:
                self.stim[shift_times, ...] = self.shift_stim( shifts, shift_times=shift_times, already_lagged=False )

        # Reduce size back to original If expanded to handle shifts
        if need2crop:
            #assert self.stim_crop is None, "Cannot crop stim at same time as shifting"
            self.crop_stim( [BUF, L-BUF-1, BUF, L-BUF-1] )  # move back to original size
            self.stim_crop = None
            L = L-2*BUF
            self.stim_dims[1] = L
            self.stim_dims[2] = L
        else:
            if self.stim_crop is not None:
                self.crop_stim()

        if time_embed is not None:
            self.time_embed = time_embed
        if time_embed > 0:
            #self.stim_dims[3] = num_lags  # this is set by time-embedding
            if time_embed == 2:
                self.stim = self.time_embedding( self.stim, nlags = num_lags )
        # now stimulus is represented as full 4-d + 1 tensor (time, channels, NX, NY, num_lags)

        self.num_lags = num_lags

        # Flatten stim 
        self.stim = self.stim.reshape([self.NT, -1])
        print( "  Done" )
    # END .construct_stimulus()

    def assemble_stimulus( self ):
        print('need to assemble the thing')
    # END .assemble_stimulus()


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


    def time_embedding( self, stim=None, nlags=None ):
        """Note this overloads SensoryBase because reshapes in full dimensions to handle folded_lags"""
        assert self.stim_dims is not None, "Need to assemble stim before time-embedding."
        if nlags is None:
            nlags = self.num_lags
        if self.stim_dims[3] == 1:
            self.stim_dims[3] = nlags
        if stim is None:
            tmp_stim = deepcopy(self.stim)
        else:
            if isinstance(stim, np.ndarray):
                tmp_stim = torch.tensor( stim, dtype=torch.float32)
            else:
                tmp_stim = deepcopy(stim)
        #if not isinstance(tmp_stim, np.ndarray):
        #    tmp_stim = tmp_stim.cpu().numpy()
    
        NT = stim.shape[0]
        print("  Time embedding...")
        if len(tmp_stim.shape) == 2:
            print( "Time embed: reshaping stimulus ->", self.stim_dims)
            tmp_stim = tmp_stim.reshape([NT] + self.stim_dims)

        assert self.NT == NT, "TIME EMBEDDING: stim length mismatch"

        tmp_stim = tmp_stim[np.arange(NT)[:,None]-np.arange(nlags), :, :, :]
        if self.folded_lags:
            #tmp_stim = np.transpose( tmp_stim, axes=[0,2,1,3,4] ) 
            tmp_stim = torch.permute( tmp_stim, (0,2,1,3,4) ) 
            print("Folded lags: stim-dim = ", self.stim.shape)
        else:
            #tmp_stim = np.transpose( tmp_stim, axes=[0,2,3,4,1] )
            tmp_stim = torch.permute( tmp_stim, (0,2,3,4,1) )
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

    def wrap_stim( self, vwrap=0, hwrap=0 ):
        """Take existing stimulus and move the whole thing around in horizontal and/or vertical dims,
        including if time_embedded"""

        assert self.stim is not None, "Must assemble the stimulus before using wrap_stim."
        orig_stim_dims = len(self.stim.shape)
        if orig_stim_dims == 5:
            tmp_stim = deepcopy(self.stim)
        elif orig_stim_dims == 4:
            tmp_stim = deepcopy(self.stim[:, :, :, :, None])  # put in lags as one dim
        else:
            tmp_stim = deepcopy(self.stim).reshape([self.NT] + self.dims)

        NY = self.stim_dims[2]
        if vwrap > 0:
            tmp2 = torch.zeros(tmp_stim.shape, device=tmp_stim.device)
            tmp2[:, :, :, :vwrap, :] = tmp_stim[:, :, :, (NY-vwrap):, :]
            tmp2[:, :, :, vwrap:, :] = tmp_stim[:, :, :, :(NY-vwrap), :]
        elif vwrap < 0:
            tmp2 = torch.zeros(tmp_stim.shape, device=tmp_stim.device)
            tmp2[:, :, :, (NY+vwrap):, :] = tmp_stim[:, :, :, :(-vwrap), :]
            tmp2[:, :, :, :(NY+vwrap), :] = tmp_stim[:, :, :, (-vwrap):, :]
        else:
            tmp2 = tmp_stim

        NX = self.stim_dims[1]
        if hwrap > 0:
            self.stim = torch.zeros(tmp2.shape, device=tmp2.device)
            self.stim[:, :, :hwrap, :, :] = tmp2[:, :, (NX-hwrap):, :, :]
            self.stim[:, :, hwrap:, :, :] = tmp2[:, :, :(NX-hwrap), :, :]
        elif hwrap < 0:
            self.stim = torch.zeros(tmp2.shape, device=tmp2.device)
            self.stim[:, :, (NX+hwrap):, :, :] = tmp2[:, :, :(-hwrap), :, :]
            self.stim[:, :, :(NX+hwrap), :, :] = tmp2[:, :, (-hwrap):, :, :]
        else:
            self.stim = deepcopy(tmp2)

        if orig_stim_dims == 3:
            self.stim = self.stim.reshape( [self.NT, -1] )
        elif orig_stim_dims == 4:
            # Take out extra lag dim
            self.stim = self.stim[:, :, :, :, 0]
    # END .wrap_stim()

    def crop_stim( self, stim_crop=None ):
        """Crop existing (torch) stimulus and change relevant variables [x1, x2, y1, y2]"""
        if stim_crop is None:
            stim_crop = self.stim_crop
        else:
            self.stim_crop = stim_crop 
        assert len(stim_crop) == 4, "stim_crop must be of form: [x1, x2, y1, y2]"
        if len(self.stim.shape) == 2:
            self.stim = self.stim.reshape([self.NT] + self.stim_dims)
            reshape=True
            #print('  CROP: reshaping stim')
        else:
            reshape=False
        #stim_crop = np.array(stim_crop, dtype=np.int64) # make sure array
        xs = np.arange(stim_crop[0], stim_crop[1]+1)
        ys = np.arange(stim_crop[2], stim_crop[3]+1)
        if len(self.stim.shape) == 4:
            self.stim = self.stim[:, :, :, ys][:, :, xs, :]
        else:  # then lagged -- need extra dim
            self.stim = self.stim[:, :, :, ys, :][:, :, xs, :, :]

        print("  CROP: New stim size: %d x %d"%(len(xs), len(ys)))
        self.stim_dims[1] = len(xs)
        self.stim_dims[2] = len(ys)
        if reshape:
            print(self.stim.shape, 'reshaping back')
            self.stim = self.stim.reshape([self.NT, -1])

    # END .crop_stim()

    def process_fixations( self, sacc_in=None ):
        """Processes fixation informatiom from dataset, but also allows new saccade detection
        to be input and put in the right format within the dataset (main use)"""
        if sacc_in is None:
            sacc_in = self.sacc_inds[:, 0]
        else:
            print( "  Redoing fix_n with saccade inputs: %d saccades"%len(sacc_in) )
            if self.start_t > 0:
                print( "  -> Adjusting timing for non-zero start time in this dataset.")
            sacc_in = sacc_in - self.start_t
            sacc_in = sacc_in[sacc_in > 0]

        fix_n = np.zeros(self.NT, dtype=np.int64) 
        fix_count = 0
        for ii in range(len(self.block_inds)):
            #if self.block_inds[ii][0] < self.NT:
            # note this will be the inds in each file -- file offset must be added for mult files
            #self.block_inds.append(np.arange( blk_inds[ii,0], blk_inds[ii,1], dtype=np.int64))

            # Parse fixation numbers within block
            rel_saccs = np.where((sacc_in > self.block_inds[ii][0]+6) & (sacc_in < self.block_inds[ii][-1]-5))[0]

            tfix = self.block_inds[ii][0]  # Beginning of fixation by definition
            for mm in range(len(rel_saccs)):
                fix_count += 1
                # Range goes to beginning of next fixation (note no gap)
                fix_n[ range(tfix, sacc_in[rel_saccs[mm]]) ] = fix_count
                tfix = sacc_in[rel_saccs[mm]]
            # Put in last (or only) fixation number
            if tfix < self.block_inds[ii][-1]:
                fix_count += 1
                fix_n[ range(tfix, self.block_inds[ii][-1]) ] = fix_count

        # Determine whether to be numpy or tensor
        if isinstance(self.robs, torch.Tensor):
            self.fix_n = torch.tensor(fix_n, dtype=torch.int64, device=self.robs.device)
        else:
            self.fix_n = fix_n
    # END: ColorClouds.process_fixations()

    def augment_dfs( self, new_dfs, cells=None ):
        """Replaces data-filter for given cells. note that new_df should be np.ndarray"""
        
        NTdf, NCdf = new_dfs.shape 
        if cells is None:
            assert NCdf == self.dfs.shape[1], "new DF is wrong shape to replace DF for all cells."
            cells = range(self.dfs.shape[1])
        if self.NT < NTdf:
            self.dfs[:, cells] *= torch.tensor(new_dfs[:self.NT, :], dtype=torch.float32)
        else:
            if self.NT > NTdf:
                # Assume dfs are 0 after new valid region
                print("Truncating valid region to new datafilter length", NTdf)
                new_dfs = np.concatenate( 
                    (new_dfs, np.zeros([self.NT-NTdf, len(cells)], dtype=np.float32)), 
                    axis=0)
            self.dfs[:, cells] *= torch.tensor(new_dfs, dtype=torch.float32)
        # END ColorClouds.replace_dfs()

    def draw_stim_locations( self, top_corner=None, L=None, row_height=5.0 ):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        lamlocs = self.stim_location
        ETlocs = self.stim_locationET
        fixloc = self.fix_location
        fixsize = self.fix_size
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

    def shift_stim(
        self, pos_shifts, metrics=None, metric_threshold=1, ts_thresh=8,
        shift_times=None, already_lagged=True ):
        """Shift stimulus given standard shifting input (TBD)
        use 'shift-times' if given shifts correspond to range of times"""
        NX = self.stim_dims[1]
        nlags = self.stim_dims[3]

        # Check if has been time-lagged yet
  
        if already_lagged:
            re_stim = deepcopy(self.stim).reshape([-1] + self.stim_dims)[..., 0]
        else:
            assert len(self.stim) > 2, "Should be already lagged, but seems not"
            re_stim = deepcopy(self.stim)

        # Apply shift-times (subset)
        if shift_times is None:
            fix_n = np.array(self.fix_n, dtype=np.int64)
        else:
            fix_n = np.array(self.fix_n[shift_times], dtype=np.int64)
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

    def shift_stim_fixation( self, stim, shift):
        """Simple shift by integer (rounded shift) and zero padded. Note that this is not in 
        is in units of number of bars, rather than -1 to +1. It assumes the stim
        has a batch dimension (over a fixation), and shifts the whole stim by the same amount."""
        print('Currently needs to be fixed to work with 2D')
        sh = round(shift)
        shstim = stim.new_zeros(*stim.shape)
        if sh < 0:
            shstim[:, -sh:] = stim[:, :sh]
        elif sh > 0:
            shstim[:, :-sh] = stim[:, sh:]
        else:
            shstim = deepcopy(stim)

        return shstim
    # END .shift_stim_fixation

    def create_val_indices(self, post_sacc_gap=None):
        """
        This creates self.valid_inds vector that is used for __get_item__ 
        -- Will default to num_lags following each saccade beginning"""

        if post_sacc_gap is None:
            post_sacc_gap = self.num_lags

        is_valid = np.zeros(self.NT, dtype=np.int64)
        for ff in range(self.Nexpts):
            # assemble all current val_inds from each experiment
            is_valid[self.file_tstart[ff]+self.file_info[ff]['valid_inds']] = 1

            # Now invalid post_sacc_gap following saccades
            if self.sacc_inds[ff] is not None:
                for nn in range(len(self.sacc_inds[ff])):
                    #print(self.sacc_inds[nn, :])
                    sts = self.sacc_inds[nn, :]
                    is_valid[self.file_tstart[ff]+np.arange(sts[0], np.minimum(sts[0]+post_sacc_gap, self.NT))] = 0

        # But cancel if dfs says there are no valid cells
        is_valid[torch.sum(self.dfs, axis=1) == 0] = 0
        self.valid_inds = np.where(is_valid > 0)[0]
    # END MultiCloud.create_val_indices()

    def crossval_setup(self, folds=5, random_gen=False, test_set=True, verbose=False):
        """This sets the cross-validation indices up We can add featuers here. Many ways to do this
        but will stick to some standard for now. It sets the internal indices, which can be read out
        directly or with helper functions. Perhaps helper_functions is the best way....
        
        Inputs: 
            random_gen: whether to pick random fixations for validation or uniformly distributed
            test_set: whether to set aside first an n-fold test set, and then within the rest n-fold train/val sets
        Outputs:
            None: sets internal variables test_inds, train_inds, val_inds
        """
        assert self.valid_inds is not None, "Must first specify valid_indices before setting up cross-validation."

        # Partition data by saccades, and then associate indices with each
        te_fixes, tr_fixes, val_fixes = [], [], []
        for ee in range(len(self.fixation_grouping)):  # Loops across experiments
            fixations = np.array(self.fixation_grouping[ee])  # fixations associated with each experiment
            val_fix1, tr_fix1 = self.fold_sample(len(fixations), folds, random_gen=random_gen)
            if test_set:
                te_fixes += list(fixations[val_fix1])
                val_fix2, tr_fix2 = self.fold_sample(len(tr_fix1), folds, random_gen=random_gen)
                val_fixes += list(fixations[tr_fix1[val_fix2]])
                tr_fixes += list(fixations[tr_fix1[tr_fix2]])
            else:
                val_fixes += list(fixations[val_fix1])
                tr_fixes += list(fixations[tr_fix1])

        if verbose:
            print("Partitioned %d fixations total: tr %d, val %d, te %d"
                %(len(te_fixes)+len(tr_fixes)+len(val_fixes),len(tr_fixes), len(val_fixes), len(te_fixes)))  

        # Now pull  indices from each saccade 
        tr_inds, te_inds, val_inds = [], [], []
        for nn in tr_fixes:
            tr_inds += range(self.sacc_inds[nn][0], self.sacc_inds[nn][1])
        for nn in val_fixes:
            val_inds += range(self.sacc_inds[nn][0], self.sacc_inds[nn][1])
        for nn in te_fixes:
            te_inds += range(self.sacc_inds[nn][0], self.sacc_inds[nn][1])

        if verbose:
            print( "Pre-valid data indices: tr %d, val %d, te %d" %(len(tr_inds), len(val_inds), len(te_inds)) )

        # Finally intersect with valid indices
        self.train_inds = np.array(list(set(tr_inds) & set(self.valid_inds)))
        self.val_inds = np.array(list(set(val_inds) & set(self.valid_inds)))
        self.test_inds = np.array(list(set(te_inds) & set(self.valid_inds)))

        if verbose:
            print( "Valid data indices: tr %d, val %d, te %d" %(len(self.train_inds), len(self.val_inds), len(self.test_inds)) )

    # END MultiDatasetFix.crossval_setup

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

        if self.preload:

            if self.time_embed == 1:
                print("get_item time embedding not implemented yet")
                # if self.folded_lags:
                #    stim = np.transpose( tmp_stim, axes=[0,2,1,3,4] ) 
                #else:
                #    stim = np.transpose( tmp_stim, axes=[0,2,3,4,1] )
    
            else:
                if len(self.cells_out) == 0:
                    out = {'stim': self.stim[idx, :],
                        'robs': self.robs[idx, :],
                        'dfs': self.dfs[idx, :]}
                    if len(self.fix_n) > 0:
                        out['fix_n'] = self.fix_n[idx]
                        # missing saccade timing vector -- not specified
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
                    if len(self.fix_n) > 0:
                        out['fix_n'] = self.fix_n[idx]

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
        else:
            inds = self.valid_inds[idx]
            stim = []
            robs = []
            dfs = []
            num_dims = self.stim_dims[0]*self.stim_dims[1]*self.stim_dims[2]

            """ Stim """
            # need file handle
            f = 0
            #f = self.file_index[inds]  # problem is this could span across several files

            stim = torch.tensor(self.fhandles[f]['stim'][inds,:], dtype=torch.float32)
            # reshape and flatten stim: currently its NT x NX x NY x Nclrs
            stim = stim.permute([0,3,1,2]).reshape([-1, num_dims])
                
            """ Spikes: needs padding so all are B x NC """ 
            robs = torch.tensor(self.fhandles[f]['Robs'][inds,:], dtype=torch.float32)
            if self.include_MUs:
                robs = torch.cat(
                    (robs, torch.tensor(self.fhandles[f]['RobsMU'][inds,:], dtype=torch.float32)), 
                    dim=1)

                """ Datafilters: needs padding like robs """
            dfs = torch.tensor(self.fhandles[f]['DFs'][inds,:], dtype=torch.float32)
            if self.include_MUs:
                dfs = torch.cat(
                    (dfs, torch.tensor(self.fhandles[f]['DFsMU'][inds,:], dtype=torch.float32)),
                    dim=1)

            out = {'stim': stim, 'robs': robs, 'dfs': dfs, 'fix_n': self.fix_n[inds]}

        # Addition whether-or-not preloaded
        if self.Xdrift is not None:
            out['Xdrift'] = self.Xdrift[idx, :]
        if self.binocular:
            out['binocular'] = self.binocular_gain[idx, :]
            
        if len(self.covariates) > 0:
            self.append_covariates( out, idx)

        ### THIS IS NOT NEEDED WITH TIME-EMBEDDING: needs to be on fixation-process side...
        # cushion DFs for number of lags (reducing stim)
        #if (self.num_lags > 0) &  ~utils.is_int(idx):
        #    if out['dfs'].shape[0] > self.num_lags:
        #        out['dfs'][:self.num_lags, :] = 0.0
        #    else: 
        #        print( "Warning: requested batch smaller than num_lags %d < %d"%(out['dfs'].shape[0], self.num_lags) )
     
        return out
    # END: CloudDataset.__get_item__

    #@property
    #def NT(self):
    #    return len(self.used_inds)

    def __len__(self):
        return self.robs.shape[0]