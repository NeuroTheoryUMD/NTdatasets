import os
import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import Dataset
import NDNT.utils as utils
#from NDNT.utils import download_file, ensure_dir
from copy import deepcopy
import h5py
from NTdatasets.sensory_base import SensoryBase

class WhiskerData(SensoryBase):
    """
    WhiskerData is a class for handling whisker data from the lab of Scott Pluta.
    """

    def __init__(self, expt_name=None, hemi=0, num_lags=30, **kwargs):
        """
        Args:
            expt_name: name of the experiment directory within the datadir
            hemi: 0=left, 1=right, 2=both
            num_lags: number of lags to include in the design matrix
            **kwargs: additional arguments to pass to the parent class
        """
        assert expt_name is not None, "Must specify expt_name, which is the directory name within the datadir."
        # call parent constructor
        super().__init__(filenames=expt_name, num_lags=num_lags, **kwargs)
        #self.expt_name = expt_name

        matdat = sio.loadmat( self.datadir+expt_name+'/ExpInfo.mat')
        ExptMat = matdat[list(matdat.keys())[-1]]
        NeurMat = sio.loadmat( self.datadir+expt_name+'/NeuralData.mat')['NeuralData'].astype(np.float32)
        CellLocs = sio.loadmat( self.datadir+expt_name+'/Location.mat')['Location']

        NT = ExptMat.shape[0]
        self.NT = NT

        # Read basic properties
        pistons = ExptMat[:, range(1,5)]
        self.touchfull = ExptMat[:, range(5, 9)]
        self.angles = ExptMat[:, range(9, 13)]
        self.curves = ExptMat[:, range(13, 17)]
        self.phases = ExptMat[:, range(17, 21)]
        behavior = ExptMat[:, range(22,26)]
        self.run_speed = ExptMat[:, 21]
        self.licks = ExptMat[:, 26]
        self.include_multitouches = False
        self.mtouches = None

        # Make trial-blocks and throw in used_inds
        self.used_inds = np.ones(NT)
        TRinds = self.trial_parse(ExptMat[:,0])
        Ntr = TRinds.shape[0]
        for bb in range(Ntr):
            self.block_inds.append(np.arange(TRinds[bb,0], TRinds[bb,1], dtype=np.int64))
            self.used_inds[TRinds[bb,0]+np.arange(num_lags)] = 0  # Zero out beginning of every trial

        # Process locations
        self.num_cells, self.electrode_info = self.process_locations( CellLocs )
        self.NC = np.sum(self.num_cells)
        self.Rparse = [list(np.arange(self.num_cells[0])), list(np.arange(self.num_cells[0], self.NC))]

        # Process neurons 
        assert NeurMat.shape[1]-1 == self.NC, "Cell count problem"

        self.robs = torch.tensor( NeurMat[:, 1:], dtype=torch.float32 )
        self.dfs = torch.tensor( np.ones([NT, self.NC]) * self.used_inds[:, None], dtype=torch.float32)

        self.cells_in = []
        self.set_hemispheres(out_config=hemi, in_config=2)   # default settings

        # Assign XV indices
        Xtr = np.arange(2, Ntr, 5, dtype='int64')
        Utr = np.array(list(set(np.arange(Ntr, dtype='int64'))-set(Xtr)))
        Ui, Xi, used_inds = np.zeros(0, dtype='int64'), np.zeros(0, dtype='int64'), np.zeros(0, dtype='int64')
        for tr in Utr:
            Ui = np.concatenate( (Ui, self.block_inds[tr]), axis=0)
        for tr in Xtr:
            Xi = np.concatenate( (Xi, self.block_inds[tr]), axis=0)

        self.train_inds = Ui
        self.val_inds = Xi

        ##### Additional Stim processing #####
        self.touches = np.zeros([NT, 4])
        for ww in range(4):
            self.touches[np.where(np.diff(self.touchfull[:, ww]) > 0)[0]+1, ww] = 1

        # Coincident touches -- original (pre 2024)
        #multitouches = np.zeros([NT, 4])
        #for ipsi in range(2):  # AC, AD, BC, BD (all cross-side pairings)
        #    multitouches[:,2*ipsi] = self.touchfull[:, 0] * self.touchfull[:, 2+ipsi]
        #    multitouches[:,1+2*ipsi] = self.touchfull[:, 1] * self.touchfull[:, 2+ipsi]
        # New multi-touches based on scotts advice
        #self.multitouches = np.zeros([NT, 8])
        # Extract onset for both single and multitouches
        #self.multitouches = np.zeros([NT, 4])
        #for ww in range(4):
        #    self.multitouches[np.where(np.diff(multitouches[:, ww]) > 0)[0]+1, ww] = 1

        self.TRpistons, self.TRoutcomes = self.trial_classify(TRinds, pistons, behavior)
        self.TRhit = np.where(self.TRoutcomes == 1)[0]
        self.TRmiss = np.where(self.TRoutcomes == 2)[0]
        self.TRfpos = np.where(self.TRoutcomes == 3)[0]
        self.TRcrej = np.where(self.TRoutcomes == 4)[0]
        self.TRuni = np.where(self.TRoutcomes == 5)[0]
        print("Hits: %d\tMisses: %d\nFalse Pos: %d\tCorrect rej %d\nUnilateral stim: %d"%(len(self.TRhit), 
                                                                                        len(self.TRmiss), 
                                                                                        len(self.TRfpos), 
                                                                                        len(self.TRcrej),
                                                                                        len(self.TRuni)))
        # Make drift matrix
        self.construct_drift_design_matrix() 

        # Auto-encoder design matrix
        self.ACinput = None

        # Configure stimulus  # default is just touches (onset)
        #self.prepare_stim()
    # END WhiskerData.__init__()

    def prepare_stim(self, stim_config=0, num_lags=None, temporal_basis=None, 
                     include_multitouches=False, pre_window=10, post_window=0, pre_post_window=2 ):

        self.include_multitouches = include_multitouches

        nvar_per_whisker = 0
        if post_window > 0:
            nvar_per_whisker += 2
        if pre_window > 0:
            nvar_per_whisker += 2
        self.multitouches = np.zeros([self.NT, nvar_per_whisker*4])
        print(self.multitouches.shape)
        if include_multitouches:
            self.touches[:pre_window, :] = 0  # just so no crashing -- retro
            self.touches[-np.maximum(pre_post_window, post_window):, :] = 0  # just so no crashing -- post
            
            for hem in range(2):
                oppo_ws = (1-hem)*2 + np.arange(2)
                for pw in range(2):
                    ww = 2*hem+pw
                    pwtouches = np.where(self.touches[:, ww] > 0)[0]
                    for tt in pwtouches:
                        uni_touch=True
                        # IPSI FIRST
                        if pre_window > 0:
                            ipsi_touch_count = np.sum(self.touches[range(tt-pre_window+1, tt+pre_post_window+1), :][:, oppo_ws], axis=0)
                            if ipsi_touch_count[0] > 0:
                                self.multitouches[tt, nvar_per_whisker*ww] = 1.0
                            elif ipsi_touch_count[1] > 0:
                                self.multitouches[tt, nvar_per_whisker*ww+1] = 1.0
                            if np.sum(ipsi_touch_count) > 0:
                                uni_touch=False
                        # IPSI SECOND
                        if post_window > 0:
                            if pre_window > 0:
                                # then zero-point is already included in pre
                                ipsi_touch_count = np.sum(self.touches[range(tt+1, tt+post_window), :][:,oppo_ws], axis=0)
                            else:
                                ipsi_touch_count = np.sum(self.touches[range(tt, tt+post_window), :][:,oppo_ws], axis=0)

                            if ipsi_touch_count[0] > 0:
                                self.multitouches[tt, nvar_per_whisker*ww+2] = 1.0
                            elif ipsi_touch_count[1] > 0:
                                self.multitouches[tt, nvar_per_whisker*ww+3] = 1.0
                            if np.sum(ipsi_touch_count) > 0:
                                uni_touch=False
                        
                        if not uni_touch:
                            self.touches[tt, ww] = 0.0

        #self.stim = torch.tensor( self.touches, dtype=torch.float32, device=device )
        if num_lags is None:
            num_lags = self.num_lags

        self.stim_dims = [2, 1, 1, 1]
        if stim_config == 0:
            self.stim = self.time_embedding( stim=self.touches[:, :2], nlags=num_lags, verbose=False )
            self.stimA = self.time_embedding( stim=self.touches[:, 2:], nlags=num_lags )
        elif stim_config == 1:
            self.stimA = self.time_embedding( stim=self.touches[:, :2], nlags=num_lags, verbose=False )
            self.stim = self.time_embedding( stim=self.touches[:, 2:], nlags=num_lags )
        else:
            self.stim_dims = [4, 1, 1, 1]
            self.stim = self.time_embedding( stim=self.touches, nlags=num_lags )
            self.stimA = None

        if include_multitouches:
            self.mtouches = self.time_embedding( stim=self.multitouches, nlags=num_lags )
        if temporal_basis is not None:
            # then temporal basis gives the doubling time
            self.stim_anchors = self.anchor_set(num_lags, temporal_basis)
            self.TB = torch.tensor(self.temporal_basis(num_lags, self.stim_anchors), dtype=torch.float32)
        else:
            self.TB = None

        # Process with temporal basis, if relevant
        if self.TB is not None:
            self.stim = torch.einsum(
                'bxt,tf->bxf', 
                self.stim.reshape([self.NT, self.stim_dims[0], -1]), 
                self.TB ).reshape([self.NT, -1])
            if self.stimA is not None:
                self.stimA = torch.einsum(
                    'bxt,tf->bxf', 
                    self.stimA.reshape([self.NT, self.stim_dims[0], -1]), 
                    self.TB).reshape([self.NT, -1])
            if include_multitouches:
                multiT_dims = self.multitouches.shape[1]
                self.mtouches = torch.einsum(
                    'bxt,tf->bxf', self.mtouches.reshape([self.NT, multiT_dims, -1]), self.TB).reshape([self.NT, -1])
                    #torch.tensor(self.multitouches.reshape([self.NT, 4, -1]), dtype=torch.float32), 
                    #self.TB).reshape([self.NT, -1])
            self.stim_dims[3] = self.TB.shape[1]
        else:
            self.stim_dims[3] = num_lags
    # END WhiskerData.prepare_stim()

    def set_hemispheres( self, out_config=0, in_config=0 ):
        """
        This sets cells_out and cells_in based on hemisphere, making things easy. 
        Can also set cells_out and cells_in by hand.
        
        Args:
            out_config, in_config: 0=left outputs, 1=right outputs, 2=both
        
        Returns:
            None
        """
        if out_config < 2:
            self.set_cells(self.Rparse[out_config])
        else:
            self.set_cells()

        if in_config < 2:
            self.cells_in = self.Rparse[in_config]
        else:
            self.cells_in = []
    # END WhiskerData.set_hemispheres

    def construct_XLV( self, tent_spacing=12, cueduncued=False ):
        """
        Constructs tent-basis-style trial-based tent function
        
        Args: 
            num_tents: default 11
            cueduncued: whether to fit separate kernels to cued/uncued
        
        Returns:
            None
        """
        # automatically wont have any anchors past min_trial_size
        anchors = np.arange(0, self.min_trial_size, tent_spacing) 
        # Generate master tent_basis
        trial_tents = self.design_matrix_drift(
            self.max_trial_size, anchors, zero_left=False, zero_right=True, const_right=False)
        num_tents = trial_tents.shape[1]

        if cueduncued:
            self.Xadapt = torch.zeros((self.NT, 2*num_tents), dtype=torch.float32)
        else:
            self.Xadapt = torch.zeros((self.NT, num_tents), dtype=torch.float32)

        for tr in range(self.Ntr):
            L = len(self.block_inds[tr])
            if cueduncued:
                tmp = torch.zeros([L, 2*num_tents], dtype=torch.float32) 
                if self.TRcued[tr] < 0:
                    tmp[:, range(num_tents, 2*num_tents)] = torch.tensor(trial_tents[:L, :], dtype=torch.float32)
                else:
                    tmp[:, range(num_tents)] = torch.tensor(trial_tents[:L, :], dtype=torch.float32)
                self.Xadapt[self.block_inds[tr], :] = deepcopy(tmp)
            else:
                self.Xadapt[self.block_inds[tr], :] = torch.tensor(trial_tents[:L, :], dtype=torch.float32)
    # END HNdataset.construct_Xadapt()

    def __getitem__(self, idx):
        if len(self.cells_out) == 0:
            out = {'stim': self.stim[idx, :],
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
                'robs': robs_tmp[idx, :],
                'dfs': dfs_tmp[idx, :]}
            
            if self.speckled:
                M1tmp = self.Mval[:, self.cells_out]
                M2tmp = self.Mtrn[:, self.cells_out]
                out['Mval'] = M1tmp[idx, :]
                out['Mtrn'] = M2tmp[idx, :]
            
        if self.stimA is not None:
            out['stimA'] = self.stimA[idx, :]

        if self.Xdrift is not None:
            out['Xdrift'] = self.Xdrift[idx, :]

        if self.include_multitouches:
            out['mtouches'] = self.mtouches[idx, :]

        if self.ACinput is not None:
            out['ACinput'] = self.ACinput[idx, :]

        if len(self.covariates) > 0:
            self.append_covariates( out, idx)

        return out
    # END WhiskerData.__getitem()

    # Make temporal basis possible
    def temporal_basis( self, nlags, anchors ):
        """
        Make temporal basis for use in temporal basis expansion.

        Args:
            nlags: number of lags
            anchors: list of anchor points for the temporal basis

        Returns:
            KB: temporal basis
        """
        if anchors[-1] < nlags:
            anchors.append(nlags)
        anchors = np.array(anchors, dtype=np.int64)-1
        num_tk=len(anchors)-1
        KB = np.zeros([nlags, num_tk])
        KB[anchors[0]+1,0]=1
        for ii in range(1,num_tk):
            if anchors[ii]-anchors[ii-1] > 1:  # then need to slope
                dx = anchors[ii]-anchors[ii-1]
                KB[np.arange(anchors[ii-1]+1, anchors[ii]+1)+1, ii] = (np.arange(dx)+1)/dx
            else:
                KB[anchors[ii]+1,ii] = 1 
            if anchors[ii+1]-anchors[ii] > 1:  # then need to slope
                dx = anchors[ii+1]-anchors[ii]
                KB[np.arange(anchors[ii], anchors[ii+1])+1, ii] = 1-(np.arange(dx))/dx
        print(num_tk, 'basis elements')
        return KB

    def anchor_set( self, nlags, doubling_time, offset=0 ):
        """
        Make anchor set for temporal basis expansion.

        Args:
            nlags: number of lags
            doubling_time: doubling time for temporal basis
            offset: offset for anchor set

        Returns:
            a: anchor set
        """
        sp = 1
        a = []
        x = offset
        while x < nlags:
            for ii in range(doubling_time):
                x += sp
                if x < nlags:
                    a.append(x)
            sp *= 2
        if a[-1] >= nlags-2:
            a[-1] = nlags
        else:
            a.append(nlags)
        return a
    
    # Autoencoders
    def autoencoder_design_matrix( self, pre_win=0, post_win=0, blank=0, cells=None ):
        """
        Makes auto-encoder input using windows described above, and including the
        chosen cells. Will put as additional covariate "ACinput" in __get_item__
        
        Args:
            pre_win: how many time steps to include before origin
            post_win: how many time steps to include after origin
            blank: how many time steps to blank in each direction, including origin
            cells: which cells to include in the auto-encoder design matrix

        Returns:
            None
        """

        if cells is None:
            cells = np.arange(self.NC)
        Rraw = deepcopy(self.robs[:, cells])
        self.ACinput = torch.zeros(Rraw.shape, dtype=torch.float32)
        nsteps = 0
        if blank == 0:
            self.ACinput += Rraw
            nsteps = 1
        for ii in range(blank, (pre_win+1)):
            self.ACinput[ii:, :] += Rraw[:(-ii), :]
            nsteps += 1
        for ii in range(blank, (post_win+1)):
            self.ACinput[:(-ii), :] += Rraw[ii:, :]
            nsteps += 1
        assert nsteps > 0, "autoencoder design: invalid parameters"
        self.ACinput *= 1.0/nsteps
        self.ACinput *= self.dfs[:, cells]
    # END autoencoder_design_matrix


    def WTAs( self, r0=5, r1=30):
        """
        Args: 
            Ton: list of touch onsets for all 4 whiskers
            Rs: Robs 
            r0, r1: how many lags before and after touch onset to include (and block out)

        Returns:
            wtas: whisker-triggered averages of firing rate
            nontouchFRs: average firing rate (spike prob) away from all four whisker touches
        """
        L = r1+r0
        wtas = np.zeros([L,4, self.NC])
        wcounts = deepcopy(wtas)
        #valws = np.zeros([NT,4])
        nontouchFRs = np.zeros([5, self.NC])
        for ww in range(4):
            #valws[val_inds.astype(int),ww] = 1.0
            wts = np.where(self.touches[:, ww] > 0)[0]
            print( "w%d: %d touches"%(ww+1, len(wts)))
            for tt in wts:
                t0 = np.maximum(tt-r0,0)
                t1 = np.minimum(tt+r1, self.NT)
                if t1-t0 == L: # then valid event
                    footprint = np.expand_dims(valws[range(t0,t1), ww], 1) # All touches probably valid, but just in case
                    wcounts[:,ww,:] += footprint
                    wtas[:,ww,:] += Rs[range(t0,t1),:]*footprint
                #valws[range(t0,t1),ww] = 0
            wtas[:,ww,:] = wtas[:,ww,:] / wcounts[:,ww,:] 
            #nontouchFRs[ww,:] = np.sum(valws[:,[ww]]*Rs,axis=0) /np.sum(valws[:,ww])

        # Stats where there are no touches from any whisker
        #valtot = np.expand_dims(np.prod(valws, axis=1), 1)
        #nontouchFRs[4,:] = np.sum(valtot*Rs,axis=0)/np.sum(valtot)
        
        return wtas, nontouchFRs

    @staticmethod
    def create_NLmap_design_matrix( x, num_bins, val_inds=None, thresh=5, 
                                borderL=None, borderR=None, anchorL=True, rightskip=False):
        """
        Make design matrix of certain number of bins that maps variable of interest
        anchorL is so there is not an overall bias fit implicitly.
        
        Args:
            x: variable of interest
            num_bins: number of bins
            val_inds: indices to use for thresholding
            thresh: threshold for determining borders
            borderL, borderR: left and right borders
            anchorL: whether to anchor the left side at zero
            rightskip: whether to skip the rightmost bin

        Returns:
            XNL: design matrix
        """
        from NDNT.utils.NDNutils import design_matrix_tent_basis

        NT = x.shape[0]
        if val_inds is None:
            val_inds = range(NT)    
        #m = np.mean(x[val_inds])
        # Determine 5% and 95% intervals (related to thresh)
        h, be = np.histogram(x[val_inds], bins=100)
        h = h/np.sum(h)*100
        cumu = 0
        if borderL is None:
            borderL = np.nan
        if borderR is None:
            borderR = np.nan
        for nn in range(len(h)):
            cumu += h[nn]
            if np.isnan(borderL) and (cumu >=  thresh):
                borderL = be[nn]
            if np.isnan(borderR) and (cumu >= 100-thresh):
                borderR = be[nn]
        # equal divisions between 95-max
        if rightskip:
            bins = np.arange(num_bins)*(borderR-borderL)/num_bins + borderL
        else:
            bins = np.arange(num_bins+1)*(borderR-borderL)/num_bins + borderL
        print(bins)
        XNL = design_matrix_tent_basis( x, bins, zero_left=anchorL )
        return XNL

    @staticmethod
    def find_first_locmin(trace, buf=0, sm=0):
        """
        Find first local minimum in trace, starting from buf.

        Args:
            trace: trace to analyze
            buf: starting point
            sm: smoothing window

        Returns:
            loc: location of first local minimum
        """
        der = np.diff(trace)
        loc = np.where(np.diff(trace[buf:]) >= 0)[0][0]+buf
        return loc

    @staticmethod
    def prop_distrib(events, prop_name):
        """
        Extracts distribution of a property from events.

        Args:
            events: events
            prop_name: property name

        Returns:
            distrib: distribution of the property
        """
        assert prop_name in events[0], 'Invalid property name.'
        distrib = np.zeros(len(events))
        for tt in range(len(events)):
            distrib[tt] = events[tt][prop_name]
        return distrib

    @staticmethod
    def trial_parse( frames ):
        """
        Parse trials from frames.
        
        Args:
            frames: frames

        Returns:
            blks: trial blocks
        """
        trial_starts = np.where(frames == 1)[0]
        num_trials = len(trial_starts)
        blks = np.zeros([num_trials, 2], dtype='int64')
        for nn in range(num_trials-1):
            blks[nn, :] = [trial_starts[nn], trial_starts[nn+1]]
        blks[-1, :] = [trial_starts[-1], len(frames)]
        return blks

    @staticmethod
    def trial_classify( blks, pistons, outcomes=None ):
        """
        Args:
            blks: trial blocks
            pistons: pistons
            outcomes: 1=hit, 2=miss, 3=false alarm, 4=correct reject

        Returns:
            TRpistons: trial pistons
            TRoutcomes: trial outcomes
        """
        #assert pistons is not None, "pistons cannot be empty"
        Ntr = blks.shape[0]
        #= np.mean(blk_inds, axis=1).astype('int64')
        TRpistons = np.zeros(Ntr, dtype='int64')
        TRoutcomes = np.zeros(Ntr, dtype='int64')
        for nn in range(Ntr):
            ps = np.max(pistons[range(blks[nn,0], blks[nn,1]),:], axis=0)
            TRpistons[nn] = (ps[0] + 2*ps[1] + 4*ps[2] + 8*ps[3]).astype('int64')
            if outcomes is not None:
                os = np.where(np.max(outcomes[range(blks[nn,0], blks[nn,1]),:], axis=0) > 0)[0]
                if len(os) != 1:
                    print("Warning: trial %d had unclear outcome."%nn)
                else:
                    TRoutcomes[nn] = os[0]+1
        # Reclassify unilateral (or no) touch trials as Rew = 5
        unilateral = np.where((TRpistons <=2) | (TRpistons == 4) | (TRpistons==8))[0]
        TRoutcomes[unilateral] = 5
        
        return(TRpistons, TRoutcomes)

    @staticmethod
    def process_locations( clocs ):
        """
        Process locations.

        electrode_info: first column is shank membership, second column is electrode depth

        Args:
            clocs: cell locations

        Returns:
            num_cells: number of cells
        """
        hemis = np.where(clocs[:,0] == 1)[0]
        NC = clocs.shape[0]
        if len(hemis) > 1:
            num_cells = [hemis[1], NC-hemis[1]]
        else:
            num_cells = [NC, 0]
        electrode_info = [None]*2
        for hh in range(len(hemis)):
            if hh == 0:
                crange = range(num_cells[0])
            else:
                crange = range(num_cells[0], NC)
            ei = clocs[crange, :]
            electrode_info[hh] = ei[:, 1:]
        return num_cells, electrode_info
        