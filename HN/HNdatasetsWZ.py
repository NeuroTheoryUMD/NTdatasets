import os
import numpy as np
import scipy.io as sio
import warnings
import math
import torch
from torch.utils.data import Dataset
import NDNT.utils as utils
import NDNT.NDN as NDN
import matplotlib.pyplot as plt

# from NDNT.utils import download_file, ensure_dir
from copy import deepcopy
import h5py
from NTdatasets.sensory_base import SensoryBase

"""
Patch notes: 
 - data filters (dfs) now loaded instead of created
 - used_inds no longer supported
 - removed automatic stim aseembly (assemble_stim in init), since specifying number of lags should be a conscious choice
 - Xsacc input now consists of 5 rows, corresponding to: onsets, offsets, direction (in radians), categorical direction (lrup) and saccade amplitude
 New methods
 - added static method get_sacc_dir, which categorizes sacc direction based on angle
 - added mark_good_cells method, which creastes an array of channel numbers which pass the 3-part var criterium (although it's already deprecated, since we filter the data prior in matlab)
 - added get_psycurve method, helpful for ploting psychophysical curves
 - added trial_to_timepoint method which broadcasts a trial-level array to a timepoint-level one.
   TODO:
   - add sta method?
    - skip_lags -> shift_lags
   - assemble_stimulus should be depracated(?) since assemble_stim is in use
   - correct misleading naming convention for stimL and stimR
   - implement str, repr methods
"""


class HNdataset(SensoryBase):
    """
    Class for handling HN data.
    """

    def __init__(
        self, filename=None, #which_stim="left", skip_lags=2, # no longer intializes stim here
        pad_length=None, # if filled in, will automatically pad (but can be done later)
        mask_lags=4,
        num_lags=10,
        **kwargs
    ):
        """
        Args:
            filename: currently the pre-processed matlab file from Dan's old-style format
            which_stim: which stim is relevant for the neurons in this dataset (default 'left')
            skip_lags: shift stim to throw out early lags
            **kwargs: non-dataset specific arguments that get passed into SensoryBase
        """
        # call parent constructor
        super().__init__(filename, num_lags=num_lags, **kwargs)

        print(self.datadir + filename)
        matdat = sio.loadmat(self.datadir + filename)
        print("Loaded " + filename)

        self.mask_lags = mask_lags
        self.disp_list = matdat["disp_list"].squeeze()
        self.stimlist = np.unique(matdat["stimL"])
        self.Nstim = len(self.stimlist)
        self.hemi_left = matdat["hemisphere"]  # probe in left (1) or right (-1) hemisphere
        self.TRcued = matdat["cued"].squeeze()  # Ntr
        self.TRchoice = matdat["choice"].squeeze()  # Ntr
        self.TRsignal = matdat["signal"]  # Ntr x 2 (sorted by RF)
        self.TRstrength = matdat["strength"]  # Ntr x 2 (sorted by RF)
        self.TRstim = matdat["cued_stim"]  # Ntr x 4 (sorted by cued, then uncued)
        # Detect disparities used for decision (indexed by stimulus number)
        # decision_stims = np.where(matdat['disp_list'] == np.unique(matdat['cued_stim'][:,0]))[0]

        self.TRstim = np.multiply(
            self.TRsignal, self.TRstrength
        )  # stim strength and direction combined

        # XX new: adding list of all possible frame-lvl disp. vals
        # #!! different from disp_list, which contains trial-average disp vals
        self.slist = np.unique(self.TRstim)
        ### Process neural data
        self.robs = torch.tensor(matdat["Robs"], dtype=torch.float32)
        self.NT, self.NC = self.robs.shape
        # Make/load datafilters
        if "dfs" in matdat.keys():
            self.dfs = torch.tensor(
                matdat["dfs"], dtype=torch.float32
            )  # XX is it necessary to float here?
        else:
            warnings.warn(
                "Warning: Data filters matrix 'dfs' not provided. A ones matrix will be created in it's place"
            )
            self.dfs = torch.zeros([self.NT, self.NC], dtype=torch.float32)
            self.used_inds = matdat["used_inds"].squeeze().astype(np.int64) - 1
            self.dfs[self.used_inds, :] = 1.0
        self.modvars = matdat["moduvar"]

        # High resolution stimuli: note these are already one-hot matrices
        self.stimL = matdat["stimL"]
        self.stimR = matdat["stimR"]

        # Saccade info
        self.Xsacc = torch.tensor(matdat["Xsacc"], dtype=torch.float32)
        self.Xadapt = None
        self.ACinput = None  # autoencoder input
        # saccdirs = matdat['sacc_dirs']

        # Make block_inds
        blks = matdat["blks"]
        self.Ntr = blks.shape[0]
        self.Nframes = np.min(np.diff(blks))
        for bb in range(self.Ntr):
            self.block_inds.append( np.arange(blks[bb, 0]-1, blks[bb, 1], dtype=np.int64) )
            self.dfs[np.arange(blks[bb,0]-1, blks[bb,0]+self.mask_lags-1), :] = 0.0
            # Take out the first num_lags part of each data-filter
            #!! Here
        # self.dfs[np.arange(blks[bb,0]-1, blks[bb,0]+np.maximum(10,self.num_lags+1)), :] = 0.0

        # for some reason there is stuff at the end that is not associated with trial -- cut!
        self.NT = self.block_inds[-1][-1] + 1
        self.robs = self.robs[: self.NT, :]
        self.dfs = self.dfs[: self.NT, :]
        self.stimL = self.stimL[: self.NT, :]
        self.stimR = self.stimR[: self.NT, :]
        self.Xsacc = self.Xsacc[: self.NT, :]

        self.CHnames = [None] * self.NC
        for cc in range(self.NC):
            self.CHnames[cc] = matdat["CHnames"][0][cc][0]
        # expt_info = {'exptname':filename, 'CHnames': CHname, 'blks':blks, 'dec_stims': decision_stims,
        #            'DispList': dislist, 'StimList': stimlist, #'Xsacc': Xsacc, 'sacc_dirs': saccdirs,
        #            'stimL': stimL, 'stimR':stimR, 'Robs':Robs, 'used_inds': used_inds}

        twin = np.arange(25, self.Nframes, dtype=np.int64)
        self.Rtr = np.zeros([self.Ntr, self.NC], dtype="float32")
        for ii in range(self.Ntr):
            self.Rtr[ii, :] = torch.sum(self.robs[twin + blks[ii, 0], :], axis=0)

        print(
            "%d frames, %d units, %d trials with %d frames each"
            % (self.NT, self.NC, self.Ntr, self.Nframes)
        )

        chansVX = np.hstack(
            [
                matdat["ChnamesV1"],
                matdat["ChnamesV2"],
                matdat["ChnamesV3"],
                matdat["ChnamesV4"],
                matdat["ChnamesXX"],
            ]
        )
        chansVX_id = np.hstack(
            [
                np.repeat("v1", matdat["ChnamesV1"].shape[1]),
                np.repeat("v2", matdat["ChnamesV2"].shape[1]),
                np.repeat("v3", matdat["ChnamesV3"].shape[1]),
                np.repeat("v4", matdat["ChnamesV4"].shape[1]),
                np.repeat("xx", matdat["ChnamesXX"].shape[1]),
            ]
        )
        chansVX = (
            chansVX + np.random.rand(len(chansVX)) * 0.001
        )  # so that the numbers don't repeat
        indices = np.argsort(chansVX)
        self.chan_area = chansVX_id[indices]
        
        # Figure out minimum and max trial size, and then make function that fits anywhere
        self.min_trial_size = len(self.block_inds[0])
        self.max_trial_size = len(self.block_inds[0])
        for ii in range(self.Ntr):
            if len(self.block_inds[ii]) < self.min_trial_size:
                self.min_trial_size = len(self.block_inds[ii])
            elif len(self.block_inds[ii]) > self.max_trial_size:
                self.max_trial_size = len(self.block_inds[ii])

        # Additional processing check
        # Cued and uncued stim
        # Cstim = np.multiply(TRstim[:,1], np.sign(TRstim[:,0])) # Cued stim
        # Ustim = np.multiply(TRstim[:,3], np.sign(TRstim[:,2]))  # Uncued stim
        # f_far = np.zeros([Nstim,2])
        # for nn in range(Nstim):
        #    tr1 = np.where(Cstim == stimlist[nn])[0]
        #    tr2 = np.where(Ustim == stimlist[nn])[0]
        #    f_far[nn,0] = np.sum(TRchoice[tr1] > 0)/len(tr1)
        #    f_far[nn,1] = np.sum(TRchoice[tr2] > 0)/len(tr2)

        # Prepare stimulus using input argument 'which_stim'
        # self.assemble_stim( which_stim=which_stim, skip_lags=skip_lags )

        # Make drift-design matrix using anchor points at each cycle
        if self.drift_interval is None:
            cued_transitions = np.where(abs(np.diff(self.TRcued)) > 0)[0]
            anchors = [0] + list(cued_transitions[range(1, len(cued_transitions), 2)])
            self.construct_drift_design_matrix(block_anchors=anchors, zero_right=True)
        else:
            raise ValueError(
                "drift_interval is not supported for this dataset. The drift matrix is constructed based on cue transitions. Please omit the drift_intervsal argument and try creating a data instance again."
            )
            # self.construct_drift_design_matrix()

        # This needs to be an internal variable
        if pad_length is None:
            self.trial_padding = 0
        else:
            self.pad_trials( pad_length=pad_length )
    # END HNdata.__init__()

    def assing_train_test_inds(self, which_fold=2, fold_num=5, use_random=False):
        # Generate cross-validation
        use_random = use_random
        # Cued and uncued trials
        trC = np.where(self.TRcued > 0)[0]
        trU = np.where(self.TRcued < 0)[0]
        # zero-strength trials
        tr0 = np.where(self.TRstrength[:, 0] == 0)[0]
        # sort by cued/uncued
        tr0C = np.where((self.TRstrength[:, 0] == 0) & (self.TRcued > 0))[0]
        tr0U = np.where((self.TRstrength[:, 0] == 0) & (self.TRcued < 0))[0]
        # for purposes of cross-validation, do the same for non-zero-strength trials
        trXC = np.where((self.TRstrength[:, 0] != 0) & (self.TRcued > 0))[0]
        trXU = np.where((self.TRstrength[:, 0] != 0) & (self.TRcued < 0))[0]

        # Assign train and test indices sampled evenly from each subgroup (note using default 4-fold)
        Ut0C, Xt0C = self.train_test_assign(
            tr0C, use_random=use_random, which_fold=which_fold, fold_num=fold_num, 
        )
        Ut0U, Xt0U = self.train_test_assign(
            tr0U, use_random=use_random, which_fold=which_fold, fold_num=fold_num, 
        )
        UtXC, XtXC = self.train_test_assign(
            trXC, use_random=use_random, which_fold=which_fold, fold_num=fold_num, 
        )
        UtXU, XtXU = self.train_test_assign(
            trXU, use_random=use_random, which_fold=which_fold, fold_num=fold_num, 
        )

        # Putting together for larger groups
        Ut0 = np.sort(np.concatenate((Ut0C, Ut0U), axis=0))
        Xt0 = np.sort(np.concatenate((Xt0C, Xt0U), axis=0))
        UtC = np.sort(np.concatenate((Ut0C, UtXC), axis=0))
        XtC = np.sort(np.concatenate((Xt0C, XtXC), axis=0))
        UtU = np.sort(np.concatenate((Ut0U, UtXU), axis=0))
        XtU = np.sort(np.concatenate((Xt0U, XtXU), axis=0))

        Ut = np.sort(np.concatenate((Ut0, UtXC, UtXU), axis=0))
        Xt = np.sort(np.concatenate((Xt0, XtXC, XtXU), axis=0))

        self.trs = {"c": trC, "u": trU, "0": tr0, "0c": tr0C, "0u": tr0U}
        self.Utr = {"all": Ut, "0": Ut0, "c": UtC, "u": UtU, "0c": Ut0C, "0u": Ut0U}
        self.Xtr = {"all": Xt, "0": Xt0, "c": XtC, "u": XtU, "0c": Xt0C, "0u": Xt0U}

        train_inds, val_inds = [], []
        for tr in Ut:
            train_inds = np.concatenate(
                (train_inds, self.block_inds[tr]), axis=0
            ).astype(np.int64)
        for tr in Xt:
            val_inds = np.concatenate(
                (val_inds, self.block_inds[tr]), axis=0
            ).astype(np.int64)
        return train_inds, val_inds
    # END HNdataset.assing_train_test_inds()

    # new! (added by WZ)
    def mask_dfs(self, mask_length):
        '''Masks initial number of tp of each trial by setting
        dfs to 0'''
        for t in range(len(self.block_inds)):
            self.dfs[self.block_inds[t][0:mask_length],:]=0
    def pad_trials(self, pad_length=10):
        '''
        Pads the end of each trial with additional empty timepoints. (could be the end if we want)
        Modfies the following:
            -- block_inds, and sets trial_padding
            -- robs and dfs
            -- stimL and stimR (and nulls stimulus to make sure it is reassembled)
            -- Xdrift and Xsacc
            -- modvars: whatever that is
            -- will null any covariates, which have to be remade

        Args:
            pad_length: how much to pad after each trial (default 10)

        Returns:
            None, although modifies all the design matrices and data listed above
        '''

        self.trial_padding = pad_length
        tot_rows = self.robs.shape[0] + pad_length*len(self.block_inds)

        indsx = deepcopy(self.block_inds)
        #robsx=np.full((tot_rows, self.robs.shape[1]), np.nan)
        #dfx=np.full((tot_rows, self.robs.shape[1]), np.nan)
        robsx = torch.zeros((tot_rows, self.robs.shape[1]), dtype=torch.float32) # changed from nans to zeros
        dfx = torch.zeros((tot_rows, self.robs.shape[1]), dtype=torch.float32) # changed from nans to zeros
        # Unassembled stimulus stored as numpy (not torch)
        stimLx = np.zeros((tot_rows, self.stimL.shape[1]), dtype=np.float32) 
        stimRx = np.zeros((tot_rows, self.stimR.shape[1]), dtype=np.float32) 

        driftx = torch.zeros((tot_rows, self.Xdrift.shape[1]), dtype=torch.float32) # changed from nans to zeros
        saccx = torch.zeros((tot_rows, self.Xsacc.shape[1]), dtype=torch.float32) # changed from nans to zeros

        modvarx = np.zeros((tot_rows, self.modvars.shape[1]), dtype=np.float32) 

        if len(self.stim) > 0:
            print("PADDING: Erasing existing stimulus: must reaseemble")
            self.stim = []
        if len(self.covariates) > 0:
            print("PADDING: Erasing existing covariates: must rebuild")
            self.covariates = {}

        total_pad = 0 # total padding accumulator
        for t in range(len(self.block_inds)):   
            #add padding
            
            #indsx[t] = indsx[t]+total_pad # commented out: new trials must include gap
            new_inds = indsx[t]+total_pad
            #pad_array=np.arange(indsx[t][-1]+1, indsx[t][-1]+pad_length+1)
            #indsx[t] = np.concatenate((indsx[t], pad_array))
            
            robsx[new_inds, :] = self.robs[self.block_inds[t], :]
            dfx[new_inds, :] = self.dfs[self.block_inds[t], :]
            stimLx[new_inds, :] = self.stimL[self.block_inds[t], :]
            stimRx[new_inds, :] = self.stimR[self.block_inds[t], :]
            saccx[new_inds, :] = self.Xsacc[self.block_inds[t], :]
            driftx[new_inds, :] = self.Xdrift[self.block_inds[t], :]
            modvarx[new_inds, :] = self.modvars[self.block_inds[t], :]
            #for cell in range(self.robs.shape[1]):
            #    robi = self.robs[self.block_inds[t],cell]
            #    robsx[indsx[t], cell] = torch.cat(
            #        (robi,
            #        torch.zeros(len(range(pad_length)))
            #        )
            #    )
            #    dfi = self.dfs[self.block_inds[t],cell]
            #    dfx[indsx[t],cell] = torch.cat(
            #        (dfi,
            #        torch.zeros(len(range(pad_length)))
            #        )
            #    )

            # Now that data is written in, need to make each trial include the gap so
            # that datasets can be assembled trial-by-trial (preserving the gaps)
            indsx[t] = np.arange(new_inds[0], new_inds[-1]+1+pad_length)
            total_pad += pad_length

        self.block_inds = indsx
        #self.robs=torch.from_numpy(robsx)
        #self.dfs=torch.from_numpy(dfx)
        self.robs = robsx
        self.dfs = dfx
        self.stimL = stimLx
        self.stimR = stimRx
        self.modvars = modvarx  # not sure what this is but needs to be expanded
        self.Xdrift = driftx
        self.Xsacc = saccx
        self.NT = tot_rows
    # END HNdatasets.pad_trials()

    def split_CU(self, design_matrix):
        """
        Splits design matrix by cued / uncued conditions and then concatenates columns
        """
        Xcued = deepcopy(design_matrix)
        Xuncued = deepcopy(design_matrix)

        for t in range(self.Ntr):
            if t in self.trs["c"]:
                Xuncued[self.block_inds[t], :] = 0.0
            elif t in self.trs["u"]:
                Xcued[self.block_inds[t], :] = 0.0
        XCU = np.concatenate((Xcued, Xuncued), axis=1)
        return XCU

    def mark_good_cells(self, thresh_reject_var: float=None, thresh_reject_mean:float=None) -> None:
        """
        Creates a 'keepers' atribute, which is a list of 'good' cells,
            based on mean & variance threshold criteria.
        thresh_reject_mean: specifies how much mean activity can vary across
            the length of the session (which is divided into 3 equal parts).
        thresh_reject_var: specifies how much variance can vary across the length 
          of the session
        """
        if thresh_reject_var is not None:
            keepers_var = self.var_test(thresh_reject_var)
        else:
            keepers_var = range(self.NC)    
        if thresh_reject_mean is not None:
            keepers_mean = self.mean_test(thresh_reject_mean)    
        else:
            keepers_mean = range(self.NC)
        self.keepers_var = keepers_var
        self.keepers_mean = keepers_mean
        self.keepers = np.intersect1d(keepers_var, keepers_mean)
   
    def mean_test(self, thresh_reject: float=2.0) -> np.ndarray:
        """
         Mark good cells based on mean criterium,
         using dfs to account for partial missing data.
         Provides a simple interpretation: the variance in any third
           part of the session cannot be more than twice the variance in the other two thirds.
        """
        trial_bool = np.zeros([self.Ntr, self.NC], dtype=bool)
        meantest = np.zeros([self.NC, 3])
        num, deno = np.zeros(self.NC), np.zeros(self.NC)
        for cc in range(self.NC):
            dfs = self.dfs[:, cc]
            tp_used = np.where(dfs==1)[0]
            for (i, tr) in enumerate(self.block_inds):
                trial_bool[i,cc]=(np.isin(tr, tp_used)).all()
            rtr_cc = self.Rtr[trial_bool[:,cc], cc]
            meantest[cc, 0] = np.mean(rtr_cc[range(len(rtr_cc) // 3)])
            meantest[cc, 1] = np.mean(rtr_cc[range(len(rtr_cc) // 3, 2 *len(rtr_cc) // 3)])
            meantest[cc, 2] = np.mean(rtr_cc[range(2 *len(rtr_cc) // 3,len( rtr_cc))])
            num[cc] = np.max([meantest[cc, 0], meantest[cc, 1] , meantest[cc, 2]])
            deno[cc] = np.min([meantest[cc, 0],meantest[cc, 1], meantest[cc, 2]])

        criteria_reject = np.divide(num,deno)
        if thresh_reject > 0:
            out= np.where(criteria_reject < thresh_reject)[0]
        else: 
            out= criteria_reject
        return out

    def var_test(self, thresh_reject:float=1.5) -> np.ndarray:
        """
         Mark good cells based on variance criterium,
         using dfs to account for partial missing data.
         Provides a simple interpretation: the variance in any third
           part of the session cannot be more than twice the variance in the other two thirds.
        """
        trial_bool = np.zeros([self.Ntr, self.NC], dtype=bool)
        vartest = np.zeros([self.NC, 3])
        num, deno = np.zeros(self.NC), np.zeros(self.NC)
        for cc in range(self.NC):
            dfs = self.dfs[:, cc]
            tp_used = np.where(dfs==1)[0]
            for (i, tr) in enumerate(self.block_inds):
                trial_bool[i,cc]=(np.isin(tr, tp_used)).all()
            rtr_cc = self.Rtr[trial_bool[:,cc], cc]
            vartest[cc, 0] = np.std(rtr_cc[range(len(rtr_cc) // 3)])
            vartest[cc, 1] = np.std(rtr_cc[range(len(rtr_cc) // 3, 2 *len(rtr_cc) // 3)])
            vartest[cc, 2] = np.std(rtr_cc[range(2 *len(rtr_cc) // 3,len( rtr_cc))])
            num[cc] = np.max([vartest[cc, 0], vartest[cc, 1] , vartest[cc, 2]])
            deno[cc] = np.min([vartest[cc, 0],vartest[cc, 1], vartest[cc, 2]])

        criteria_reject = np.divide(num,deno)
        if thresh_reject > 0:
            out= np.where(criteria_reject < thresh_reject)[0]
        else: 
            out= criteria_reject
        return out
    
    def get_psycurve(self) -> None:
        """
        Returns a 2D array specifying contrast (disparity) level (rows)
            for attended and unattended (columns) stimuli. Used for plotting
            psychophysiological curves.
        """
        Ntr = len(self.block_inds)
        TRstims, TRstimsOPP = np.zeros(Ntr), np.zeros(Ntr)
        cued_inds = 1 - (self.TRcued + 1) // 2  # transforms to 0-1 for indexing
        for ii in range(Ntr):
            TRstims[ii] = self.TRstim[ii, cued_inds[ii]]
            TRstimsOPP[ii] = self.TRstim[ii, 1 - cued_inds[ii]]

        #
        psycurves = np.zeros([len(self.slist), 2])
        for ss in range(len(self.slist)):
            trs = np.where(TRstims == self.slist[ss])[0]
            psycurves[ss, 0] = np.mean(self.TRchoice[trs])
            trs = np.where(TRstimsOPP == self.slist[ss])[0]
            psycurves[ss, 1] = np.mean(self.TRchoice[trs])
        self.psycurves = psycurves

    def trial_to_timepoint(self, TRvar: np.ndarray, col_num: int) -> np.ndarray:
        """
        Takes a trial-level dataset variable (e.g. average disparity per trial)
            and broadcasts it across time-points, creating a tp-level variable
        col_num: specifies which column of the trial-lvl variable to transform
        """
        out = np.zeros(self.robs.shape[0])
        for tr, blk in enumerate(self.block_inds):
            out[blk[0] : blk[-1]] = TRvar[tr, col_num]
        return out

    @staticmethod
    def get_sacc_dir(rad_angle: torch.Tensor, type: str) -> torch.Tensor:
        """
        Discretizes an array of angles "rad_angle" (in radians) in 3 ways:
        1. type='lr' categorizes directions into left and right
            and outputs a 2 column binary array, for
            whether the direction was left (col 1) or right (col 2)
        2. type = 'ud': same as 1, but for up/down saccades
        3. type = 'lrud': categorizes directions into 4 types: left (col 1),
        right (col 2), up (col 3) and down (col 4).
        """
        deg_angle = 180 / math.pi * rad_angle
        if type.lower() == "lr":
            right = ((deg_angle > 0) & (deg_angle < 90) | (deg_angle > 270)).long()
            left = (deg_angle > 90) & (deg_angle < 270).long()
            out = torch.column_stack((left, right))
        elif type.lower() == "ud":
            up = ((deg_angle > 0) & (deg_angle < 180)).long()
            down = ((deg_angle > 180) & (deg_angle < 360)).long()
            out = torch.column_stack((up, down))
        elif type.lower() == "lrud" or type.lower() == "udlr":
            right = ((deg_angle > 315) | ((deg_angle > 0) & (deg_angle < 45))).long()
            left = ((deg_angle > 135) & (deg_angle < 225)).long()
            up = ((deg_angle > 45) & (deg_angle < 135)).long()
            down = ((deg_angle > 225) & (deg_angle < 315)).long()
            out = torch.column_stack((left, right, up, down))
        return out

    def set_num_lags(self, stim_interval, verbose=False):
        samples_per_trial = self.NT / len(self.block_inds)
        sampling_rate = 2000 / samples_per_trial
        if verbose:
            print("Number of lags: ", np.ceil(stim_interval/ sampling_rate))
        return np.ceil(stim_interval / sampling_rate)

    def assemble_stim(self, which_stim="left", skip_lags=None, num_lags=None):
        """
        Prepares stimulus for dataset.

        Args:
            which_stim: 'left' or 'right' (default 'left')
            skip_lags: how many lags to skip (default 2)
            num_lags: how many lags to include in stimulus (default None, will use dataset value)

        Returns:
            None
        """
        if skip_lags is not None:
            self.skip_lags = skip_lags
        # otherwise will use already set value

        if which_stim in ["left", "L", "Left"]:
            stim = torch.tensor(self.stimL, dtype=torch.float32)
        else:
            stim = torch.tensor(self.stimR, dtype=torch.float32)

        # XX no longer needed?
        # Zero out invalid time points (disp=0) before time embedding
        # df_generic = torch.zeros( stim.shape, dtype=torch.float32 )
        # df_generic[self.used_inds, :] = 1.0
        # stim = stim * df_generic

        self.stim_dims = [
            1,
            stim.shape[1],
            1,
            1,
        ]  # Put one-hot on first spatial dimension

        # Shift stimulus by skip_lags (note this was prev multiplied by DF so will be valid)
        if self.skip_lags > 0:
            stim[self.skip_lags :, :] = deepcopy(
                stim[: -self.skip_lags, :]
            )  # shifts stims by skip_lags
            stim[: self.skip_lags, :] = 0.0
        elif self.skip_lags < 0:
            print(
                "Currently cannot use negative skip_lags, and doesnt make sense anyway"
            )
            self.skip_lags = 0

        if num_lags is None:
            # then read from dataset (already set):
            num_lags = self.num_lags

        self.stim = self.time_embedding(stim=stim, nlags=num_lags)
        self.stim_dims[3] = num_lags
        # This will return a torch-tensor

    # END HNdata.assemble_stim()
    def assemble_sac(self, stim: [np.ndarray], skip_lags=0, num_lags=None):
        """
        Prepares stimulus for dataset.

        Args:
            which_stim: 'left' or 'right' (default 'left')
            skip_lags: how many lags to skip (default 2)
            num_lags: how many lags to include in stimulus (default None, will use dataset value)

        Returns:
            None
        """
        if skip_lags is not None:
            self.skip_lags = skip_lags

        stim = torch.tensor(stim, dtype=torch.float32)
        self.sac_dims = [
            1,
            stim.shape[0],
            1,
            1,
        ]  # Put one-hot on first spatial dimension

        # Shift stimulus by skip_lags (note this was prev multiplied by DF so will be valid)
        if self.skip_lags > 0:
            stim[self.skip_lags :, :] = deepcopy(
                stim[: -self.skip_lags, :]
            )  # shifts stims by skip_lags
            stim[: self.skip_lags, :] = 0.0
        elif self.skip_lags < 0:
            print(
                "Currently cannot use negative skip_lags, and doesnt make sense anyway"
            )
            self.skip_lags = 0

        if num_lags is None:
            # then read from dataset (already set):
            num_lags = self.num_lags
        else:
            self.num_lags = num_lags

        self.sac = self.time_embedding(stim=stim, nlags=num_lags)
        self.sac_dims[3] = num_lags
        # This will return a torch-tensor

    # END HNdata.assemble_stim()

    ###### DAN RECENT ADD #########
    def design_matrix_attentuation( self, num_segs=8):
        """
        blksI is the data.block_inds information.
        num_segs says how many tent_bases should be used across trial
        onset_period (0-1): proportion of the trial modelled as stim onset
        onset_seg_len (int): how long should be the segments during onset period
        """

        # Compute max trial length
        blksI = self.block_inds
        num_trials = len(blksI)
        NTtrial = np.zeros(num_trials)
        for ii in range(num_trials):
            NTtrial[ii] = len(blksI[ii])

        # make one-trial design-matrix
        L = np.max(NTtrial).astype(int)
        #Tseg = np.mean(L * (1 - onset_period) / num_segs).astype(int)

        ## SQRT-spacing
        anchors = np.round(self.mask_lags + np.linspace(0,np.sqrt(L-self.mask_lags),num_segs+1)**2).astype(int)[:-1]
        print('Anchors:', anchors)
        #anchors = np.concatenate(
        #    (
        #        np.arange(0, L * onset_period, onset_seg_len),
        #        np.arange(L * onset_period, L + 1, Tseg),
        #    )
        #).astype(int)

        Xtrial_time = self.design_matrix_drift(L, anchors=anchors, 
                                                const_left=True, zero_left=False, zero_right=True)
        # place them on every trial
        X = np.zeros([blksI[-1][-1] + 1, Xtrial_time.shape[1]])
        for ii in range(num_trials):
            X[blksI[ii], :] = deepcopy(Xtrial_time[range(len(blksI[ii])), :])
        return X
    ####### DAN RECENT ADD END #######

    def construct_Xadapt(self, tent_spacing=12, cueduncued=False):
        """
        Constructs adaptation-within-trial tent function

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
            self.max_trial_size,
            anchors,
            zero_left=False,
            zero_right=True,
            const_right=False,
        )
        num_tents = trial_tents.shape[1]

        if cueduncued:
            self.Xadapt = torch.zeros((self.NT, 2 * num_tents), dtype=torch.float32)
        else:
            self.Xadapt = torch.zeros((self.NT, num_tents), dtype=torch.float32)

        for tr in range(self.Ntr):
            L = len(self.block_inds[tr])
            if cueduncued:
                tmp = torch.zeros([L, 2 * num_tents], dtype=torch.float32)
                if self.TRcued[tr] < 0:
                    tmp[:, range(num_tents, 2 * num_tents)] = torch.tensor(
                        trial_tents[:L, :], dtype=torch.float32
                    )
                else:
                    tmp[:, range(num_tents)] = torch.tensor(
                        trial_tents[:L, :], dtype=torch.float32
                    )
                self.Xadapt[self.block_inds[tr], :] = deepcopy(tmp)
            else:
                self.Xadapt[self.block_inds[tr], :] = torch.tensor(
                    trial_tents[:L, :], dtype=torch.float32
                )

    # END HNdataset.construct_Xadapt()

    def autoencoder_design_matrix(self, pre_win=0, post_win=0, blank=0, cells=None):
        """
        Makes auto-encoder input using windows described above, and including the
        chosen cells. Will put as additional covariate "ACinput" in __get_item__

        Args:
            pre_win: how many time steps to include before origin
            post_win: how many time steps to include after origin
            blank: how many time steps to blank in each direction, including origin

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
        for ii in range(1, (pre_win + 1)):
            self.ACinput[ii:, :] += Rraw[:(-ii), :]
            nsteps += 1
        for ii in range(1, (post_win + 1)):
            self.ACinput[:(-ii), :] += Rraw[ii:, :]
            nsteps += 1
        assert nsteps > 0, "autoencoder design: invalid parameters"
        self.ACinput *= 1.0 / nsteps
        self.ACinput *= self.dfs[:, cells]

    # END HNdataset.autoencoder_design_matrix

    # Moved to SensoryBase
    # def trial_psths( self, trials=None, R=None ):
    #    """Computes average firing rate of cells_out at bin-resolution"""

    #    if R is None:  #then use [internal] Robs
    #        if len(self.cells_out) > 0:
    #            ccs = self.cells_out
    #        else:
    #            ccs = np.arange(self.NC)
    #        R = deepcopy( self.robs[:, ccs].detach().numpy() )
    #    if len(R.shape) == 1:
    #        R = R[:, None]
    #    num_psths = R.shape[1]  # otherwise use existing input

    #    T = self.min_trial_size
    #    psths = np.zeros([T, num_psths])

    #    if trials is None:
    #        trials = np.arange(self.Ntr)

    #    if len(trials) > 0:
    #        for ii in trials:
    #            psths += R[self.block_inds[ii][:T]]
    #        psths *= 1.0/len(trials)

    #    return psths
    # END HNdataset.calculate_psths()

    @staticmethod
    def train_test_assign(
        trial_ns, fold_num=5, which_fold=2, use_random=False
    ):  # this should be a static function
        """
        Assigns trials to training and test sets.

        Args:
            trial_ns: trial numbers
            fold_num: number of folds (default 4)
            use_random: whether to use random assignment (default True)

        Returns:
            utr: trial numbers for training set
            xtr: trial numbers for test set
        """
        num_tr = len(trial_ns)
        if use_random:
            permu = np.random.permutation(num_tr)
            xtr = np.sort(trial_ns[permu[range(np.floor(num_tr / fold_num).astype(int))]])
            utr = np.sort(
                trial_ns[permu[range(np.floor(num_tr / fold_num).astype(int), num_tr)]]
            )
        else:
            xlist = np.arange(which_fold, num_tr, fold_num, dtype="int32")
            ulist = np.setdiff1d(np.arange(num_tr), xlist)
            xtr = trial_ns[xlist]
            utr = trial_ns[ulist]
        return utr, xtr

    # END HNdata.train_test_assign()

    @staticmethod
    def channel_list_scrub(
        fnames, subset=None, display_names=True
    ):  # This should also be a static function
        """
        Scrubs channel names from filenames.

        Args:
            fnames: list of filenames
            subset: subset of filenames (default None)
            display_names: whether to display names (default True)

        Returns:
            chnames: list of channel names
        """
        chnames = []
        if subset is None:
            subset = list(range(len(fnames)))
        for nn in subset:
            fn = fnames[nn]
            a = fn.find("c")  # finds the 'ch'
            b = fn.find("s") - 1  # finds the 'sort'
            chn = deepcopy(fn[a:b])
            chnames.append(chn)
        if display_names:
            print(chnames)
        return chnames

    def __getitem__(self, idx):

        if len(self.cells_out) == 0:
            out = {
                "stim": self.stim[idx, :],
                "robs": self.robs[idx, :],
                "dfs": self.dfs[idx, :],
            }
            if self.speckled:
                out["Mval"] = self.Mval[idx, :]
                out["Mtrn"] = self.Mtrn[idx, :]

        else:
            assert isinstance(self.cells_out, list), "cells_out must be a list"
            robs_tmp = self.robs[:, self.cells_out]
            dfs_tmp = self.dfs[:, self.cells_out]
            out = {
                "stim": self.stim[idx, :],
                "robs": robs_tmp[idx, :],
                "dfs": dfs_tmp[idx, :],
            }
            if self.speckled:
                if self.Mtrn_out is None:
                    M1tmp = self.Mval[:, self.cells_out]
                    M2tmp = self.Mtrn[:, self.cells_out]
                    out["Mval"] = M1tmp[idx, :]
                    out["Mtrn"] = M2tmp[idx, :]
                else:
                    out["Mval"] = self.Mtrn_out[idx, :]
                    out["Mtrn"] = self.Mtrn_out[idx, :]

        if self.Xdrift is not None:
            out["Xdrift"] = self.Xdrift[idx, :]

        if self.Xadapt is not None:
            out["Xadapt"] = self.Xadapt[idx, :]

        if self.ACinput is not None:
            out["ACinput"] = self.ACinput[idx, :]

        if len(self.covariates) > 0:
            self.append_covariates(out, idx)

        return out

    # END HNdata.__getitem()


class MotionNeural(SensoryBase):
    """
    Class for handling motion neural data.
    """

    def __init__(self, filename=None, num_lags=30, tr_gap=10, **kwargs):
        """
        Args:
            num_lags: number of lags to include in stimulus (default 30)
            tr_gap: number of frames to blank out after each trial (default 10)
            filename: currently the pre-processed matlab file from Dan's old-style format
            datadir: directory for data (goes directly into SensoryBase)
            **kwargs: non-dataset specific arguments that get passed into SensoryBase
        """

        # Call parent constructor
        super().__init__(filename, **kwargs)
        matdat = sio.loadmat(self.datadir + filename)
        print("Loaded " + filename)

        # Essential variables
        self.robs = torch.tensor(matdat["robs"], dtype=torch.float32)
        self.NT = len(self.robs)
        self.dfs = torch.tensor(matdat["rf_criteria"], dtype=torch.float32)
        self.NC = self.robs.shape[1]
        self.ACinput = None  # autoencoder input

        # Use stim onsets for stim timing (starts with black)
        self.stimB = matdat["stimB"][:, 0].astype(np.float32)
        self.stimW = matdat["stimW"][:, 0].astype(np.float32)
        self.trial_stim = matdat["trial_stim"].astype(np.float32)
        self.stimts = np.where(np.diff(self.stimB) > 0)[0] + 1
        self.flash_stim = torch.zeros([self.NT, 1], dtype=torch.float32)
        self.flash_stim[self.stimts] = 1.0
        self.dims = [1, 1, 1, 1]
        self.Xadapt = None

        # Make trial block and classify trials
        blocks = matdat["blocks"] - 1  # Convert to python indexing
        trial_type_t = matdat["trial_type"][:, 0]
        Ntr = len(blocks)
        self.trial_type = np.zeros(Ntr, dtype=np.int64)  # 1=free viewing, 2=fixation
        for bb in range(Ntr):
            if bb < Ntr - 1:
                self.block_inds.append(np.arange(blocks[bb], blocks[bb + 1]))
            else:
                self.block_inds.append(np.arange(blocks[bb], self.NT))
            self.trial_type[bb] = np.median(trial_type_t[self.block_inds[bb]])

        # Mask out blinks using data_filters +/ 4 frames around each blink
        self.blinks = matdat["blinks"][:, 0].astype(np.float32)
        valid_data = 1 - self.blinks
        blink_onsets = np.where(np.diff(self.blinks) > 0)[0] + 1
        blink_offsets = np.where(np.diff(self.blinks) > 0)[0] + 1
        for ii in range(len(blink_onsets)):
            valid_data[
                range(
                    np.maximum(0, blink_onsets[ii] - 4),
                    np.minimum(self.NT, blink_offsets[ii] + 4),
                )
            ] = 0.0
        self.dfs *= valid_data[:, None]
        if tr_gap > 0:
            for bb in range(Ntr):
                self.dfs[self.block_inds[bb][:tr_gap], :] = 0.0

        # Saccades
        self.saccades = matdat["saccades"][:, 0].astype(np.float32)
        saccade_onsets = np.where(np.diff(self.saccades) > 0)[0] + 1
        saccade_offsets = np.where(np.diff(self.saccades) < 0)[0] + 1
        self.sacc_on = torch.zeros([self.NT, 1], dtype=torch.float32)
        self.sacc_on[saccade_onsets] = 1.0
        self.sacc_off = torch.zeros([self.NT, 1], dtype=torch.float32)
        self.sacc_off[saccade_offsets] = 1.0

        # Non-essential
        self.eye_pos = matdat["eye_pos"]
        self.eye_speed = matdat["eye_speed"][:, 0]
        self.framerate = matdat["framerate"][0, 0]
        # Bharath's used time points
        self.BTtimepts = matdat["timepts2use"][0, :] - 1  # Convert to python indexing

        # Make drift term -- need anchors at every other cycle
        transitions = np.where(abs(np.diff(self.trial_type)) > 0)[0] + 1
        drift_anchors = [0] + list(
            transitions[np.arange(1, len(transitions), 2)]
        )  # every other transition
        self.construct_drift_design_matrix(block_anchors=drift_anchors)

        self.crossval_setup(test_set=False)

        if num_lags is not None:
            self.assemble_stimulus(num_lags=num_lags)

    # END MotionNeural.__init__

    def assemble_stimulus(self, which_stim="all", use_trial_type=False, num_lags=180):
        """
        Assembles stimulus for dataset.

        Args:
            which_stim: could be 'trial' (default) using just the first stim onset" or could
                mark each flash (calling 'which_stim' anything else), although this would not implicitly
                take things like adapation into account
            use_trial_type: could fit different stim responses in the two conditions to gauge the effects
            num_lags: number of lags to include in stimulus (default 180)
        """

        self.num_lags = num_lags
        if which_stim == "trial":
            stim = self.trial_stim
        else:
            stim = self.flash_stim.detach().numpy()

        if use_trial_type:
            stim2 = np.concatenate((deepcopy(stim), deepcopy(stim)), axis=1)
            for bb in range(len(self.block_inds)):
                stim2[self.block_inds[bb], self.trial_type[bb] - 1] = 0.0
                # this will have stim present in 0 for fixation and 1 for free viewing
            self.stim = self.time_embedding(stim=stim2, nlags=num_lags)
            self.sac_dims = [2, 1, 1, num_lags]
        else:
            self.stim = self.time_embedding(stim=stim, nlags=num_lags)
            self.sac_dims = [1, 1, 1, num_lags]

    # END MotionNeural.assemble_stimulus()

    def construct_Xadapt(self, tent_start=18 + 8, tent_spacing=20):
        """
        Constructs adaptation-within-trial tent function
        Args:
            tent_start: starting point for tent basis
            tent_spacing: spacing between tent basis

        Returns:
            None
        """
        # Put first anchor
        anchors = tent_start + tent_spacing * np.arange(
            0, 6
        )  # 5 pulses (ignoring first)

        # Generate master tent_basis
        trial_tents = self.design_matrix_drift(
            len(self.block_inds[0]),
            anchors,
            zero_left=True,
            const_left=True,
            zero_right=False,
            const_right=True,
        )
        num_tents = trial_tents.shape[1]

        self.Xadapt = torch.zeros((self.NT, num_tents), dtype=torch.float32)

        for tr in range(len(self.block_inds)):
            L = len(self.block_inds[tr])
            self.Xadapt[self.block_inds[tr], :] = torch.tensor(
                trial_tents[:L, :], dtype=torch.float32
            )

    # END MotionNeural.construct_Xadapt()

    def autoencoder_design_matrix(
        self, trial_level=False, pre_win=0, post_win=0, blank=0, cells=None
    ):
        """
        Makes auto-encoder input using windows described above, and including the
        chosen cells. Will put as additional covariate "ACinput" in __get_item__

        Args:
            trial_level: whether to average over trials
            pre_win: how many time steps to include before origin
            post_win: how many time steps to include after origin
            blank: how many time steps to blank in each direction, including origin
            cells: which cells to include
        """

        if cells is None:
            cells = np.arange(self.NC)
        Rraw = deepcopy(self.robs[:, cells])
        self.ACinput = torch.zeros(Rraw.shape, dtype=torch.float32)

        if trial_level:
            assert (
                len(self.block_inds) > 0
            ), "No trials: trial-level autoencoder will not work."
            for tt in range(len(self.block_inds)):
                ts = self.block_inds[tt]
                self.ACinput[ts, :] = (
                    torch.ones([len(ts), 1]) @ torch.mean(Rraw[ts, :], axis=0)[None, :]
                )
        else:
            nsteps = 0
            if blank == 0:
                self.ACinput += Rraw
                nsteps = 1
            for ii in range(1, (pre_win + 1)):
                self.ACinput[ii:, :] += Rraw[:(-ii), :]
                nsteps += 1
            for ii in range(1, (post_win + 1)):
                self.ACinput[:(-ii), :] += Rraw[ii:, :]
                nsteps += 1
            assert nsteps > 0, "autoencoder design: invalid parameters"
            self.ACinput *= 1.0 / nsteps

        self.ACinput *= self.dfs[:, cells]

    # END MotionNeural.autoencoder_design_matrix

    def __getitem__(self, idx):

        if len(self.cells_out) == 0:
            out = {
                "stim": self.stim[idx, :],
                "robs": self.robs[idx, :],
                "dfs": self.dfs[idx, :],
            }
            if self.speckled:
                out["Mval"] = self.Mval[idx, :]
                out["Mtrn"] = self.Mtrn[idx, :]

        else:
            assert isinstance(self.cells_out, list), "cells_out must be a list"
            robs_tmp = self.robs[:, self.cells_out]
            dfs_tmp = self.dfs[:, self.cells_out]
            out = {
                "stim": self.stim[idx, :],
                "robs": robs_tmp[idx, :],
                "dfs": dfs_tmp[idx, :],
            }
            if self.speckled:
                M1tmp = self.Mval[:, self.cells_out]
                M2tmp = self.Mtrn[:, self.cells_out]
                out["Mval"] = M1tmp[idx, :]
                out["Mtrn"] = M2tmp[idx, :]

        out["sacc_on"] = self.sacc_on[idx, :]
        out["sacc_off"] = self.sacc_off[idx, :]
        out["Xdrift"] = self.Xdrift[idx, :]
        if self.Xadapt is not None:
            out["Xadapt"] = self.Xadapt[idx, :]
        if self.ACinput is not None:
            out["ACinput"] = self.ACinput[idx, :]

        if len(self.covariates) > 0:
            self.append_covariates(out, idx)

        return out

    # END MotionNeural.__get_item__()
