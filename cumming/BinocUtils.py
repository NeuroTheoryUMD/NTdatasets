import numpy as np
import scipy.io as sio
import NDNT.utils as utils
import NDNT.NDN as NDN
from time import time
from copy import deepcopy
import matplotlib.pyplot as plt


def varDF( s, df=None, mean_adj=True ):
    """
    Calculates variance over valid data. 
    mean_adj means true variance, but take away and becomes average squared deviation.
    
    Args:
        s: signal to calculate variance over
        df: data filter (if None, will use all data)
        mean_adj: whether to subtract mean before squaring

    Returns:
        variance of signal
    """

    s = s.squeeze()
    if df is None:
        df = np.ones(s.shape)
    df = df.squeeze()
    nrm = np.sum(df)
    assert nrm > 0, "df: no valid data"

    if mean_adj:
        sbar = np.sum(df*s)/nrm
    else:
        sbar = 0
    return np.sum(np.square(s-sbar)*df)/nrm
     

def explainable_variance( Edata, cell_n, fr1or3=None, inds=None, verbose=True ):
    """
    Explainable variance calculation: binocular-specific because of the data structures
    
    Args:
        Edata: binocular dataset (one experiment with a certain number of cells recorded)
        cell_n: cell number (in python numbering, i.e. starting with 0)
        fr1or3: whether to use fr1==1, fr3==3, or both (leave as None) to calculate disparity. Should choose 1 or 3
        inds (def: None): indices to calculate variances over, if None will use all inds
            generally will pass in the fr3 indices, for example
        verbose (def: True): self-explanatory
    
    Returns:
        totvar: literally the variance of the binned spike counts (resp) -- will be dom by spike stoch.
        explvar: explainable variance: repeatable variance (small fraction of totvar)

    Note: will return total variance (and a warning) if repeats not present in the dataset."""

    assert cell_n < Edata.NC, "cell_n is too large for this experiment (num cells = %d)"%Edata.NC

    resp = Edata.robs[:, cell_n].detach().numpy()
    
    if inds is None:
        inds = np.where(Edata.dfs[:, cell_n] > 0)[0]
    else:
        inds = np.intersect1d( inds, np.where(Edata.dfs[:, cell_n] > 0)[0] )

    if (fr1or3 == 3) or (fr1or3 == 1):
        inds = np.intersect1d(inds, np.where(Edata.frs == fr1or3)[0])

    if Edata.rep_inds is None:
        if verbose:
            print( 'No repeats in this dataset -- using total variance.')
        return np.var(resp[inds]), np.var(resp[inds])
            
    rep1inds = np.intersect1d(inds, Edata.rep_inds[cell_n][:,0])
    rep2inds = np.intersect1d(inds, Edata.rep_inds[cell_n][:,1])
    allreps = np.concatenate((rep1inds, rep2inds), axis=0)

    totvar = np.var(resp[allreps])
    explvar = np.mean(np.multiply(resp[rep1inds]-np.mean(resp[allreps]), resp[rep2inds]-np.mean(resp[allreps]))) 
    
    return explvar, totvar


def predictive_power( pred, Edata, cell_n, inds=None, verbose=True ):
    """
    Predictive power calculation (R2 adjusted by dividing by explainable (rather than total) variance
    (binocular-specific because of the data structures)
    
    Args:
        pred: model prediction: must have same size as robs (unindexed)
        Edata: binocular dataset (one experiment with a certain number of cells recorded)
        cell_n: cell number (in python numbering, i.e. starting with 0)
        inds (def: None): indices to calculate variances over, if None will use all inds
            BUT it should pass in XVinds generally, and for example could also focus on fr3
        verbose (def: True): self-explanatory
    
    Returns: 
        predictive power of cell
    """

    assert cell_n < Edata.NC, "cell_n is too large for this experiment (num cells = %d)"%Edata.NC
    pred=pred.squeeze()

    if not type(pred) == np.ndarray:
        pred = pred.detach().numpy()

    if inds is None:
        inds = np.where(Edata.dfs[:, cell_n] > 0)[0]
    else:
        inds = np.intersect1d( inds, np.where(Edata.dfs[:, cell_n] > 0)[0] )

    Robs = Edata.robs[:, cell_n].detach().numpy()
    mod_indxs = deepcopy(inds)
    if Edata.rep_inds is not None:
        rep1inds = np.intersect1d(inds, Edata.rep_inds[cell_n][:,0])
        rep2inds = np.intersect1d(inds, Edata.rep_inds[cell_n][:,1])
        allreps = np.concatenate((rep1inds, rep2inds), axis=0)
        mod_indxs = np.intersect1d( mod_indxs, allreps )
        r1a = Robs[rep1inds]
        r1b = Robs[rep2inds]
        r2 = pred[rep1inds]  # pred will be the same on both
    else:
        r1a = Robs[mod_indxs]
        r1b = r1a
        r2 = pred[mod_indxs]

    # Now assuming that r (Robs) is length of indxs, and pred is full res
    expl_var,_ = explainable_variance( Edata, cell_n, inds=mod_indxs, verbose=verbose )

    # explained_power = np.var(r1)-np.mean(np.square(r1-r2))  # WOW I THINK THIS IS WRONG -- WAS ORIGINAL FORMULA
    explained_power = expl_var -np.mean(np.multiply(r1a-r2, r1b-r2))

    # calculate other way
    #crosscorr = np.mean(np.multiply(r1-np.mean(r1), r2-np.mean(r2)))
    #print( (crosscorr**2/expl_var/np.var(r2)) )

    return explained_power/expl_var


def disparity_matrix( dispt, corrt=None ):
    """
    Create one-hot representation of disparities: NT x 2*ND+2 (ND = num disparities)
    -- Columns (0,ND-1):  correlated
    -- Columns (ND, 2*ND-1): anticorrelated
    -- Column -2: uncorrelated
    -- Column -1: blank

    Args:
        dispt: disparity values
        corrt: correlation values (if None, will not use)

    Returns:
        dmat: one-hot representation of disparities
    """
    dlist_raw = np.unique(dispt) 
    if np.max(abs(dlist_raw)) > 100:
        # last two colums will be uncorrelated (-1005) and blank (-1009)
        dlist = dlist_raw[2:]  # this will exclude -1009 (blank) and -1005 (uncor)
        num_blanks = 2
    else:
        dlist = dlist_raw
        num_blanks = 0

    ND = len(dlist)
    if corrt is None:
        dmat = np.zeros([dispt.shape[0], ND+num_blanks])
    else:
        dmat = np.zeros([dispt.shape[0], 2*ND+num_blanks])

    if num_blanks > 0:
        dmat[np.where(dispt == -1009)[0], -1] = 1
        dmat[np.where(dispt == -1005)[0], -2] = 1

    for dd in range(len(dlist)):
        if corrt is None:
            dmat[np.where(dispt == dlist[dd])[0], dd] = 1
        else:
            dmat[np.where((dispt == dlist[dd]) & (corrt > 0))[0], dd] = 1
            dmat[np.where((dispt == dlist[dd]) & (corrt < 0))[0], ND+dd] = 1

    return dmat


def disparity_tuning( data, r, cell_n=None, num_dlags=8, fr1or3=3, to_plot=False ):
    """
    Compute disparity tuning (disparity vs time) -- returned in dictionary object
    -> include cell_n to use data_filters from the actual cell

    Args:
        data: binocular dataset (NTdatasets.binocular.single)
        r: response of the cell (or model) to use
        cell_n: cell number (in python numbering, i.e. starting with 0)
        num_dlags: number of lags to use in the time embedding
        fr1or3: whether to use fr1==1, fr3==3, or both (leave as None) to calculate disparity. Should choose 1 or 3
        to_plot: whether to plot the tuning curve

    Returns:
        Dinfo: dictionary with all the information about the disparity tuning curve
    """
    import torch

    dmat = disparity_matrix( data.dispt, data.corrt)
    ND = (dmat.shape[1]-2) // 2

    # Weight all by their frequency of occurance    
    if (fr1or3 == 3) or (fr1or3 == 1):
        inds = np.where(data.frs == fr1or3)[0]
    else:
        inds = np.where(data.frs > 0)[0]

    if isinstance(r, torch.Tensor):
        r = r.cpu().detach().numpy()
    r = r.squeeze()
    
    if cell_n is not None:
        resp = np.multiply( deepcopy(r), data.dfs[:, cell_n].detach().numpy())[inds]
    else:
        resp = deepcopy(r[inds])

    resp = resp[:, None]
    dmatN = dmat / np.mean(dmat[inds, :], axis=0)  # will be stim rate

    # if every stim resulted in 1 spk, the would be 1 as is
    #nrms = np.mean(dmat[used_inds[to_use],:], axis=0) # number of stimuli of each type
    Xmat = utils.create_time_embedding( dmatN[:, range(ND*2)], num_dlags)[inds, :]
    # uncorrelated response
    Umat = utils.create_time_embedding( dmatN[:, [-2]], num_dlags)[inds, :]
                          
    #Nspks = np.sum(resp[to_use, :], axis=0)
    Nspks = np.sum(resp, axis=0)  # this will end up being number of spikes associated with each stim 
    # at different lags, divided by number of time points used. (i.e. prob of spike per bin)

    Dsta = np.reshape( Xmat.T@resp, [2*ND, num_dlags] ) / Nspks
    Usta = (Umat.T@resp)[:,0] / Nspks
    
    # Rudimentary analysis
    best_lag = np.argmax(np.max(Dsta[range(ND),:], axis=0))
    Dtun = np.reshape(Dsta[:, best_lag], [2, ND]).T
    uncor_resp = Usta[best_lag]
    
    Dinfo = {'Dsta':Dsta, 'Dtun': Dtun, 'uncor_resp': uncor_resp, 
            'best_lag': best_lag, 'uncor_sta': Usta, 'disp_list': data.disp_list[2:]}

    if to_plot:
        utils.subplot_setup(1,2, fig_width=10, row_height=2.8)
        plt.subplot(1,2,1)
        utils.imagesc(Dsta-uncor_resp, cmap='bwr')
        plt.plot([ND-0.5,ND-0.5], [-0.5, num_dlags-0.5], 'k')
        plt.plot([-0.5, 2*ND-0.5], [best_lag, best_lag], 'k--')
        plt.subplot(1,2,2)
        plt.plot(Dtun)
        plt.plot(-Dtun[:,1]+2*uncor_resp,'m--')

        plt.plot([0, ND-1], [uncor_resp, uncor_resp], 'k')
        plt.xlim([0, ND-1])
        plt.show()
        
    return Dinfo

## Not  working yet ##

def disparity_predictions( 
    data, resp=None, cell_n=None, 
    fr1or3=None, indxs=None, 
    num_dlags=8,  spiking=True, rectified=True ):
    """
    Calculates a prediction of the disparity (and timing) signals that can be inferred from the response
    by the disparity input alone. This puts a lower bound on how much disparity is driving the response, although
    practically speaking it generates the same disparity tuning curves for neurons.
    
    Args:
        data: dataset (NTdatasets.binocular.single)
        resp: either Robs or predicted response across whole dataset -- leave blank if want neurons Robs
        cell_n: cell number (in python numbering, i.e. starting with 0)
        fr1or3: whether to use fr1==1, fr3==3, or both (leave as None) to calculate disparity. Should choose 1 or 3
        indxs: subset of data -- probably will not use given dfs and fr1or3
        num_dlags: how many lags to compute disparity/timing predictions using (default 8 is sufficient)
        spiking: whether to use Poisson loss function (spiking data) or Gaussian (continuous prediction): default True
        rectified: whether to rectify the predictions using softplus (since predicting spikes, generally)
    
    Returns: 
        Dpred: full disparity+timing prediction
        Tpred: prediction using just frame refresh and blanks
        Note that both will predict over the whole dataset, even if only used 1 part to fit
    """

    assert cell_n is not None, "Need to specify which neuron modeling for valid comparisons"

    import torch
    from NTdatasets.generic import GenericDataset
    from NDNT.modules.layers import NDNLayer

    # use Robs if response is blank
    if resp is None:
        resp = data.robs[:, cell_n]
    # Make sure response is formatted correctly
    if not isinstance( resp, torch.Tensor):
        resp = torch.tensor( resp, dtype=torch.float32 )
    if len(resp.shape) == 1:
        resp = resp[:, None]

    if indxs is None:
        indxs = range(resp.shape[0])
    if (fr1or3 == 3) or (fr1or3 == 1):
        mod_indxs = np.intersect1d(indxs, np.where(data.frs == fr1or3)[0])
    else:
        mod_indxs = indxs

    # Process disparity into disparty and timing design matrix
    dmat = disparity_matrix( data.dispt, data.corrt )  # all information about disparity
    
    # This keeps track of changes in disparity (often every 3)
    switches = np.expand_dims(np.concatenate( (np.sum(abs(np.diff(dmat, axis=0)),axis=1), [0]), axis=0), axis=1)/2
    # Switches are not relevant during FR1 -- would at best be constant offset
    switches[np.where(data.frs == 1)[0]] = 0.0
    
    # Append switches to full disparity-oracle dataset
    dmat = np.concatenate( (dmat, switches), axis=1 )
    ND2 = dmat.shape[1]  # this includes uncorrelated (second to last) and blanks (last column

    blanks = dmat[:, -1][:, None]
    tmat = np.concatenate( (blanks, switches), axis=1 )

    # Make models that use disparity to predict response, and timing alone
    Ddata = GenericDataset( {
        'stim': torch.tensor(utils.create_time_embedding( dmat, num_dlags), dtype=torch.float32),
        'timing': torch.tensor(utils.create_time_embedding( tmat, num_dlags), dtype=torch.float32),
        'robs': resp,
        'dfs': data.dfs[:, cell_n][:, None]})

    lbfgs_pars = utils.create_optimizer_params(
        optimizer_type='lbfgs',
        tolerance_change=1e-8, tolerance_grad=1e-8,
        history_size=100, batch_size=4000, max_epochs=3, max_iter=500)
 
    if rectified:
        nltype = 'softplus'
    else:
        nltype = 'lin'
    if spiking:
        losstype = 'poisson'
    else:
        losstype = 'gaussian'
        
    dpred_layer = NDNLayer.layer_dict(
        input_dims=[1,ND2,1,num_dlags], num_filters=1, bias=True, NLtype=nltype)

    tpred_layer = NDNLayer.layer_dict(
        input_dims=[2,1,1,num_dlags], num_filters=1, bias=True, NLtype=nltype)

    dpredmod = NDN.NDN( layer_list=[dpred_layer], loss_type=losstype )
    tpredmod = NDN.NDN( layer_list=[tpred_layer], loss_type=losstype )
    tpredmod.networks[0].xstim_n = 'timing'

    #dpredmod.fit( Ddata, force_dict_training=True, train_inds=mod_indxs, val_inds=mod_indxs, 
    #            **lbfgs_pars, verbose=0, version=1)
    utils.iterate_lbfgs( dpredmod, Ddata, lbfgs_pars, train_inds=mod_indxs, val_inds=mod_indxs, verbose=False )
    utils.iterate_lbfgs( tpredmod, Ddata, lbfgs_pars, train_inds=mod_indxs, val_inds=mod_indxs, verbose=False )
    #tpredmod.fit( Ddata, force_dict_training=True, train_inds=mod_indxs, val_inds=mod_indxs, 
    #            **lbfgs_pars, verbose=0, version=1)

    tpred = tpredmod(Ddata[:]).detach().numpy()
    dpred = dpredmod(Ddata[:]).detach().numpy()

    return dpred, tpred


def binocular_model_performance( data=None, cell_n=None, Rpred=None, valset=None, verbose=True ):
    """
    Current best-practices for generating prediction quality of neuron and binocular tuning. Currently we
    are not worried about using cross-validation indices only (as they are based on much less data and tend to
    otherwise be in agreement with full measures, but this option could be added in later versions.
    valset can be None (use all val_inds, 'a' or 'b': use subset)
    
    Args:
        data: binocular dataset (NTdatasets.binocular.single)
        cell_n: cell number (in python numbering, i.e. starting with 0)
        Rpred: predicted response of the model
        valset: which validation set to use (None, 'a', 'b')
        verbose: whether to print out results

    Returns:
        BMP: dictionary with all the information about the binocular model performance
    """

    assert data is not None, 'Need to include dataset'
    assert cell_n is not None, 'Must specify cell to check'

    #import torch
    #if not isinstance( Rpred, torch.Tensor):
    #    Rpred = torch.tensor( Rpred, dtype=torch.float32 )
    if len(Rpred.shape) == 1:
        Rpred = Rpred[:, None]
    
    ## GENERAL COMPUTATIONS on data (cell-specific but not yet model-specific, using as much data as can)
    # make disparity predictions for all conditions
    dobs0, tobs0 = disparity_predictions( data, cell_n=cell_n, spiking=True, rectified=True )
    dmod0, tmod0 = disparity_predictions( data, resp=Rpred, cell_n=cell_n, spiking=True, rectified=True )

    dobs3, tobs3 = disparity_predictions( data, cell_n=cell_n, fr1or3=3, spiking=True, rectified=True )
    dmod3, tmod3 = disparity_predictions( data, resp=Rpred, cell_n=cell_n, fr1or3=3, spiking=True, rectified=True )

    dobs1, tobs1 = disparity_predictions( data, cell_n=cell_n, fr1or3=1, spiking=True, rectified=True )
    dmod1, tmod1 = disparity_predictions( data, resp=Rpred, cell_n=cell_n, fr1or3=1, spiking=True, rectified=True )

    # This necessarily takes data-filters into account, but not cross-validation inds
    ev, tv = explainable_variance( data, cell_n=cell_n, verbose=verbose )
    ev3, tv3 = explainable_variance( data, cell_n=cell_n, fr1or3=3, verbose=verbose )
    ev1, tv1 = explainable_variance( data, cell_n=cell_n, fr1or3=1, verbose=verbose )
    
    if ev == tv:
        ev_valid = False
    else:
        ev_valid = True

    if verbose:
        print( "  Overall explainable variance fraction: %0.3f"%(ev/tv) )

    #### Model and data properties (not performance yet)
    indxs3 = np.where(data.frs == 3)[0]
    indxs1 = np.where(data.frs == 1)[0]
    df = data.dfs[:, cell_n].detach().numpy()

    # Have to use same (df) inds as overall explainable variance to make fractions directly valid
    dv_obs = varDF(dobs0-tobs0, df=df)
    dv_obs3 = varDF(dobs0[indxs3]-tobs0[indxs3], df=df[indxs3])
    dv_obs1 = varDF(dobs0[indxs1]-tobs0[indxs1], df=df[indxs1])

    dv_pred = varDF(dmod0-tmod0, df=df)
    dv_pred3 = varDF(dmod3[indxs3]-tmod3[indxs3], df=df[indxs3])
    dv_pred1 = varDF(dmod1[indxs1]-tmod1[indxs1], df=df[indxs1])
    
    if verbose:
        print( "  Obs disparity variance fraction (DVF): %0.3f (FR3: %0.3f)"%(dv_obs/ev, dv_obs3/ev3) )
    vars_obs = [tv, ev, dv_obs, ev-dv_obs ]  # total, explainable, disp_var, pattern_var
    vars_obs_FR3 = [tv3, ev3, dv_obs3, ev3-dv_obs3 ]  # total, explainable, disp_var, pattern_var
    DVfrac_obs = [dv_obs/ev, dv_obs3/ev3, dv_obs1/ev1 ]

    # use numpy version of Rpred from here on
    Rpred = Rpred.detach().numpy()
    var_pred = varDF(Rpred, df=df)
    vars_mod = [varDF(Rpred, df=df), dv_pred, var_pred-dv_pred]
    DVfrac_mod = [dv_pred/var_pred, 
                  dv_pred3/varDF(Rpred[indxs3], df=df[indxs3]), 
                  dv_pred1/varDF(Rpred[indxs1], df=df[indxs1])]


    ## Model-based performance measures - need cross-validation indices only
    if valset is None:
        v_inds = data.val_inds
    elif valset in ['a', 'A']:
        v_inds = data.val_indsA
        #print( '  Using val set A')
    else:
        v_inds = data.val_indsB
        #print( '  Using val set B')

    #indxs3xv = np.intersect1d( data.val_inds, indxs3 )
    #indxs1xv = np.intersect1d( data.val_inds, indxs1 )
    if data.rep_inds is None:
        allreps = np.arange(data.NT)
    else:
        rep1inds = np.intersect1d(v_inds, data.rep_inds[cell_n][:,0])
        rep2inds = np.intersect1d(v_inds, data.rep_inds[cell_n][:,1])
        allreps = np.concatenate((rep1inds, rep2inds), axis=0)

    indxs3xv = np.intersect1d( allreps, indxs3 )
    indxs1xv = np.intersect1d( allreps, indxs1 )

    #DVfrac_mod_alt = [dv_mod/np.var(Rpred[indxs]), 
    #                  dv_pred3b/np.var(Rpred[indxs3]), 
    #                  dv_pred1b/np.var(Rpred[indxs1])]
    
    # Predictive powers (model performance): full response, fr3, fr1
    pps = [predictive_power( Rpred, data, cell_n=cell_n, inds=allreps ),  # predict variance of full response
           predictive_power( Rpred, data, cell_n=cell_n, inds=indxs3xv ),
           predictive_power( Rpred, data, cell_n=cell_n, inds=indxs1xv )]

    Dpps = np.zeros(3)
    #Dpps[0] = 1-varDF(dobs0[data.val_inds]-dmod0[data.val_inds], df=df[data.val_inds]) / \
    #    varDF(dobs0[data.val_inds], df=df[data.val_inds])
    #Dpps[1] = 1-varDF(dobs3[indxs3xv]-dmod3[indxs3xv], df=df[indxs3xv]) / \
    #    varDF(dobs3[indxs3xv], df=df[indxs3xv])
    #Dpps[2] = 1-varDF(dobs1[indxs1xv]-dmod1[indxs1xv], df=df[indxs1xv]) / \
    #    varDF(dobs1[indxs1xv], df=df[indxs1xv])
    dobs_only0 = dobs0-tobs0
    dobs_only3 = dobs3-tobs3
    dobs_only1 = dobs1-tobs1
    dmod_only0 = dmod0-tmod0
    dmod_only3 = dmod3-tmod3
    dmod_only1 = dmod1-tmod1

    Dpps[0] = 1-varDF(dobs_only0[v_inds]-dmod_only0[v_inds], df=df[v_inds], mean_adj=False) / \
        varDF(dobs_only0[v_inds], df=df[v_inds])
    Dpps[1] = 1-varDF(dobs_only3[indxs3xv]-dmod_only3[indxs3xv], df=df[indxs3xv], mean_adj=False) / \
        varDF(dobs_only3[indxs3xv], df=df[indxs3xv])
    Dpps[2] = 1-varDF(dobs_only1[indxs1xv]-dmod_only1[indxs1xv], df=df[indxs1xv], mean_adj=False) / \
        varDF(dobs_only1[indxs1xv], df=df[indxs1xv])

    if verbose:
        print( "  Pred powers: %0.3f  disp %0.3f (FR3 %0.3f, FR1 %0.3f)"%(pps[0], Dpps[0], Dpps[1], Dpps[2]))

    # Add general tuning of each
    Dtun_obs = [disparity_tuning( data, data.robs[:, cell_n], cell_n=cell_n, fr1or3=3 ),
                     disparity_tuning( data, data.robs[:, cell_n], cell_n=cell_n, fr1or3=1 )]
    Dtun_pred = [disparity_tuning( data, Rpred, cell_n=cell_n, fr1or3=3),
                      disparity_tuning( data, Rpred, cell_n=cell_n, fr1or3=1)]
    # Tuning curve consistency
    DtuningR2 = [1-np.mean(np.square(Dtun_obs[0]['Dtun']-Dtun_pred[0]['Dtun']))/np.var(Dtun_obs[0]['Dtun']),
                 1-np.mean(np.square(Dtun_obs[1]['Dtun']-Dtun_pred[1]['Dtun']))/np.var(Dtun_obs[1]['Dtun'])]
    if verbose:
        print( "  Tuning consistency R2s: FR3 %0.3f  FR1 %0.3f"%(DtuningR2[0], DtuningR2[1]))

    BMP = {'EVfrac': ev/tv, 'EVvalid': ev_valid, 
           'vars_obs': vars_obs, 'vars_mod': vars_mod, 'vars_obs_FR3': vars_obs_FR3,
           'DVfrac_obs': DVfrac_obs, 'DVfrac_mod': DVfrac_mod, 
           'pred_powers': pps, 'disp_pred_powers': Dpps,
           'Dtun_obs': Dtun_obs, 'Dtun_pred': Dtun_pred, 'DtuningR2': DtuningR2}
    
    return BMP


## Binocular model utilities
def compute_Mfilters( tkerns=None, filts=None, mod=None, to_plot=True, to_output=False):
    """
    Compute a spatiotemporal filter from temporal kernels convolved with second-layer filter
    
    Args:
        tkerns: temporal kernels (NT x num_kernels)
        filts: second-layer filters (num_kernels x num_space x num_filters)
        mod: model to extract weights from
        to_plot: whether to plot the filter
        to_output: whether to return the filter

    Returns:
        Mfilts: spatiotemporal filter
    """
    if mod is not None:
        tkerns = mod.get_weights(layer_target=0)
        filts = mod.get_weights(layer_target=1)
    else:
        assert filts is not None, "Cant have everything be none"
    ntks = tkerns.shape[1]
    nfilts = filts.shape[-1]
    if len(filts.shape) == 2:
        filts=filts.reshape([ntks,-1, nfilts])
    else:
        assert filts.shape[0] == ntks, "Issue with filter shapes"
    Mfilts = np.einsum('tb,bxl->xtl', tkerns, filts)
    if to_plot:
        nrows = (nfilts-1)//8 + 1
        ncols = np.minimum(nfilts, 8)
        utils.ss(nrows, ncols, rh=1.5)
        for ii in range(nfilts):
            plt.subplot(nrows, ncols, ii+1)
            utils.imagesc(Mfilts[:,:, ii])
        plt.show()
    if to_output:
        return Mfilts


def compute_binocular_filters(binoc_mod, to_plot=True, cmap=None, time_reverse=True,
                              num_space=None, ffnet_n=None, mfilters=None ):
    """
    Using standard binocular model, compute filters. defaults to first ffnet and
    num_space = 36. Set num_space=None to go to minimum given convolutional constraints
    
    Args:
        binoc_mod: binocular model
        to_plot: whether to plot the filters
        cmap: colormap to use
        time_reverse: whether to reverse the time axis
        num_space: number of spatial dimensions to use
        ffnet_n: which feed-forward network to use
        mfilters: monocular filters to use (if None, will use first layer of model)

    Returns:
        bifilts: binocular filters
    """

    # Find binocular layer within desired (or whole) model
    from NDNT.modules.layers import BiConvLayer1D, BiSTconv1D #, ChannelConvLayer

    if ffnet_n is not None:
        networks_to_search = [ffnet_n]
    else:
        networks_to_search = range(len(binoc_mod.networks))        
    blayer, bnet = None, None
    for mm in networks_to_search:
        for nn in range(len(binoc_mod.networks[mm].layers)):
            if isinstance(binoc_mod.networks[mm].layers[nn], BiConvLayer1D) or isinstance(binoc_mod.networks[mm].layers[nn], BiSTconv1D): # this is layer output to bi-layer
                if nn < len(binoc_mod.networks[mm].layers) - 1:
                    bnet, blayer = mm, nn + 1
                elif mm < len(binoc_mod.networks) - 1:
                    bnet, blayer = mm + 1, 0  # split in hierarchical network
    assert blayer is not None, 'biconv layer not found'

    if mfilters is None:
        if (bnet == 0) & (blayer == 1): # then mfilters are just first layer
            mfilters = binoc_mod.get_weights(time_reverse=time_reverse)
        else:  # construct using tkerns
            mfilters = compute_Mfilters(mod=binoc_mod, to_output=True, to_plot=False) 
            if time_reverse:
                print('Warning: time_reverse not implemented in this case.')
    NXM, num_lags, NF = mfilters.shape
    #print(NXM, num_lags, NF)
    ws = binoc_mod.get_weights(ffnet_target=bnet, layer_target=blayer)
    NF2, NXC, NBF = ws.shape 
    num_cspace = NXM+NXC-1

    eye_filts = np.zeros([num_cspace, num_lags, 2, NBF])

    #if isinstance(binoc_mod.networks[bnet].layers[blayer], ChannelConvLayer):
        #print('Input filters:', NXM, NF)
        #print(NF2, 'weights per binocular filter, LRLR ->', NF2//2, 'monocular filters involved')
        #print(NXM,NXC,num_cspace)
    #    assert (NF2//2)*NBF == NF, "ChannelConv monocular filter mismatch"
    #    for ff in range(NBF):
    #        frange = (NF2//2)*ff + np.arange(NF2//2) 
    #        for xx in range(NXC):
    #            for eye in range(2):
    #                eye_filts[np.arange(NXM)+xx, :, eye, ff] += np.einsum(
    #                    'xtm,m->xt', mfilters[:, :, frange], ws[np.arange(eye,NF2,2), xx, ff] )
                    #eye_filts[np.arange(NXM)+xx, :, eye, ff] += mfilters[:, :, frange+eye] @ ws[np.arange(eye,NF2,2), xx, ff][:,None,None]
    #else: # Standard
    assert NF2 == 2*NF, "Monocular filter number mismatch"
    for xx in range(NXC):
        for eye in range(2):
            eye_filts[np.arange(NXM)+xx, :, eye, :] += np.einsum(
                'xtm,mf->xtf', mfilters, ws[np.arange(eye,NF2,2), xx, :] )
                #'xtm,mf->xtf', mfilters, ws[np.arange(NF)+eye*NF, xx, :] )
    if num_space is None:
        # Cast into desired num_space
        num_space = num_cspace
    if num_space == num_cspace:
        Bfilts = eye_filts
    elif num_space > num_cspace:
        Bfilts = np.zeros([num_space, num_lags, 2, NBF])
        padding = (num_space-num_cspace) // 2
        Bfilts[padding+np.arange(num_cspace), :, :, :] = eye_filts
    else:  # crop
        unpadding = (num_cspace-num_space) // 2
        Bfilts = eye_filts[unpadding+np.arange(num_space), :, :, :]

    bifilts = np.concatenate((Bfilts[:, :, 0, :], Bfilts[:, :, 1, :]), axis=0)
    if to_plot:
        from NDNT.utils import plot_filters_ST1D
        if isinstance(binoc_mod.networks[bnet].layers[blayer-1], BiSTconv1D):
            bifilts = np.flip(bifilts, axis=1)  # Then time-reverse 
        plot_filters_ST1D( bifilts, num_cols=5, cmap=cmap, )
    else:
        return bifilts


def plot_sico_readout( sico ):
    """
    Plot the readout weights of the SICO model

    Args:
        sico: SICO model

    Returns:
        None
    """
    from NDNT.utils import imagesc
    w = sico.networks[0].layers[-1].get_weights().squeeze().T
    NF = w.shape[1]
    NI = sico.networks[0].layers[-2].num_inh
    NE = NF-NI
    w[:, NE:] *= -1

    utils.subplot_setup(1,1,fig_width=6, row_height=0.15*NF)
    imagesc(w, cmap='bwr')
    plt.show()

