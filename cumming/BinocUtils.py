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

    dpredmod = NDN( layer_list=[dpred_layer], loss_type=losstype )
    tpredmod = NDN( layer_list=[tpred_layer], loss_type=losstype )
    tpredmod.networks[0].xstim_n = 'timing'

    #dpredmod.fit( Ddata, force_dict_training=True, train_inds=mod_indxs, val_inds=mod_indxs, 
    #            **lbfgs_pars, verbose=0, version=1)
    #utils.iterate_lbfgs( dpredmod, Ddata, lbfgs_pars, train_inds=mod_indxs, val_inds=mod_indxs, verbose=False )
    #utils.iterate_lbfgs( tpredmod, Ddata, lbfgs_pars, train_inds=mod_indxs, val_inds=mod_indxs, verbose=False )
    utils.fit_lbfgs( dpredmod, Ddata[mod_indxs], verbose=False )
    utils.fit_lbfgs( tpredmod, Ddata[mod_indxs], verbose=False )
    #tpredmod.fit( Ddata, force_dict_training=True, train_inds=mod_indxs, val_inds=mod_indxs, 
    #            **lbfgs_pars, verbose=0, version=1)

    tpred = tpredmod(Ddata[:]).detach().numpy()
    dpred = dpredmod(Ddata[:]).detach().numpy()

    return dpred, tpred


def binocular_model_performance( data=None, cell_n=0, Rpred=None, valset=None, verbose=True ):
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
    #assert cell_n is not None, 'Must specify cell to check'

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
def compute_mfilters( tkerns=None, filts=None, mod=None, to_plot=True, to_output=False):
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


def compute_binocular_filters_old(binoc_mod, to_plot=True, cmap=None, time_reverse=True,
                              num_space=None, ffnet_n=None, mfilters=None ):
    """
    Using standard binocular model, compute filters. defaults to first ffnet and
    num_space = 36. Set num_space=None to go to minimum given convolutional constraints
    -> this is older function that might be outdated (see new function below)
    
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
            mfilters = compute_mfilters(mod=binoc_mod, to_output=True, to_plot=False) 
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


def plot_sico_readout( sico, cell_n=None, rh=None, row_mult=0.2 ):
    """
    Plot the readout weights of the SICO model

    Args:
        sico: SICO model
        cell_n: select which clone to plot if clone model (default: None, assumes single model)
        rh: height of plot, overriding scaling-per line (row_mult) (default: None)
        row_mult: normal way to determine depth of plot: inches per line (default: 0.4)

    Returns:
        None
    """
    from NDNT.utils import imagesc
    if cell_n is None:
        w = sico.networks[0].layers[-1].get_weights().squeeze().T
        if sico.networks[0].layers[-1].input_dims[0] == 1:
            w = w[None,:].T
    else:
        w = sico.networks[0].layers[-1].get_weights()[..., cell_n].T
    NF = w.shape[1]
    NI = sico.networks[0].layers[-2].num_inh
    NE = NF-NI
    w[:, NE:] *= -1

    if rh is None:
        rh = row_mult*NF+0.55 #np.minimum(0.4*NF,2)
        # added 0.55 is estimated size of x-axis ticks and other vertical space
    if cell_n is None:
        utils.subplot_setup(1,1, row_height=rh, fig_width=6)

    imagesc(w, cmap='bwr', balanced=True, axis_labels=False)
    if cell_n is None:
        plt.show()
# END plot_sico_readout()


### NEW FUNCTIONS ADDED 2024 ###
def compute_mfilters( mod, tkerns=None):
    """
    Calculates the filters of the first (monocular) stage of model given tkerns

    Args: 
        mod: binocular model with monocular subspace and implicit temporal basis embedded in stim
        tkerns: temporal kernels, if none (default) than just returns first layer (but shouldn't)

    Returns:
        mfilts: 3-d array space x lags x number of filters
    """
    assert tkerns is not None, "No temporal basis specified."
    k = mod.get_weights()
    if tkerns is not None:
        mfilts = np.einsum('tc,xcf->xtf', tkerns, k)
    else:
        print(  "Warning: no temporal kernels specified, just using get_weights()" )
        mfilts = k    
    return mfilts
# END compute_mfilters()


def dist_mean( p ):
    xs = np.arange(len(p))
    return np.sum(xs*p)/np.sum(p)


def compute_bfilters( binoc_mod, tkerns=None, width=None, newlags=None, skip_lags=0, center=True ):
    """
    Calculates binocular filter array for monocular-subspace model, including accounting for temporal kernels
    used to preprocess the input.
    Adjusts for skip_lags, although must enter by hand (default=0)
    
    Args:
        binoc_mod: assumes temporal kernels processing input, and monocular subspace before binocular filters
        tkerns: temporal kernels that pre-processed stimulus
        width: how wide each monocular filter should be displayed: its convolutional so not set. Default
               is based on model itself: monocular filter with plus what binocular convolutions brings
        newlags: lags past what is given by filter (default none)
        center: centers each binocular filter based on mean temporal power (since it will be convolved too)
    
    Returns:
        eye_filts: binocular filters by eye with dims [2, width, lags, num_filters]        
    """

    if tkerns is None:
        mfilters = binoc_mod.get_weights(layer_target=0)
        if skip_lags != 0:
            mfilters = np.roll(mfilters, skip_lags, axis=0)
    else:
        # shift 
        if skip_lags != 0:
            tks = np.roll(tkerns, skip_lags, axis=0)
        else:
            tks = tkerns
        mfilters = compute_mfilters(binoc_mod, tkerns=tks)
    
    NXM, num_lags, NF = mfilters.shape
    if newlags is None:
        newlags = num_lags

    ws = binoc_mod.get_weights(ffnet_target=0, layer_target=1)
    NF2, NXC, NBF = ws.shape 
    num_cspace = NXM+NXC-1
    #print('top', NF2, NXC, NBF, num_cspace)
    #print(mfilters.shape, tkerns.shape, NXM)
    eye_filts = np.zeros([num_cspace, newlags, 2, NBF])
    assert NF2 == 2*NF, "Monocular filter number mismatch"
    for xx in range(NXC):
        for eye in range(2):
            eye_filts[np.arange(NXM)+xx, :, eye, :] += np.einsum(
                'xtm,mf->xtf', mfilters[:,:newlags,:], ws[np.arange(eye,NF2,2), xx, :] )#[:, :newlags]

    if center:
        sp_maps = np.sum(np.std(eye_filts,axis=1),axis=1)
        for ii in range(NBF):
            sh = int(np.round(num_cspace/2-dist_mean(sp_maps[:,ii])))                
            eye_filts[..., ii] = np.roll(eye_filts[..., ii], sh, axis=0)

    if width is None:
        width = num_cspace
    if width < num_cspace:
        shrinky = (num_cspace-width)//2
        print('  shrinking width', num_cspace,'->', width)
        eye_filts = eye_filts[np.arange(width)+shrinky, ...]
    else:
        if width > num_cspace:
            print('width too large -- havent set up yet')
    
    return eye_filts.transpose([2,0,1,3])


def plot_mfilters( model, tkerns=None, flip=True, axis_labels=False, max_lags=12, rh=2 ):
    """
    Plots mfilters (first-layer monocular filters) of sico models, using compute_mfilters to calcuate. 

    Inputs:
        model: sico model to plot monocular filters of
        tkerns: temporal kernels used in sico model (default: None assumes no temporal filters used)
        flip: whether to have lag-0 at bottom and go up (default: flip=True) or opposite
        axis_labels: whether to display axis_labels (default: False=No)
        max_lags: number of lags to trim at (default: 12)
        rh: row height in units of inches (default: 2) 
    """
    mfilts = compute_mfilters( model, tkerns=tkerns)
    NF = mfilts.shape[-1]

    nrows = int(np.ceil(NF/6))
    utils.ss(nrows,6, rh=rh)
    for ii in range(NF):
        plt.subplot(nrows,6,ii+1)
        if flip:
            utils.imagesc(np.flip(mfilts[:, :max_lags, ii],axis=1), axis_labels=axis_labels)
        else:
            utils.imagesc(mfilts[:, :max_lags, ii], axis_labels=axis_labels)
    plt.show()
# END plot_mfilters()


def plot_bfilters( model, tkerns=None, flip=True, axis_labels=False, max_lags=12, rh=2 ):
    """
    Plots bfilters (second-layer binocular filters) of sico models, using compute_bfilters to calcuate. 

    Inputs:
        model: sico model to plot monocular filters of
        tkerns: temporal kernels used in sico model (default: None assumes no temporal filters used)
        flip: whether to have lag-0 at bottom and go up (default: flip=True) or opposite
        axis_labels: whether to display axis_labels (default: False=No)
        max_lags: number of lags to trim at (default: 12)
        rh: row height in units of inches (default: 2) 
    """

    kb = compute_bfilters( model, tkerns )
    _, fw, nlags, NF = kb.shape
    nexc = NF-model.networks[0].layers[1].num_inh

    nrows = int(np.ceil(NF/4))
    utils.ss(nrows,4, rh=rh)
    #m = np.max(abs(kb))
    for ii in range(NF):
        plt.subplot(nrows,4,ii+1)
        if flip:
            utils.imagesc( np.fliplr(kb[:,:,:max_lags,ii].reshape([-1,max_lags])), axis_labels=axis_labels )
        else:
            utils.imagesc( kb[:,:,:max_lags,ii].reshape([-1,max_lags]) )
        plt.plot(np.ones(2)*(fw-0.5),[-0.5,max_lags-0.5],'k')
        if ii < nexc:
            plt.title("EXC %d"%ii)
        else:
            plt.title("%d: INH %d "%(ii, ii-nexc))
    plt.show()
# END plot_bfilters()


#### NEW CLONE UTILS ####
def clone_model_selection( LLs, thresh=0.99, LLthresh=None, verbose=False ):
    """
    Model selection from regularization -- find smallest regularization value within thresh of max
    This is made for first-stage clone-sico models that have an ordering (from small to big)
    Choses earliest model that has LL greater than threshold

    Input:
        LLs: LL list for all the clones
        thresh: fraction of max that threshold is selected for (default: 0.99)
        LLthresh: explicit LL-threshold: overrides value of thresh, but default=None
        verbose: to print inner workings or not
    Returns:
        selection: index of clone that fits criteria
    """
    if LLthresh is None:
        LLthresh = thresh*max(LLs)
    a = np.where(LLs >= LLthresh)[0]
    selection = min(a)
    if verbose:
        print("  %d LLs meet criteria (%0.5f out of %0.5f)"%(len(a), LLthresh, max(LLs)))
        print("  Using %d: (max at %d out of %d)"%(selection, np.argmax(LLs), len(LLs)))
    return selection
# END model_selection()


def reconstitute_bmodel( clone_mod, cc, verbose=True ):
    """
    SiCo-model specific: makes reduced single-neuron model from larger-scale clone model,
    which generates many different instances of a single neuron model.

    Args:
        clone_mod: clone-sico model with many instances of single neuron models
        cc: which clone to take
        verbose: whether to suppress output (default: verbose=True)
    
    Returns:
        single_mod: sico single-neuron model
    """
    ### start with exact copy 
    import torch
    clone_mod = clone_mod.to(torch.device('cpu'))
    num_networks = len(clone_mod.ffnet_list)
    stim_net = deepcopy(clone_mod.ffnet_list[0])
    
    # Reconstitute mask
    NF = clone_mod.networks[0].layers[-1].input_dims[0]
    NX = clone_mod.networks[0].layers[0].input_dims[1]
    NXmod = clone_mod.networks[0].layers[-1].input_dims[1]
    NC = clone_mod.networks[-1].layers[0].output_dims[0]
    numE = NF - clone_mod.networks[0].layers[-2].num_inh
    mask = deepcopy(clone_mod.networks[0].layers[-1].mask.detach().numpy()).reshape([NF, NXmod, NC])
    
    # Extract how many filters getting pulled
    ne =  int(np.sum(mask[:numE,0,cc]))
    ni =  int(np.sum(mask[numE:,0,cc]))
    if verbose:
        print( "Model extraction: %d exc, %d inh filters"%(ne, ni))

    # modification of binocular layer
    stim_net['layer_list'][1]['num_filters'] = ne+ni
    stim_net['layer_list'][1]['num_inh'] = ni

    # modification of readout layer
    stim_net['layer_list'][2]['layer_type'] = 'normal'
    stim_net['layer_list'][2]['num_filters'] = 1
    #del stim_net['layer_list'][2]['mask']
    if num_networks == 1:
        
        single_mod = NDN( ffnet_list=[stim_net] )
    else:
        
        drift_net = deepcopy(clone_mod.ffnet_list[1])
        drift_net['layer_list'][0]['num_filters'] = 1
        
        comb_net = deepcopy(clone_mod.ffnet_list[2])
        comb_net['layer_list'][0]['layer_type'] = 'normal'
        comb_net['layer_list'][0]['num_filters'] = 1
    
        single_mod = NDN( ffnet_list=[stim_net, drift_net, comb_net] )
        single_mod.networks[1].layers[0].weight.data[:,0] = \
            clone_mod.networks[1].layers[0].weight.data[:,cc].clone()

    # Copy stim-processing in layer-0
    single_mod.networks[0].layers[0] = deepcopy(clone_mod.networks[0].layers[0])
    # Get relevant filters in layer-1
    non_zeroBs = np.where(mask[:,0,cc] > 0)[0]
    single_mod.networks[0].layers[1].weight.data = deepcopy(
        clone_mod.networks[0].layers[1].weight.data[:, non_zeroBs]).clone()
    if single_mod.networks[0].layers[1].output_norm is not None:
        single_mod.networks[0].layers[1].output_norm.running_mean.data = \
            clone_mod.networks[0].layers[1].output_norm.running_mean.data[non_zeroBs].clone()
        single_mod.networks[0].layers[1].output_norm.running_var.data = \
            clone_mod.networks[0].layers[1].output_norm.running_var.data[non_zeroBs].clone()
        single_mod.networks[0].layers[1].output_norm.weight.data = \
            clone_mod.networks[0].layers[1].output_norm.weight.data[non_zeroBs].clone()
        single_mod.networks[0].layers[1].output_norm.bias.data = \
            clone_mod.networks[0].layers[1].output_norm.bias.data[non_zeroBs].clone()
    # get relevant input weights in readout
    bweights = clone_mod.networks[0].layers[2].weight.data.clone().reshape([NF, NXmod, NC])
    single_mod.networks[0].layers[2].weight.data = deepcopy(bweights[non_zeroBs,:, cc].reshape([-1,1]))
    single_mod.eval()
    return single_mod
# END reconstitute_bmodel()


def bmp_check( mod, dataset, LLs=None, num_cells=None, valset=None, cell_list=None ):
    """
    calculate binocular model performance (pred power and disparity power) for top num_cells in clone-model
    it uses LLs to sort from top, but also can explicitly enter 'cell_list', which supercedes
    cell_list trumps everything else

    Args:
        mod: clone binocular model with lots of outputs
        dataset: dataset that can push through mod
        LLs: LLs of the mod. Should be able to generate internally, but would need null models
        num_cells: how many of the top cells/outputs to calculate with
        valset: whether to use 'a' or 'b' (devault is 'b')
        cell_list: superceded ordering by LLs, and just says which ccs to calculate performance for

    Returns:
        pps: predictive powers of whole clone array, but non-zero for only cells that were computed here
        dps: disparity-predictive powers for same deal as pps
        cell_order: list of cells that were probed
    """
    import torch

    if valset is None:
        valset='b'
        #print('  Testing valset b')
    mod = mod.to(torch.device('cpu'))
    NF = mod.networks[0].layers[-1].input_dims[0]
    NXmod = mod.networks[0].layers[-1].input_dims[1]
    NC = mod.networks[-1].layers[0].output_dims[0]
    numE = NF - mod.networks[0].layers[-2].num_inh
    mask = deepcopy(mod.networks[0].layers[-1].mask.detach().numpy()).reshape([NF, NXmod, NC])
    
    if num_cells is None:
        num_cells = NC
    if cell_list is None:
        assert LLs is not None, "Need to pass in LLs"
        assert len(LLs) == NC, "LLs size mismatch given model"
        cell_order = np.argsort(LLs)[range(NC-1, -1, -1)][:num_cells]
    else:
        if utils.is_int(cell_list):
            cell_list = [cell_list]
        cell_order = cell_list
        num_cells = len(cell_list)
        if LLs is None:
            LLs = np.zeros(NC)-1
    rs = mod(dataset[:])  # predicted responses acriss time
    
    pps = np.zeros([NC])
    dps = np.zeros([NC,2])
    
    for ii in range(num_cells):
        cc = cell_order[ii]
        bmp = binocular_model_performance( 
            data=dataset, cell_n=0, Rpred=rs[:,cc].detach(), valset=valset, verbose=False )
        # Compute aspects of this model
        ne =  int(np.sum(mask[:numE,0,cc]))
        ni =  int(np.sum(mask[numE:,0,cc]))
        pps[cc] = bmp['pred_powers'][0]
        dps[cc,:] = deepcopy(bmp['disp_pred_powers'][:2])
        if cell_list is not None:
            print("%3d (%d,%d):\tpp = %6.4f  dp3 = %6.4f"%(cc, ne, ni, pps[cc], dps[cc,1]))
        else:
            print("%3d %7.5f (%d,%d):\tpp = %6.4f  dp3 = %6.4f"%(cc, LLs[cc], ne, ni, pps[cc], dps[cc,1]))
    return pps, dps, cell_order


def subsample_mask( numM, numB, alpha, conv_width=21, flatten=True, resample=False ):
    """
    Generate mask to give binocular filters (numB) access to some number of monocular filters
    Args
        numM: size of monocular pool
        numB: number of binocular filters
        alpha: how many monocular filters each gets to sample 
    
    Returns:
        mask that is [numM, conv_width, numB] with alpha ones in each row (randomly selected)
    """
    mask = np.zeros([numM, conv_width, numB])
    
    if resample:
        for ii in range(numB):
            a = np.argsort(np.random.rand(numM))
            mask[a[:alpha], :, ii] = 1.0
    else:  # make more distributed by going through order explicitly once each time -- each filter used same amt
        filt_sample = []
        for ii in range(int(np.ceil(alpha*numB/numM))):
            filt_sample = np.concatenate( (filt_sample, np.argsort(np.random.rand(numM)))).astype(int)
        pos = 0
        for ii in range(numB):
            mask[filt_sample[pos+np.arange(alpha)], :, ii] = 1.0
            pos += alpha
            
    if flatten:
        return mask.reshape([-1, numB])
    else:
        return mask
    
    
def EImask( numEIavail, numE, numI, num_filters, width=36, flatten=True, resample=False ):
    
    mask = np.zeros([numEIavail*2, width, num_filters])
    mask[:numEIavail, :, :] = subsample_mask( 
        numEIavail, num_filters, numE, conv_width=width, flatten=False, resample=resample)
    if numI > 0:
        mask[numEIavail:, :, :] = subsample_mask( 
            numEIavail, num_filters, numI, conv_width=width, flatten=False, resample=resample)

    if flatten:
        return mask.reshape([-1, num_filters])
    else:
        return mask
    
    
def convert2ST( mod0, temporal_basis=None, new_lags=16 ):
    """
    Convert temporal-basis-based model to spatiotemporal with new_lags
    """
    from NDNT.modules.layers.bilayers import BiConvLayer1D
    import torch

    assert temporal_basis is not None, "need to enter temporal basis"
    converted_model = deepcopy(mod0)
    mks = compute_mfilters( mod0, temporal_basis )
    #print(mks.shape)
    mfilt_layer_pars = deepcopy(mod0.ffnet_list[0]['layer_list'][0])
    mfilt_layer_pars['input_dims'][-1] = new_lags
    mfilt_layer_pars['filter_dims'][-1] = new_lags
    # ALSO CONVERT relevant parts of ffnet_list
    converted_model.ffnet_list[0]['layer_list'][0]['input_dims'][-1] = new_lags
    converted_model.ffnet_list[0]['layer_list'][0]['filter_dims'][-1] = new_lags

    Mlayer = BiConvLayer1D(**mfilt_layer_pars)
    Mlayer.weight.data = torch.tensor(mks[:,:new_lags,:].reshape([-1, mks.shape[-1]]))
    converted_model.networks[0].layers[0] = Mlayer
    return converted_model


def bmodel_regpath(
    model, train_ds, val_ds, reg_type=None, reg_vals=None, ffnet_target=0, layer_target=0, 
    nullLL=None, couple_xt=0.5, hard_reset=True, extended_loop=True, average_pool=1,
    verbose=True, device=None):
    """
    regularization-path for NDN model -- standard I think other than using specific details of the model

    Args:
        model: model to be regularized
        train_ds: dataset to be used for training (generic, on device already)
        val_ds: dataset for validation, complementary to train_ds
        reg_type: type of regularization (required)
        reg_vals: list of regularization values (default: [1e-6, 0.0001, 0.001, 0.01, 0.1])
        ffnet_target: which ffnetwork that containes layer to regularize (default: 0)
        layer_target: which layer to regularize (default 0)
        nullLL: give LL-null of model so that can correctly compute LLs relative to null model. Default is
            None, where it will use the embedded null_adjusted=True of eval_models
        couple_xt: whether to make d2t=d2x/2 coupled to 'd2x' when its the reg_type (default: 0.5)
        extended_loop: whether to continue with reg_vals of factors of 10 if best value is last (default: True)
        hard_reset: when set to 'True': will zero out drift model to force model to fit longer (default: False)
        average_pool: if population model, which fraction of LLs to average over to determine 
            the best-reg (default: 1=all)
        verbose: self-explanatory (default: True)
        device: which device to use (default cuda:0)

    Returns:
        model_select: selected model
        reg_val: selected reg value from the reg_vals list
        export_dict: dictionary with detailed information of reg path
    """
    import torch
    from NDNT.utils import fit_lbfgs

    assert reg_type is not None, "ERROR: Must specify reg_type"
    assert average_pool <= 1, "average_pool mis-set"
    MAX_REGS = 16

    if reg_vals is None:
        reg_vals = [1e-6, 0.0001, 0.001, 0.01, 0.1, 1]
    if device is None:
        device=torch.device('cuda:0')

    num_regs = len(reg_vals)
    Rmods = []
    LLsR = np.zeros(num_regs)
    LLcells = []
    if verbose:
        print( "Reg path %s:"%reg_type, reg_vals)
    if isinstance(reg_vals, list):
        reg_vals = np.array(reg_vals, dtype=np.float32)

    ## This is a loop so that regularization path can be extended until LLs go down
    LLbest, rr = 0, 0
    #for rr in range(num_regs):
    while rr < np.minimum(len(reg_vals), MAX_REGS):

        sico_iter = deepcopy(model)
        sico_iter.networks[ffnet_target].layers[layer_target].reg.vals[reg_type] = reg_vals[rr]
        
        if (reg_type == 'd2x') and (couple_xt > 0):  # then couple d2t
            sico_iter.networks[ffnet_target].layers[layer_target].reg.vals['d2t'] = reg_vals[rr]*couple_xt
            
        if hard_reset:
            #sico_iter.networks[1].layers[0].weight.data[:,:] = 0.0
            sico_iter.networks[ffnet_target].layers[layer_target].weight.data[:,:] *= 0.9
            
        sico_iter = sico_iter.to(device)
        fit_lbfgs( sico_iter, train_ds[:], verbose=False)
        
        if nullLL is None:
            LLs = sico_iter.eval_models(val_ds[:], null_adjusted=True)
        else:
            LLs = nullLL-sico_iter.eval_models(val_ds[:])
        sico_iter = sico_iter.to(torch.device('cpu'))
        LLcells.append(deepcopy(LLs))

        if len(LLs) == 1:
            LL = LLs[0]
        else:
            if average_pool == 1:
                LL = np.mean(LLs)
            else:
                num_skip = int(np.round(len(LLs)*(1-average_pool)))
                LL = np.mean(np.sort(LLs)[num_skip:])
                
        Rmods.append(deepcopy(sico_iter))
        LLsR[rr] = LL
        if (LL > LLbest):
            print( "  %s-%d: %9.6f **"%(reg_type, rr, LL))
            #BU.plot_mfilters(sico_iter, TB)
            LLbest = LL
            if (rr == len(reg_vals)-1) and extended_loop:
                reg_vals = np.concatenate( (reg_vals, [reg_vals[-1]*10]) )
                LLsR = np.concatenate( (LLsR, [0]) ) # so loop continues, adding value
                if verbose:
                    print( '    Adding additional reg_val:', reg_vals[-1] )
        else:
            print( "  %s-%d: %9.6f"%(reg_type, rr, LL))
        rr += 1

    #if thresh < 1.0:
    #    reg_select = model_selection( LLsR, thresh, verbose=verbose )
    #else:
    reg_select = np.argmax(LLsR)

    reg_val = reg_vals[reg_select]
    print( "Finished %s: selected reg"%reg_type, reg_val, "(%d)"%reg_select )
    model_select = Rmods[reg_select]
    export_dict = {
        'Rmods': deepcopy(Rmods), 'LLsR': deepcopy(LLsR), "reg_vals": reg_vals,
        'LLcells': LLcells}
    return deepcopy(model_select), reg_val, export_dict
# END bmodel_regpath()    


def sico_ffnetworks( 
        num_mfilters=None, num_clones=None, numBE=None, numBI=None, mask=True,  # must enter
        monoc_width=21, binoc_width=13, num_tkerns=8, NX=36, pos_constraint=False,  # default values
        XregM=0.0001, CregM=0.001, MregB=0.001, LOCregR=0.1):  # regularization

    from NDNT.modules.layers import BiConvLayer1D, NDNLayer, ConvLayer, MaskLayer, ChannelLayer
    from NDNT.networks import FFnetwork

    monoc_basis_par = BiConvLayer1D.layer_dict( 
        input_dims=[1, 2*NX, 1, num_tkerns], num_filters=num_mfilters, 
        filter_dims=[1, monoc_width, 1, num_tkerns],
        norm_type=1, bias=False, initialize_center=True, NLtype='lin',
        #output_norm='batch', window='hamming', 
        reg_vals={'d2x':XregM, 'center':CregM })
        #reg_vals={'d2x':XregM, 'd2t':TregM, 'center':CregM })

    bfilt_par = ConvLayer.layer_dict( 
        num_filters=(numBE+numBI), num_inh=numBI, filter_dims=binoc_width, 
        norm_type=1, pos_constraint=pos_constraint,
        window='hamming', #output_norm='batch', #padding='valid',
        bias=False, initialize_center=True,
        NLtype='relu', reg_vals={'max_space':MregB})

    if mask is None:
        readout_par = NDNLayer.layer_dict(
            num_filters=num_clones, bias=False, initialize_center=True, pos_constraint=True,
            NLtype='lin', reg_vals={'glocalx': LOCregR})
    else:
        readout_par = MaskLayer.layer_dict(
            num_filters=num_clones, bias=False, initialize_center=True, pos_constraint=True,
            NLtype='lin', reg_vals={'glocalx': LOCregR})

    if num_clones > 1:
        comb_layer = ChannelLayer.layer_dict(num_filters=num_clones, NLtype='softplus', bias=False)
    else: 
        comb_layer = NDNLayer.layer_dict(num_filters=1, NLtype='softplus', bias=False)
    comb_layer['weights_initializer'] = 'ones'

    stim_net = FFnetwork.ffnet_dict( xstim_n='stim', layer_list = [monoc_basis_par, bfilt_par, readout_par])
    net_comb = FFnetwork.ffnet_dict( xstim_n=None, ffnet_n=[0,1], layer_list=[comb_layer], ffnet_type='add')

    return stim_net, net_comb


def clone_path_prepare_data(ee, cc, TB, nlags=12, clone_model=None, num_clones=None, check_performance=0, 
                            dirname=[], datadir=[], device=None):
    """
    Load dataset and clone model from respective directories for regularization
    So also translates models into 
    Uses already-defined datadir and dirname respectively
    
    Args:
        ee: expt number starting with zero
        cc: cell number starting with zero
        TB: temporal bases used to fit clones
        
    Returns:
        data_clone: dataset made to fit clone models
        data1: dataset made to fit single copy of cell
        clone_path_info: dictionary with the following info (if applicable)
            LLnull: null model LL computed by fitting drift model
            drift_terms: drift terms for number of clones specified
            base_model: base clone model with spatiotemporal filters and nlags, if included
            LLs: null-adjusted LLs of clone model, if included
    """
    import torch
    from NTdatasets.generic import GenericDataset
    from NTdatasets.cumming.binocular import binocular_single
    from NDNT.modules.layers.ndnlayer import NDNLayer
    
    if device is None:
        device = torch.device('cpu')

    Dreg = 0.5
    if clone_model is None:
        if num_clones is None: # then assume its a file loading
            filename = "E%sc%sclones.ndn"%(utils.filename_num2str(ee), utils.filename_num2str(cc))
            base_model = NDN.load_model_zip(dirname+filename)
            base_model.networks[0].layers[2].mask_is_set = True
        else:
            base_model = None
            NX = 36
    else:
        base_model = clone_model
    if base_model is not None:
        num_clones = base_model.networks[-1].layers[0].output_dims[0]
        NX = base_model.networks[0].layers[2].input_dims[1]
        
    old_nlags, num_tk = TB.shape

    # Make datasets
    #data_clone = binocular_single( expt_num=ee+1, datadir=datadir, time_embed=2, skip_lags=1, num_lags=old_nlags )
    data_clone = binocular_single( expt_num=ee, datadir=datadir, time_embed=2, skip_lags=1, num_lags=old_nlags )
    if nlags is not None:
        data1 = binocular_single( expt_num=ee, datadir=datadir, time_embed=2, skip_lags=1, num_lags=nlags, verbose=False )
    else:
        data1 = binocular_single( expt_num=ee, datadir=datadir, time_embed=2, skip_lags=1, num_lags=old_nlags, verbose=False )
        
    robs = deepcopy(data1.robs)
    dfs = deepcopy(data1.dfs)
    data_clone.robs = np.repeat(deepcopy(robs[:,[cc]]), num_clones, axis=1)
    data_clone.dfs = np.repeat(deepcopy(dfs[:,[cc]]), num_clones, axis=1)
    data1.set_cells(cc)

    # Add drift capacities and calc drift model
    block_length = 3600  # 60 sec
    anchors = np.arange(0,data_clone.NT, block_length)
    drift_tents = data_clone.design_matrix_drift(
        data_clone.NT, anchors, zero_left=False, zero_right=True, const_right=False)
    data_clone.Xdrift = torch.tensor(drift_tents, dtype=torch.float32)
    data1.Xdrift = torch.tensor(drift_tents, dtype=torch.float32)
    NA = drift_tents.shape[1]

    # Make tent-basis stim for clone model LL extraction (with-tent-basis thing)
    if (base_model is not None) or (nlags is None):
        Xstim = torch.einsum('bxt,tf->bxf', 
                             data_clone.stim.reshape([-1,2*NX,old_nlags]), 
                             torch.tensor(TB, dtype=torch.float32) )
        data_clone.stim = Xstim.reshape([len(data_clone), -1])
        data_clone.stim_dims[-1] = num_tk

    # Calculate LLs because why-not: need it but dont need drift models after that
    train_ds = GenericDataset(data_clone[data_clone.train_inds], device=device)

    drift_pars1N = NDNLayer.layer_dict( 
        input_dims=[1,1,1,NA], num_filters=num_clones, bias=False, norm_type=0, 
        NLtype='softplus', reg_vals={'d2t': Dreg, 'bcs':{'d2t':0} } )
    drift_mod = NDN(layer_list = [drift_pars1N], loss_type='poisson').to(device)
    drift_mod.networks[0].xstim_n = 'Xdrift'
    utils.fit_lbfgs( drift_mod, train_ds[:], verbose=False)
    drift_mod = drift_mod.to(device)
    LLnull = drift_mod.eval_models(data_clone[data1.val_indsA], null_adjusted=False)[0]
    print('  c%d: LLnull = %0.6f'%(cc, LLnull))
    drift_terms = drift_mod.networks[0].layers[0].weight.data.clone().cpu()

    clone_path_info = {'drift_terms': drift_terms, 'LLnull': LLnull, 'num_lags': nlags}
    if base_model is not None:
        # Calculate LLs of original model before converting 
        LLs0 = LLnull-base_model.eval_models(data_clone[data1.val_indsA], null_adjusted=False)
        print('  LLs0 original clone mean:', np.mean(LLs0))
        del train_ds
        torch.cuda.empty_cache()    
        
        if check_performance > 0:
            _ = bmp_check( base_model, data_clone, LLs0, check_performance ) 
        
        # Convert dataset back to handle spatiotemporal
        if nlags is not None:
            data_clone.stim = data1.stim.clone()
            data_clone.stim_dims[-1] = nlags

            base_model = convert2ST( base_model, TB, nlags )  # note this will not be normalized correctly
        else:
            data1.stim = data_clone.stim.clone()
            data1.stim_dims[-1] = data_clone.stim_dims[-1]
            
        clone_path_info['base_model'] = base_model 
        clone_path_info['LLs'] = LLs0
    
    return data_clone, data1, clone_path_info


def spatiotemporal_box_std( filts, t_edge, x_edge, filt_ns=None, to_plot=True, display_cc=None, subplot_info=None):
    nx, nt, nf = filts.shape
    if filt_ns is None:
        filt_ns = range(nf)
    ks = deepcopy(filts[:,:,filt_ns])
    m = np.max(abs(ks))
    tlags = np.arange(nt-t_edge, nt)
    #print(tlags)
    box1 = np.arange(x_edge)
    box2 = np.arange(nx-x_edge, nx)
    #print(box1,box2)
    stds = np.mean([np.std(ks[:,tlags,:][box1,:,:]), np.std(ks[:,tlags,:][box2,:,:])])
    #print(stds,m)
    if to_plot:
        import matplotlib.patches as patches
        if display_cc is None:
            utils.ss(1,6)
            for ii in range(len(filt_ns)):
                ax = plt.subplot(1,6,ii+1)
                utils.imagesc(ks[...,ii], max=m)
                ax.add_patch(patches.Rectangle(
                    (box1[0]-0.5,tlags[0]-0.5), x_edge, t_edge, linewidth=1, edgecolor='r', facecolor='none'))
                ax.add_patch(patches.Rectangle(
                    (box2[0]-0.5,tlags[0]-0.5), x_edge, t_edge, linewidth=1, edgecolor='r', facecolor='none'))
            plt.show()

    return stds/m


def spatiotemporal_std_display( filt2d, t_edge, x_edge, ax_handle, display_norm=None ):
    import matplotlib.patches as patches
    if display_norm is None:
        display_norm = np.max(abs(filt2d))
    nx,nt = filt2d.shape
    tlags = np.arange(nt-t_edge, nt)
    box1 = np.arange(x_edge)
    box2 = np.arange(nx-x_edge, nx)

    utils.imagesc(filt2d, max=display_norm)
    ax_handle.add_patch(patches.Rectangle(
        (box1[0]-0.5,tlags[0]-0.5), x_edge, t_edge, linewidth=1, edgecolor='r', facecolor='none'))
    ax_handle.add_patch(patches.Rectangle(
        (box2[0]-0.5,tlags[0]-0.5), x_edge, t_edge, linewidth=1, edgecolor='r', facecolor='none'))


def smoothness_select0( reg_info, LLthresh=None, to_plot=True, tbasis=None ):
    """
    Selects best smoothness based on where transition occurs rather than maximizing LL.
    Will automaticall display unless display_n is set and < 0
    """        
    LLs = reg_info['LLsR']
    num_regs = len(LLs)
    LLpos = np.argmax(LLs)
    if LLthresh is None:
        LLthresh = np.max(LLs)*0.90
        print( "  No LLthresh set: using 90\% of max: ", LLthresh )
    a = np.where(LLs[LLpos:] > LLthresh)[0]
    #print(' check', LLs, LLpos)
    if len(a) == 0:
        print('Warning: did not achieve LLthresh90 in regpath')
        print( "  LLthresh %0.4f > %0.4f"%(LLthresh, np.max(LLs)) )
        a = [0]
    #print(a, np.max(a), a[-1])
    #print(np.where(LLs[LLpos:] > LLthresh*np.max(LLs))[0])
    #print(np.where(LLs[LLpos] > LLthresh*np.max(LLs))[0][-1])
    #print(LLpos, np.max( np.where(LLs[LLpos] > LLthresh*np.max(LLs))[0] ), LLthresh*np.max(LLs))
    reg = LLpos + a[-1]
    Xreg = reg_info['reg_vals'][reg]

    if to_plot:
        #print( "Threshold: %0.4f"%LLthresh)
        ws = []
        for ii in range(num_regs):
            if tbasis is None:
                ws.append(reg_info['Rmods'][ii].get_weights())
            else:
                ws.append(compute_mfilters(reg_info['Rmods'][ii], tbasis))
        NF = ws[0].shape[-1]

        utils.ss(NF, 4, rh=2.8)
        for ii in range(NF):
            #plt.subplot(NF,6,ii*6+1)
            #utils.imagesc(kmasks[ii])
            plt.subplot(NF,4,ii*4+1)
            utils.imagesc(ws[0][..., ii])
            plt.title('Lowest d2x-reg')
            plt.subplot(NF,4,ii*4+2)
            utils.imagesc(ws[-1][..., ii]) 
            plt.title('Highest d2x-reg')
            plt.subplot(NF,4,ii*4+3)
            utils.imagesc(ws[reg][..., ii])
            plt.title('Chosen d2x-reg')

        plt.subplot(NF,4,4)
        plt.plot(reg_info['LLsR'],'b')
        plt.plot(reg_info['LLsR'],'b.')
        plt.plot(reg, reg_info['LLsR'][reg],'go')
        plt.title('LLs')
        ys = plt.ylim()
        xs = plt.xlim()
        plt.plot(np.ones(2)*reg, ys, 'c--')
        plt.plot(xs, np.ones(2)*LLthresh,'r--' )
        plt.ylim(ys)
        plt.xlim(xs)
        plt.show()
    print('d2xt reg:', Xreg)
    return deepcopy(reg_info['Rmods'][reg])
# END smoothness_select0

def smoothness_select( reg_info, t_edge=6, x_edge=6, display_n=None, tbasis=None ):
    """
    Selects best smoothness based on where transition occurs rather than maximizing LL.
    Will automaticall display unless display_n is set and < 0
    """        
    num_regs = len(reg_info['LLsR'])
    stds = np.zeros(num_regs)
    ws = []
    for ii in range(num_regs):
        if tbasis is None:
            ws.append(reg_info['Rmods'][ii].get_weights())
        else:
            ws.append(compute_mfilters(reg_info['Rmods'][ii], tbasis))
        stds[ii] = spatiotemporal_box_std(ws[ii], t_edge, x_edge, to_plot=False ) 
    thresh = np.mean([max(stds),min(stds)]) 
    reg = np.where(stds < thresh)[0][0]
    Xreg = reg_info['reg_vals'][reg]
    
    # Cancel display by making display_n < 0
    if display_n is not None:
        if display_n < 0:
            print('d2xt reg:', Xreg)
            return deepcopy(reg_info['Rmods'][reg])

    # STATUS DISPLAY
    if display_n is None:
        NF = ws[0].shape[-1]
        utils.ss(NF,5, rh=3)
        for ii in range(NF):
            ax = plt.subplot(NF,5,1+ii*5)
            spatiotemporal_std_display( ws[0][..., ii], t_edge, x_edge, ax )
            plt.title('Lowest d2x-reg')
            ax = plt.subplot(NF,5,2+ii*5)
            spatiotemporal_std_display( ws[-1][..., ii], t_edge, x_edge, ax )
            plt.title('Highest d2x-reg')
            ax = plt.subplot(NF,5,3+ii*5)
            spatiotemporal_std_display( ws[reg][..., ii], t_edge, x_edge, ax )
            plt.title('Chosen d2x-reg')
    else:
        utils.ss(1,5, rh=3)
        ax = plt.subplot(1,5,1)
        spatiotemporal_std_display( ws[0][..., display_n], t_edge, x_edge, ax )
        plt.title('Lowest d2x-reg')
        ax = plt.subplot(1,5,2)
        spatiotemporal_std_display( ws[-1][..., display_n], t_edge, x_edge, ax )
        plt.title('Highest d2x-reg')
        ax = plt.subplot(1,5,3)
        spatiotemporal_std_display( ws[reg][..., display_n], t_edge, x_edge, ax )
        plt.title('Chosen d2x-reg')
        NF = 1

    plt.subplot(NF,5,4)
    plt.plot(stds,'b')
    plt.plot(stds,'b.')
    plt.plot(reg,stds[reg],'go')
    xs = plt.xlim()
    plt.plot(xs, np.ones(2)*thresh,'r--')
    plt.title('Box stdevs')

    plt.subplot(NF,5,5)
    plt.plot(reg_info['LLsR'],'b')
    plt.plot(reg_info['LLsR'],'b.')
    plt.plot(reg, reg_info['LLsR'][reg],'go')
    plt.title('LLs')
    xs = plt.xlim()
    plt.show()
    print('d2xt reg:', Xreg)
    return deepcopy(reg_info['Rmods'][reg])


def mask_filter_noise( k, area_fraction=0.4, thresh=None, verbose=False ):
    """
    Generates mask over 2-d filter that includes all points greater than 0.1 of the filter max
    Returns the standard dev of the filter outside of the mask, and possibly the mask too
    """
    from skimage import measure
    from ColorDataUtils.RFutils import get_mask_from_contour  

    kabs = np.pad( abs(k)/np.max(abs(k)), 1) # Padding to get rid of edge effects with contours
    A = np.prod(k.shape[:2])

    if thresh is None:
        PREC = 0.05
        MAX_ITER = 10
        # Search for threshold to achieve area_fraction not occupied by mask
        mfrac, niter = 0, 0
        thresh, inc = 1, 0.5
        while ((mfrac < area_fraction-PREC) | (mfrac > area_fraction)) & (niter < MAX_ITER):
            if mfrac < area_fraction:
                thresh += -inc
            else:
                thresh += inc

            contours = measure.find_contours(kabs, thresh)
            kmask = np.zeros(kabs.shape[:2])
            for contour in contours:
                #plt.plot(contour[:, 0], contour[:, 1], linewidth=2)
                kmask += get_mask_from_contour(kabs, contour)
            #plt.show()
            #kmask[kmask > 1] = 1.0
            kmask = kmask[1:-1,:][:,1:-1]  # trim kmask now that padding is done
            mfrac = np.sum(kmask > 0)/A
            niter += 1
            inc *= 0.5
            if verbose:
                print( "  Iter %2d: thresh %0.4f area: %0.3f"%(niter, thresh, mfrac))
    else:
        contours = measure.find_contours(kabs, thresh)
        kmask = np.zeros(kabs.shape[:2])
        for contour in contours:
            #plt.plot(contour[:, 0], contour[:, 1], linewidth=2)
            kmask += get_mask_from_contour(kabs, contour)
        #plt.show()
        kmask = kmask[1:-1,:][:,1:-1]  # trim kmask now that padding is done
    kmask[kmask > 1] = 1.0
    
    #utils.imagesc(kmask)
    #plt.show()
    return kmask


def smoothness_select_contour( reg_info, thresh=0.5, to_plot=True, tbasis=None ):
    """
    Selects best smoothness based on where transition occurs rather than maximizing LL 
    threshold now corresponds to area-fraction as applied by mask
    """     
    num_regs = len(reg_info['LLsR'])
    stds = np.zeros(num_regs)

    # First, determin masks on most-smoothed filter
    ws = []
    for ii in range(num_regs):
        if tbasis is None:
            ws.append(reg_info['Rmods'][ii].get_weights())
        else:
            ws.append(compute_mfilters(reg_info['Rmods'][ii], tbasis))

    NF = ws[0].shape[-1]
    kmasks = []
    for jj in range(NF):
        kmasks.append( mask_filter_noise(ws[-1][..., jj], area_fraction=thresh) )

    for ii in range(num_regs):
        for jj in range(NF): 
            k = ws[ii][..., jj]
            stds[ii] += np.std( k[kmasks[jj] == 0] )
    stds *= 1/NF
    #print(stds, (max(stds)+min(stds))/2 )
    selection_thresh = (max(stds)+min(stds))/2
    #reg_min = np.where(stds > selection_thresh)[0][-1]   # wants before smoothness gets crazy
    reg_min = np.where(stds < selection_thresh)[0][0] # wants after smoothness transition
    # This (above) is the minimum reg -- now pick best LL better than this:
    reg = np.argmax(reg_info['LLsR'][reg_min:]) + reg_min

    Xreg = reg_info['reg_vals'][reg]

    if to_plot:
        utils.ss(NF, 6, rh=2.8)
        for ii in range(NF):
            plt.subplot(NF,6,ii*6+1)
            utils.imagesc(kmasks[ii])
            plt.subplot(NF,6,ii*6+2)
            utils.imagesc(ws[0][..., ii])
            plt.title('Lowest d2x-reg')
            plt.subplot(NF,6,ii*6+3)
            utils.imagesc(ws[-1][..., ii]) 
            plt.title('Highest d2x-reg')
            plt.subplot(NF,6,ii*6+4)
            utils.imagesc(ws[reg][..., ii])
            plt.title('Chosen d2x-reg')

        plt.subplot(NF,6,5)
        plt.plot(stds,'b')
        plt.plot(stds,'b.')
        plt.plot(reg,stds[reg],'go')
        xs = plt.xlim()
        plt.plot(xs, np.ones(2)*selection_thresh,'r--')
        plt.xlim(xs)
        plt.title('Box stdevs')

        plt.subplot(NF,6,6)
        plt.plot(reg_info['LLsR'],'b')
        plt.plot(reg_info['LLsR'],'b.')
        plt.plot(reg, reg_info['LLsR'][reg],'go')
        plt.title('LLs')
        ys = plt.ylim()
        plt.plot(np.ones(2)*reg_min, ys, 'c--')
        plt.ylim(ys)
        plt.show()
    print('d2xt reg:', Xreg)
    return deepcopy(reg_info['Rmods'][reg])


def smoothness_select_contour2( reg_info, thresh=0.5, to_plot=True ):
    """
    Selects best smoothness based on where transition occurs rather than maximizing LL 
    threshold now corresponds to area-fraction as applied by mask
    """     
    num_regs = len(reg_info['LLsR'])
    stds = np.zeros(num_regs)

    # First, determin masks on most-smoothed filter
    ws = []
    NF = reg_info['Rmods'][0].networks[0].layers[0].num_filters
    area_frac = np.zeros([num_regs, NF])
    for ii in range(num_regs):
        ws.append( reg_info['Rmods'][ii].get_weights() )
        for jj in range(NF):
            m = mask_filter_noise(ws[ii][..., jj], thresh=0.2)
            area_frac[ii,jj] = np.sum(m)/np.prod(m.shape)
            
    #print(area_frac)
    area_frac = np.min(area_frac, axis=1)
    print(area_frac, np.sum(area_frac < 0.5))
    num_regs= np.sum(area_frac < 0.5)
    NF = ws[0].shape[-1]
    kmasks = []
    for jj in range(NF):
        #kmasks.append( mask_filter_noise(ws[-1][..., jj], area_fraction=thresh) )
        kmasks.append( mask_filter_noise(ws[num_regs-1][..., jj], area_fraction=thresh) )

    for ii in range(num_regs):
        for jj in range(NF): 
            k = ws[ii][..., jj]
            stds[ii] += np.std( k[kmasks[jj] == 0] )
    stds *= 1/NF
    #print(stds, (max(stds)+min(stds))/2 )
    #print(stds)
    #selection_thresh = 0.25*np.max(stds)+ 0.75*np.min(stds)
    selection_thresh = 0.5*np.max(stds)+ 0.5*np.min(stds)
    # Pick best LL after regularization is at least 50% there
    reg_min = np.where(stds < selection_thresh)[0][0] # wants after smoothness transition
    # ORIGINAL
    #reg_min = np.where(stds > selection_thresh)[0][-1]   # wants highest LL after smoothness starts to trans

    # This (above) is the minimum reg -- now pick best LL better than this:
    reg = np.argmax(reg_info['LLsR'][reg_min:]) + reg_min

    Xreg = reg_info['reg_vals'][reg]

    if to_plot:
        utils.ss(NF, 6, rh=2.8)
        for ii in range(NF):
            plt.subplot(NF,6,ii*6+1)
            utils.imagesc(kmasks[ii])
            plt.subplot(NF,6,ii*6+2)
            utils.imagesc(ws[0][..., ii])
            plt.title('Lowest d2x-reg')
            plt.subplot(NF,6,ii*6+3)
            utils.imagesc(ws[-1][..., ii]) 
            plt.title('Highest d2x-reg')
            plt.subplot(NF,6,ii*6+4)
            utils.imagesc(ws[reg][..., ii])
            plt.title('Chosen d2x-reg')

        plt.subplot(NF,6,5)
        plt.plot(stds,'b')
        plt.plot(stds,'b.')
        plt.plot(reg,stds[reg],'go')
        xs = plt.xlim()
        plt.plot(xs, np.ones(2)*selection_thresh,'r--')
        plt.xlim(xs)
        plt.title('Box stdevs')

        plt.subplot(NF,6,6)
        plt.plot(reg_info['LLsR'],'b')
        plt.plot(reg_info['LLsR'],'b.')
        plt.plot(reg, reg_info['LLsR'][reg],'go')
        plt.title('LLs')
        ys = plt.ylim()
        plt.plot(np.ones(2)*reg_min, ys, 'c--')
        plt.ylim(ys)
        plt.show()
    print('d2xt reg:', Xreg)
    return deepcopy(reg_info['Rmods'][reg])
# smoothness_select_contour2()


def binocular_filter_shift( sico0, verbose=True ):
    import torch    
    wB = sico0.get_weights(layer_target=1)
    ni, fw, NBF = wB.shape
    Nout = sico0.networks[-1].output_dims[0]

    #clrs='bgr'
    shifts = np.zeros(NBF, dtype=int) #[0,0,0]
    for ii in range(NBF):
        dist = np.sum(abs(wB[...,ii]),axis=0)
        shifts[ii] = -int(np.round(np.sum((np.arange(fw)-(fw-1)/2)*dist)/np.sum(dist)))
    if verbose:
        print('  Binocular shifts:', shifts)
    
    # Move the torch-weights
    wB = sico0.networks[0].layers[1].weight.data.detach().reshape([ni, fw, NBF])
    wB2 = torch.zeros(wB.shape, dtype=torch.float32)
    for ii in range(NBF):
        wB2[..., ii] = torch.roll(wB[..., ii].clone(), shifts[ii], dims=1)

    wR = sico0.networks[0].layers[2].weight.data.detach().reshape([NBF, -1, Nout])
    wR2 = torch.zeros(wR.shape, dtype=torch.float32)
    for ii in range(NBF):
        wR2[ii,:,:] = torch.roll(abs(wR[ii,:,:].clone()), -shifts[ii], dims=0)
        
    sico1 = deepcopy(sico0)
    sico1.networks[0].layers[1].weight.data = wB2.reshape([-1,NBF])
    sico1.networks[0].layers[2].weight.data = wR2.reshape([-1,Nout])
    return sico1
# END binocular_filter_shift()


def monocular_filter_shift( sico0, verbose=True ):
    """
    Shifts the filters in the monocular layer, and adjusts the filters in the binocular layer as a result.
    """
    import torch    
    wM = sico0.get_weights(layer_target=0)
    fw, nt, NMF = wM.shape
    Nout = sico0.networks[0].layers[1].output_dims[0]

    shifts = np.zeros(NMF, dtype=int) #[0,0,0]
    for ii in range(NMF):
        dist = np.sum(abs(wM[...,ii]),axis=1)
        shifts[ii] = -int(np.round(np.sum((np.arange(fw)-(fw-1)/2)*dist)/np.sum(dist)))
    if verbose:
        print('  Monocular shifts:', shifts)
    
    # Move the torch-weights
    wM = sico0.networks[0].layers[0].weight.data.detach().reshape([fw, nt, NMF])
    wM2 = torch.zeros(wM.shape, dtype=torch.float32)
    for ii in range(NMF):
        wM2[..., ii] = torch.roll(wM[..., ii].clone(), shifts[ii], dims=0)

    wB = sico0.networks[0].layers[1].weight.data.detach().reshape([NMF, -1, Nout])
    wB2 = torch.zeros(wB.shape, dtype=torch.float32)
    for ii in range(NMF):
        wB2[ii,:,:] = torch.roll(abs(wB[ii,:,:].clone()), -shifts[ii], dims=0)
        
    sico1 = deepcopy(sico0)
    sico1.networks[0].layers[0].weight.data = wM2.reshape([-1, NMF])
    sico1.networks[0].layers[1].weight.data = wB2.reshape([-1, Nout])
    return sico1
# END monocular_filter_shift()
