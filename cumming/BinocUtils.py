import numpy as np
import scipy.io as sio
import NDNT.utils as utils
import NDNT.NDNT as NDN
from time import time
from copy import deepcopy

def explainable_variance_binocular( Einfo, resp, indxs, cell_num=None ):
    """This is the explainable variance calculation, and is binocular-specific because of the data structures
    in Einfo only. Note: will return total variance (and a warning) if repeats not present in the dataset."""

    if cell_num is None:
        print( 'Warning: cell-specific repeats not given: using cc=0.')
        cell_num = 1
    else:
        if cell_num == 0:
            print('Warning: must use matlab-style cell numbers (starting at 1)')

    if Einfo.rep_inds is None:
        print( 'No repeats in this dataset.')
        return np.var(resp[indxs]), np.var(resp[indxs])
            
    rep1inds = np.intersect1d(indxs, Einfo.rep_inds[cell_num-1][:,0])
    rep2inds = np.intersect1d(indxs, Einfo.rep_inds[cell_num-1][:,1])
    allreps = np.concatenate((rep1inds, rep2inds), axis=0)

    totvar = np.var(resp[allreps])
    explvar = np.mean(np.multiply(resp[rep1inds]-np.mean(resp[allreps]), resp[rep2inds]-np.mean(resp[allreps]))) 
    
    return explvar, totvar


def predictive_power_binocular( Robs, pred, indxs=None, expl_var=None, Einfo=None, cell_num=1, suppress_warnings=False ):
    """Use Einfo to modify indices used to rep_inds (and can fill in expl_var) -- otherwise not used for anything
    Robs sometimes is already indexed nby indxs, so check that it needs to be indexed. Will be assuming that
    pred is full range -- otherwise no need to pass indx in."""
    
    if indxs is None:
        indxs = np.arange(len(Robs))
    mod_indxs = deepcopy(indxs)
    if Einfo is not None:
        if Einfo.rep_inds is not None:
            rep1inds = np.intersect1d(indxs, Einfo.rep_inds[cell_num-1][:,0])
            rep2inds = np.intersect1d(indxs, Einfo.rep_inds[cell_num-1][:,1])
            allreps = np.concatenate((rep1inds, rep2inds), axis=0)
            mod_indxs = np.intersect1d( mod_indxs, allreps )
        if expl_var is None:
            expl_var,_ = explainable_variance_binocular( Einfo, Robs.numpy(), indxs, cell_num )
            
    
    r1 = deepcopy(Robs[mod_indxs].numpy())
    r2 = pred[mod_indxs].detach().numpy()

    # Now assuming that r (Robs) is length of indxs, and pred is full res
    if expl_var is None:
        # this means repeat info not passed in, so just use total variance
        if ~suppress_warnings:
            print( '  Using total variance for normalization')
        expl_var = np.var(Robs)
    
    explained_power = np.var(r1)-np.mean(np.square(r1-r2))
    
    # calculate other way
    #crosscorr = np.mean(np.multiply(r1-np.mean(r1), r2-np.mean(r2)))
    #print( (crosscorr**2/expl_var/np.var(r2)) )
    return explained_power/expl_var


def disparity_matrix( dispt, corrt=None ):

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

## Not  working yet ##

def disparity_predictions( 
    Einfo, resp, 
    indxs=None, num_dlags=8, fr1or3=None, spiking=True, rectified=True, opt_params=None ):
    """Calculates a prediction of the disparity (and timing) signals that can be inferred from the response
    by the disparity input alone. This puts a lower bound on how much disparity is driving the response, although
    practically speaking will generate the same disparity tuning curves.
    
    Usage: Dpred, Tpred = disparity_predictions( Einfo, resp, indxs, num_dlags=8, spiking=True, rectified=True, opt_params=None )

    Inputs: Indices gives data range to fit to.
    Outputs: Dpred and Tpred will be length of entire experiment -- not just indxs
    """

    # Process disparity into disparty and timing design matrix
    dmat = disparity_matrix( Einfo.dispt, Einfo.corrt )
    ND2 = dmat.shape[1]
    if indxs is None:
        indxs = range(dmat.shape[0])
        
    #Fb added: new way to fit model
    # everything but blank
    #Xd = NDNutils.create_time_embedding( dmat[:, :-1], [num_dlags, ND2-1, 1])
    data_xd={"stim":dmat[:,:-1]}
    data_xd.stim_dims = [num_dlags, ND2-1, 1]
    
    # blank
    #Xb = NDNutils.create_time_embedding( dmat[:, -1], [num_dlags, 1, 1])
    data_xb.stim = dmat[:, -1]
    data_xb.stim_dims = [num_dlags, 1, 1]
    
    # timing
    switches = np.expand_dims(np.concatenate( (np.sum(abs(np.diff(dmat, axis=0)),axis=1), [0]), axis=0), axis=1)
    #Xs = NDNutils.create_time_embedding( switches, [num_dlags, 1, 1])
    data_xs.stim = switches
    
    #tpar = NDNutils.ffnetwork_params( 
    #    xstim_n=[0], input_dims=[1,1,1, num_dlags], layer_sizes=[1], verbose=False,
    #    layer_types=['normal'], act_funcs=['lin'], reg_list={'d2t':[None],'l1':[None ]})
    tpar = NDNLayer.layer_dict( 
    input_dims=[num_dlags, 1, 1], num_filters=1, bias=True, initialize_center = True, NLtype='lin')
    
    bpar = deepcopy(tpar)
    bpar['xstim_n'] = [1]
    dpar = NDNutils.ffnetwork_params( 
        xstim_n=[2], input_dims=[1,ND2-1,1, num_dlags], layer_sizes=[1], verbose=False,
        layer_types=['normal'], act_funcs=['lin'], reg_list={'d2xt':[None],'l1':[None]})
    if rectified:
        comb_parT = NDNutils.ffnetwork_params( 
            xstim_n=None, ffnet_n=[0,1], layer_sizes=[1], verbose=False,
            layer_types=['normal'], act_funcs=['softplus'])
    else:
        comb_parT = NDNutils.ffnetwork_params( 
            xstim_n=None, ffnet_n=[0,1], layer_sizes=[1], verbose=False,
            layer_types=['normal'], act_funcs=['lin'])

    comb_par = deepcopy(comb_parT)
    comb_par['ffnet_n'] = [0,1,2]

    if spiking:
        nd = 'poisson'
    else:
        nd = 'gaussian'
        
    Tglm = NDN.NDN( [tpar, bpar, comb_parT], noise_dist=nd, tf_seed = 5)
    DTglm = NDN.NDN( [tpar, bpar, dpar, comb_par], noise_dist=nd, tf_seed = 5)
    v2fT = Tglm.fit_variables( layers_to_skip=[2], fit_biases=False)
    v2fT[2][0]['fit_biases'] = True
    v2f = DTglm.fit_variables( layers_to_skip=[3], fit_biases=False)
    v2f[3][0]['fit_biases'] = True

    if (fr1or3 == 3) or (fr1or3 == 1):
        mod_indxs = np.intersect1d(indxs, np.where(Einfo.frs == fr1or3)[0])
        #frs_valid = Einfo['frs'] == fr1or3
    else:
        mod_indxs = indxs
    
    _= Tglm.train(
        input_data=[Xs[mod_indxs,:], Xb[mod_indxs,:]], output_data=resp[mod_indxs], # fit_variables=v2fT,
        learning_alg='lbfgs', opt_params=opt_params)
    _= DTglm.train(
        input_data=[Xs[mod_indxs,:], Xb[mod_indxs,:], Xd[mod_indxs,:]], # fit_variables=v2f, 
        output_data=resp[mod_indxs], learning_alg='lbfgs', opt_params=opt_params)
    
    # make predictions of each
    predT = Tglm.generate_prediction( input_data=[Xs, Xb] )
    predD = DTglm.generate_prediction( input_data=[Xs, Xb, Xd] )
    
    return predD, predT

def binocular_model_performance( Einfo=None, Robs=None, Rpred=None, indxs=None, cell_num=1, opt_params=None ):
    """Current best-practices for generating prediction quality of neuron and binocular tuning. Currently we
    are not worried about using cross-validation indices only (as they are based on much less data and tend to
    otherwise be in agreement with full measures, but this option could be added in later versions."""

    assert Einfo is not None, 'Need to include Einfo'
    assert indxs is not None, 'Need to include valid indxs'
    if opt_params is None:
        opt_params = NDN.NDN.optimizer_defaults(opt_params={'use_gpu': True, 'display': True}, learning_alg='lbfgs')
    
    if len(Robs.shape) == 1:
        Robs = np.expand_dims(Robs, axis=1)
    if len(Rpred.shape) == 1:
        Rpred = np.expand_dims(Rpred, axis=1)

    indxs3 = np.intersect1d(indxs, np.where(Einfo.frs == 3)[0])
    indxs1 = np.intersect1d(indxs, np.where(Einfo.frs == 1)[0])
    
    # make disparity predictions for all conditions
    # -- actually checked and best to fit both data simultaneously. So: do all possibilities
    dobs0, tobs0 = disparity_predictions( Einfo, Robs, indxs, spiking=True, opt_params=opt_params )
    dmod0, tmod0 = disparity_predictions( Einfo, Rpred, indxs, spiking=False, opt_params=opt_params )

    dobs1, tobs1 = disparity_predictions(Einfo, Robs, indxs1, spiking=True, opt_params=opt_params )
    dmod1, tmod1 = disparity_predictions( Einfo, Rpred, indxs1, spiking=False, opt_params=opt_params )

    dobs3, tobs3 = disparity_predictions(Einfo, Robs, indxs3, spiking=True, opt_params=opt_params )
    dmod3, tmod3 = disparity_predictions( Einfo, Rpred, indxs3, spiking=False, opt_params=opt_params )

    # Calculate overall
    ev, tv = explainable_variance_binocular( Einfo, Robs, indxs=indxs, cell_num=cell_num)
    ev3, tv3 = explainable_variance_binocular( Einfo, Robs, indxs=indxs3, cell_num=cell_num)
    ev1, tv1 = explainable_variance_binocular( Einfo, Robs, indxs=indxs1, cell_num=cell_num) 
    
    if ev == tv:
        ev_valid = False
    else:
        ev_valid = True
    dv_obs = np.var(dobs0[indxs]-tobs0[indxs])
    dv_obs3 = np.var(dobs0[indxs3]-tobs0[indxs3])
    dv_obs1 = np.var(dobs0[indxs1]-tobs0[indxs1])
    dv_pred = np.var(dmod0[indxs]-tmod0[indxs])
    dv_pred3a = np.var(dmod3[indxs3]-tmod3[indxs3])
    #dv_pred3b = np.var(dmod3[indxs3]-tmod3[indxs3])
    dv_pred1a = np.var(dmod1[indxs1]-tmod1[indxs1])
    #dv_pred1b = np.var(dmod1[indxs1]-tmod1[indxs1])
    
    print( "\nOverall explainable variance fraction: %0.2f"%(ev/tv) )
    print( "Obs disparity variance fraction: %0.2f (FR3: %0.2f)"%(dv_obs/ev, dv_obs3/ev3) )
    vars_obs = [tv, ev, dv_obs, ev-dv_obs ]  # total, explainable, disp_var, pattern_var
    vars_obs_FR3 = [tv3, ev3, dv_obs3, ev3-dv_obs3 ]  # total, explainable, disp_var, pattern_var
    DVfrac_obs = [dv_obs/ev, dv_obs3/ev3, dv_obs1/ev1 ]

    vars_mod = [np.var(Rpred[indxs]), dv_pred, np.var(Rpred[indxs])-dv_pred]
    DVfrac_mod = [dv_pred/np.var(Rpred[indxs]), 
                  dv_pred3a/np.var(Rpred[indxs3]), 
                  dv_pred1a/np.var(Rpred[indxs1])]
    #DVfrac_mod_alt = [dv_mod/np.var(Rpred[indxs]), 
    #                  dv_pred3b/np.var(Rpred[indxs3]), 
    #                  dv_pred1b/np.var(Rpred[indxs1])]
    
    # Predictive powers (model performance): full response and then disparity
    pps = [predictive_power_binocular( Robs, Rpred, indxs=indxs, Einfo=Einfo, cell_num=cell_num ),
           #predictive_power_binocular(dobs0, dmod0, indxs=indxs, Einfo=Einfo, cell_num=cell_num),
           predictive_power_binocular(dobs0-tobs0, dmod0-tmod0, indxs=indxs, Einfo=Einfo, cell_num=cell_num)]
    # bound possible pps 
    
    pps_dispFR3 = [
        predictive_power_binocular( dobs0-tobs0, dmod0-tmod0, indxs=indxs3, Einfo=Einfo, cell_num=cell_num ),
        predictive_power_binocular( dobs3-tobs3, dmod3-tmod3, indxs=indxs3, Einfo=Einfo, cell_num=cell_num )]
    pps_dispFR1 = [
        predictive_power_binocular( dobs0-tobs0, dmod0-tmod0, indxs=indxs1, Einfo=Einfo, cell_num=cell_num ),
        predictive_power_binocular( dobs1-tobs1, dmod1-tmod1, indxs=indxs1, Einfo=Einfo, cell_num=cell_num )]

    print( "Pred powers: %0.3f  disp %0.3f (FR3 %0.3f)"%(pps[0], pps[1], pps_dispFR3[0]))

    # Add general tuning of each
    Dtun_info_obs = [disparity_tuning( Einfo, Robs, indxs, fr1or3=3, to_plot=False),
        disparity_tuning( Einfo, Robs, indxs, fr1or3=1, to_plot=False)]
    Dtun_info_pred = [disparity_tuning( Einfo, Rpred, indxs, fr1or3=3, to_plot=False),
        disparity_tuning( Einfo, Rpred, indxs, fr1or3=1, to_plot=False)]

    BMP = {'EVfrac': ev/tv, 'EVvalid': ev_valid, 
           'vars_obs': vars_obs, 'vars_mod': vars_mod, 'vars_obs_FR3': vars_obs_FR3,
           'DVfrac_obs': DVfrac_obs, 'DVfrac_mod': DVfrac_mod, 
           'pred_powers': pps, 'pps_disp_FR3': pps_dispFR3, 'pps_disp_FR1': pps_dispFR1,
           'Dtun_obs': Dtun_info_obs, 'Dtun_pred': Dtun_info_pred}
    
    return BMP

