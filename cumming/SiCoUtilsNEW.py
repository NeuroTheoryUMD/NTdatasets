import numpy as np
import scipy.io as sio
import NDNT.utils as utils
import NDNT.NDN as NDN
import torch
from time import time
from copy import deepcopy
import matplotlib.pyplot as plt
from NTdatasets.cumming.BinocUtils import plot_sico_readout


def baseline_bem(NE, NI, LorR=0, seed=100, XTreg=0.01, logXTmult=0, Greg=0.001, Dreg=0.001, nlags=None,
                num_anchors=0, target_rate=None ):
    """
    We are taking in passing in the drift terms away. and let if fit both drift and beta-drift terms, with an overall
    firing rate adjustment to alphas off the bat. 
    logXTmult=0 means that d2x and d2t are the same, so use d2xt, otherwise separately define them based on factor of 10
    """
    from NDNT.NDN import NDN
    from NDNT.networks import FFnetwork, SoftplusNetwork
    from NDNT.modules.layers import NDNLayer, ConvLayer, MaskConvLayer, BinocShiftLayer, SoftplusLayer

    if nlags is None:
        nlags = 10
        print("  Using default nlags = %d"%nlags)

    # define reg values for first layer
    reg_vals = {'center':CregM}
    if logXTmult == 0:
        reg_vals['d2xt'] = XTreg
    else:
        reg_vals['d2x'] = XTreg
        reg_vals['d2t'] = XTreg*(10.0**logXTmult)

    ### DNN PARAMETERS ###        
    monoc_basis_par = ConvLayer.layer_dict( 
        input_dims=[1,72,1,nlags], num_filters=num_mfilts, filter_dims=[1, mfw, 1, nlags],
        norm_type=1,bias=False, initialize_center=True, NLtype='lin',
        reg_vals=reg_vals)    

    if sample_layer:
        bfilt_par = BinocShiftLayer.layer_dict( 
            num_inh= NI, LRdom=LorR, xdoms=shift_start, bias=bi_bias, NLtype='relu')
    else:
        bfilt_par = MaskConvLayer.layer_dict( 
            input_dims=[num_mfilts*2,36,1,1], # reinterprets convolutional output above
            num_filters=NE+NI, num_inh= NI, filter_dims=bfw, 
            num_groups=num_mfilts, norm_type=1, pos_constraint=True, #window='hamming',
            bias=bi_bias, initialize_center=True, NLtype='relu')

        masks = [np.ones( [2, bfw, num_mfilts], dtype=np.float32 ), 
                np.ones( [2, bfw, num_mfilts], dtype=np.float32 )]
        zfw = bfw//2
        masks[0][0,:zfw,:] = 0
        masks[0][0,-zfw:,:] = 0
        masks[1][1,:zfw,:] = 0
        masks[1][1,-zfw:,:] = 0

    # Two versions: first is linear because softplus comes later, second has this as last layer
    readout_par = NDNLayer.layer_dict(
            num_filters=1, bias=False, initialize_center=True, pos_constraint=True,
            NLtype='lin', reg_vals={'glocalx': Greg })

    stim_net = FFnetwork.ffnet_dict( layer_list = [monoc_basis_par, bfilt_par, readout_par] )

    if time_covariates > 0:
        time_pars = NDNLayer.layer_dict( 
            input_dims=[1,1,1,time_covariates], num_filters=1, bias=False, norm_type=0, NLtype='lin')
        frame_net = FFnetwork.ffnet_dict( xstim_n='Xframe_switch', layer_list=[time_pars] )
        ffnets = [0,1]
    else:
        ffnets = [0]

    # No matter what fit softplus network at the end, with num_anchors determining drift term or not
    if num_anchors == 0:
        beta_drift = False
    else:
        beta_drift = True

    comb_net = SoftplusNetwork.ffnet_dict(ffnet_n=ffnets, num_anchors=num_anchors, beta_drift=beta_drift, drift_reg=Dreg, beta_reg=Dreg)

    if time_covariates > 0:
        sico = NDN(ffnet_list=[stim_net, frame_net, comb_net], seed=seed)
    else:
        sico = NDN(ffnet_list=[stim_net, comb_net], seed=seed)

    if not sample_layer:
        sico.networks[0].layers[1].set_mask(masks[LorR])

    if target_rate is not None:
        sico.networks[-1].layers[0].set_scale([target_rate])

    return sico
# END baseline_sico2()


def bem_reg_path(
    ds_trn, ds_val, NE=2, NI=2, XTreg0=None, logXTmult=0, XTcoupled=True, Greg0=None, 
    thresh=0.95, Gthresh=None, sample_layer=True, num_anchors=0,
    nlags=None, time_covariates=0, LLn=0, drift_term=None, to_plot=True, device=None ):
    """reg0 is if want centered -- test order of mag in each direction"""
    if num_anchors == 0:
        assert drift_term is not None, "Need to enter 'drift_term'"

    # Figure out average firing rate for scaling softplus layer (if entered)
    avrate = (torch.sum(ds_trn[:]['robs']*ds_trn[:]['dfs'],axis=0)/torch.sum(ds_trn[:]['dfs'],axis=0)).detach().cpu().numpy()

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("  Regpath WARNING: device not entered, using device:", device)
    device0 = torch.device("cpu") # for storing models on CPU

    if Gthresh is None:
        Gthresh = thresh
    #if sample_layer:
    #    ln_search = None
    #else:
    ln_search = 'strong_wolfe'

    if nlags is None:
        nlags = 12
        print("  Using default nlags = %d"%nlags)

    # Determine LR
    LR = ocular_dominance( ds_trn, verbose=False )

    # Regularize full sweep or just around a given value
    if XTreg0 is None:
        Rvals = [1e-6, 1e-4, 0.001, 0.01, 0.1, 1]
    else:
        Rvals = [XTreg0*0.1, XTreg0, XTreg0*10.0]

    # d2xt Reg path (Xreg and Treg coupled at certain ratio)
    LLsRx = np.zeros(len(Rvals))
    mods = []
    print('  Initial XT-regpath:', utils.string_convert(Rvals) )
    for ii in range(len(Rvals)): # same seed
        #sico_iter = baseline_sico(NE, NI, LorR=LR, seed=101, XTreg=Rvals[ii], logXTmult=logXTmult, nlags=nlags,
        #                          sample_layer=sample_layer,
        #                          drift_term=drift_term, time_covariates=time_covariates).to(device)
        sico_iter = baseline_sico2(NE, NI, LorR=LR, seed=101, XTreg=Rvals[ii], logXTmult=logXTmult, nlags=nlags,
                                  sample_layer=sample_layer, target_rate=avrate,
                                  num_anchors=num_anchors, time_covariates=time_covariates).to(device)
        utils.fit_lbfgs( sico_iter, ds_trn[:], verbose=0, max_iter=2000, line_search=ln_search)
        LL = LLn - sico_iter.eval_models(ds_val[:], null_adjusted=False)[0]
        mods.append(deepcopy(sico_iter))  # this will still be on GPU
        LLsRx[ii] = LL
        print( "    %2d  %9.6f"%(ii, LLsRx[ii]) )

    bestXr = np.where(LLsRx > (np.nanmax(LLsRx)*thresh))[0][-1]
    # really if its better by 1 to go higher... 
    #if bestr > 0:
    #    if LLsRx[bestr-1] > (LLsRx[bestr]):
    #        bestr = bestr-1
    #print('Chosen Reg 1-1 (%d)'%bestr)
    XTreg = Rvals[bestXr]
    LLprev = LLsRx[bestXr]
    del sico_iter
    torch.cuda.empty_cache()
    mod0 = deepcopy(mods[bestXr]).to(torch.device('cpu'))
    
    print('  Chosen d2xt =', utils.string_convert(XTreg), '(%d)'%bestXr)

    if not XTcoupled: 
        #log_mult_list = np.array([logXTmult-1, logXTmult+1], dtype=int)
        #log_mult_list = log_mult_list[log_mult_list >= -2]
        #log_mult_list = log_mult_list[log_mult_list <= 2]
        log_mult_list = np.array([-1, 0, 1], dtype=int)  # limit to within one order of magnitude of d2x
        #print('  T-regpath:', utils.string_convert(log_mult_list) )
        print('  T-regpath:' )
        #mods = []
        #LLs = np.zeros(len(log_mult_list))
        mod1 = deepcopy(mod0)
        for ii, log_mult in enumerate(log_mult_list): 
            #sico_iter = baseline_sico(NE, NI, LorR=LR, seed=101, XTreg=XTreg, logXTmult=log_mult, nlags=nlags, 
            #                          sample_layer=sample_layer,
            #                          time_covariates=time_covariates, drift_term=drift_term).to(device) 
            sico_iter = baseline_sico2(NE, NI, LorR=LR, seed=101, XTreg=XTreg, logXTmult=log_mult, nlags=nlags, 
                                      sample_layer=sample_layer, target_rate=avrate,
                                      time_covariates=time_covariates, num_anchors=num_anchors).to(device) 

            utils.fit_lbfgs( sico_iter, ds_trn[:], verbose=0, max_iter=2000, line_search=ln_search)
            #LLs[ii] = LLn - sico_iter.eval_models(ds_val[:], null_adjusted=False)[0]
            LL = LLn - sico_iter.eval_models(ds_val[:], null_adjusted=False)[0]
            #mods.append(deepcopy(sico_iter))
            print( "  d2t = 1e%d:\t%9.6f"%(log_mult+int(np.log10(XTreg)), LL ), end='' ) 
            if LL > LLprev:  
                print(' *')
                LLprev = LL
                logXTmult = log_mult
                mod1 = deepcopy(sico_iter)
            else:
                print('')
        #bestr = np.where(LLs > (np.nanmax(LLs)*thresh))[0][-1]
        #mod1 = deepcopy(mods[bestr]).to(torch.device('cpu'))
        print('  Chosen d2x, d2t =', utils.string_convert(XTreg), utils.string_convert(XTreg*(10.0**logXTmult)))
    else:
        mod1 = deepcopy(mod0)

    # Center and refine model, and then pick best Greg
    if sample_layer:
        mod1 = center_model( mod1, include_binoc=True ).to(device)
        mod1.networks[0].layers[1].fit_shifts(val=True, fixed_sigmas=True, sigma0 = 0.9) # make sure sigmas not too big
        utils.fit_lbfgs( mod1, ds_trn[:], verbose=0, max_iter=2000, line_search=ln_search)
    else:
        mod1 = refine_binocular(center_model(mod1, include_binoc=False), ds_trn, #ds_val, LLnull=LLn, 
                                device=device, to_plot=False )
        mod1 = center_model( mod1, include_binoc=True ).to(torch.device("cpu"))
    LL = LLn - mod1.eval_models(ds_val[:], null_adjusted=False)[0]
    #print("  Refined sico%d-%d LL = %0.6f"%(NE, NI, LL) )

    # now glocalx
    if Greg0 is None:
        Rvals = [1e-6, 1e-4, 0.001, 0.01, 0.1, 1, 10]
    else:
        Rvals = [Greg0*0.1, Greg0, Greg0*10.0]

    print('  glocalx-regpath:', utils.string_convert(Rvals))
    LLsRg = np.zeros(len(Rvals))
    mods = []
    for ii in range(len(Rvals)):
        sico_iter = deepcopy(mod1).to(device)
        sico_iter.networks[0].layers[2].reg.vals['glocalx'] = Rvals[ii]
        utils.fit_lbfgs( sico_iter, ds_trn[:], verbose=0, max_iter=2000, line_search=ln_search)
        mods.append(deepcopy(sico_iter))
        LL = LLn - sico_iter.eval_models(ds_val[:], null_adjusted=False)[0]
        print( "    %2d  %9.6f"%(ii, LL), end='' )
        if LL > np.nanmax(LLsRg):
            #mod2 = deepcopy(sico_iter).to(device0)
            print(' *')
            #bestr = ii
        else:
            print('')
        LLsRg[ii] = LL

    #bestr = np.argmax(LLsRg)
    # overwrite bestr with (compromise) threshold that is very close to max LL but not necessarily the max (to avoid overfitting)
    try:
        bestGr = np.where(LLsRg > (np.nanmax(LLsRg)*Gthresh))[0][-1]
    except IndexError:
        bestGr = np.nanargmax(LLsRg)
    bestGr = np.where(LLsRg > (np.max(LLsRg)*Gthresh))[0][-1]
    Greg = Rvals[bestGr]
    mod2 = mods[bestGr].to(torch.device('cpu'))
    print('  Chosen glocalx = ', utils.string_convert(Greg), '(%d)'%bestGr, '\n')

    if to_plot:
        utils.subplot_setup( 1, 2, row_height=3, fig_width=8)
        plt.subplot(1,2,1)
        plt.plot(LLsRx,'b')
        plt.plot(LLsRx,'bo')
        plt.axhline(np.nanmax(LLsRx)*thresh, color='k', linestyle='--')
        plt.axvline(bestXr, color='c')

        plt.subplot(1,2,2)
        plt.plot(LLsRg,'g')
        plt.plot(LLsRg,'go')
        plt.axhline(np.nanmax(LLsRg)*thresh, color='k', linestyle='--')
        plt.axvline(bestGr, color='c')
        plt.show()

        if not sample_layer:
            mod2.plot_filters()
            plot_conv_layer(mod2)
            plot_sico_readout(mod2)
        else:
            display_sampler_model(mod2)
    # Temporary saves in case craps out
    return {'XTreg': XTreg, 'logXTmult': logXTmult, 'Greg': Greg, 'model': mod2}
# END bem_reg_path
