import numpy as np
import scipy.io as sio
import NDNT.utils as utils
import NDNT.NDN as NDN
import torch
from time import time
from copy import deepcopy
import matplotlib.pyplot as plt
from NTdatasets.cumming.BinocUtils import plot_sico_readout

############## SiCo Fitting Pathways ##############
def sico_path(ds_trn, ds_val, LLn_trn=0, LLn_val=0, drift_term=None,
              XTreg=None, Greg=None, XTcoupled=False, logXTmult=0, sample_layer=True,
              n_iter=8, time_covariates=True, device=None,
              save_dir=None, expt_n=None, cell_n=None, id=None):
    """
    Fit a series of SICO models with increasing numbers of excitatory and inhibitory filters

    Args:
        ds_trn: training dataset
        ds_val: validation dataset
        LLn_trn: null log-likelihood for training data
        LLn_val: null log-likelihood for validation data
        drift_term: numpy array of drift term to use in model (must be entered)
        XTreg: initial regularization for d2x and d2t (if XTcoupled=True) or d2x (if XTcoupled=False)
        Greg: initial regularization for glocalx
        XTcoupled: if True, d2x and d2t are coupled, otherwise they are separate
        logXTmult: if XTcoupled=False, this is the log10 multiplier for d2t relative to d2x
        sample_layer: if True, use BinocShiftLayer, otherwise use MaskConvLayer
        n_iter: number of iterations to fit each model (default=8)
        time_covariates: if True, include time covariates in the model (default=True)
        device: torch device to use for fitting (default=None, will use cuda if available)
        expt_n: experiment number for saving models (optional)
        cell_n: cell number for saving models (optional)
        id: additional optional identifier string to add to filename for saving models
        save_dir: directory to save models (optional, default=None, will save in current directory)
    """
    assert drift_term is not None, "Need to enter 'drift_term'"
    nlags = ds_trn[0]['stim'].shape[-1]//72
    print("  Detected num lags = %d"%nlags)

    # Make temporary save models: can index by experiment/cell_n (if expt_n, cell_n is entered) or not
    save_name = "sico"
    if expt_n is not None:
        assert cell_n is not None, "If entering expt_n, must also enter cell_n for saving models"
        save_name += str(expt_n)+'c'+str(cell_n)
    elif cell_n is not None:
        save_name += str(cell_n)
    if id is not None:
        save_name += str(id)
    else:
        save_name += 'path'
    if save_dir is not None:
        if save_dir[-1] != '/':
            save_dir += '/'
        # Check if directory exists, if not create it
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print('  Creating directory for saving models:', save_dir)
        save_name = save_dir + '/' + save_name
    
    if time_covariates: # replace boolean with number (messy I know but has good default behavior)
        time_covariates = ds_trn[0]['Xframe_switch'].shape[-1]

    # Determine LR
    LR = ocular_dominance( ds_trn, verbose=False )
    #Rvals = [1e-6, 1e-4, 0.001, 0.01, 0.1, 1, 10]

    NE, NI = 1, 1
    # d2xt Reg path for beginner model all the way through
    if (XTreg is None) or (Greg is None):
        regs = sico_reg_path(ds_trn, ds_val, NE=1, NI=1, thresh=0.95, XTreg0=XTreg, Greg0=Greg, sample_layer=sample_layer,
                             time_covariates=time_covariates, XTcoupled=XTcoupled, logXTmult=logXTmult, nlags=nlags,
                             LLn=LLn_val, drift_term=drift_term, device=device, to_plot=False )
        XTreg = regs['XTreg']
        Greg = regs['Greg']
        logXTmult = regs['logXTmult']

    # Find best model for 1-1 over n_iters
    print('NE, NI = %d, %d'%(NE, NI))
    if sample_layer:
        mod_path = [produce_best_sampler_model(
            ds_trn, ds_val, drift_term, LR, XTreg, logXTmult, Greg, NE=1, NI=1, time_covariates=time_covariates,
            n_iter=n_iter, nlags=nlags, LLn_trn=LLn_trn, LLn_val=LLn_val, device=device, to_plot=True)]
    else:
        mod_path = [produce_best_model(
            ds_trn, ds_val, drift_term, LR, XTreg, logXTmult, Greg, NE=1, NI=1, time_covariates=time_covariates,
            n_iter=n_iter, nlags=nlags, LLn_trn=LLn_trn, LLn_val=LLn_val, device=device, to_plot=True)]
    
    LLprev = LLn_val - mod_path[0].eval_models(ds_val[:], null_adjusted=False)[0]
    #print("1-1: LL = %0.5f"%LLprev)
    mod_path[0].save_model(save_name+"1_1.ndn")
    
    no_stop=True
    iter = 0 # number of adds to E and/or I

    while no_stop and (iter < 6):
        no_stop = False

        # Check best regularization on previous model (from last iteration)
        if iter > 0:  
            regs = sico_reg_path(
                ds_trn, ds_val, NE=NE, NI=NI, 
                time_covariates=time_covariates, thresh=0.95, XTreg0=XTreg, XTcoupled=XTcoupled, logXTmult=logXTmult, Greg0=Greg,
                sample_layer=sample_layer,
                nlags=nlags, LLn=LLn_val, drift_term=drift_term, device=device, to_plot=False )
            XTreg = regs['XTreg']
            Greg = regs['Greg']
            logXTmult = regs['logXTmult']
            prev_mod = regs['model']
            LLprev = LLn_val - prev_mod.eval_models(ds_val[:], null_adjusted=False)[0]

        # plus one excitation
        NE += 1
        print('NE, NI = %d, %d'%(NE, NI))
        if sample_layer:
            sicoE1 = produce_best_sampler_model(ds_trn, ds_val, drift_term, LR, XTreg, logXTmult, Greg, NE=NE, NI=NI, time_covariates=time_covariates,
                                                n_iter=n_iter, nlags=nlags, LLn_trn=LLn_trn, LLn_val=LLn_val, device=device, to_plot=False)
            #sicoE1 = produce_best_sampler_model(ds_trn, ds_val, model=mod_path[-1], LR=LR, addEorI=0, time_covariates=time_covariates,
            #                                     n_iter=n_iter, LLn_trn=LLn_trn, LLn_val=LLn_val, device=device, to_plot=False)
        else:
            sicoE1 = produce_best_model(ds_trn, ds_val, drift_term, LR, XTreg, logXTmult, Greg, NE=NE, NI=NI, 
                                        time_covariates=time_covariates, nlags=nlags, n_iter=n_iter, LLn_trn=LLn_trn, LLn_val=LLn_val, device=device,
                                        to_plot=False)
            
        LL = LLn_val - sicoE1.eval_models(ds_val[:], null_adjusted=False)[0]
        if LL > LLprev:
            no_stop = True
            LLprev = LL
            mod_path.append(deepcopy(sicoE1))

            # Plot model here
            print("Keeping (%d,%d): %0.5f"%(NE, NI, LL))
            if sample_layer:
                display_sampler_model(sicoE1) 
            else:
                sicoE1.plot_filters()
                plot_conv_layer(sicoE1)
                plot_sico_readout(sicoE1)
            sicoE1.save_model(save_name+"%d_%d.ndn"%(NE, NI))
        else:
            print("EXC+1 (%d,%d) no good: %0.5f < %0.5f\n"%(NE, NI, LL, LLprev))
            NE += -1

        # plus one inhibition
        NI += 1
        print('NE, NI = %d, %d'%(NE, NI))
        if sample_layer:
            sicoI1 = produce_best_sampler_model(ds_trn, ds_val, drift_term, LR, XTreg, logXTmult, Greg, NE=NE, NI=NI, time_covariates=time_covariates,
                                                n_iter=n_iter, nlags=nlags, LLn_trn=LLn_trn, LLn_val=LLn_val, device=device, to_plot=False)
            #sicoI1 = produce_best_sampler_model2(ds_trn, ds_val, model=mod_path[-1], LR=LR, addEorI=1, time_covariates=time_covariates,
            #                                     n_iter=n_iter, LLn_trn=LLn_trn, LLn_val=LLn_val, device=device, to_plot=False)
        else:
            sicoI1 = produce_best_model(ds_trn, ds_val, drift_term, LR, XTreg, logXTmult, Greg, NE=NE, NI=NI, 
                                        time_covariates=time_covariates, nlags=nlags, n_iter=n_iter, LLn_trn=LLn_trn, LLn_val=LLn_val, device=device,
                                        to_plot=False)
            
        LL = LLn_val - sicoI1.eval_models(ds_val[:], null_adjusted=False)[0]
        if LL > LLprev:
            no_stop = True
            LLprev = LL
            mod_path.append(deepcopy(sicoI1))

            # Plot model here
            print("Keeping (%d,%d): %0.5f"%(NE, NI, LL))
            if sample_layer:
                display_sampler_model(sicoI1) 
            else:
                sicoI1.plot_filters()
                plot_conv_layer(sicoI1)
                plot_sico_readout(sicoI1)
            sicoI1.save_model(save_name+"%d_%d.ndn"%(NE, NI))
        else:
            print("INH+1 (%d,%d) no good: %0.5f < %0.5f\n"%(NE, NI, LL, LLprev))
            NI += -1

        iter += 1

    return mod_path
# END sico_path()


def sico_path_parallel(ds_trn, ds_val, LLn_trn=0, LLn_val=0, drift_term=None,
              XTreg=None, Greg=None, XTcoupled=False, logXTmult=0, sample_layer=True,
              n_iter=8, time_covariates=True, expt_n=None, cell_n=None, id=None, device=None):
    """
    Fit a series of SICO models with increasing numbers of excitatory and inhibitory filters

    Args:
        ds_trn: training dataset
        ds_val: validation dataset
        LLn_trn: null log-likelihood for training data
        LLn_val: null log-likelihood for validation data
        drift_term: numpy array of drift term to use in model (must be entered)
        XTreg: initial regularization for d2x and d2t (if XTcoupled=True) or d2x (if XTcoupled=False)
        Greg: initial regularization for glocalx
        XTcoupled: if True, d2x and d2t are coupled, otherwise they are separate
        logXTmult: if XTcoupled=False, this is the log10 multiplier for d2t relative to d2x
        sample_layer: if True, use BinocShiftLayer, otherwise use MaskConvLayer
        n_iter: number of iterations to fit each model (default=8)
        time_covariates: if True, include time covariates in the model (default=True)
        expt_n: experiment number for saving models (optional)
        cell_n: cell number for saving models (optional)
        id: additional optional identifier string to add to filename for saving models
        device: torch device to use for fitting (default=None, will use cuda if available)
    """
    assert drift_term is not None, "Need to enter 'drift_term'"
    nlags = ds_trn[0]['stim'].shape[-1]//72
    print("  Detected num lags = %d"%nlags)

    # Make temporary save models: can index by experiment/cell_n (if expt_n, cell_n is entered) or not
    save_name = "sico"
    if expt_n is not None:
        assert cell_n is not None, "If entering expt_n, must also enter cell_n for saving models"
        save_name += str(expt_n)+'c'+str(cell_n)
    elif cell_n is not None:
        save_name += str(cell_n)
    if id is not None:
        save_name += str(id)
    else:
        save_name += 'path'
    
    if time_covariates: # replace boolean with number (messy I know but has good default behavior)
        time_covariates = ds_trn[0]['Xframe_switch'].shape[-1]

    # Determine LR
    LR = ocular_dominance( ds_trn, verbose=False )
    #Rvals = [1e-6, 1e-4, 0.001, 0.01, 0.1, 1, 10]

    NE, NI = 1, 1
    # d2xt Reg path for beginner model all the way through
    if (XTreg is None) or (Greg is None):
        regs = sico_reg_path(ds_trn, ds_val, NE=1, NI=1, thresh=0.95, XTreg0=XTreg, Greg0=Greg, sample_layer=sample_layer,
                             time_covariates=time_covariates, XTcoupled=XTcoupled, logXTmult=logXTmult, nlags=nlags,
                             LLn=LLn_val, drift_term=drift_term, device=device, to_plot=False )
        XTreg = regs['XTreg']
        Greg = regs['Greg']
        logXTmult = regs['logXTmult']

    # Find best model for 1-1 over n_iters
    print('NE, NI = %d, %d'%(NE, NI))
    if sample_layer:
        best_mod, mod0s, LL0s = produce_best_sampler_model(
            ds_trn, ds_val, drift_term, LR, XTreg, logXTmult, Greg, NE=1, NI=1, time_covariates=time_covariates,
            n_iter=n_iter, nlags=nlags, LLn_trn=LLn_trn, LLn_val=LLn_val, device=device, to_plot=True, save_models=True)
    else:
        best_mod, mod0s, LL0s = produce_best_model(
            ds_trn, ds_val, drift_term, LR, XTreg, logXTmult, Greg, NE=1, NI=1, time_covariates=time_covariates,
            n_iter=n_iter, nlags=nlags, LLn_trn=LLn_trn, LLn_val=LLn_val, device=device, to_plot=True, save_models=True)
    
    LLprev = LLn_val - best_mod.eval_models(ds_val[:], null_adjusted=False)[0]
    #print("1-1: LL = %0.5f"%LLprev)
    best_mod.save_model(save_name+"1_1.ndn")
    
    no_stop=True
    iter = 0 # number of adds to E and/or I
    model_iterations = [mod0s]
    LL_iterations = [LL0s]
    while no_stop and (iter < 6):
        no_stop = False

        # Check best regularization on previous model (from last iteration)
        if iter > 0:  
            regs = sico_reg_path(
                ds_trn, ds_val, NE=NE, NI=NI, 
                time_covariates=time_covariates, thresh=0.95, XTreg0=XTreg, XTcoupled=XTcoupled, logXTmult=logXTmult, Greg0=Greg,
                sample_layer=sample_layer,
                nlags=nlags, LLn=LLn_val, drift_term=drift_term, device=device, to_plot=False )
            XTreg = regs['XTreg']
            Greg = regs['Greg']
            logXTmult = regs['logXTmult']
            prev_mod = regs['model']
            LLprev = LLn_val - prev_mod.eval_models(ds_val[:], null_adjusted=False)[0]

        # plus one excitation
        NE += 1
        print('NE, NI = %d, %d'%(NE, NI))
        sicoE1, next_mods, next_LLs = increment_models(ds_trn, ds_val, modlist=model_iterations[-1], addEorI=0)

        exc_keeper_list, excLLs = [], []
        for ii in range(len(next_mods)):
            if next_mods[ii].networks[0].layers[0].num_filters > model_iterations[-1][ii].networks[0].layers[0].num_filters:
                exc_keeper_list.append(next_mods[ii])
                excLLs.append(deepcopy(next_LLs[ii]))
        #model_iterations.append(deepcopy(keeper_list))  # do this later
        if len(exc_keeper_list) > 0:
            if sample_layer:
                display_sampler_model(sicoE1) 
            else:
                sicoE1.plot_filters()
                plot_conv_layer(sicoE1)
                plot_sico_readout(sicoE1)
            sicoE1.save_model(save_name+"%d_%d.ndn"%(NE, NI))  # only saves best model
        else:
            print("EXC+1 (%d,%d) no good throughout -- wholesale rejection"%(NE, NI))
            NE += -1

        # plus one inhibition
        NI += 1
        print('NE, NI = %d, %d'%(NE, NI))

        # note this takes the full list (some of which might not have been incremented)
        sicoI1, next_mods, next_LLs = increment_models(ds_trn, ds_val, modlist=next_mods, addEorI=1)

        inh_keeper_list, inhLLs = [], []
        for ii in range(len(next_mods)):
            if next_mods[ii].networks[0].layers[0].num_filters > model_iterations[-1][ii].networks[0].layers[0].num_filters:
                inh_keeper_list.append(next_mods[ii])
                inhLLs.append(deepcopy(next_LLs[ii]))
        if len(exc_keeper_list) > 0:
            model_iterations.append(deepcopy(exc_keeper_list))
            LL_iterations.append(deepcopy(excLLs))
            no_stop = True
        if len(inh_keeper_list) > 0:
            model_iterations.append(deepcopy(inh_keeper_list))
            LL_iterations.append(deepcopy(inhLLs))
            no_stop = True
            if sample_layer:
                display_sampler_model(sicoI1) 
            else:
                sicoI1.plot_filters()
                plot_conv_layer(sicoI1)
                plot_sico_readout(sicoI1)
            sicoI1.save_model(save_name+"%d_%d.ndn"%(NE, NI))  # only saves best model
        else:
            print("INH+1 (%d,%d) no good throughout -- wholesale rejection"%(NE, NI))
            NI += -1
            
        iter += 1

    return model_iterations, LL_iterations
# END sico_path_parallel()


def sico_reg_path(ds_trn, ds_val, NE=2, NI=2, XTreg0=None, logXTmult=0, XTcoupled=True, Greg0=None, thresh=0.95,
                  sample_layer=True,
                  nlags=None, time_covariates=0, LLn=0, drift_term=None, to_plot=True, device=None ):
    """reg0 is if want centered -- test order of mag in each direction"""
    assert drift_term is not None, "Need to enter 'drift_term'"

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("  Regpath WARNING: device not entered, using device:", device)
    device0 = torch.device("cpu") # for storing models on CPU

    if sample_layer:
        ln_search = None
    else:
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
        sico_iter = baseline_sico(NE, NI, LorR=LR, seed=101, XTreg=Rvals[ii], logXTmult=logXTmult, nlags=nlags,
                                  sample_layer=sample_layer,
                                  drift_term=drift_term, time_covariates=time_covariates).to(device)
        utils.fit_lbfgs( sico_iter, ds_trn[:], verbose=0, max_iter=2000, line_search=ln_search)
        LL = LLn - sico_iter.eval_models(ds_val[:], null_adjusted=False)[0]
        mods.append(deepcopy(sico_iter).to(device0))
        LLsRx[ii] = LL
        print( "    %2d  %9.6f"%(ii, LLsRx[ii]) )

    if to_plot:
        utils.subplot_setup( 1, 1, row_height=3, fig_width=5)
        plt.plot(LLsRx,'b')
        plt.plot(LLsRx,'bo')
        plt.axhline(np.max(LLsRx)*thresh, color='k', linestyle='--')
        plt.show()

    bestr = np.where(LLsRx > (np.nanmax(LLsRx)*thresh))[0][-1]
    # really if its better by 1 to go higher... 
    if bestr > 0:
        if LLsRx[bestr-1] > (LLsRx[bestr]):
            bestr = bestr-1
    #print('Chosen Reg 1-1 (%d)'%bestr)
    XTreg = Rvals[bestr]
    LLprev = LLsRx[bestr]
    mod0 = deepcopy(mods[bestr]).to(device0)
    
    print('  Chosen d2xt =', utils.string_convert(XTreg), '(%d)'%bestr)

    if not XTcoupled: 
        log_mult_list = np.array([logXTmult-1, logXTmult+1], dtype=int)
        log_mult_list = log_mult_list[log_mult_list >= -2]
        log_mult_list = log_mult_list[log_mult_list <= 2]
        #log_mult_list = np.array([-1, 1], dtype=int)
        #print('  T-regpath:', utils.string_convert(log_mult_list) )
        print('  T-regpath:' )

        mod1 = deepcopy(mod0)
        for log_mult in log_mult_list: 
            sico_iter = baseline_sico(NE, NI, LorR=LR, seed=101, XTreg=XTreg, logXTmult=log_mult, nlags=nlags, 
                                      sample_layer=sample_layer,
                                      time_covariates=time_covariates, drift_term=drift_term).to(device) 
            utils.fit_lbfgs( sico_iter, ds_trn[:], verbose=0, max_iter=2000, line_search=ln_search)
            LL = LLn - sico_iter.eval_models(ds_val[:], null_adjusted=False)[0]
            print( "  d2t = 1e%d:\t%9.6f"%(log_mult+int(np.log10(XTreg)), LL ), end='' ) 
            if LL > LLprev:
                print(' *')
                LLprev = LL
                logXTmult = log_mult
                mod1 = deepcopy(sico_iter).to(device0)
            else:
                print('')
        print('  Chosen d2x, d2t =', utils.string_convert(XTreg), utils.string_convert(XTreg*(10.0**logXTmult)))
    else:
        mod1 = deepcopy(mod0)

    # Center and refine model, and then pick best Greg
    if sample_layer:
        mod1 = center_model( mod1, include_binoc=True ).to(device)
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
        mods.append(deepcopy(sico_iter).to(device0))
        LL = LLn - sico_iter.eval_models(ds_val[:], null_adjusted=False)[0]
        print( "    %2d  %9.6f"%(ii, LL), end='' )
        if LL > np.nanmax(LLsRg):
            #mod2 = deepcopy(sico_iter).to(device0)
            print(' *')
        else:
            print('')
        LLsRg[ii] = LL
    if to_plot:
        utils.subplot_setup( 1, 1, row_height=3, fig_width=5)
        plt.plot(LLsRg,'g')
        plt.plot(LLsRg,'go')
        plt.axhline(np.max(LLsRg)*thresh, color='k', linestyle='--')
        plt.show()

    #bestr = np.argmax(LLsRg)
    bestr = np.where(LLsRg > (np.max(LLsRg)*thresh))[0][-1]
    Greg = Rvals[bestr]
    mod2 = mods[bestr]
    print('  Chosen glocalx = ', utils.string_convert(Greg), '(%d)'%bestr, '\n')

    if to_plot:
        if not sample_layer:
            mod2.plot_filters()
            plot_conv_layer(mod2)
            plot_sico_readout(mod2)
        else:
            display_sampler_model(mod2)
    # Temporary saves in case craps out
    return {'XTreg': XTreg, 'logXTmult': logXTmult, 'Greg': Greg, 'model': mod2}
# END sico_reg_path


# Produce best model (based on training data) from XX iterations
def produce_best_model( 
    ds_trn, ds_val, drift, LorR, XTreg, logXTmult, Greg, NE=2, NI=2, n_iter=16, LLn_trn=0, LLn_val=0, nlags=None,
    time_covariates=0, device=None, save_models=False, to_plot=True ):
    """
    This implements a simple model selection procedure where we fit n_iter models 
    with the same parameters and pick the best one based on validation data
    """
    if nlags is None:
        nlags = 12
        print("  Using default nlags = %d"%nlags)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("  PBM WARNING: device not entered, using device:", device)

    #print('NE, NI = %d, %d'%(NE, NI))
    mods = []
    LLs = np.zeros([n_iter, 2])
    for ii in range(n_iter):
        t0=time()

        # Initial model: unconstrained mask on one side (less dominant eye)
        sico_iter = baseline_sico(NE,NI, LorR=LorR, seed=101+ii, bi_bias=True, nlags=nlags, time_covariates=time_covariates,
                                  sample_layer=False,
                                  XTreg=XTreg, logXTmult=logXTmult, Greg=Greg/10, drift_term=drift ).to(device)  
        utils.fit_lbfgs( sico_iter, ds_trn[:], verbose=0, max_iter=2000)

        # Brings in mask so selecting best disparity for each filter
        sico_iter = refine_binocular( sico_iter, ds_trn, #ds_val, LLnull=LLn_val, 
                                     to_plot=False, device=device )

        # Centers and refits
        sico_iter = center_model( sico_iter, include_binoc=True ).to(device)
        sico_iter.networks[0].layers[2].reg.vals['glocalx'] = Greg
        utils.fit_lbfgs( sico_iter, ds_trn[:], verbose=0, max_iter=2000)
        LLs[ii,0] = LLn_val - sico_iter.eval_models(ds_val[:], null_adjusted=False)[0]
        LLs[ii,1] = LLn_trn - sico_iter.eval_models(ds_trn[:], null_adjusted=False)[0] 
        t1 = time()
        print("  %2d  %8.5f  %8.5f (%0.2f min)"%(ii, LLs[ii,1], LLs[ii,0],(t1-t0)/60 ))
        mods.append(deepcopy(sico_iter))
            
    a = np.argmax(LLs[:,1])
    print( "  %d-%d best model (%d) LLs (val/trn): "%(NE, NI, a), LLs[a,:])
    best_mod = deepcopy(mods[a].to(torch.device("cpu")))
    if to_plot:
        best_mod.plot_filters()
        plot_conv_layer(best_mod)
        plot_sico_readout(best_mod)
    if save_models:
        return best_mod, mods, LLs
    else:
        return best_mod
# END produce_best_model()

def extend_binocular_model( mod0, addEorI=0, LorR=0, seed=101 ):
    """
    Extend a binocular model by adding one excitatory or inhibitory unit. This preserves the previous model structure
    but generally not needing to be repeated too much since very little randomness except for new filter.

    Args:
        mod0: previous model to extend
        addEorI: 0 to add excitatory, 1 to add inhibitory
        LorR: ocular dominance for new filter (default=0)
        seed: random seed for new filter (default=101)
    """
    from NDNT.modules.layers import BinocShiftLayer

    if isinstance(mod0.networks[0].layers[1], BinocShiftLayer):
        sample_layer = True
        LorR = mod0.networks[0].layers[1].LorR
    else:
        sample_layer = False
    NI0 = mod0.networks[0].layers[1].num_inh
    NE0 = mod0.networks[0].layers[1].num_filters - NI0

    if addEorI == 0: # add one excitation
        NE = NE0 + 1
        NI = NI0
    else: # add one inhibition
        NE = NE0
        NI = NI0 + 1

    XTcoupled = True
    logXTmult = 0
    if 'd2x' in mod0.networks[0].layers[0].reg.vals:
        if mod0.networks[0].layers[0].reg.vals['d2x'] > 0:
            XTcoupled = False
            logXTmult = np.log10(mod0.networks[0].layers[0].reg.vals['d2t']/mod0.networks[0].layers[0].reg.vals['d2x'])

    mod1 = baseline_sico(NE, NI, LorR=LorR, seed=seed, 
                         XTreg=mod0.networks[0].layers[0].reg.vals['d2xt'], Greg=mod0.networks[0].layers[2].reg.vals['glocalx'],
                         logXTmult=logXTmult,
                         nlags=mod0.networks[0].layers[0].filter_dims[-1],
                         sample_layer=sample_layer,
                         drift_term=mod0.networks[1].layers[0].weight.data.cpu().numpy(),
                         time_covariates=False).to(mod0.device)

    # Copy model parameters appropriately
    weight_mapping = np.concatenate( (np.arange(NE0), np.arange(NE, NE+NI0)) )
    mod1.networks[0].layers[0].weight.data[:, weight_mapping] = mod0.networks[0].layers[0].weight.data.clone()

    mod1.networks[0].layers[1].weight.data[:, weight_mapping] = mod0.networks[0].layers[1].weight.data.clone()
    mod1.networks[0].layers[1].bias.data[weight_mapping] = mod0.networks[0].layers[1].bias.data.clone()
    if sample_layer:
        mod1.networks[0].layers[1].shifts.data[weight_mapping] = mod0.networks[0].layers[1].shifts.data.clone()
        mod1.networks[0].layers[1].sigmas.data[weight_mapping] = mod0.networks[0].layers[1].sigmas.data.clone()
    else:
        mod1.networks[0].layers[1].mask.data[:, weight_mapping] = mod0.networks[0].layers[1].mask.data.clone()
    # Not currently doing readout layer -- would have to reshape first

    return mod1
# END extend_binocular_model()


def prune_binocular_model( mod0, subunit_n=None ):
    """
    Extend a binocular model by adding one excitatory or inhibitory unit. This preserves the previous model structure
    but generally not needing to be repeated too much since very little randomness except for new filter.
    
    Args:
        mod0: previous model to extend
        subunit_n: index of the subunit to remove
    """
    from NDNT.modules.layers import BinocShiftLayer

    assert subunit_n is not None, "Must enter subunit_n to prune model"
    NI0 = mod0.networks[0].layers[1].num_inh
    NE0 = mod0.networks[0].layers[1].num_filters - NI0
    assert subunit_n < (NE0+NI0), "subunit_n must be less than total number of subunits"
    
    if isinstance(mod0.networks[0].layers[1], BinocShiftLayer):
        sample_layer = True
        LorR = mod0.networks[0].layers[1].LorR
    else:
        sample_layer = False
        LorR=0  # doesnt matter since mask is copied
    
    if subunit_n < NE0:
        NE = NE0 - 1
        NI = NI0
    else:
        NE = NE0
        NI = NI0 - 1

    XTcoupled = True
    logXTmult = 0
    time_cov = len(mod0.networks) > 3

    if 'd2x' in mod0.networks[0].layers[0].reg.vals:
        d2xt = mod0.networks[0].layers[0].reg.vals['d2x']
        if mod0.networks[0].layers[0].reg.vals['d2x'] > 0:
            XTcoupled = False
            logXTmult = np.log10(mod0.networks[0].layers[0].reg.vals['d2t']/mod0.networks[0].layers[0].reg.vals['d2x'])
    else:
        d2xt = None

    mod1 = baseline_sico(NE, NI, LorR=LorR, nlags=mod0.networks[0].layers[0].filter_dims[-1],
                         XTreg=d2xt, Greg=mod0.networks[0].layers[2].reg.vals['glocalx'], logXTmult=logXTmult,
                         sample_layer=sample_layer,
                         drift_term=mod0.networks[1].layers[0].weight.data.cpu().numpy(),
                         time_covariates=time_cov)

    # Copy model parameters appropriately
    weight_mapping = np.array(list(set(np.arange(NE0+NI0)) - set([subunit_n])), dtype=int)
    mod1.networks[0].layers[0].weight.data = mod0.networks[0].layers[0].weight.data[:, weight_mapping].clone()

    mod1.networks[0].layers[1].weight.data = mod0.networks[0].layers[1].weight.data[:, weight_mapping].clone()
    mod1.networks[0].layers[1].bias.data = mod0.networks[0].layers[1].bias.data[weight_mapping].clone()
    if sample_layer:
        mod1.networks[0].layers[1].shifts.data = mod0.networks[0].layers[1].shifts.data[weight_mapping].clone()
        mod1.networks[0].layers[1].sigmas.data = mod0.networks[0].layers[1].sigmas.data[weight_mapping].clone()
    else:
        mod1.networks[0].layers[1].mask.data = mod0.networks[0].layers[1].mask.data[:, weight_mapping].clone()
    # Also do readout layer -- have to reshape first
    tar_dims = mod0.networks[0].layers[2].filter_dims[:2]
    mod1.networks[0].layers[2].weight.data = mod0.networks[0].layers[2].weight.data.clone().reshape(tar_dims)[weight_mapping, :].reshape([-1,1])
    if len(mod0.networks) == 1:
        mod1.networks[0].layers[2].bias.data = mod0.networks[0].layers[2].bias.data.clone()
    else:
        for ii in range(1, len(mod0.networks)):
            mod1.networks[ii] = deepcopy(mod0.networks[ii]).to(torch.device("cpu"))

    return mod1
# END prune_binocular_model()


def prune_binocular_subunit( mod0, ds_trn, LLthresh=0.05, refit=True, LLnullTR=None, verbose=True ):
    """
    Will reduce model by one subunit, refit (or not) and judge change of each in training performance.
    Will activate LLnull if want decision to be returned rather than whole dictionary.
    LLthresh should be a fraction, but needs LLnullTR if so. Leave it at none if wants to remove the 
    highest unit regardless.
    """
    device = ds_trn[0]['robs'].device
    NI = mod0.networks[0].layers[1].num_inh
    NE = mod0.networks[0].layers[1].num_filters - NI
    NF = NE+NI
    mod0 = mod0.to(device)
    mods, dLLs = [], []
    # Compute training LL (no need for null)
    LL0 = mod0.eval_models(ds_trn[:])
    LLs = np.zeros([NF,2], dtype=np.float32)
    for ii in range(NE+NI):
        mod_iter = prune_binocular_model( mod0, subunit_n=ii ).to(device)
        LLs[ii,0] = mod_iter.eval_models(ds_trn[:])[0]
        if refit:
            mod_iter.networks[0].layers[0].set_parameters(val=False)
            mod_iter.networks[0].layers[1].set_parameters(val=False)
            for jj in range(1, len(mod0.networks)):
                mod_iter.set_parameters(ffnet_target=jj, val=False)
            mod_iter.networks[-1].layers[-1].set_parameters(name='bias', val=True)               
            utils.fit_lbfgs( mod_iter, ds_trn[:], verbose=0, max_iter=2000) #, line_search=None)
            LLs[ii,1] = mod_iter.eval_models(ds_trn[:])[0]
            dLLs.append(deepcopy(LL0-LLs[ii,:]))
        else:
            dLLs.append(deepcopy(LL0-LLs[ii,0]))
        mods.append(deepcopy(mod_iter.to(torch.device('cpu'))))
        if verbose:
            print("  Subunit %2d:"%ii, dLLs[-1])

    return {'mods':mods, 'dLLs': np.array(dLLs, dtype=np.float32)}
# END prune_binocular_subunit()


def binocular_cull_path( mod0, ds_trn, LLn, verbose=True ):
    mod_iter = deepcopy(mod0)
    if isinstance(LLn, dict):
        LLn = LLn['train']
    LL0 = LLn - mod0.eval_models(ds_trn[:])[0]
    dLLlist = []
    mods = [deepcopy(mod0)]
    NI = mod0.networks[0].layers[1].num_inh
    NE = mod0.networks[0].layers[1].num_filters-NI
    keep_going = True
    LLiter = LL0
    while keep_going:
        print("Evaluate pruning E%d I%d:"%(NE, NI), LLiter)
        cull_dict = prune_binocular_subunit( mod_iter, ds_trn, verbose=verbose )
        best_clip = np.argmax(cull_dict['dLLs'][:,1])
        mod_iter = deepcopy(cull_dict['mods'][best_clip])
        mods.append(deepcopy(mod_iter))
        dLLlist.append(cull_dict['dLLs'][best_clip,1])
        NI = mod_iter.networks[0].layers[1].num_inh
        NE = mod_iter.networks[0].layers[1].num_filters-NI
        LLiter = LLn - mod_iter.eval_models(ds_trn[:])[0]
        print("E%d I%d (elim sub%d: frac change %5.2f, cumulative %5.2f: "%(NE, NI, best_clip, dLLlist[-1]/LL0, (LL0-LLiter)/LL0) )
        if NE+NI <= 2:
            keep_going = False
    return {'mods':mods, 'dLLs':dLLlist, 'LLmax': LL0}
# END binocular_cull_path()


def load_sicos( dataloc, ee, cc, id=None, verbose=False ):
    """
    Load a saved SICO model from a directory. If id is None, will load the first one found.
    """
    import re, os
    from NDNT.NDN import NDN
    mods = []

    relevant_fns = []
    fn_list = os.listdir(dataloc)
    for fn in fn_list:
        if fn.__contains__('ndn'):
            m = re.search(r"(\d{1,2})c(\d{1,2})", fn)
            if m:
                expt_n, cell_n = map(int, m.groups())
                if (expt_n == ee) and (cell_n == cc):
                    if id is None:
                        relevant_fns.append(fn)
                    elif fn.__contains__(id):
                        relevant_fns.append(fn)
    if len(relevant_fns) == 0:
        print('No files with appropriate formating found in', dataloc) 
    else:
        fns = sorted(relevant_fns)
        if verbose:
            print('Found %d relevant models:'%(len(relevant_fns)))
            for ii in range(len(relevant_fns)):
                print('  %d: %s'%(ii, fns[ii]))
        else:
            print('Found %d relevant models, e.g.,'%(len(relevant_fns)), fns[0])
        for fn in fns:
            mods.append( NDN.load_model(os.path.join(dataloc, fn)).to(torch.device('cpu')) )
    return mods
# END load_sicos()


def increment_models(ds_trn, ds_val, modlist=None, addEorI=0, cull_list=True):
    """
    This implements a simple model selection procedure where we fit n_iter models

    Args:
        ds_trn: training dataset
        ds_val: validation dataset
        modlist: list of models to increment (if None, will use the first model in modlist)
        addEorI: 0 to add excitatory, 1 to add inhibitory
        cull_list: if True, will cull the list of models to only those that improved
    """
    from NDNT.modules.layers import BinocShiftLayer

    has_sample_layer = isinstance(modlist[0].networks[0].layers[1], BinocShiftLayer)
    device = ds_trn[:3]['robs'].device
    #print('NE, NI = %d, %d'%(NE, NI))
    num_copies = len(modlist)
    mods = []
    LLprev = np.zeros([num_copies, 2])
    LLnew = np.zeros([num_copies, 2])
    #running_shifts = None  # track shifts to save time to not have to center all the time
    new_mods = []
    for ii in range(num_copies):
        t0=time()
        # Initial model: unconstrained mask on one side (less dominant eye)
        LLprev[ii,0] = modlist[ii].eval_models(ds_trn[:], null_adjusted=False)[0]  # training LL (used for decisions)
        LLprev[ii,1] = modlist[ii].eval_models(ds_val[:], null_adjusted=False)[0]  # validation LL (just as additional info)

        sico_iter = extend_binocular_model(modlist[ii], addEorI=addEorI, seed=101+ii).to(device) 
        sico_iter.networks[0].layers[2].reg.vals['glocalx'] *= 0.1
        #if running_shifts is not None:
        utils.fit_lbfgs( sico_iter, ds_trn[:], verbose=0, max_iter=2000, line_search=None)  # this seems fragile w Strong-Wolfe

        # Centers and refits
        sico_iter.networks[0].layers[2].reg.vals['glocalx'] *= 10
        if has_sample_layer:
            sico_iter = center_model(sico_iter, include_binoc=True, verbose=False)
            # ratchet in sigmas over three steps
            if np.max(sico_iter.networks[0].layers[1].sigmas.data.cpu().numpy()) > 1.0:
                num_over = np.sum(sico_iter.networks[0].layers[1].sigmas.data.cpu().numpy() > 1.0)
                print("       Highest sigma %0.2f (%d over 1.0). Decreasing to 1"%(np.max(sico_iter.networks[0].layers[1].sigmas.data.cpu().numpy()), num_over))
                sico_iter.networks[0].layers[1].fit_shifts(val=True, fixed_sigmas=True, sigma0 = 1.0)
                utils.fit_lbfgs( sico_iter, ds_trn[:], verbose=0, max_iter=2000, line_search=None)
            if np.max(sico_iter.networks[0].layers[1].sigmas.data.cpu().numpy()) > 0.6:
                num_over = np.sum(sico_iter.networks[0].layers[1].sigmas.data.cpu().numpy() > 0.6)
                print("       Highest sigma %0.2f (%d over 0.6). Decreasing to 0.6"%(np.max(sico_iter.networks[0].layers[1].sigmas.data.cpu().numpy()), num_over))
                sico_iter.networks[0].layers[1].fit_shifts(val=True, fixed_sigmas=True, sigma0 = 0.6)
                utils.fit_lbfgs( sico_iter, ds_trn[:], verbose=0, max_iter=2000, line_search=None)
            sico_iter.networks[0].layers[1].fit_shifts(val=False)
        else:
            sico_iter = refine_binocular( sico_iter, ds_trn, to_plot=False, device=device )
            sico_iter = center_model(sico_iter, include_binoc=True, verbose=False)
        
        utils.fit_lbfgs( sico_iter, ds_trn[:], verbose=0, max_iter=2000, line_search=None)
        #print("   Refinement time: %0.2f min"%( (t2-t1)/60 ))
        LLnew[ii,0] = sico_iter.eval_models(ds_trn[:], null_adjusted=False)[0] 
        LLnew[ii,1] = sico_iter.eval_models(ds_val[:], null_adjusted=False)[0]
        t2 = time()
        print("  %2d deltas: tr %8.5f  val %8.5f (%0.2f min): "%(ii, LLprev[ii,0]-LLnew[ii,0], LLprev[ii,1]-LLnew[ii,1],(t2-t0)/60), end='')
        NI = sico_iter.networks[0].layers[1].num_inh
        NE = sico_iter.networks[0].layers[1].num_filters - NI
        if cull_list is not None:
            if LLnew[ii,0] < LLprev[ii,0]:
                print('  Keeper (%d-%d)'%(NE, NI))
                new_mods.append(deepcopy(sico_iter).to(torch.device("cpu")))
            else:
                print('  Reject' )
                new_mods.append(deepcopy(modlist[ii]).to(torch.device("cpu")))  # keep old version to check for inh
        #running_shifts = deepcopy(sico_iter.networks[0].layers[1].x_fixed.data.cpu().numpy())

    a = np.nanargmin(LLnew[:,0])
    best_mod = deepcopy(new_mods[a].to(torch.device("cpu")))
    NI = best_mod.networks[0].layers[1].num_inh
    NE = best_mod.networks[0].layers[1].num_filters - NI
    print( "  %d-%d best model (%d) dLLs (trn/val): "%(NE, NI, a), LLnew[a,:]-LLprev[a,:])
    #if to_plot:
    #    display_sampler_model(best_mod)
    #if save_models:
    return best_mod, new_mods, LLnew
# END increment_models()


def produce_best_sampler_model( 
    ds_trn, ds_val, drift, LorR, XTreg, logXTmult, Greg, NE=2, NI=2, n_iter=8, LLn_trn=0, LLn_val=0, nlags=None,
    time_covariates=0, device=None, save_models=False, to_plot=True ):
    """
    This implements a simple model selection procedure where we fit n_iter models 
    with the same parameters and pick the best one based on validation data
    """
    if nlags is None:
        nlags = 12
        print("  Using default nlags = %d"%nlags)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("  PBM WARNING: device not entered, using device:", device)

    #print('NE, NI = %d, %d'%(NE, NI))
    mods = []
    LLs = np.zeros([n_iter, 2])
    running_shifts = 0  # track shifts to save time to not have to center all the time
    for ii in range(n_iter):
        t0=time()

        # Initial model: unconstrained mask on one side (less dominant eye)
        sico_iter = baseline_sico(NE,NI, LorR=LorR, seed=101+ii, bi_bias=True, nlags=nlags, time_covariates=time_covariates,
                                  sample_layer=True, shift_start=running_shifts,
                                  XTreg=XTreg, logXTmult=logXTmult, Greg=Greg/10, drift_term=drift ).to(device)
        #sico_iter.list_parameters()
        utils.fit_lbfgs( sico_iter, ds_trn[:], verbose=0, max_iter=2000, line_search=None)  # this seems fragile w Strong-Wolfe
        # Centers and refits
        sico_iter.networks[0].layers[2].reg.vals['glocalx'] = Greg
        sico_iter = center_model(sico_iter, include_binoc=True, verbose=False)
        t1 = time()
        # ratchet in sigmas over three steps
        #if np.max(sico_iter.networks[0].layers[1].sigmas.data.cpu().numpy()) > 1.0:
        #    print("       Highest sigma %0.2f. Decreasing to 1"%np.max(sico_iter.networks[0].layers[1].sigmas.data.cpu().numpy()))
        #    sico_iter.networks[0].layers[1].fit_shifts(val=True, fixed_sigmas=True, sigma0 = 1.0)
        #    utils.fit_lbfgs( sico_iter, ds_trn[:], verbose=0, max_iter=2000, line_search=None)
        #if np.max(sico_iter.networks[0].layers[1].sigmas.data.cpu().numpy()) > 0.6:
        #    print("       Highest sigma %0.2f. Decreasing to 0.6"%np.max(sico_iter.networks[0].layers[1].sigmas.data.cpu().numpy()))
        #    sico_iter.networks[0].layers[1].fit_shifts(val=True, fixed_sigmas=True, sigma0 = 0.6)
        #    utils.fit_lbfgs( sico_iter, ds_trn[:], verbose=0, max_iter=2000, line_search=None)
        
        sico_iter.networks[0].layers[1].fit_shifts(val=False)
        utils.fit_lbfgs( sico_iter, ds_trn[:], verbose=0, max_iter=2000, line_search=None)
        t2 = time()
        #print("   Refinement time: %0.2f min"%( (t2-t1)/60 ))
        LLs[ii,0] = LLn_val - sico_iter.eval_models(ds_val[:], null_adjusted=False)[0]
        LLs[ii,1] = LLn_trn - sico_iter.eval_models(ds_trn[:], null_adjusted=False)[0] 
        t2 = time()
        print("  %2d  %8.5f  %8.5f (%0.2f min)"%(ii, LLs[ii,1], LLs[ii,0],(t2-t0)/60 ))
        mods.append(deepcopy(sico_iter).to(torch.device("cpu")))
        running_shifts = deepcopy(sico_iter.networks[0].layers[1].x_fixed.data.cpu().numpy())

    a = np.nanargmax(LLs[:,1])
    print( "  %d-%d best model (%d) LLs (val/trn): "%(NE, NI, a), LLs[a,:])
    best_mod = deepcopy(mods[a].to(torch.device("cpu")))
    if to_plot:
        display_sampler_model(best_mod)
    if save_models:
        return best_mod, mods, LLs
    else:
        return best_mod
# END produce_best_sampler_model()


def refine_binocular( mod0, train_data, to_plot=True, device=None ):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("  RefBinoc WARNING: device not entered, using device:", device)

    mod = deepcopy(mod0).to(device)
    # determine where the mask needs to be
    w = mod.get_weights(layer_target=1)
    LR = int( len(np.where(np.sum(w[0,...],axis=1) > 0)[0]) == 1 )
    _, NX, NF = w.shape
    #if LLnull is None:
    #    LL = mod0.eval_models(val_data[:], null_adjusted=True)[0]
    #else:
    #    LL = LLnull - mod0.eval_models(val_data[:], null_adjusted=False)[0]
    #if to_plot:
    #    print("  Initial:%9.6f"%LL)
    for maskwidth in [2,1,0]:
        w = mod.get_weights(layer_target=1)
        mod.networks[0].layers[1].mask.data[NX*LR+np.arange(NX),:] = 0.0
        for ii in range(NF):
            x = np.argmax(w[LR, :, ii])
            if maskwidth > 0:
                rng = np.arange(np.maximum(x-maskwidth,0), np.minimum(x+maskwidth,NX-1))
                mod.networks[0].layers[1].mask.data[NX*LR+rng, ii] = 1.0
            else:
                mod.networks[0].layers[1].mask.data[NX*LR+x, ii] = 1.0
        utils.fit_lbfgs( mod, train_data[:], verbose=0, max_iter=1000)
        #if LLnull is None:
        #    LL = mod.eval_models(val_data[:], null_adjusted=True)[0]
        #else:
        #    LL = LLnull - mod.eval_models(val_data[:], null_adjusted=False)[0]

        if to_plot:
            #print("  Mask w%d:%9.6f"%(maskwidth,LL))
            plot_conv_layer(mod)
    #mod = mod.to(torch.device("cpu"))
    #if not to_plot:
    #    print("LL =%9.6f"%(LL))
    return mod


############## SICO CREATION / MANIPULATION FUNCTIONS ##############
def baseline_sico(NE, NI, LorR=0, seed=100, XTreg=0.01, logXTmult=0, Greg=0.001, Dreg=1.0, nlags=None,
                  sample_layer=True, shift_start=0, bi_bias=True,
                  drift_term=None, time_covariates=0 ):
    """
    Make standard binocular model with given size and defaults -- with drift term
    logXTmult=0 means that d2x and d2t are the same, so use d2xt, otherwise separately define them based on factor of 10
    """
    from NDNT.NDN import NDN
    from NDNT.networks import FFnetwork
    from NDNT.modules.layers import NDNLayer, ConvLayer, MaskConvLayer, BinocShiftLayer

    if nlags is None:
        nlags = 12
        print("  Using default nlags = %d"%nlags)

    num_mfilts = NE+NI
    mfw = 21
    bfw = 11
    CregM = 0.0001

    # define reg values for first layer
    reg_vals = {'center':CregM}
    if logXTmult == 0:
        reg_vals['d2xt'] = XTreg
    else:
        reg_vals['d2x'] = XTreg
        reg_vals['d2t'] = XTreg*(10.0**logXTmult)
        
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

    readout_par = NDNLayer.layer_dict(
        num_filters=1, bias=True, initialize_center=True, pos_constraint=True,
        NLtype='softplus', reg_vals={'glocalx': Greg }) 
    
    if drift_term is not None:
        # Stim net
        readout_par['NLtype'] = 'lin'
        readout_par['bias'] = False
        stim_net = FFnetwork.ffnet_dict( layer_list = [monoc_basis_par, bfilt_par, readout_par] )
        # Drift net
        drift_pars = NDNLayer.layer_dict( 
            input_dims=[1,1,1,len(drift_term)], num_filters=1, bias=False, norm_type=0, NLtype='lin',
            reg_vals = {'d2t': Dreg, 'bcs':{'d2t':0} })
        drift_net = FFnetwork.ffnet_dict( xstim_n='Xdrift', layer_list=[drift_pars] )

        if time_covariates > 0:
            time_pars = NDNLayer.layer_dict( 
                input_dims=[1,1,1,time_covariates], num_filters=1, bias=False, norm_type=0, NLtype='lin',
            reg_vals = {'d2t': Dreg, 'bcs':{'d2t':0} })
            frame_net = FFnetwork.ffnet_dict( xstim_n='Xframe_switch', layer_list=[time_pars] )

        # Comb net
        comb_par = NDNLayer.layer_dict(num_filters=1, NLtype='softplus', bias=True, weights_initializer='ones')
        if time_covariates > 0:
            comb_net = FFnetwork.ffnet_dict( xstim_n=None, ffnet_n=[0,1,2], layer_list = [comb_par], ffnet_type='add')
            sico = NDN(ffnet_list=[stim_net, drift_net, frame_net, comb_net], seed=seed)
            sico.set_parameters(val=False, name='weight', ffnet_target=3)
        else:
            comb_net = FFnetwork.ffnet_dict( xstim_n=None, ffnet_n=[0,1], layer_list = [comb_par], ffnet_type='add')
            sico = NDN(ffnet_list=[stim_net, drift_net, comb_net], seed=seed)
            sico.set_parameters(val=False, name='weight', ffnet_target=2)
        
        # Fix drift term and do not fit
        sico.networks[1].layers[0].weight.data[:,0] = torch.tensor(drift_term.squeeze(), dtype=torch.float32)
        sico.set_parameters(val=False, ffnet_target=1)
    else:
        sico = NDN(layer_list=[monoc_basis_par, bfilt_par, readout_par], seed=seed)

    if not sample_layer:
        sico.networks[0].layers[1].set_mask(masks[LorR])
    
    return sico
# END baseline_sico()


def center_filter( k0, dfloor=0.2 ):
    # Make centered filter list based on disparity
    g = np.var(k0, axis=1 )
    #plt.plot(g)
    g = np.maximum(g-np.max(g)*dfloor, 0)
    NX = len(g)
    #plt.plot(g)
    #plt.show()
    sh = -int(np.round(utils.dist_mean(g)-NX/2))
    #print(sh)
    return deepcopy(np.roll(k0, sh,axis=0))


def center_binoc_mask( mod, filters=None ):
    # Center binocular filter and masks
    new_mod = deepcopy(mod)
    NF = mod.networks[0].layers[0].num_filters
    NX = mod.networks[0].layers[1].filter_dims[1]
    mask = np.zeros([2, NX, NF], dtype=np.float32)
    bws = np.zeros([2, NX, NF], dtype=np.float32)

    ds = disparities( mod )
    for ii in range(NF):
        xL = NX//2 - ds[ii]//2
        xR = xL + ds[ii]
        mask[0,xL,ii] = 1.0
        mask[1,xR,ii] = 1.0
        bws[0,xL,ii] = np.sqrt(2)
        bws[1,xR,ii] = np.sqrt(2)

    new_mod.networks[0].layers[1].weight.data = torch.tensor( bws.reshape([-1,NF]), dtype=torch.float32, device=mod.device)
    new_mod.networks[0].layers[1].mask.data = torch.tensor( mask.reshape([-1,NF]), dtype=torch.float32, device=mod.device)
    if filters is not None:
        assert np.prod(filters.shape) == np.prod(mod.networks[0].layers[0].filter_dims)*NF, "wrong size"
        new_mod.networks[0].layers[0].weight.data = torch.tensor( filters.reshape([-1, NF]), dtype=torch.float32, device=mod.device)
    return new_mod
# END center_binoc_mask()


def center_model( mod0, include_binoc=False, verbose=True ):
    """
    Centers filters and potentially binocular filters of model
    """
    from NDNT.modules.layers import MaskConvLayer, BinocShiftLayer

    ks0 = mod0.get_weights()
    NF = ks0.shape[-1]
    ks1 = deepcopy(ks0)
    for ii in range(NF):
        ks1[..., ii] = center_filter(ks0[...,ii])
    mod1 = deepcopy(mod0)
    mod1.networks[0].layers[0].weight.data = torch.tensor( 
        ks1.reshape([-1, NF]), dtype=torch.float32, device=mod0.device)

    if include_binoc:
        if isinstance(mod1.networks[0].layers[1], MaskConvLayer):
            return center_binoc_mask( mod1, filters=ks1 )
        elif isinstance(mod1.networks[0].layers[1], BinocShiftLayer):
            mod1.networks[0].layers[1].center_filters(round_shifts=True, verbose=verbose)
        else:
            print("WARNING: Binocular layer unidentified: not centering.")
    return mod1


############# GENERAL MODEL MEASUREMENT UTILITY FUNCTIONS #############
def ocular_dominance( data, frac=False, verbose=False, nlags=12 ):
    """
    Compute rough ocular dominance (which eye more strongly drives) based on STA. This is mostly to 
    prevent using an eye where there is no signal rather than be a tie-breaker (where either would work)"""
    spatial_power = np.var((data[:]['stim'].T@data[:]['robs']).detach().cpu().numpy().reshape([-1,nlags]), axis=1)
    a = [np.max(spatial_power[:36]), np.max(spatial_power[36:])]
    a = a/np.sum(a)
    dom = int(a[0] < a[1])
    if verbose:
        print(a, dom)
    if frac:
        return a[0] # fractional dominance of left eye
    else:
        return dom # which eye is stronger
# END ocular dominance


def disparities( mod, return_ws=False ):
    w = mod.get_weights(layer_target=1)
    ds = np.argmax(w[1,:,:], axis=0)-np.argmax(w[0,:,:], axis=0)
    NF = w.shape[-1]
    if return_ws:
        ws = np.max(w, axis=1)
        return ds, ws
    else:
        return ds


def display_sampler_model( mod, sample=True ):
    mod.plot_filters()
    mod.networks[0].layers[1].plot_filters(sample=sample)
    plot_sico_readout(mod)
# END display_sampler_model()


def plot_conv_layer( model, layer_target=1 ):
    """Plots binocular convolutional layer in Bi2026 model-type"""
    w = model.get_weights(layer_target=layer_target)
    utils.ss(rh=2)
    NF = w.shape[-1]
    utils.imagesc(w.reshape([-1,NF]), cmap='seismic', balanced=True)
    L = w.shape[0]*w.shape[1]
    plt.axvline(L//2-0.5,color='k',linewidth=2)
    plt.axvline((L//2)/2-0.5,color='c',linestyle='--',linewidth=2)
    plt.axvline(L//2+(L//2)/2-0.5,color='c',linestyle='--',linewidth=2)
    plt.show()
# END plot_conv_layer() 
# mod.plot_filters() gets first layer
# BU.plot_sico_readout() gets final layer