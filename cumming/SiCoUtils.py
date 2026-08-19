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
              n_iter=16, time_covariates=True, expt_n=None, cell_n=None, device=None):
    """
    Fit a series of SICO models with increasing numbers of excitatory and inhibitory filters
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
    save_name += "path"
    
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
    if sample_layer:
        mod_path = [produce_best_sampler_model(ds_trn, ds_val, drift_term, LR, XTreg, logXTmult, Greg, NE=1, NI=1, time_covariates=time_covariates,
                                    n_iter=n_iter, nlags=nlags, LLn_trn=LLn_trn, LLn_val=LLn_val, device=device, to_plot=True)]
    else:
        mod_path = [produce_best_model(ds_trn, ds_val, drift_term, LR, XTreg, logXTmult, Greg, NE=1, NI=1, time_covariates=time_covariates,
                                    n_iter=n_iter, nlags=nlags, LLn_trn=LLn_trn, LLn_val=LLn_val, device=device, to_plot=True)]
    
    LLprev = LLn_val - mod_path[0].eval_models(ds_val[:], null_adjusted=False)[0]
    #print("1-1: LL = %0.5f"%LLprev)
    mod_path[0].save_model(save_name+"1_1.ndn")
    
    no_stop=True
    NE, NI = 1,1
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
        if sample_layer:
            sicoE1 = produce_best_sampler_model(ds_trn, ds_val, drift_term, LR, XTreg, logXTmult, Greg, NE=NE, NI=NI, time_covariates=time_covariates,
                                                n_iter=n_iter, nlags=nlags, LLn_trn=LLn_trn, LLn_val=LLn_val, device=device, to_plot=False)
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
        if sample_layer:
            sicoI1 = produce_best_sampler_model(ds_trn, ds_val, drift_term, LR, XTreg, logXTmult, Greg, NE=NE, NI=NI, time_covariates=time_covariates,
                                                n_iter=n_iter, nlags=nlags, LLn_trn=LLn_trn, LLn_val=LLn_val, device=device, to_plot=False)
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
        mod1 = refine_binocular(center_model(mod1, include_binoc=False), ds_trn, ds_val, LLnull=LLn, 
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
    print('  Chosen glocalx = ', utils.string_convert(Greg), '(%d)'%bestr)

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

    print('NE, NI = %d, %d'%(NE, NI))
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
        sico_iter = refine_binocular( sico_iter, ds_trn, ds_val, LLnull=LLn_val, to_plot=False, device=device )

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

    print('NE, NI = %d, %d'%(NE, NI))
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
        if np.max(sico_iter.networks[0].layers[1].sigmas.data.cpu().numpy()) > 1.0:
            print("      Highest sigma %0.2f. Decreasing to 1"%np.max(sico_iter.networks[0].layers[1].sigmas.data.cpu().numpy()))
            sico_iter.networks[0].layers[1].fit_shifts(val=True, fixed_sigmas=True, sigma0 = 1.0)
            utils.fit_lbfgs( sico_iter, ds_trn[:], verbose=0, max_iter=2000, line_search=None)
        if np.max(sico_iter.networks[0].layers[1].sigmas.data.cpu().numpy()) > 0.6:
            print("      Highest sigma %0.2f. Decreasing to 0.6"%np.max(sico_iter.networks[0].layers[1].sigmas.data.cpu().numpy()))
            sico_iter.networks[0].layers[1].fit_shifts(val=True, fixed_sigmas=True, sigma0 = 0.6)
            utils.fit_lbfgs( sico_iter, ds_trn[:], verbose=0, max_iter=2000, line_search=None)
        
        sico_iter.networks[0].layers[1].fit_shifts(val=False)
        utils.fit_lbfgs( sico_iter, ds_trn[:], verbose=0, max_iter=2000, line_search=None)
        t2 = time()
        print("  Refinement time: %0.2f min"%( (t2-t1)/60 ))
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


def refine_binocular( mod0, train_data, val_data, to_plot=True, LLnull=None, device=None ):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("  RefBinoc WARNING: device not entered, using device:", device)

    mod = deepcopy(mod0).to(device)
    # determine where the mask needs to be
    w = mod.get_weights(layer_target=1)
    LR = int( len(np.where(np.sum(w[0,...],axis=1) > 0)[0]) == 1 )
    _, NX, NF = w.shape
    if LLnull is None:
        LL = mod0.eval_models(val_data[:], null_adjusted=True)[0]
    else:
        LL = LLnull - mod0.eval_models(val_data[:], null_adjusted=False)[0]
    if to_plot:
        print("  Initial:%9.6f"%LL)
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
        if LLnull is None:
            LL = mod.eval_models(val_data[:], null_adjusted=True)[0]
        else:
            LL = LLnull - mod.eval_models(val_data[:], null_adjusted=False)[0]

        if to_plot:
            print("  Mask w%d:%9.6f"%(maskwidth,LL))
            plot_conv_layer(mod)
    mod = mod.to(torch.device("cpu"))
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

        #mod.networks[0].layers[1].mask.clone().reshape(
        #list(mod.networks[0].layers[1].filter_dims) + [10]).squeeze()
    ds = disparities( mod )
    for ii in range(NF):
        xL = NX//2 - ds[ii]//2
        xR = xL + ds[ii]
        mask[0,xL,ii] = 1.0
        mask[1,xR,ii] = 1.0
        bws[0,xL,ii] = np.sqrt(2)
        bws[1,xR,ii] = np.sqrt(2)

    new_mod.networks[0].layers[1].weight.data = torch.tensor( bws.reshape([-1,NF]), dtype=torch.float32)
    new_mod.networks[0].layers[1].mask.data = torch.tensor( mask.reshape([-1,NF]), dtype=torch.float32)
    if filters is not None:
        assert np.prod(filters.shape) == np.prod(mod.networks[0].layers[0].filter_dims)*NF, "wrong size"
        new_mod.networks[0].layers[0].weight.data = torch.tensor( filters.reshape([-1, NF]), dtype=torch.float32)
    return new_mod


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