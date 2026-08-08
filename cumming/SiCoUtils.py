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
def sico_path(ds_trn, ds_val, LLn_trn=0, LLn_val=0, drift_term=None, XTreg=None, Greg=None, 
              n_iter=16, nlags=None, device=None ):
    assert drift_term is not None, "Need to enter 'drift_term'"
    if nlags is None:
        nlags = 12
        print("  Using default nlags = %d"%nlags)

    # Determine LR
    LR = ocular_dominance( ds_trn, verbose=False )
    #Rvals = [1e-6, 1e-4, 0.001, 0.01, 0.1, 1, 10]

    NE, NI = 1, 1
    # d2xt Reg path for beginner model all the way through
    if (XTreg is None) or (Greg is None):
        XTreg, Greg = sico_reg_path(ds_trn, ds_val, NE=1, NI=1, thresh=0.95, Xreg0=XTreg, Greg0=Greg,
                                    nlags=nlags, LLn=LLn_val, drift_term=drift_term, device=device,
                                    return_model=False, to_plot=True )

    # Find best model for 1-1 over n_iters
    mod_path = [produce_best_model(ds_trn, ds_val, drift_term, LR, XTreg, Greg, NE=1, NI=1, 
                                   n_iter=n_iter, nlags=nlags, LLn_trn=LLn_trn, LLn_val=LLn_val,
                                   device=device)]
    
    LLprev = LLn_val - mod_path[0].eval_models(ds_val[:], null_adjusted=False)[0]
    print("1-1: LL = %0.5f"%LLprev)
    
    no_stop=True
    NE, NI = 1,1
    iter = 0 # number of adds to E and/or I
    
    while no_stop and (iter < 6):
        no_stop = False
        if iter > 0:  # check best regularization on previous model (from last iteration)
            XTreg, Greg = sico_reg_path(ds_trn, ds_val, NE=NE, NI=NI, thresh=0.95, Xreg0=XTreg, Greg0=Greg,
                                        nlags=nlags, LLn=LLn_val, drift_term=drift_term, device=device,
                                        return_model=False, to_plot=False )

        # plus one excitation
        NE += 1
        sicoE1 = produce_best_model(ds_trn, ds_val, drift_term, LR, XTreg, Greg, NE=NE, NI=NE, nlags=nlags,
                                    n_iter=n_iter, LLn_trn=LLn_trn, LLn_val=LLn_val, device=device)
        LL = LLn_val - sicoE1.eval_models(ds_val[:], null_adjusted=False)[0]
        if LL > LLprev:
            no_stop = True
            LLprev = LL
            mod_path.append(deepcopy(sicoE1))
            print("Keeping (%d,%d): %0.5f"%(NE, NI, LL))
        else:
            print("  EXC+1 (%d,%d) no good: %0.5f < %0.5f"%(NE, NI, LL, LLprev))
            NE += -1

        # plus one inhibition
        NI += 1
        sicoI1 = produce_best_model(ds_trn, ds_val, drift_term, LR, XTreg, Greg, NE=NE, NI=NI, 
                                    nlags=nlags, n_iter=n_iter, LLn_trn=LLn_trn, LLn_val=LLn_val, device=device)
        LL = LLn_val - sicoI1.eval_models(ds_val[:], null_adjusted=False)[0]
        if LL > LLprev:
            no_stop = True
            LLprev = LL
            mod_path.append(deepcopy(sicoI1))
            print("Keeping (%d,%d): %0.5f"%(NE, NI, LL))
        else:
            print("  INH+1 (%d,%d) no good: %0.5f < %0.5f"%(NE, NI, LL, LLprev))
            NI += -1

        iter += 1

    return mod_path
# END sico_path()


def sico_reg_path(ds_trn, ds_val, NE=2, NI=2, Xreg0=None, Greg0=None, thresh=0.95, nlags=None,
                  LLn=0, drift_term=None, return_model=True, to_plot=True, device=None ):
    """reg0 is if want centered -- test order of mag in each direction"""
    assert drift_term is not None, "Need to enter 'drift_term'"

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("  Regpath WARNING: device not entered, using device:", device)
    device0 = torch.device("cpu") # for storing models on CPU

    if nlags is None:
        nlags = 12
        print("  Using default nlags = %d"%nlags)

    # Determine LR
    LR = ocular_dominance( ds_trn, verbose=False )
    if Xreg0 is None:
        Rvals = [1e-6, 1e-4, 0.001, 0.01, 0.1, 1, 10]
    else:
        Rvals = [Xreg0*0.1, Xreg0, Xreg0*10.0]

    # d2xt Reg path for beginer model all the way through
    LLsRx = np.zeros(len(Rvals))
    mods = []
    print('  XT-regpath:', Rvals)
    for ii in range(len(Rvals)): # same seed
        sico_iter = baseline_sico(NE, NI, LorR=LR, seed=101, XTreg=Rvals[ii], nlags=nlags,
                                  drift_term=drift_term).to(device) 
        utils.fit_lbfgs( sico_iter, ds_trn[:], verbose=0, max_iter=2000)
        LL = LLn - sico_iter.eval_models(ds_val[:], null_adjusted=False)[0]
        #if LL > np.max(LLsRx):
        #    mod0 = deepcopy(sico_iter).to(device0)
        mods.append(deepcopy(sico_iter).to(device0))
        LLsRx[ii] = LL
        print( "    %2d  %9.6f"%(ii, LLsRx[ii]) )
    if to_plot:
        plt.plot(LLsRx,'b')
        plt.plot(LLsRx,'bo')
        plt.axhline(np.max(LLsRx)*thresh, color='k', linestyle='--')
        plt.show()
    # Best reg
    #a = np.argmax(LLsRx)
    #print('Highest Reg 1-1 (%d):'%a, LLsRx[a])
    #mods[a].plot_filters()
    bestr = np.where(LLsRx > (np.max(LLsRx)*thresh))[0][-1]
    #print('Chosen Reg 1-1 (%d)'%bestr)
    XTreg = Rvals[bestr]
    print('  Chosen d2xt =', XTreg, '(%d)'%bestr)
    mod0 = deepcopy(mods[bestr])
    #mod0.plot_filters()
    
    # Center and refine model, and then pick best Greg
    mod1 = refine_binocular(center_model(mod0, center_binoc=False), 
                            ds_trn, ds_val, LLnull=LLn, to_plot=False )
    LL = LLn - mod1.eval_models(ds_val[:], null_adjusted=False)[0]
    print("  Refined sico%d-%d LL = %0.6f"%(NE, NI, LL) )
    mod1 = center_model( mod1, center_binoc=True )

    # now glocalx
    if Greg0 is None:
        Rvals = [1e-6, 1e-4, 0.001, 0.01, 0.1, 1, 10]
    else:
        Rvals = [Greg0*0.1, Greg0, Greg0*10.0]

    print('  glocalx-regpath:', Rvals)
    LLsRg = np.zeros(len(Rvals))
    mods = []
    for ii in range(len(Rvals)):
        sico_iter = deepcopy(mod1).to(device)
        sico_iter.networks[0].layers[2].reg.vals['glocalx'] = Rvals[ii]
        utils.fit_lbfgs( sico_iter, ds_trn[:], verbose=0, max_iter=2000)
        mods.append(deepcopy(sico_iter).to(device0))
        LL = LLn - sico_iter.eval_models(ds_val[:], null_adjusted=False)[0]
        print( "    %2d  %9.6f"%(ii, LL), end='' )
        if LL > np.max(LLsRg):
            #mod2 = deepcopy(sico_iter).to(device0)
            print(' *')
        else:
            print('')
        LLsRg[ii] = LL
    if to_plot:
        plt.plot(LLsRg,'g')
        plt.plot(LLsRg,'go')
        plt.axhline(np.max(LLsRg)*thresh, color='k', linestyle='--')
        plt.show()
    #bestr = np.argmax(LLsRg)
    bestr = np.where(LLsRg > (np.max(LLsRg)*thresh))[0][-1]
    Greg = Rvals[bestr]
    mod2 = mods[bestr]
    print('  Chosen glocalx = ', Greg, '(%d)'%bestr)

    mod2.plot_filters()
    plot_conv_layer(mod2)
    plot_sico_readout(mod2)
    if return_model:
        return XTreg, Greg, mod2
    return XTreg, Greg
# END sico_reg_path


# Produce best model (based on training data) from XX iterations
def produce_best_model( 
    ds_trn, ds_val, drift, LorR, XTreg, Greg, NE, NI, n_iter=16, LLn_trn=0, LLn_val=0, nlags=None,
    device=None, save_models=False, to_plot=True ):

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
        sico_iter = baseline_sico(NE,NI, LorR=LorR, seed=101+ii, bi_bias=True, nlags=nlags,
                                  XTreg=XTreg, Greg=Greg/10, drift_term=drift ).to(device)  
        utils.fit_lbfgs( sico_iter, ds_trn[:], verbose=0, max_iter=2000)
        #LL = LLn_val - sico_iter.eval_models(ds_val[:], null_adjusted=False)[0]
        #print( "  iter %2d:%9.6f"%(ii, LL), end='' )
        sico_iter = refine_binocular( sico_iter, ds_trn, ds_val, LLnull=LLn_val, to_plot=False )
        #LLs = LLn_val - sico_iter.eval_models(ds_val[:], null_adjusted=False)[0]
        sico_iter = center_model( sico_iter, center_binoc=True ).to(device)
        sico_iter.networks[0].layers[2].reg.vals['glocalx'] = Greg
        utils.fit_lbfgs( sico_iter, ds_trn[:], verbose=0, max_iter=2000)
        LLs[ii,0] = LLn_val - sico_iter.eval_models(ds_val[:], null_adjusted=False)[0]
        LLs[ii,1] = LLn_trn - sico_iter.eval_models(ds_trn[:], null_adjusted=False)[0] 
        t1 = time()
        print("  %2d  %8.5f  %8.5f (%0.2f min)"%(ii, LLs[ii,1], LLs[ii,0],(t1-t0)/60 ))
        mods.append(deepcopy(sico_iter))
            
    a = np.argmax(LLs[:,1])
    print( "  %d-%d best model (%d) LLs (val/trn): "%(NE, NI, a), LLs[a,:])
    best_mod = deepcopy(mods[a])
    if to_plot:
        best_mod.plot_filters()
        plot_conv_layer(best_mod)
        plot_sico_readout(best_mod)
    if save_models:
        return best_mod, mods, LLs
    else:
        return best_mod
# END produce_best_model()


def refine_binocular( mod0, train_data, val_data, to_plot=True, LLnull=None, device=None ):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("  refine_binocular WARNING: device not entered, using device:", device)

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
def baseline_sico(NE, NI, LorR=0, seed=100, XTreg=0.01, Greg=0.001, Dreg=1.0, nlags=None,
                  drift_term=None, bi_bias=True ):
    """Make standard binocular model with given size and defaults -- with drift term"""

    from NDNT.NDN import NDN
    from NDNT.networks import FFnetwork
    from NDNT.modules.layers import NDNLayer, ConvLayer, MaskConvLayer

    if nlags is None:
        nlags = 12
        print("  Using default nlags = %d"%nlags)

    num_mfilts = NE+NI
    # other constants
    mfw = 21
    bfw = 11
    CregM = 0.0001

    monoc_basis_par = ConvLayer.layer_dict( 
        input_dims=[1,72,1,nlags], num_filters=num_mfilts, filter_dims=[1, mfw, 1, nlags],
        norm_type=1,bias=False, initialize_center=True, NLtype='lin',
        reg_vals={'center':CregM, 'd2xt':XTreg})

    bfilt_par = MaskConvLayer.layer_dict( 
        input_dims=[num_mfilts*2,36,1,1], # reinterprets convolutional output above
        num_filters=NE+NI, num_inh= NI, filter_dims=bfw, 
        num_groups=num_mfilts, norm_type=1, pos_constraint=True, #window='hamming',
        bias=bi_bias, initialize_center=True, NLtype='relu')

    readout_par = NDNLayer.layer_dict(
        num_filters=1, bias=True, initialize_center=True, pos_constraint=True,
        NLtype='softplus', reg_vals={'glocalx': Greg }) 
    
    masks = [np.ones( [2, bfw, num_mfilts], dtype=np.float32 ), 
             np.ones( [2, bfw, num_mfilts], dtype=np.float32 )]
    zfw = bfw//2
    masks[0][0,:zfw,:] = 0
    masks[0][0,-zfw:,:] = 0
    masks[1][1,:zfw,:] = 0
    masks[1][1,-zfw:,:] = 0
    
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
        # Comb net
        comb_par = NDNLayer.layer_dict(num_filters=1, NLtype='softplus', bias=True, weights_initializer='ones')
        comb_net = FFnetwork.ffnet_dict( xstim_n=None, ffnet_n=[0,1], layer_list = [comb_par], ffnet_type='add')
        
        # Define model
        sico = NDN(ffnet_list=[stim_net, drift_net, comb_net], seed=seed)
        sico.networks[1].layers[0].weight.data[:,0] = torch.tensor(drift_term.squeeze(), dtype=torch.float32)
        sico.set_parameters(val=False, ffnet_target=1)
        sico.set_parameters(val=False, name='weight', ffnet_target=2)
    else:
        sico = NDN(layer_list=[monoc_basis_par, bfilt_par, readout_par], seed=seed)
   
    sico.networks[0].layers[1].set_mask(masks[LorR])
    return sico


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


def center_model( mod0, center_binoc=False ):
    
    ks0 = mod0.get_weights()
    NF = ks0.shape[-1]
    ks1 = deepcopy(ks0)
    for ii in range(NF):
        ks1[..., ii] = center_filter(ks0[...,ii])
    if center_binoc:
        return center_binoc_mask( mod0, filters=ks1 )
    else:
        mod1 = deepcopy(mod0)
        mod1.networks[0].layers[0].weight.data = torch.tensor( 
            ks1.reshape([-1, NF]), dtype=torch.float32, device=mod0.device)
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