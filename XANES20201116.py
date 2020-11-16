# ======================================================================
# Title         : PyMC3 code for XANES spectra
# Source file   : XANES20201116.py
# Creation Date : 2020/11/16
# Version       : 1.0
# Used Package  : PyMC3, https://docs.pymc.io/
# Author        : Ichiro AKAI
# Maintainer    : Kazunori IWAMITSU <ayuohs.5780@gmail.com> 
# Description   : This is a package for spectral analysis of XANES using the PyMC3 package.
#
# Ichiro AKAI, Copyright (c) 2020. All rights reserved.
# ====================================================================== 

############################################
# System initialization
############################################
import os 
import pickle
#
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# 
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html#scipy.stats.norm
from scipy import stats
# 
from scipy import integrate
# 
from math import gamma as gamma_func
#
# %matplotlib inline
#
import numpy as np
#
# https://docs.pymc.io/
import pymc3 as pm
import theano
import time
from pymc3.backends.base import merge_traces
#
import datetime
#
# for reloading 
import importlib
#
# print('numpy      Ver.', np.__version__)
# print('matplotlib Ver.', mpl.__version__)
# print('pymc3      Ver.', pm.__version__)
# print('theano     Ver.', theano.__version__)
#
################################
# Import smtplib for the actual sending function
import smtplib
import datetime
#
# Import the email modules we'll need
from email.message import EmailMessage
#
try:
    NodeName = os.uname().nodename
except:
    NodeName = "local"
# print('nodename = ', NodeName )
# addr_from = NodeName + ' <iakai@kumamoto-u.ac.jp>'

######################################
LinestyleTab  = [ 'solid', 'dashed', 'dashdot', 'dotted' ]
ColorCodesTab = [ 'blue', 'red', 'green', 'yellow', 'orange', 'brown', 'violet', 'gray', 'white', 'black']
#
LinestyleTab_ref_value  = [ 'dashed', 'dashdot', 'dotted', 'solid' ]
ColorCodesTab_ref_value = [ 'blue', 'red', 'green', 'yellow', 'orange', 'brown', 'violet', 'gray', 'white', 'black']
#
ColorCodesTab_posterior = [ 'brown', 'red', 'darkorange', 'gold', 'green', 'navy', 'purple', 'dimgray', 'sienna', 'magenta']

############################################
# Spectral functions
############################################
######################################
psdVoigt1_c1 = 2.0 / np.pi
psdVoigt1_c2 = 4.0 * np.log(2.0)
psdVoigt1_c3 = np.sqrt( psdVoigt1_c2 / np.pi ) 
#
def psdVoigt1(x,A,xc,w,mu):
    return A * ( \
                mu * psdVoigt1_c1 * w / ( (x-xc)**2 + w**2 ) + \
                (1-mu) * psdVoigt1_c3 / w * np.exp( -psdVoigt1_c2 * ((x-xc)/w)**2 ) \
               )

######################################
Gaussian_c1 = 4.0 * np.log(2.0)
Gaussian_c2 = np.sqrt( Gaussian_c1 / np.pi ) 
#
def Gaussian(x, F, E, W):
    return F * Gaussian_c2 / W * np.exp( -Gaussian_c1 * ((x-E)/W)**2 )

######################################
STEParctan_c1 = 1.0/np.pi
#
def STEParctan(x,H, E0, Gamma):
    return H * (0.5 + STEParctan_c1 * np.arctan( (x-E0)/(Gamma/2) ) )
#
def STEParctanBL(x,H, E0, Gamma, y0):
    return y0 + H * (0.5 + STEParctan_c1 * np.arctan( (x-E0)/(Gamma/2) ) )

######################################
def Lorentzian(x, F, E, FWHM):
    return F * FWHM / (np.pi) / ( (x-E)**2 + FWHM**2 )

######################################
def LorentzianPeak(x, PeakInt, E, FWHM):
    return PeakInt * FWHM**2 / ( (x-E)**2 + FWHM**2 )

############################################
# Prior probability 
############################################
######################################
# Type of prior probability
PriorFunc_Unknown                = -1
PriorFunc_Uniform                = 0 # arg1 = lower, arg2 = upper
PriorFunc_Norm                   = 1 # arg1 = mean,  arg2 = sd
PriorFunc_Gamma                  = 2 # arg1 = alpha, arg2 = beta
PriorFunc_Beta                   = 3 # arg1 = alpha, arg2 = beta
PriorFunc_TruncatedNorm          = 4 # arg1 = mean,  arg2 = sd, arg3 = lower, arg4 = upper
PriorFunc_TruncatedNorm_sd_ratio = 5 # arg1 = mean,  arg2 = sd, arg3 = ratio

######################################
# Graph for prior probability
######################################
######################################
# Graph for prior probability
######################################
def _grfprior( axis, label_x, prior_def, ref_value, \
               color_tab=ColorCodesTab, linestyle_tab=LinestyleTab, linewidth=2.0, \
               color_tab_ref_value=ColorCodesTab_ref_value, linestyle_tab_ref_value=LinestyleTab_ref_value, linewidth_ref_value=1.0 ):
    #
    prior_type = prior_def[0]
    arg1       = prior_def[1]
    arg2       = prior_def[2]
    #
    # Making data for prior
    #
    if(   prior_type == PriorFunc_Uniform ):
        #
        # def priorUniform( x, lower=0, upper=1 ):
        #        
        xx_center = (arg2 + arg1) / 2.0
        xx_width  = (arg2 - arg1) / 2.0
        xx        = np.linspace( xx_center - 1.1 * xx_width, xx_center + 1.1 * xx_width, num=500 )
        prior     = priorUniform( xx, lower=arg1, upper=arg2 )
        #
    elif( prior_type == PriorFunc_Norm ):
        #
        # def priorNorm( x, mu=0.0, sd=1.0 ):
        #    
        xx    = np.linspace( arg1 - 3.0 * arg2, arg1 + 3.0 * arg2 )
        prior = priorNorm( xx, mu=arg1, sd=arg2 )
        #
    elif( prior_type == PriorFunc_Gamma ):
        #
        #  def priorGamma( x, alpha=None, beta=None ):
        #
        x_max = arg1 / arg2 + 3 * np.sqrt(arg1) / arg2
        xx    = np.linspace( 0, x_max, num=500 )
        prior = priorGamma( xx, alpha=arg1, beta=arg2 )
        #
    elif( prior_type == PriorFunc_Beta ):
        #
        # def priorBeta( x, alpha=2.0, beta=2.0 ):
        #
        xx    = np.linspace( 0, 1.0, num=500 )
        prior = priorBeta( xx, alpha=arg1, beta=arg2 )
        #
    elif( prior_type == PriorFunc_TruncatedNorm ):
        #
        # def priorTruncatedNorm( x, mu=0.0, sd=1.0, lower=-1.0, upper=1.0 ):
        #
        arg3 = prior_def[3]
        arg4 = prior_def[4]
        #        
        xx_center = (arg4 + arg3) / 2.0
        xx_width  = (arg4 - arg3) / 2.0
        xx        = np.linspace( xx_center - 1.1 * xx_width, xx_center + 1.1 * xx_width, num=500 )
        #
        prior = priorTruncatedNorm( xx, mu=arg1, sd=arg2, lower=arg3, upper=arg4 )
        #
    elif( prior_type == PriorFunc_TruncatedNorm_sd_ratio ):
        #
        # def priorTruncatedNorm_sd_ratio( x, mu=0.0, sd=1.0, ratio=1.0 ):
        # 
        arg3 = prior_def[3]
        lower_ = arg1 - arg3 * arg2
        upper_ = arg1 + arg3 * arg2
        xx_center = (upper_ + lower_) / 2.0
        xx_width  = (upper_ - lower_) / 2.0
        xx        = np.linspace( xx_center - 1.1 * xx_width, xx_center + 1.1 * xx_width, num=500 )
        prior = priorTruncatedNorm_sd_ratio( xx, mu=arg1, sd=arg2, ratio=arg3 )
    #
    xx_max = np.amax( xx )
    xx_min = np.amin( xx )
    #
    # Drawing Graph of prior 
    #
    axis.set_xlabel( label_x )
    x_center_ = ( xx_max + xx_min ) / 2.0
    x_width_  = ( xx_max - xx_min ) / 2.0
    x_lim = axis.set_xlim( x_center_ - x_width_, x_center_ + x_width_ )
    plt.xticks( rotation=90 )
    #
    axis.set_ylabel('Prior Probability')
    y_max = np.nanmax( prior )
    y_lim = axis.set_ylim( 0, 1.2 * y_max)
    #
    color_idx     = 0
    linestyle_idx = 0
    #
    axis.plot( xx, prior, label='Prior', \
               color='blue', linestyle='solid', linewidth=1.5 )
    #
    # Drawing a vertical line for ref_value
    # 
    color_ref_value_idx     = 0
    linestyle_ref_value_idx = 0
    #
    if( ref_value is not None ):
        if( ref_value[1] is not None ):
            label_ref_value_   = ref_value[0]
            numeric_ref_value_ = ref_value[1]
            #
            label_ref_ = '%s(%+.6E)' % (label_ref_value_, numeric_ref_value_)
            # print(label_ref_ )
            #
            axis.vlines( numeric_ref_value_, y_lim[0], y_lim[1], label=label_ref_value_, \
                         colors='blue', linestyles='dashed', linewidth=1.0 )
    #
    # Getting legend handle
    #
    legend_handle, legend_label = axis.get_legend_handles_labels()    
    #
    return x_lim, legend_handle, legend_label

######################################
# PriorTrial
######################################
def PriorTrial( prior_def, ref_value ):
    #
    plt.rcParams['font.size'] = 14
    #
    fig = plt.figure( figsize=(7.0, 5.5) )
    fig.subplots_adjust( left=0.10, right=0.70, top=0.93,  bottom=0.20 )
    #
    axis = fig.add_subplot( 1, 1, 1 )    
    #
    _grfprior( axis, 'x', prior_def, ['ref', ref_value] )    


######################################
# Graph for prior/posterior probability
######################################
def GrfParamsProbabilities01( param, \
                fontsize=14, figsize=(7.0,5.5), \
                adjust_left=0.1, adjust_right=0.70, adjust_top=0.93,  adjust_bottom=0.20, \
                legend_bbox_to_anchor=(1.15, 1), legend_loc='upper left', legend_borderaxespad=0, \
                color_tab=ColorCodesTab, linestyle_tab=LinestyleTab ):
    #
    param_expression = param[1]
    param_ref_value  = param[2]
    param_prior_def  = param[3]
    #
    plt.rcParams['font.size'] = fontsize
    #
    fig = plt.figure( figsize=figsize )
    fig.subplots_adjust( left=adjust_left, right=adjust_right, top=adjust_top,  bottom=adjust_bottom )
    #
    # Prior probability
    #
    axis1L = fig.add_subplot(1,1,1)
    #
    (prior_x_lim, prior_legend_handle, prior_legend_label) = _grfprior( axis1L, param_expression, param_prior_def, param_ref_value )
    #
    legend_h = prior_legend_handle
    legend_l = prior_legend_label
    #
    # Posterior probability
    #
    # (7) histgram.   [bins, hists], the histgram of the trace
    if param[7] is not None:
        #
        histgram = param[ 7 ]
        bins_    = histgram[ 0 ]
        hists_   = histgram[ 1 ]
        #
        bins_min  = np.amin( bins_ )
        bins_max  = np.amax( bins_ )
        hists_max = np.amax( hists_ )
        #
        bins_center = (bins_max + bins_min)/2.0
        bins_width  = (bins_max - bins_min)/2.0
        #
        x_lim = axis1L.set_xlim( bins_center - 1.2* bins_width, bins_center + 1.2 * bins_width )
        #
        axis1R = axis1L.twinx()
        #
        axis1R.set_ylabel('Posterior Probability')
        axis1R_ylim = axis1R.set_ylim( 0, 1.2 * hists_max )
        #
        axis1R.fill_between( bins_, hists_, color="lightgreen", alpha=1, label='Posterior')
        #
        if( param[5] is not None ):
            mean_value_   = param[5]  # (5) mean_value, the mean value of the trace
            #
            axis1R.vlines( mean_value_, axis1R_ylim[0], axis1R_ylim[1], label='Mean(%+.6E)' % mean_value_, \
                           colors='green', linestyles='solid', linewidth=1.0 )
        #
        if( param[9] is not None ):
            MLE_value_   = param[9]   # (9) MLE_value,  the value at the MLE_index
            #
            axis1R.vlines( MLE_value_, axis1R_ylim[0], axis1R_ylim[1], label='MLE(%+.6E)' % MLE_value_, \
                           colors='green', linestyles='dashed', linewidth=1.0 )
        #
        # Getting legend handle
        #
        legend_handle1R, legend_label1R = axis1R.get_legend_handles_labels()    
        #
        legend_h = prior_legend_handle + legend_handle1R
        legend_l = prior_legend_label  + legend_label1R
    #
    # Title & legend
    # 
    # plt.legend()
    plt.legend(legend_h, legend_l, bbox_to_anchor=legend_bbox_to_anchor, loc=legend_loc )
    #
    plt.title( 'Probabilities for ' + param_expression)
    #
    return fig

######################################
# Graph for prior/posterior probability
######################################
def GrfFuncProbabilities01( fit_func, \
                            xlim_posterior = True, \
                            fontsize=14, figsize=(7.0,5.5), \
                            adjust_left=0.10, adjust_right=0.70, adjust_top=0.93,  adjust_bottom=0.20, adjust_wspace=0.5, \
                            legend_bbox_to_anchor=(1.15, 1), legend_loc='upper left', legend_borderaxespad=0, \
                            color_tab=ColorCodesTab, linestyle_tab=LinestyleTab ):
    #
    func_name_    = fit_func[ 0 ]
    func_params_  = fit_func[ 1 ]
    func_paramsC_ = len( func_params_ )
    #
    plt.rcParams['font.size'] = fontsize
    #
    figsize_x = figsize[0]
    figsize_y = figsize[1]
    figsize_  = ( func_paramsC_ * figsize_x * 1.25, figsize_y )
    #
    fig = plt.figure( figsize=figsize_ )
    fig.subplots_adjust( left=adjust_left, right=adjust_right, top=adjust_top,  bottom=adjust_bottom, wspace=adjust_wspace )
    #
    param_i_ = 0
    axisL_ = []
    axisR_ = []
    legend_h_ = None
    legend_l_ = None
    color_idx_max = len( ColorCodesTab_posterior )
    color_idx = 0
    prior_flag = False
    ##########################
    for param_ in func_params_:
        #
        # Prior probability
        #
        axisL_.append( fig.add_subplot( 1, func_paramsC_, param_i_+1 ) )
        #
        param_expression_ = param_[1]
        param_ref_value_  = param_[2]
        param_prior_def_  = param_[3]
        #
        (prior_x_lim_, prior_legend_handle_, prior_legend_label_) = _grfprior( axisL_[ param_i_ ], \
                                                                               param_expression_, \
                                                                               param_prior_def_, \
                                                                               param_ref_value_ )
        #
        if( legend_h_ is None ):
            legend_h_ = prior_legend_handle_
            legend_l_ = prior_legend_label_
        #
        #
        # Posterior probability
        #
        color_idx = color_idx % color_idx_max
        # (7) histgram.   [bins, hists], the histgram of the trace
        if param_[7] is not None:
            #
            histgram = param_[ 7 ]
            bins_    = histgram[ 0 ]
            hists_   = histgram[ 1 ]
            #
            bins_min  = np.amin( bins_ )
            bins_max  = np.amax( bins_ )
            hists_max = np.amax( hists_ )
            #
            bins_center = (bins_max + bins_min)/2.0
            bins_width  = (bins_max - bins_min)/2.0
            #
            if xlim_posterior:
                x_lim = axisL_[ param_i_ ].set_xlim( bins_center - 1.2* bins_width, bins_center + 1.2 * bins_width )
            #
            axisR_.append( axisL_[ param_i_ ].twinx() )
            #
            axisR_[ param_i_ ].set_ylabel('Posterior Probability')
            axisR_ylim = axisR_[ param_i_ ].set_ylim( 0, 1.2 * hists_max )
            #
            axisR_[ param_i_ ].fill_between( bins_, hists_, color=ColorCodesTab_posterior[color_idx], alpha=0.5, label='Posterior')
            #
            if( param_[5] is not None ):
                mean_value_   = param_[5]  # (5) mean_value, the mean value of the trace
                #
                axisR_[ param_i_ ].vlines( mean_value_, axisR_ylim[0], axisR_ylim[1], label='Mean(%+.6E)' % mean_value_, \
                                           colors=ColorCodesTab_posterior[color_idx], linestyles='solid', linewidth=1.0 )
            #
            if( param_[9] is not None ):
                MLE_value_   = param_[9]   # (9) MLE_value,  the value at the MLE_index
                #
                axisR_[ param_i_ ].vlines( MLE_value_, axisR_ylim[0], axisR_ylim[1], label='MLE(%+.6E)' % MLE_value_, \
                                           colors=ColorCodesTab_posterior[color_idx], linestyles='dashed', linewidth=1.0 )
            #
            # Getting legend handle
            #
            legend_handle1R_, legend_label1R_ = axisR_[ param_i_ ].get_legend_handles_labels()    
            #
            legend_h_ = legend_h_ + legend_handle1R_
            legend_l_ = legend_l_ + legend_label1R_
        #
        plt.title( 'Probabilities for ' + param_expression_ + ' in ' + func_name_ )
        #
        # The end of for param_ in func_params_:
        param_i_  = param_i_ + 1
        color_idx = color_idx + 1
    #
    # Title & legend
    # 
    # plt.legend()
    plt.legend(legend_h_, legend_l_, bbox_to_anchor=legend_bbox_to_anchor, loc=legend_loc )
    #
    # plt.title( 'Probabilities for ' + func_name_ )
    #
    return fig


######################################
# Making histgram
######################################
def TraceHistgram( trace, bins=100 ):
    #
    hist, bins_ = np.histogram( trace, bins=bins, density=True )
    #
    # bins_ are th edge values of the respective histgram.
    # We have to obtain the center value of the bin edges.
    #
    bins = np.zeros_like( hist )
    for i in range(bins.size):
        bins[i] = ( bins_[i] + bins_[ i + 1 ])/2.0
    #
    return [bins, hist]

######################################
# Searching a param by name
######################################
def SearchParam( param_name, Params_ ):
    #
    for param_ in Params_:
        if( param_name == param_[0] ):
            return param_
    #
    print( '%s is not found.' % param_name )
    exit()

######################################
# Prior Model of the param
######################################
def ParamModel( param_name, Params_ ):
    #
    param = SearchParam( param_name, Params_ )
    #
    prior_def = param[3] #   prior_def,  definition for the prior probability    
    #
    # [ XA.PriorFunc_????, arg1, arg2, arg3, ... ]
    prior_type = prior_def[0]
    arg1       = prior_def[1]
    arg2       = prior_def[2]
    #
    # XA.PriorFunc_Unknown                = -1
    # XA.PriorFunc_Uniform                = 0 # arg1 = lower, arg2 = upper
    # XA.PriorFunc_Norm                   = 1 # arg1 = mean,  arg2 = sd
    # XA.PriorFunc_Gamma                  = 2 # arg1 = alpha, arg2 = beta
    # XA.PriorFunc_Beta                   = 3 # arg1 = alpha, arg2 = beta
    # XA.PriorFunc_TruncatedNorm          = 4 # arg1 = mean,  arg2 = sd, arg3 = lower, arg4 = upper
    # XA.PriorFunc_TruncatedNorm_sd_ratio = 5 # arg1 = mean,  arg2 = sd, arg3 = ratio
    #
    # Making data for prior
    #
    if(   prior_type == PriorFunc_Uniform ):
        #
        return pm.Uniform( param_name, lower = arg1, upper = arg2 )
        #
    elif( prior_type == PriorFunc_Norm ):
        #
        return pm.Normal( param_name, mu = arg1, sd = arg2 )
        #
    elif( prior_type == PriorFunc_Gamma ):
        #
        return pm.Gamma( param_name, alpha = arg1, beta = arg2 )
        #
    elif( prior_type == PriorFunc_Beta ):
        #
        return pm.Beta( param_name, alpha = arg1, beta = arg2 )
        #
    elif( prior_type == PriorFunc_TruncatedNorm ):
        #
        arg3 = prior_def[3]
        arg4 = prior_def[4]
        #        
        return pm.TruncatedNormal( param_name, mu = arg1, sd = arg2, lower = arg3, upper = arg4 )
        #
    elif( prior_type == PriorFunc_TruncatedNorm_sd_ratio ):
        #
        arg3 = prior_def[3]
        lower_ = arg1 - arg3 * arg2
        upper_ = arg1 + arg3 * arg2
        #
        return pm.TruncatedNormal( param_name, mu = arg1, sd = arg2, lower = lower_, upper = upper_ )

    
######################################
# Getting the prior model of the param
######################################
def ParamPriorModel( param ): 
    #
    param_name_ = param[0]
    prior_def_  = param[3] #   prior_def,  definition for the prior probability    
    #
    # [ XA.PriorFunc_????, arg1, arg2, arg3, ... ]
    prior_type_ = prior_def_[0]
    arg1_       = prior_def_[1]
    arg2_       = prior_def_[2]
    #
    # XA.PriorFunc_Unknown                = -1
    # XA.PriorFunc_Uniform                = 0 # arg1 = lower, arg2 = upper
    # XA.PriorFunc_Norm                   = 1 # arg1 = mean,  arg2 = sd
    # XA.PriorFunc_Gamma                  = 2 # arg1 = alpha, arg2 = beta
    # XA.PriorFunc_Beta                   = 3 # arg1 = alpha, arg2 = beta
    # XA.PriorFunc_TruncatedNorm          = 4 # arg1 = mean,  arg2 = sd, arg3 = lower, arg4 = upper
    # XA.PriorFunc_TruncatedNorm_sd_ratio = 5 # arg1 = mean,  arg2 = sd, arg3 = ratio
    #
    # Making data for prior
    #
    if(   prior_type_ == PriorFunc_Uniform ):
        #
        print( '%-10s: Uniform( lower=%g, upper=%g )' % (param_name_, arg1_, arg2_) )
        return pm.Uniform( param_name_, lower = arg1_, upper = arg2_ )
        #
    elif( prior_type_ == PriorFunc_Norm ):
        #
        print( '%-10s: Normal( mu=%g, sd=%g )' % (param_name_, arg1_, arg2_) )
        return pm.Normal( param_name_, mu = arg1_, sd = arg2_ )
        #
    elif( prior_type_ == PriorFunc_Gamma ):
        #
        print( '%-10s: Gamma( alpha=%g, beta=%g )' % (param_name_, arg1_, arg2_) )
        return pm.Gamma( param_name_, alpha = arg1_, beta = arg2_ )
        #
    elif( prior_type_ == PriorFunc_Beta ):
        #
        print( '%-10s: Beta( alpha=%g, beta=%g )' % (param_name_, arg1_, arg2_) )
        return pm.Beta( param_name_, alpha = arg1_, beta = arg2_ )
        #
    elif( prior_type_ == PriorFunc_TruncatedNorm ):
        #
        arg3_ = prior_def_[3]
        arg4_ = prior_def_[4]
        #        
        print( '%-10s: TruncatedNormal( mu=%g, sd=%g, lower=%g, upper=%g )' % (param_name_, arg1_, arg2_, arg3_, arg4_ ) )
        return pm.TruncatedNormal( param_name_, mu = arg1_, sd = arg2_, lower = arg3_, upper = arg4_ )
        #
    elif( prior_type_ == PriorFunc_TruncatedNorm_sd_ratio ):
        #
        arg3_ = prior_def_[3]
        lower_ = arg1_ - arg3_ * arg2_
        upper_ = arg1_ + arg3_ * arg2_
        #
        print( '%-10s: TruncatedNormal( mu=%g, sd=%g, lower=%g, upper=%g )' % (param_name_, arg1_, arg2_, lower_, upper_ ) )
        return pm.TruncatedNormal( param_name_, mu = arg1_, sd = arg2_, lower = lower_, upper = upper_ )
    #    
    
######################################
# pymc3 sampling
######################################
def MCMC_findMAP_NUTS_v6( Xdata, Ydata, phys_model, rmsd_param, fit_funcs, iters=50000, chains=4, burns=5000, tunes=5000 ):
    #
    print( 'MCMC_findMAP_NUTS_v6: iters=%d, chains=%d, burns=%d, tunes=%d' % (iters, chains, burns, tunes) )
    #
    with pm.Model() as model:
        #
        ####################### 
        # rmsd
        rmsd       = ParamPriorModel( rmsd_param )
        #
        ####################### 
        # params in fit_funcs
        # ---------------------
        # The parameters in the params_ are registered according to the order of 
        # the registered functions (Fit_Funcs) and their variables, 
        # and the params_ is passed directly to the physical model.
        # Therefore, in the physical model, we have to use these parameters by 
        # the ordered indexes.
        #
        params_ = []
        #
        for func_ in fit_funcs:
            func_name_   = func_[ 0 ]
            func_params_ = func_[ 1 ]
            #
            for param_ in func_params_:
                params_.append( ParamPriorModel( param_ ) )
        #
        ####################### 
        # y
        y = pm.Normal('y', \
                      mu=phys_model( Xdata, params_ ), \
                      sd=rmsd, \
                      observed=Ydata )
        #
        ####################### 
        # start = pm.find_MAP(method='powell')
        start = pm.find_MAP()
        #
        step  = pm.NUTS()
        #
        trace = pm.sample( iters+burns, step, start=start, chains= chains, tune=tunes ) 
        #
        return trace

######################################
# Selection of chain, which has the smallest mean value of rmsd 
# and statistical analysis of the rmds param.
#------------------
# When more than one spectral component with the same prior probability 
# and spectral function are included, the assignments of their spectral 
# components are exchanged randomly in each of the parallel computing 
# chains, so that, the warning of "The rhat statistic is larger than ...
# The sampler did not converge." arises.
# 
# However, that is a natural result. So that, we employ the trace of 
# the chain with the smallest mean value of rmsd among the chains and 
# perform statistical processing.
#
# An alternative selection method would be according to the criterion 
# of maximum likelihood estimation / MAP and select at the rmsd minimum 
# / the posterior probability maximum.
#
def SelectChain_StatsRMSD( trace_, rmsd_param, chains_, burn_ ):
    #
    name_rmsd_param_ = rmsd_param[0]
    #
    rmsd_min       = float('inf')
    #
    for chain in range(chains_):
        #
        rmsd_trace_ = trace_.get_values( name_rmsd_param_, combine=False, chains=chain, burn=burn_ )
        #
        rmsd_trace_mean_ = np.mean( rmsd_trace_ )
        #
        print( '         chain=%2d, rmsd_mean=%.6E' % (chain, rmsd_trace_mean_) )
        #
        if rmsd_trace_mean_ < rmsd_min:
            rmsd_min = rmsd_trace_mean_
            chain_selected = chain
    #
    print( '-------------' )
    print( 'Selected chain=%2d, rmsd_mean=%.6E' % (chain_selected, rmsd_min) )
    print( '-------------' )
    #
    # (4) trace,      the selected trace
    rmsd_param[4] = trace_.get_values( name_rmsd_param_, combine=False, chains=chain_selected, burn=burn_ )
    #
    # (5) mean_value, the mean value of the trace
    rmsd_param[5] = np.mean( rmsd_param[4] ) 
    print( 'RMSD: mean      = %.6E' % rmsd_param[5] )
    #
    # (6) sd,         the sandard deviation of the trace
    rmsd_param[6] = np.std( rmsd_param[4] )  
    print( 'RMSD: sd        = %.6E' % rmsd_param[6] )
    #
    # (7) histgram.   [bins, hists], the histgram of the trace
    rmsd_param[7] = TraceHistgram( rmsd_param[4] )
    #
    # (8) MLE_index,  the trace index having maximum likelihood (ML estimation)
    MLE_Index     = rmsd_param[4].argmin()
    rmsd_param[8] = MLE_Index
    print( 'RMSD: MLE_index = %d' % rmsd_param[8] )
    #
    # (9) MLE_value,  the value at the MLE_index
    rmsd_param[9] = rmsd_param[4][ MLE_Index ]
    print( 'RMSD: MLE_value = %.6E' % rmsd_param[9] )
    #
    return chain_selected, MLE_Index

######################################
# Statistical analysis of the params in fit_funcs.
######################################
def Stats_fit_funcs( trace_, fit_funcs_, chain_selected, MLE_index_, burn_ ):
    #
    print( 'Number of the fitting functions = %d' % len( fit_funcs_ ) )
    #
    for func_ in fit_funcs_:
        #
        func_name_   = func_[ 0 ]
        func_params_ = func_[ 1 ]
        #
        for param_ in func_params_:
            #
            param_name_  = param_[0]
            #
            param_trace_ = trace_.get_values( param_name_, combine=False, chains=chain_selected, burn=burn_ )
            #
            param_[4] = param_trace_               # (4) trace,      the selected trace
            param_[5] = np.mean( param_trace_ )    # (5) mean_value, the mean value of the trace
            param_[6] = np.std( param_trace_ )     # (6) sd,         the sandard deviation of the trace
            param_[7] = TraceHistgram( param_[4] ) # (7) histgram.   [bins, hists], the histgram of the trace
            param_[8] = MLE_index_                 # (8) MLE_index,  the trace index having maximum likelihood (ML estimation)
            param_[9] = param_trace_[ MLE_index_ ] # (9) MLE_value,  the value at the MLE_index
    #
    # Report for Mean values
    #
    print( '---------- Mean values ---------')
    #
    for func_ in fit_funcs_:
        #
        func_name_   = func_[ 0 ]
        print( '# [\'%s\',' % func_name_ )
        print( '#   [' )
        #
        func_params_ = func_[ 1 ]
        #
        for param_ in func_params_:
            #
            param_name_     = param_[0] # (0) name,       string used as the identifier in pymc3
            param_name__    = '\'%s\'' % param_name_
            prama_ref_value = param_[5] # (5) mean_value, the mean value of the trace
            # prama_ref_value = param_[9] # (9) MLE_value,  the value at the MLE_index
            #
            print( '#     [%-15s, [\'mean\', %+.6E] ]' % (param_name__, prama_ref_value) )
            #
        print( '#   ]' )
        print( '# ]')
    #
    # Report for MLE values
    #
    print( '---------- MLE values  ---------')
    #
    for func_ in fit_funcs_:
        #
        func_name_   = func_[ 0 ]
        print( '# [\'%s\',' % func_name_ )
        print( '#   [' )
        #
        func_params_ = func_[ 1 ]
        #
        for param_ in func_params_:
            #
            param_name_     = param_[0] # (0) name,       string used as the identifier in pymc3
            param_name__    = '\'%s\'' % param_name_
            # prama_ref_value = param_[5] # (5) mean_value, the mean value of the trace
            prama_ref_value = param_[9] # (9) MLE_value,  the value at the MLE_index
            #
            print( '#     [%-15s, [\'MLE\', %+.6E] ]' % (param_name__, prama_ref_value) )
        #
        print( '#   ]' )
        print( '# ]')
    #
    # Making Report List
    #
    reports = [ '-------------------' ]
    reports.append( 'Func:Param           =      mean     +/-      sd      ,     MLE       ' )
    #
    for func_ in fit_funcs_:
        #
        func_name_   = func_[ 0 ]
        func_params_ = func_[ 1 ]
        #
        for param_ in func_params_:
            #
            param_name_  = param_[0]
            param_name__ = '%s:%s' % ( func_name_, param_name_ )
            #
            param_mean_  = param_[5]
            param_sd_    = param_[6]
            param_MLE_   = param_[9]
            #
            reports.append( '%-20s = %+.6E +/- %+.6E, %+.6E ' % (param_name__, param_mean_, param_sd_, param_MLE_ ) )
    #
    reports.append( '-------------------' )    
    #
    # print out the command line
    #
    for rep in reports:
        print( rep )
    #
    return reports

######################################
# Regression
######################################
RegCriterion_Mean = 0
RegCriterion_MLE  = 1
######################################
LinestyleTab_RegComps  = [ 'dashed', 'dashdot', 'dotted', 'solid' ]
ColorCodesTab_RegComps = [ 'brown', 'darkorange', 'gold', 'green', 'navy', 'purple', 'dimgray', 'sienna', 'magenta']
######################################
def GrfRegression( data_x, data_y, reg_func, fit_funcs, title, xlabel, ylabel,
                   regcriterion=RegCriterion_Mean, \
                   color_tab=ColorCodesTab_RegComps, linestyle_tab=LinestyleTab_RegComps):
    #
    if regcriterion == RegCriterion_Mean:
        #
        print( 'Criterion for Regression: Mean')
        #
        label_plot = 'Mean'
        title_     = title + ' / Mean'
        #   (5) mean_value, the mean value of the trace
        idx        = 5
        #
    elif regcriterion == RegCriterion_MLE:
        #
        print( 'Criterion for Regression: MLE')
        #
        label_plot = 'MLE'
        title_     = title + ' / MLE'
        #   (9) MLE_value,  the value at the MLE_index
        idx        = 9
        #
    #
    ####################### 
    # params for regression
    # ---------------------
    # The parameters in the params_ are registered according to the order of 
    # the registered functions (Fit_Funcs) and their variables, 
    # and the params_ is passed directly to 'reg_func', which is the physical 
    # model for regression.
    # Therefore, in the 'reg_func', we have to use these parameters by the 
    # ordered indexes.
    #
    params_ = []
    print( '----------------------' )
    print( 'Func:Param           =' )
    #
    for func_ in fit_funcs:
        #
        func_name_   = func_[ 0 ]
        func_params_ = func_[ 1 ]
        #
        for param_ in func_params_:
            param_name_     = param_[0]   # (0) name,       string used as the identifier in pymc3
            param_name__    = '%s:%s' % (func_name_, param_name_)
            prama_ref_value = param_[idx]
            params_.append( prama_ref_value )
            #
            print( '%-20s = %+.6E (%s)' % (param_name__, prama_ref_value, label_plot ) )
    #
    ####################### 
    # Calling 'reg_func'
    #
    (reg_, Comps_) = reg_func( data_x, params_ )
    #
    ####################### 
    # Drawing Graph
    #
    plt.rcParams['font.size'] = 16
    #
    fig = plt.figure( figsize=(11,6) )
    fig.subplots_adjust( left=0.15, right=0.6, top=0.93,  bottom=0.1 )
    #
    axis = fig.add_subplot(1,1,1)
    axis.set_xlabel( xlabel )
    axis.set_ylabel( ylabel )
    #
    # Data
    #
    axis.plot( data_x, data_y, color='black', linewidth=2.0, label='Exp.' )
    #
    # Regression
    #
    residual_ = reg_ - data_y
    reg_rmsd_ = np.std( residual_ )
    label_plot_ = label_plot + '(RMSD=%.6E)' % reg_rmsd_ 
    #
    axis.plot( data_x, reg_, color='red', linewidth=1.0, label=label_plot_ )
    #
    # Components
    #
    color_tab_len = len( color_tab )
    # 
    CompI = 0
    for Comp in Comps_:
        # 
        color     = color_tab[ CompI % color_tab_len ]
        linestyle = linestyle_tab[ CompI // color_tab_len ]
        #
        axis.plot( data_x, Comp[1], color=color, linewidth=1.0, linestyle=linestyle, label=Comp[0] )
        CompI = CompI + 1
    #
    plt.title( title_ )
    #
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    #
    return reg_rmsd_, fig

######################################
######################################
######################################
def makeFolder( folder ):
    if( True != os.path.isdir( folder ) ):
        print('making ', folder )
        os.mkdir( folder )
    else:
        print( folder, ' is exist.' )

######################################
def PickleTrace( file_fpath, trace ):
    print( 'Saving pickle file = %s' % file_fpath )
    with open( file_fpath, mode='wb') as f:
        pickle.dump(trace, f)

######################################
def SpectralStats( x, y ):
    #
    x_min = np.amin( x )
    x_max = np.amax( x )
    #
    y_min = np.amin( y )
    y_max = np.amax( y )
    #
    x_mean = np.mean( x )
    x_sd   = np.std( x )
    #
    y_mean = np.mean( y )
    y_sd   = np.std( y )
    #
    return (x_min, x_max, y_min, y_max, x_mean, x_sd, y_mean, y_sd)

######################################
def SpectralMoments( x, y ):
    #
    S0 = integrate.simps( y, x )
    M0 = S0
    #
    xy = x * y
    S1 = integrate.simps( xy, x )
    M1 = S1 / M0
    #
    xxy = x * x * y
    SS2 = integrate.simps( xxy, x )
    S2  = SS2 - 2 * S1 * M1 + S0 * M1**2
    M2  = S2 / M0
    # 
    return (M0, M1, M2)
#

######################################
def Grf1( X,Y, title='Title', xlabel='x', ylabel='y', Xvalues=None, \
          color_tab=ColorCodesTab, linestyle_tab=LinestyleTab ):
    #
    plt.rcParams['font.size'] = 16
    #
    fig = plt.figure( figsize=(6,6) )
    fig.subplots_adjust( \
            left=0.15, right=0.98, 
            top=0.93,  bottom=0.1 )
    #
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_xlabel( xlabel )
    ax1.set_ylabel( ylabel )
    # ax1.set_yscale('log')
    #
    ax1.plot(X, Y, color='black', \
             linewidth=2.0, \
             label='Exp.' )
    #
    # Xvalues
    y_lim = ax1.set_ylim()
    if Xvalues!=None:
        info_c = 0
        for info in Xvalues:
            # 
            color     = color_tab[ info_c % 7 ]
            linestyle = linestyle_tab[ info_c // 7 ]
            #
            ax1.vlines( info[1], y_lim[0], y_lim[1], colors=color, linestyles=linestyle, label=info[0] )
            #
            info_c = info_c + 1
    
    #
    plt.title( title )
    #
    plt.legend()
    #
    return fig
    #

######################################
def GrfPdfPages( file_fpath, fig ):
    print( 'Saving GrfPdfPages = %s' % file_fpath )
    with PdfPages( file_fpath ) as pp:
        pp.savefig( fig )
        # pp.close()

######################################
# Gamma
# %
# \begin{equation}
#   f(x|\alpha,\beta)
#   =
#   \frac{
#     \beta^{\alpha}
#     x^{\alpha-1}
#   }{
#     \Gamma(\alpha)
#   }
#   \exp
#   \left(
#     -\beta
#     x
#   \right)
#   \eqnlbl{pymc3-distribution-gamma}
# \end{equation}
#
def priorGamma( x, alpha=None, beta=None ):
    res = beta**alpha * x**(alpha-1) * np.exp(-beta*x)/ gamma_func(alpha)
    res[(x<0)]=0
    return res
######################################
# Normal
# %
#     \begin{equation}
#       f(x)
#       =
#       \frac{1}{\sigma\sqrt{2\pi}}
#       \exp
#       \left[
#         -
#         \frac{(x-X)^2}{2\sigma^2}
#       \right]
#       \eqnlbl{GaussianDistribution01}
#     \end{equation}
def priorNorm( x, mu=0.0, sd=1.0 ):
    return np.exp( -(x-mu)**2/(2 * sd**2) )/(sd * np.sqrt(2*np.pi))
######################################
# Truncated Normal
def priorTruncatedNorm( x, mu=0.0, sd=1.0, lower=-1.0, upper=1.0 ):
    prior = np.exp( -(x-mu)**2/(2 * sd**2) )
    area  = np.sqrt( np.pi / 2.0 ) * sd * ( erf_func( (mu-lower)/( np.sqrt(2)*sd ) ) - erf_func( (mu-upper)/( np.sqrt(2)*sd ) ) )
    prior = prior / area
    prior[ (x<lower)] = 0
    prior[ (x>upper)] = 0
    return prior
######################################
# Truncated Normal
# def priorTruncatedNormRatio( x, mu=0.0, sd=1.0, ratio=1.0 ):
def priorTruncatedNorm_sd_ratio( x, mu=0.0, sd=1.0, ratio=1.0 ):
    lower_ = mu - ratio * sd
    upper_ = mu + ratio * sd
    return priorTruncatedNorm( x, mu=mu, sd=sd, lower=lower_, upper=upper_ )
######################################
# Uniform
def priorUniform( x, lower=0, upper=1 ):
    return (1-np.maximum( np.sign( (x-lower)*(x-upper) ), 0 ))/(upper-lower)
######################################
# beta
# %
def priorBeta( x, alpha=2.0, beta=2.0 ):
    res = x**(alpha-1.0) * (1.0-x)**(beta-1.0)/ beta_func(alpha,beta)
    res[(x<0)|(x>1)]=0
    return res
#    return x**(alpha-1.0) * (1.0-x)**(beta-1.0)/ beta_func(alpha,beta)


######################################
def Email_Send( str_body, str_subject, addr_from, addr_to ):
    msg = EmailMessage()
    # dt_now = datetime.datetime.now()
    msg.set_content( str_body )
    # msg['Subject'] = str_subject + dt_now.strftime(' %Y/%m/%d %H:%M:%S')
    msg['Subject'] = str_subject
    msg['From']    = addr_from
    msg['To']      = addr_to
    #
    s = smtplib.SMTP('localhost')
    s.send_message(msg)
    s.quit()
    #
    print( "Sending Email")
################################
def Email_Initiation( name_notebook, addr_from, addr_to ):
    #
    dt_now = datetime.datetime.now()
    # 
    txt = name_notebook + " is initiated at " + dt_now.strftime(' %Y/%m/%d %H:%M:%S')
    Email_Send( txt, txt, addr_from, addr_to)
################################
def Email_Termination( name_notebook, addr_from, addr_to ):
    #
    dt_now = datetime.datetime.now()
    # 
    txt = name_notebook + " is terminated at " + dt_now.strftime(' %Y/%m/%d %H:%M:%S')
    Email_Send( txt, txt, addr_from, addr_to)
################################
def Email_Report( name_notebook, str_reports, addr_from, addr_to ):
    #
    dt_now = datetime.datetime.now()
    # 
    str_subject = name_notebook + "is reporting at " + dt_now.strftime(' %Y/%m/%d %H:%M:%S')
    #
    str_body    = name_notebook + " is reporting at " + dt_now.strftime(' %Y/%m/%d %H:%M:%S') + '\r\n' \
                  + '---------' + '\r\n'
    #
    for str_report in str_reports:
        str_body = str_body + str_report + '\r\n'
        print( str_report )
    #
    str_body = str_body + '---------' + '\r\n'
    #
    Email_Send( str_body, str_subject, addr_from, addr_to)
################################
def Email_ReportPDF( name_notebook, list_reports, pdf_name, pdf_path, addr_from, addr_to ):
    #
    email_addr_from = NodeName + addr_from
    #
    dt_now = datetime.datetime.now()
    #
    msg = EmailMessage()
    #
    msg['Subject'] = name_notebook + "is reporting at " + dt_now.strftime(' %Y/%m/%d %H:%M:%S')
    msg['From']    = email_addr_from
    msg['To']      = addr_to
    ###################
    str_body  = name_notebook + " is reporting at " + dt_now.strftime(' %Y/%m/%d %H:%M:%S') + '\r\n' \
                + '---------' + '\r\n'
    #
    for str_report in list_reports:
        str_body = str_body + str_report + '\r\n'
        print( str_report )
    #
    str_body = str_body + '---------' + '\r\n'
    #
    msg.set_content( str_body )
    ###################
    #
    with open( pdf_path, 'rb' ) as content_file:
        content = content_file.read()
        msg.add_attachment(content, maintype='application/pdf', subtype='pdf', filename=pdf_name)
    #
    try:
        ###################
        s = smtplib.SMTP('localhost')
        s.send_message(msg)
        s.quit()
    except:
        print('smtplib.SMTP is not available.')
        #
    #
#