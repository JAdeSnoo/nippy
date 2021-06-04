# -*- coding: utf-8 -*-
"""
Created on Sun May 23 11:18:22 2021

@author: jelle
"""

# PREPROCESSING FUNCTIONS
"""
Created on Sat May 22 15:17:21 2021

@author: jelle
"""


import numpy as np
# from sklearn.preprocessing import normalize, scale







#%% SCALING & NORML



def mean_center(spectra):
    
    mean = np.mean(spectra, axis= 0)
    def mean_center(spectra):
        ''' Mean centers the columns of the spectra '''
        
        return spectra - mean[None, :]
    
    return mean_center


def pareto(spectra):
    
    sqrt_std = np.sqrt(np.std(spectra, axis=0))
    def pareto(spectra):
        ''' Perform Pareto scaling to decrease the the importance of high variance variables 
        
        Args:
            spectra <numpy.ndarray>: NIRS data matrix.
        Returns:
            spectra <numpy.ndarray>: Scaled NIR spectra'''
        return spectra / sqrt_std[None, :]
    
    return pareto




def auto_scale(spectra):
    
    mean= np.mean(spectra, axis= 0)
    std = np.std(spectra, axis= 0)
    
    def auto_scale(spectra):
        ''' Performs auto scaling on spectra '''
        
        return (spectra - mean[None, :]) / std[None, :]
    
    return auto_scale



def norm_unit(spectra):
    
    f = (np.max(spectra) - np.min(spectra))    #Min-max-range
    def norm_unit(spectra):
        ''' Normalize absorbance units to fall between 0 and 1 
        
        Args:
            spectra <numpy.ndarray>: NIRS data matrix.
        Returns:
            spectra <numpy.ndarray>: Normalized NIR spectra.
        '''
        return spectra / f
    
    return norm_unit