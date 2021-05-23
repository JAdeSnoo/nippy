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


import scipy.signal
import numpy as np
# from sklearn.preprocessing import normalize, scale



#%% SMOOTHING & DERIVATIVES

def savgol(X):
    #EXTRA FUNCTION REDUNDANT
    
    def savgol(spectra, filter_win=9, poly_order=3, deriv_order=0, delta=1.0):
        """ Perform Savitzky–Golay filtering on the data (also calculates derivatives). This function is a wrapper for
        scipy.signal.savgol_filter.
        
        Args:
            spectra <numpy.ndarray>: NIRS data matrix.
            filter_win <int>: Size of the filter window in samples (default 11).
            poly_order <int>: Order of the polynomial estimation (default 3).
            deriv_order <int>: Order of the derivation (default 0).
        Returns:
            spectra <numpy.ndarray>: NIRS data smoothed with Savitzky-Golay filtering
        """
        return scipy.signal.savgol_filter(spectra, filter_win, poly_order, deriv_order, delta=delta, axis=0)
    return savgol


def derivate(X):
    #EXTRA FUNCTION REDUNDANT
    
    def derivatives(spectra, order=1, delta=1):
        """ Computes Nth order derivates with the desired spacing using numpy.gradient.
        Args:
            spectra <numpy.ndarray>: NIRS data matrix.
            order <float>: Order of the derivation.
            delta <int>: Delta of the derivate (in samples).
        Returns:
            spectra <numpy.ndarray>: Derivated NIR spectra.
        """
        for n in range(order):
            spectra = np.gradient(spectra, delta, axis=1)
        return spectra
    
    return derivatives


#%% SCATTER CORRECTIONS

def baseline(spectra):

    mean_cols = np.mean(spectra, axis=0)
    def baseline(spectra):
        """ Removes baseline (mean) from each spectrum.

        Args:
            spectra <numpy.ndarray>: NIRS data matrix.

        Returns:
            spectra <numpy.ndarray>: Mean-centered NIRS data matrix
        """
        return spectra - mean_cols
    
    return baseline

def detrend(X):
    #EXTRA FUNCTION REDUNDANT
    
    def detrend(spectra, bp= 0):
        """ Perform spectral detrending to remove linear trend from data.

        Args:
            spectra <numpy.ndarray>: NIRS data matrix.
            bp <list>: A sequence of break points. If given, an individual linear fit is performed for each part of data
            between two break points. Break points are specified as indices into data.
        Returns:
            spectra <numpy.ndarray>: Detrended NIR spectra
        """
        return scipy.signal.detrend(spectra, bp=bp)
    
    return detrend


def norml(X):
    #EXTRA FUNCTION REDUNDANT
    
    def norml(spectra):
        """ Perform spectral normalisation based on vector norm.

        Args:
            spectra <numpy.ndarray>: NIRS data matrix.
        Returns:
            processed_spectra <numpy.ndarray>: Normalized NIR spectra.
        """
        return spectra / np.linalg.norm(spectra, axis=0)
    
    return norml



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



def snv(X):
    #EXTRA FUNCTION REDUNDANT
    
    def snv(spectra):
        """ Perform scatter correction using the standard normal variate.

        Args:
            spectra <numpy.ndarray>: NIRS data matrix.
        Returns:
            spectra <numpy.ndarray>: NIRS data with (S/R)NV applied.
        """

        return (spectra - np.shape(np.mean(spectra, axis=0))) / np.std(spectra, axis=0)
    
    return snv



def rnv(X):
    #EXTRA FUNCTION REDUNDANT
    
    def rnv(spectra, iqr=[75, 25]):
        """ Perform scatter correction using robust normal variate.

        Args:
            spectra <numpy.ndarray>: NIRS data matrix.
            iqr <list>: IQR ranges [lower, upper] for robust normal variate.
        Returns:
            spectra <numpy.ndarray>: NIRS data with (S/R)NV applied.
        """

        return (spectra - np.median(spectra, axis=0)) / np.subtract(*np.percentile(spectra, iqr, axis=0))
    
    return rnv


def lsnv(X):
    #EXTRA FUNCTION REDUNDANT
    
    def lsnv(spectra, num_windows=6):
    
        """ Perform local scatter correction using the standard normal variate.

        Args:
            spectra <numpy.ndarray>: NIRS data matrix.
            num_windows <int>: number of equispaced windows to use (window size (in points) is length / num_windows)
        Returns:
            spectra <numpy.ndarray>: NIRS data with local SNV applied.
            """

        parts = np.array_split(spectra, num_windows, axis=0)
        for idx, part in enumerate(parts):
            parts[idx] = snv(part)

        return np.concatenate(parts, axis=0)

    return lsnv()





#%% SCALING



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