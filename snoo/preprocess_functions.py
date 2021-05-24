# PREPROCESSING FUNCTIONS
"""
Created on Sat May 22 15:17:21 2021

@author: jelle
"""


import scipy.signal
import scipy.ndimage as nd
import numpy as np
from sklearn.preprocessing import normalize, scale



#%% SMOOTHING & DERIVATIVES

def savgol(spectra, filter_win=11, poly_order=2, deriv_order=0, delta=1.0):
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
        return scipy.signal.savgol_filter(spectra, filter_win, poly_order, deriv_order, delta=delta, axis=1)

def savgol_der(spectra, filter_win=11, poly_order=2, deriv_order=0, delta=1.0):
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
        return scipy.signal.savgol_filter(spectra, filter_win, poly_order, deriv_order, delta=delta, axis=1)


def derivate(spectra, order=1, delta=1):
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


#%% BASELINE

def baseline(spectra):
    """ Removes baseline (mean) from each spectrum.

    Args:
        spectra <numpy.ndarray>: NIRS data matrix.

    Returns:
        spectra <numpy.ndarray>: Mean-centered NIRS data matrix
    """
    
    return spectra - np.mean(spectra, axis=1)[:, None]


def detrend(spectra, bp=0):
    """ Perform spectral detrending to remove linear trend from data.

    Args:
        spectra <numpy.ndarray>: NIRS data matrix.
        bp <list>: A sequence of break points. If given, an individual linear fit is performed for each part of data
        between two break points. Break points are specified as indices into data.

    Returns:
        spectra <numpy.ndarray>: Detrended NIR spectra
    """
    
    return scipy.signal.detrend(spectra, bp=bp, axis= 1)


#%% SCATTER CORRECTIONS

def norml(spectra):
    """ Perform spectral normalisation based on vector norm.

    Args:
        spectra <numpy.ndarray>: NIRS data matrix.
        udefined <bool>: use user defined limits
        imin <float>: user defined minimum
        imax <float>: user defined maximum

    Returns:
        spectra <numpy.ndarray>: Normalized NIR spectra
    """
    
    return spectra / np.linalg.norm(spectra, axis=1)[:, None]



def snv(spectra):
    """ Perform scatter correction using the standard normal variate.

    Args:
        spectra <numpy.ndarray>: NIRS data matrix.

    Returns:
        spectra <numpy.ndarray>: NIRS data with (S/R)NV applied.
    """

    return (spectra - np.shape(np.mean(spectra, axis=1))) / np.std(spectra, axis=1)   #5.7 ms ± 122 µs per loop
    #return scale(spectra, axis=0, with_mean= True, with_std= True)                   #11.8 ms ± 211 µs per loop



def rnv(spectra, iqr=[75, 25]):
    """ Perform scatter correction using robust normal variate.

    Args:
        spectra <numpy.ndarray>: NIRS data matrix.
        iqr <list>: IQR ranges [lower, upper] for robust normal variate.

    Returns:
        spectra <numpy.ndarray>: NIRS data with (S/R)NV applied.
    """

    return (spectra - np.median(spectra, axis=1)) / np.subtract(*np.percentile(spectra, iqr, axis=1))


def lsnv(spectra, num_windows=6):
    """ Perform local scatter correction using the standard normal variate.

    Args:
        spectra <numpy.ndarray>: NIRS data matrix.
        num_windows <int>: number of equispaced windows to use (window size (in points) is length / num_windows)

    Returns:
        spectra <numpy.ndarray>: NIRS data with local SNV applied.
    """

    parts = np.array_split(spectra, num_windows, axis=1)
    for idx, part in enumerate(parts):
        parts[idx] = snv(part)

    return np.concatenate(parts, axis=1)




#%% SCALING & EXTRA


def pareto(spectra):
    ''' Perform Pareto scaling to decrease the the importance of high variance variables '''
    sqrt_std = np.sqrt(np.std(spectra, axis=0))
    
    return spectra / sqrt_std[None, :]


def norm_unit(spectra):
    ''' Normalize absorbance units to fall between 0 and 1 '''
    f = (np.max(spectra) - np.min(spectra))    #Min-max-range
    
    return spectra / f















#%% Unused functions (STILL COLUMN BASED)


def savgol_OLD(X):
    #EXTRA FUNCTION REDUNDANT
    
    def savgol(spectra, filter_win=11, poly_order=2, deriv_order=0, delta=1.0):
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


def robust_baseline(spectra):
    """ Removes baseline (if robust: median, else: mean) from each spectrum.

    Args:
        spectra <numpy.ndarray>: NIRS data matrix.

    Returns:
        spectra <numpy.ndarray>: Median-centered NIRS data matrix
    """

    return spectra - np.mean(spectra, axis=0)


def old_norml(spectra, udefined=False, imin=0, imax=1):
    """ Perform spectral normalisation with user defined limits.

    Args:
        spectra <numpy.ndarray>: NIRS data matrix.
        udefined <bool>: use user defined limits
        imin <float>: user defined minimum
        imax <float>: user defined maximum

    Returns:
        spectra <numpy.ndarray>: Normalized NIR spectra
    """
    if udefined:
        f = (imax - imin)/(np.max(spectra) - np.min(spectra)) #Get min and max of all
        n = spectra.shape
        arr = np.empty((0, n[0]), dtype=float) #create empty array for spectra
        
        for i in range(0, n[1]):
            dnorm = imin + f*spectra[:,i]
            arr = np.append(arr, [dnorm], axis=0)
        return np.transpose(arr)
        
    else:
        return spectra / np.linalg.norm(spectra, axis=0)




def smooth(spectra, filter_win, window_type='flat', mode='reflect'):
    """ Smooths the spectra using convolution.

    Args:
        spectra <numpy.ndarray>: NIRS data matrix.
        filter_win <float>: length of the filter window in samples.
        window_type <str>: filtering window to use for convolution (see scipy.signal.windows)
        mode <str>: convolution mode

    Returns:
        spectra <numpy.ndarray>: Smoothed NIR spectra.
    """

    if window_type == 'flat':
        window = np.ones(filter_win)
    else:
        window = scipy.signal.windows.get_window(window_type, filter_win)
    window = window / np.sum(window)

    for column in range(spectra.shape[1]):
        spectra[:, column] = nd.convolve(spectra[:, column], window, mode=mode)

    return spectra


def trim(wavelength, spectra, bins):
    """ Trim spectra to a specified wavelength bin (or bins).

    Args:
        wavelength <numpy.ndarray>: Vector of wavelengths.
        spectra <numpy.ndarray>: NIRS data matrix.
        bins <list>: A bin or a list of bins defining the trim operation.

    Returns:
        spectra <numpy.ndarray>: NIRS data smoothed with Savitzky-Golay filtering
    """
    if type(bins[0]) != list:
        bins = [bins]

    spectra_trim = np.array([]).reshape(0, spectra.shape[1])
    wavelength_trim = np.array([])
    for wave_range in bins:
        mask = np.bitwise_and(wavelength >= wave_range[0], wavelength <= wave_range[1])
        spectra_trim = np.vstack((spectra_trim, spectra[mask, :]))
        wavelength_trim = np.hstack((wavelength_trim, wavelength[mask]))
    return wavelength_trim, spectra_trim


def resample(wavelength, spectra, resampling_ratio):
    """ Resample spectra according to the resampling ratio.

    Args:
        wavelength <numpy.ndarray>: Vector of wavelengths.
        spectra <numpy.ndarray>: NIRS data matrix.
        resampling_ratio <float>: new length with respect to original length

    Returns:
        wavelength_ <numpy.ndarray>: Resampled wavelengths.
        spectra_ <numpy.ndarray>: Resampled NIR spectra
    """

    new_length = int(np.round(wavelength.size * resampling_ratio))
    spectra_, wavelength_ = scipy.signal.resample(spectra, new_length, wavelength)
    return wavelength_, spectra_


def msc(spectra):
    """ Performs multiplicative scatter correction to the mean.

    Args:
        spectra <numpy.ndarray>: NIRS data matrix.

    Returns:
        spectra <numpy.ndarray>: Scatter corrected NIR spectra.
    """

    spectra = scale(spectra, with_std=False, axis=0) # Demean
    reference = np.mean(spectra, axis=1)

    for col in range(spectra.shape[1]):
        a, b = np.polyfit(reference, spectra[:, col], deg=1)
        spectra[:, col] = (spectra[:, col] - b) / a

    return spectra


def emsc(wave, spectra, remove_mean=False):
    """ Performs (basic) extended multiplicative scatter correction to the mean.

    Args:
        spectra <numpy.ndarray>: NIRS data matrix.

    Returns:
        spectra <numpy.ndarray>: Scatter corrected NIR spectra.
    """

    if remove_mean:
        spectra = scale(spectra, with_std=False, axis=0)

    p1 = .5 * (wave[0] + wave[-1])
    p2 = 2 / (wave[0] - wave[-1])

    # Compute model terms
    model = np.ones((wave.size, 4))
    model[:, 1] = p2 * (wave[0] - wave) - 1
    model[:, 2] = (p2 ** 2) * ((wave - p1) ** 2)
    model[:, 3] = np.mean(spectra, axis=1)

    # Solve correction parameters
    params = np.linalg.lstsq(model, spectra)[0].T

    # Apply correction
    spectra = spectra - np.dot(params[:, :-1], model[:, :-1].T).T
    spectra = np.multiply(spectra, 1 / np.repeat(params[:, -1].reshape(1, -1), spectra.shape[0], axis=0))

    return spectra


def clip(wavelength, spectra, threshold, substitute=None):
    """ Removes or substitutes values above the given threshold.

    Args:
        wavelength <numpy.ndarray>: Vector of wavelengths.
        spectra <numpy.ndarray>: NIRS data matrix.
        threshold <float>: threshold value for rejection
        substitute <float>: substitute value for rejected values (None removes values from the spectra)

    Returns:
        wavelength <numpy.ndarray>: Vector of wavelengths.
        spectra <numpy.ndarray>: NIR spectra with threshold exceeding values removed.
    """

    if substitute == None:  # remove threshold violations
        mask = np.any(spectra > threshold, axis=1)
        spectra = spectra[~mask, :]
        wavelength = wavelength[~mask]
    else:  # substitute threshold violations with a value
        spectra[spectra > threshold] = substitute
    return wavelength, spectra

    return wavelength, spectra