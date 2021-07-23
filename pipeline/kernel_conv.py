"""
This routine contains scripts for the data konvolution
"""
__author__ = "J. den Brok"

import numpy as np
from astropy.io import fits
import copy
import pandas as pd
from os import path

#import script Aniano convolution
import convolve_aniano as ca



def get_kernel(band, target_band):
    """
    Function which provides the correct kernel based on input and target band
    """
    
    if "SDSS" in band:
        band_str = "Gauss_01.5"
    elif "2MASS" in band:
        band_str = "Gauss_02.0"
    elif "SPIRE" in band:
        band_str = band
    elif "Spitzer" in band:
        if ("3.6" in band) or ("4.5" in band)or ("5.8" in band) or ("8.0" in band):
            band_str = "IRAC_"+band.split("_")[-1]
        else:
            band_str = "MIPS_"+band.split("_")[-1]
    elif "WISE" in band:
        if "12" in band:
            band_str = "WISE_ATLAS_11.6"
        elif "22" in band:
            band_str = "WISE_ATLAS_22.1"
        else:
            band_str = "WISE_ATLAS_"+band.split("_")[-1]

    elif "PACS" in band:
        band_str = band
    elif "GALEX" in band:
        band_str = band
    else:
        print("\t [ERROR] band "+band+" not found or implemented yet")
        return ""
        
    kernel_file = "Kernel_LowRes_"+band_str+"_to_"+target_band+".fits"
    
    return kernel_file
    


def kernel_conv(data, hdr, band, target_band, path_kernel):
    """
    Perform the kernel convolution on the provided data
    """
    
    # no need to convolve if band = target band
    #if band == target_band:
    #    return data
        
    kernel_file = get_kernel(band, target_band)
        
    # check if path exists
    if not path.exists(path_kernel + kernel_file):
        print("\t [Error] "+kernel_file +" not found. No convolution performed")
        return data
        
    #if file exists, we can extract it
    kernel, kernel_hdr = fits.getdata(path_kernel + kernel_file ,header=True)
    
    #do the actual convolution
    data_conv, kernel_out = ca.do_the_convolution(data, hdr, kernel, kernel_hdr)
    
    return data_conv
