"""
This routine masks foreground stars (fgs) in the data
"""
__author__ = "J. den Brok"

import numpy as np
from astropy.io import fits
import copy
import pandas as pd
import pyregion
from astropy.coordinates import SkyCoord, FK5
from astropy.wcs import WCS
import astropy.units as au

def get_spatial_res(band):
    """
    Return FWHM Resolution in arcsec
    """
    if "SDSS" in band:
        ang_res = 1.3
    elif "2MASS" in band:
        ang_res = 2
    elif "SPIRE" in band:
        if "250" in band:
            ang_res = 18
        elif "350" in band:
            ang_res = 25
        elif "500" in band:
            ang_res = 36
        else:
            print("\t [ERROR] band "+band+" not found or implemented yet")
            return 30
            
    elif "Spitzer" in band:
        if ("3.6" in band):
            ang_res = 1.66
        elif ("4.5" in band):
            ang_res = 1.72
        elif ("5.8" in band):
            ang_res = 1.88
        elif ("8.0" in band):
            ang_res = 1.98
        elif ("24" in band):
            ang_res = 6
        else:
            ang_res = 18
            
    elif "WISE" in band:
        if "3.4" in band:
            ang_res = 6.1
        elif "4.6" in band:
            ang_res = 6.4
        elif "12" in band:
            ang_res = 6.5
        elif "22" in band:
            ang_res = 12
       

    elif "PACS" in band:
        if "70" in band:
            ang_res = 9
        elif "100" in band:
            ang_res = 10
        elif "160" in band:
            ang_res = 13
    elif "GALEX" in band:
        if "NUV" in band:
            ang_res = 5.3
        elif "FUV" in band:
            ang_res = 4.3
    else:
        print("\t [ERROR] band "+band+" not found or implemented yet")
        return 30
        
    return ang_res

def do_fgs_masking(data, hdr, this_source, path, this_band):
    """
    Mask the forground stars
    
    :param r_mask: radius of mask (in arcsec)
    """
    #generate copy which we mask
    masked_data = copy.deepcopy(data)
    px_size = abs(hdr["CDELT1"])*3600
    hdu = fits.PrimaryHDU(data)
    hdu.header = hdr
    
    
    #import forground stars
    fgs_sources =  pd.read_csv(path+this_source.lower()+"_pos_stars.txt", names = ["ra", "dec"])
    
    #iterate over the fgs and mask the
    wcs = WCS(hdr)
    
    #convert wcs to pixel
    fgs_pos_px = wcs.all_world2pix(fgs_sources,0)
    
    # The masked region will have 3 times the FWHM of the angular resolution, but not be larger than 30 arcsec
    r_mask = np.min([3*get_spatial_res(this_band),20])//2
    
    # iterate over the indicidual points
    for i in range(len(fgs_pos_px)):
        r_circle = int(np.ceil(r_mask/px_size))
        region = """
            image
            circle({},{},{})
            """.format(fgs_pos_px[i][0],fgs_pos_px[i][1],r_circle)
        r = pyregion.parse(region)
        mask_fgs_reg = r.get_mask(hdu=hdu)
    
        masked_data[mask_fgs_reg]=np.nan
        
    return masked_data
