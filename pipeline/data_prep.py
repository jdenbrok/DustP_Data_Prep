"""
This routine contains scripts for the data processing
"""
__author__ = "J. den Brok"

import numpy as np
from astropy.io import fits
import copy
import pandas as pd
import matplotlib.pyplot as plt

#for bkg sub
from astropy.stats import sigma_clipped_stats
from astropy.stats import mad_std
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground

#for galaxy stats (r_gal, etc.)
from astropy.coordinates import SkyCoord, FK5
from astropy.wcs import WCS
import astropy.units as au
#from deproject import *

def convert_units(data, hdr, conv_to = "surf_br"):
    """
    Convert Data units to mJy/sr
    :param data: 2D data array
    :param hdr: fits header
    :param conv to:
                "surf_br": convert to units of mJy/sr (surface brightness)
                "flux": convert to units mJy (flux)
    """
    
    data_unit = hdr["SIGUNIT"]
    
    if "Jy/pix" in data_unit:
        # pixel size in rad
        px_x = abs(hdr["CDELT1"])*2*np.pi/360
        px_y = abs(hdr["CDELT2"])*2*np.pi/360
        
        #pixel size in arcsec
    
        px_x = abs(hdr["CDELT1"])*3600
        px_y = abs(hdr["CDELT2"])*3600
        conversion_factor = 1000/(px_x*px_y)
        
        unit_label = "mJy/arcsec^2"
        
        #in case we want to convert to units flux, we need to multiply by the size (in steradians) of a pixel.
        if conv_to == "flux":
            conversion_factor*=abs(hdr["CDELT1"])*abs(hdr["CDELT2"])*(2*np.pi/360)**2
            unit_label = "mJy"
            
        data *= conversion_factor
        hdr["SIGUNIT"] = unit_label
    else:
        print("\t[WARNING]\t "+data_unit+" conversion not yet implemented. No conversion performed.")
        
    return data, hdr
    
def do_bkg_sub(data, header, this_band):
    """
    Do background subtraction and estimate rms
    :param hdul: fits file
    :param this_band: string name of BAND
    """
    
    size_bkp_patch = 300 #arcsec
    if this_band == "Spitzer_3.6":
        size_bkp_patch = 240
    
    n_px = int(size_bkp_patch/abs(header["CDELT1"])/3600)
    
    #Perform 2D background subtraction
    sigma_clip = SigmaClip(sigma=2.5)
    bkg_estimator = MedianBackground()
    
    bkg = Background2D(data, (n_px, n_px), filter_size=(3,3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator,exclude_percentile=10)
    
    return bkg
    
def get_stats(hdul):
    mean, median, std = sigma_clipped_stats(hdul[0].data, sigma=3.0)
    
    return mean, median, std

"""
#-------------------------------------------------------------------------
Relies on private Code, please contact jdenbrok@uni-bonn.de for further info
#-------------------------------------------------------------------------

def get_galaxy_stats(hdr, glx_tbl, i_gal):
    """
    #Compute the pixelwise r_gal and theta_gal, etc.
    """
    
    wcs = WCS(hdr)
    x_px, y_px = np.where(np.ones((hdr["NAXIS2"],hdr["NAXIS1"])))

    #compute ra and dec for all pixel
   
    world_coord = wcs.all_pix2world(np.column_stack((y_px, x_px)),0)
    
    ra_coord =  world_coord[:,0]
    dec_coord =  world_coord[:,1]
    
   
    #deporject the data
    rgal_deg, theta_rad = deproject(ra_coord, dec_coord,
                                        [glx_tbl["posang_deg"][i_gal],
                                         glx_tbl["incl_deg"][i_gal],
                                         glx_tbl["ra_ctr"][i_gal],
                                         glx_tbl["dec_ctr"][i_gal]
                                        ], vector = True)
                                        
    rgal_as = rgal_deg * 3600
    rgal_kpc = np.deg2rad(rgal_deg)*glx_tbl["dist_mpc"][i_gal]*1e3
    rgal_r25 = rgal_deg/(glx_tbl["r_25"][i_gal]/60.)
    
    theta_rad = np.reshape(theta_rad,(hdr["NAXIS1"],hdr["NAXIS2"]))
    rgal_as = np.reshape(rgal_as,(hdr["NAXIS1"],hdr["NAXIS2"]))
    rgal_kpc = np.reshape(rgal_kpc,(hdr["NAXIS1"],hdr["NAXIS2"]))
    rgal_r25 = np.reshape(rgal_r25,(hdr["NAXIS1"],hdr["NAXIS2"]))

    #store the result for return
    output={}
    output["ra_deg"]=np.reshape(ra_coord,(hdr["NAXIS1"],hdr["NAXIS2"]))
    output["dec_deg"]=np.reshape(dec_coord,(hdr["NAXIS1"],hdr["NAXIS2"]))
    output["rgal_as"]=rgal_as
    output["theta_rad"]=theta_rad
    output["rgal_kpc"]=rgal_kpc
    output["rgal_r25"]=rgal_r25

    return output
"""
