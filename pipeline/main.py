"""
This routine reads in a list of multiwave datasets and prepares a  homogenized dataset:
- 2D cutout
- Stars Masked
- Convolved to common spatial resolution
- regridded onto same grid

MODIFICATION HISTORY
    - Version which works up to and including MC
    - v0.0.0: Read in FITS File and do image manipulations
    - v1.0.0: implement MC
    - v1.1.0: implement generation of cigale readable input file
    - v1.1.1: implement S/N cut of cigale readable input file
    - v1.2.0: use uniform background
    - v1.3.0: implement calibrational uncertainty
"""
__author__ = "J. den Brok"
__version__ = "v1.2.0"
__email__ = "jdenbrok@astro.uni-bonn.de"

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import copy
import pandas as pd

#2D cutout
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord, FK5
from astropy.wcs import WCS
import astropy.units as au

#unit conversion & stats
from data_prep import *

#masking foreground stars
from mask_fgs import *

#convolution with Aniano kernels
from kernel_conv import *

#for reprojection
from reproject import reproject_interp

import warnings
warnings.filterwarnings("ignore")

#----------------------------------------------------------------------
# Change these lines of code with correct directory and names
#----------------------------------------------------------------------

list_file_path = "./Listfiles/band_list.txt"

geom_file = "./Listfiles/geometry.txt"

path_kernel = "./../../data/kernels/"

path_stellar_mask_gen = "./../../data/#/orig_data/"

path_output_gen = "/Users/jdenbrok/Desktop/ISM_school/Project_08/data/#/working_data/"

#----------------------------------------------------------------------
# Main Function
#----------------------------------------------------------------------

def prep_fits_files(just_source=None, quiet=False,do_MC=False,niter_MC=10, SN_cut = None):
    """
    :param SN_cut: None or float
    """

    if quiet == False:
        print("------------------------------------------------------")
        print("[INFO]\t Pipeline version: "+__version__)
        print("------------------------------------------------------")
        print("[INFO]\t Reading in List Files")
        
    """
    First we need to import the files with the info about the galaxy and
    the list of bands we want to import
    """
    
    
    #--------------------------------------------------------------------
    # Load in the List Files and analyze the provided bands
    #--------------------------------------------------------------------
    
    # Import the galaxy info
    names_glxy = ["galaxy",
                  "ra_ctr",
                  "dec_ctr",
                  "x_width",
                  "y_width",
                  "dist_mpc",
                  "incl_deg",
                  "posang_deg",
                  "r_25",
                  "redshift"]
    glxy_data = pd.read_csv(geom_file, sep='[\s,]{2,20}',names = names_glxy,
                            comment = "#", engine='python')
                            
    n_glxy = len(glxy_data["galaxy"])
    if quiet == False:
        print("[INFO]\t {} galaxy(ies) detected".format(n_glxy))
        
    # Add the bands to the structure
    band_columns = ["band_name",
                    "band_cig_name",
                    "band_dir",
                    "band_cal_unc"]
    bands = pd.read_csv(list_file_path, names = band_columns, sep='[\s,]{2,20}', comment="#",engine='python')
    n_bands = len(bands["band_name"])
    if quiet == False:
        print("[INFO]\t {} band(s) detected".format(n_bands))
    
    #check if SPIRE band provided
    SPIRE_bands = [band_spire for band_spire in list(bands["band_name"]) if "SPIRE" in band_spire]
    
   
    if len(SPIRE_bands)==0:
        print("[ERROR]\t Provide at least one SPIRE band")
        return 0
        
    #determine target resolution (one of the SPIRE BANDS)
    if "SPIRE_500" in SPIRE_bands:
        target_band = "SPIRE_500"
    elif "SPIRE_350" in SPIRE_bands:
        target_band = "SPIRE_350"
    elif "SPIRE_250" in SPIRE_bands:
        target_band = "SPIRE_250"
    else:
        print("[ERROR]\t No valid SPIRE band provided")
        return 0
        
    #--------------------------------------------------------------------
    # Start iterating over the individual galaxies
    #--------------------------------------------------------------------
    
    #Create a large dictionary which stores all the band informations
    band_structure = {}
    for i_gal in range(n_glxy):
        this_source =glxy_data["galaxy"][i_gal]
        
        
        #in case we only want to investigate one source
        if not just_source is None:
            if this_source != just_source:
                continue
                
        if quiet == False:
            print("-------------------------------")
            print("{}: Galaxy ".format(i_gal+1)+this_source)
            if do_MC:
                print("!!!Performing MC!!! n_iter:{}".format(niter_MC))
            print("-------------------------------")
        
        #add an entry to the band_structure
        band_structure[this_source]={}
        
        #--------------------------------------------------------------------
        # Galaxy Info
        #--------------------------------------------------------------------
        #get 2D cutout dimensions for galaxy
        ra_ctr = glxy_data["ra_ctr"][i_gal]
        dec_ctr = glxy_data["dec_ctr"][i_gal]
        
        x_width = glxy_data["x_width"][i_gal]
        y_width = glxy_data["y_width"][i_gal]
        pos_center = SkyCoord(ra=ra_ctr*au.deg, dec=dec_ctr*au.deg, frame=FK5)
        sizeTrim = (y_width*au.arcmin,x_width*au.arcmin)
        
        
        # get the target header
        hdul_target = fits.open(bands["band_dir"][0]+this_source+"_"+target_band+".fits")
        hdr_target =hdul_target[0].header
        wcs = WCS(hdr_target)
        
        #2D cutout and update target header
        cutout = Cutout2D(hdul_target[0].data, position=pos_center, size=sizeTrim, wcs=wcs)
        hdr_target.update(cutout.wcs.to_header())
        dim_cutout = np.shape(cutout.data)
        hdr_target["NAXIS1"]=dim_cutout[1]
        hdr_target["NAXIS2"]=dim_cutout[0]
        hdul_target.close()
        
        #path output
        path_output = path_output_gen.replace("#",this_source.lower())
        
        #--------------------------------------------------------------------
        # Iterate over the bands provided
        #--------------------------------------------------------------------
        
        for i in range(len(bands)):
            this_band = bands["band_name"][i]
            if quiet == False:
                print("[INFO]\t Preparing Band "+this_band+"...")
            
            #add an entry to the band_structure
            band_structure[this_source][this_band]={}
            
            hdul = fits.open(bands["band_dir"][i]+this_source+"_"+this_band+".fits")
            data = hdul[0].data
            hdr_map = hdul[0].header
            
            
            # get an estimate of the rms (need for synthetic data)
            mean, median, std = get_stats(hdul)
                        
            hdul.close()
            #if no MC, only do the following steps once, else repeat n_MC times
            n_iter = 1
            if do_MC:
                n_iter+=niter_MC
                #store the data in a 3D array. Can then compute the std at end to get uncertainty
                data_struct = np.zeros((niter_MC, hdr_target["NAXIS2"],hdr_target["NAXIS1"]))
                
            for n_i in range(n_iter):
                data_i = copy.deepcopy(data)
                #add noise, if we do MC
                if n_i>0:
                    data_i+=np.random.normal(0, std, np.shape(data_i))
                    
                
                #--------------------------------------------------------------------
                # I: Perform BKG subtraction
                #--------------------------------------------------------------------
                # Perform BKG subtraction
                # Perform the BKG subtraction on the uncut images to have enough background
                
                mean_i, median_i, std_i = sigma_clipped_stats(data_i, sigma=5.0)
                #commented out: Do 2D background sub
                #bkg = do_bkg_sub(data_i,hdr_map, this_band)
                
            
                #--------------------------------------------------------------------
                # II: Do 2D cutout
                #--------------------------------------------------------------------
            
                wcs = WCS(hdr_map)
                
                #cutout = Cutout2D(data_i-bkg.background, position=pos_center, size=sizeTrim, wcs=wcs)
                cutout = Cutout2D(data_i-median_i, position=pos_center, size=sizeTrim, wcs=wcs)
                
            
                data_cutout = cutout.data
                dim_cutout = np.shape(data_cutout)
                hdr_cutout = copy.deepcopy(hdr_map)
                hdr_cutout.update(cutout.wcs.to_header())
                hdr_cutout["NAXIS1"]=dim_cutout[1]
                hdr_cutout["NAXIS2"]=dim_cutout[0]
            
                #--------------------------------------------------------------------------------
                # III: Further Preps (unit conversion to mJy)
                #--------------------------------------------------------------------------------
            
                # convert to units mJy
                data_cutout, hdr_cutout = convert_units(data_cutout, hdr_cutout)
            
            
                #--------------------------------------------------------------------------------
                # IV: MASK Forground Stars
                #--------------------------------------------------------------------------------
                
                path_stellar_mask = path_stellar_mask_gen.replace("#", this_source.lower())
                data_cutout = do_fgs_masking(data_cutout, hdr_cutout, this_source, path_stellar_mask, this_band)
            
                
                #--------------------------------------------------------------------------------
                # V: Convolve the data
                #--------------------------------------------------------------------------------
                
                data_cutout = kernel_conv(data_cutout, hdr_cutout, this_band, target_band, path_kernel)
                
                #--------------------------------------------------------------------------------
                # VI: Reproject the data and save
                #--------------------------------------------------------------------------------
                
                
                if this_band != target_band:
                    wcs = WCS(hdr_target)
                    data_cutout, footprint = reproject_interp((data_cutout, hdr_cutout), hdr_target)
                    #save the reprojected image as fits file
                    hdr_cutout.update(wcs.to_header())
                    
                    
                #if n_i = 0, we did not add noise to the image, so store this original image
                if n_i == 0:
                    hdr_cutout["NAXIS1"] = hdr_target["NAXIS1"]
                    hdr_cutout["NAXIS2"] = hdr_target["NAXIS2"]
                    fits.writeto(path_output +this_source+"_"+this_band+"_2Dcutout_res"+target_band+".fits", data = data_cutout, header = hdr_cutout, overwrite = True)
                    
                    #store data internally in the band_structure
                    band_structure[this_source][this_band]["data"]=data_cutout
                
                if n_i>=1:
                    data_struct[n_i-1,:,:]=data_cutout
                    
                #we have reached the end, compute the std and save as separate fit
                if n_i == niter_MC:
                    data_cutout_unc = np.nanstd(data_struct, axis=0)
                    
                    #add calibrational noise
                    calc_unc = bands["band_cal_unc"][i]/100*data_cutout
                    
                    data_cutout_unc = np.sqrt(data_cutout_unc**2+calc_unc**2)
                    fits.writeto(path_output +this_source+"_"+this_band+"_2Dcutout_res"+target_band+"_unc.fits", data = data_cutout_unc, header = hdr_cutout, overwrite = True)
                    
                    #store data uncertainty internally in the band_structure
                    band_structure[this_source][this_band]["data_unc"]=data_cutout_unc
            
        
        #--------------------------------------------------------------------------------
        # VII: Prepare for each galaxy a cigale input ASCII file which holds the values for the individual pixels
        #--------------------------------------------------------------------------------
        
        #get galaxy stats (r_gal, theta_gal, etc.) of individual pixel
        #glxy_px_info = get_galaxy_stats(hdr_target,glxy_data, i_gal)
        
        n_pts = hdr_target["NAXIS1"]*hdr_target["NAXIS2"]
        
        z_gal = glxy_data["redshift"][i_gal]
        cigale_input_table = pd.DataFrame((np.column_stack((np.arange(n_pts),z_gal*np.ones(n_pts)))), columns = ["id", "redshift"])
        
        #add the individual bands
        for i in range(len(bands)):
            this_band = bands["band_name"][i]
            this_band_cigale = bands["band_cig_name"][i]
            cigale_input_table[this_band_cigale]=band_structure[this_source][this_band]["data"].flatten()
            # if we did MC, we have also uncertainty values
            if do_MC:
                cigale_input_table[this_band_cigale+"_err"]=band_structure[this_source][this_band]["data_unc"].flatten()
            else:
                cigale_input_table[this_band_cigale+"_err"]=['NA']*n_pts
                
         
        #add the glxy pixel info
        """
        Private Code, please contact jdenbrok@uni-bonn.de for the code, end result will not have deprojected coordinates. These are, however, not necessary when running CIGALE.
        """
        #cigale_input_table["ra"]=glxy_px_info["ra_deg"].flatten()
        #cigale_input_table["dec"]=glxy_px_info["dec_deg"].flatten()
        #cigale_input_table["rgal_kpc"]=glxy_px_info["rgal_kpc"].flatten()
        
        #if a S/N cut is specified, we only return data points whch pass the S_N cut (2/3 of points should have S/N>thresh)
            
        cigale_input_table.to_csv(path_output+this_source+"_input_table.txt", index = False, header=True, sep = "\t", na_rep='NA')
        
        if SN_cut:
            #do not include additional info, just the band intensities and errors
            table_array = np.asarray(cigale_input_table)[:,2:-3]
            
            #compute the S/N of the individual bands and check where threshold is reached
            SN_bands = table_array[:,::2]/table_array[:,1::2] > SN_cut
            
            #number of S/N thresh reached bands per point
            n_SN_thresh = np.nansum(SN_bands,axis=1)
            
            n_signif = np.where(n_SN_thresh>np.ceil(3/4*n_bands))
            
            #return only points which pass the S/N cut:
            
            cigale_input_table = cigale_input_table.iloc[n_signif[0],:]
            
            cigale_input_table.to_csv(path_output+this_source+"_input_table_SNcut.txt", index = False, header=True, sep = "\t", na_rep='NA')
    return 1
        
if __name__ == '__main__':
    out = prep_fits_files(do_MC=True, niter_MC=6, SN_cut= 10)
    
    if out:
        print("-------------------------------")
        print("[INFO]\t Finished Successfuly")
        print("-------------------------------")
