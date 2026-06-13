# Read in some libraries
import os
import sys
import time
import numpy as np
#import pandas as pd
import glob as glob
from astropy.io import fits
from astropy.visualization import simple_norm
from astropy.nddata import NDData 
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.table import Table,QTable
from astropy.coordinates import SkyCoord, match_coordinates_sky 
from astropy import units as u 
#from hci_utils import rebin
from photutils.background import MMMBackground, MADStdBackgroundRMS, Background2D
from photutils.detection import DAOStarFinder
from photutils.psf import EPSFBuilder
from collections import OrderedDict
from matplotlib import style, pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
import pymultinest
from pymultinest import watch
import threading, subprocess
import json
from astropy.stats import SigmaClip, sigma_clipped_stats
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm
import math
from astropy.wcs import WCS
from matplotlib.colors import LogNorm
from numpy import unravel_index
from matplotlib.colors import LogNorm
from photutils.aperture import aperture_photometry, CircularAnnulus
import time
from pathlib import Path
import scipy.constants
from scipy.stats import poisson
import csv

# Access PyMultinest
os.environ['DYLD_LIBRARY_PATH'] = '/home1/10051/km52536/MultiNest/lib'

##############################################################################################################
# ################# PSF Fitting functions
# ##############################################################################################################

# Bins the model
# Define functions for MultiNest
def bicubic_interpolate_each_psfmodel(dx=None,dy=None,psf=None):
#  ; PURPOSE: Evaluates the PSF for a given [x,y] offset. Uses
#  ;          bi-cubic interpolation for the region within 4 pixels
#  ;          of the center.
#  ;          Based on Jay Anderson's Fortran routing rpsf_phot.

    rx = 50 + dx*4
    ry = 50 + dy*4

    ix = int(rx)
    iy = int(ry)

    fx = rx - ix
    fy = ry - iy

    dd = np.sqrt(dx**2 + dy**2)

    if dd <= 4:
        A1 =  psf[iy  ,ix  ]
        B1 = (psf[iy+1,ix  ]-psf[iy-1,ix  ])/2
        C1 = (psf[iy  ,ix+1]-psf[iy  ,ix-1])/2
        D1 = (psf[iy+1,ix  ]+psf[iy-1,ix  ]-2*A1)/2
        F1 = (psf[iy  ,ix+1]+psf[iy  ,ix-1]-2*A1)/2
        E1 = (psf[iy+1,ix+1]-A1)

        A2 =  psf[iy+1,ix  ]
        B2 = (psf[iy+2,ix  ]-psf[iy  ,ix  ])/2
        C2 = (psf[iy+1,ix+1]-psf[iy+1,ix-1])/2
        D2 = (psf[iy+2,ix  ]+psf[iy  ,ix  ]-2*A2)/2
        F2 = (psf[iy+1,ix+1]+psf[iy+1,ix-1]-2*A2)/2
        E2 =-(psf[iy  ,ix+1]-A2)

        A3 =  psf[iy  ,ix+1]
        B3 = (psf[iy+1,ix+1]-psf[iy-1,ix+1])/2
        C3 = (psf[iy  ,ix+2]-psf[iy  ,ix  ])/2
        D3 = (psf[iy+1,ix+1]+psf[iy-1,ix+1]-2*A3)/2
        F3 = (psf[iy  ,ix+2]+psf[iy  ,ix  ]-2*A3)/2
        E3 =-(psf[iy+1,ix  ]-A3)

        A4 =  psf[iy+1,ix+1]
        B4 = (psf[iy+2,ix+1]-psf[iy  ,ix+1])/2
        C4 = (psf[iy+1,ix+2]-psf[iy+1,ix  ])/2
        D4 = (psf[iy+2,ix+1]+psf[iy  ,ix+1]-2*A4)/2
        F4 = (psf[iy+1,ix+2]+psf[iy+1,ix  ]-2*A4)/2
        E4 = (psf[iy  ,ix  ]-A4)

        V1 = A1 + B1*( fx ) + C1*( fy ) + D1*( fx )**2 + E1*( fx )*( fy ) + F1*( fy )**2
        V2 = A2 + B2*(fx-1) + C2*( fy ) + D2*(fx-1)**2 + E2*(fx-1)*( fy ) + F2*( fy )**2
        V3 = A3 + B3*( fx ) + C3*(fy-1) + D3*( fx )**2 + E3*( fx )*(fy-1) + F3*(fy-1)**2
        V4 = A4 + B4*(fx-1) + C4*(fy-1) + D4*(fx-1)**2 + E4*(fx-1)*(fy-1) + F4*(fy-1)**2

        rpsf_phot = (1-fx)*(1-fy)*V1 + ( fx )*(1-fy)*V2 + (1-fx)*( fy )*V3 + ( fx )*( fy )*V4
    elif dd < 12:
        rpsf_phot = (1-fx)*(1-fy)*psf[iy  ,ix  ] + ( fx )*(1-fy)*psf[iy+1,ix  ] + (1-fx)*( fy )*psf[iy  ,ix+1] + ( fx )*( fy )*psf[iy+1,ix+1]
    else:
        rpsf_phot = 0
    return rpsf_phot

def linearly_interpolate_nearest_psfs(dx=None, dy=None, ix=None, iy=None):
    ixcl = [0, 358, 1032]
    iycl = [0, 512, 1024]
    # Figure out which PSF corresponds to this [ix,jy].
    hx = 0
    for xx in range(1):
        if (ix > ixcl[xx+1]): hx += 1
    hy = 0
    for yy in range(1):
        if (iy > iycl[yy+1]): hy += 1

    fx = (ix - ixcl[hx]) / (ixcl[hx+1] - ixcl[hx])
    fy = (iy - iycl[hy]) / (iycl[hy+1] - iycl[hy])

    # Linearly interpolate the value of the PSF at this [dx,dy] offset
    # among the nearest 4 PSFs.

    part1 = bicubic_interpolate_each_psfmodel(dx, dy, epsf_library[hx+hy*3, :, :])
    part2 = bicubic_interpolate_each_psfmodel(dx, dy, epsf_library[hx+(hy+1)*3, :, :])
    part3 = bicubic_interpolate_each_psfmodel(dx, dy, epsf_library[hx+1+hy*3, :, :])
    part4 = bicubic_interpolate_each_psfmodel(dx, dy, epsf_library[hx+1+(hy+1)*3, :, :])

    thispix = (1-fx)*(1-fy)*part1 + (1-fx)*fy*part2 + fx*(1-fy)*part3 + fx*fy*part4

    return thispix

def conv_to_N_photon(arr_in):
    """Convert an array of each pixel value in MJy/sr to number of photons."""
    arr_in = np.array(arr_in)
    arr_out = np.zeros_like(arr_in)
    
    wavel = float(input_im_filt)      # in microns

    if wavel == 10:
        gain = 4.44       # in e-/DN
    elif wavel == 15:
        gain = 4.77       # in e-/DN
    else:		
        print("Look for the appropriate gain on page 11 of https://www.stsci.edu/files/live/sites/www/files/home/jwst/documentation/technical-documents/_documents/JWST-STScI-008797.pdf.")

    T_exp = EFFEXPTM      #in seconds

    if arr_in.ndim == 1:
        for ind, pix in enumerate(arr_in):     # pix in MJy/sr
            N_photon =  pix * PHOTMJSR * gain * T_exp    
            arr_out[ind] = N_photon
    elif arr_in.ndim == 2:
        for i, row in enumerate(arr_in):
            for j, pix in enumerate(row):       # pix in MJy/sr
                N_photon =  pix * PHOTMJSR * gain * T_exp 
                arr_out[i, j] = N_photon

    return arr_out
    
def conv_to_MJy_sr(arr_in_1):
    """Convert an array of each pixel value in number of photons to MJy/sr."""
    arr_in_1 = np.array(arr_in_1)
    arr_out_1 = np.zeros_like(arr_in_1)
    
    wavel = float(input_im_filt)      # in microns

    if wavel == 10:
        gain = 4.44       # in e-/DN
    elif wavel == 15:
        gain = 4.77       # in e-/DN
    else:		
        print("Look for the appropriate gain on page 11 of https://www.stsci.edu/files/live/sites/www/files/home/jwst/documentation/technical-documents/_documents/JWST-STScI-008797.pdf.")

    T_exp = EFFEXPTM      #in seconds

    if arr_in_1.ndim == 1:
        for ind, N_photon in enumerate(arr_in_1):     # pix in photons count
            pix = N_photon/ (PHOTMJSR * gain * T_exp)
            arr_out_1[ind] = pix
    elif arr_in_1.ndim == 2:
        for i, row in enumerate(arr_in_1):
            for j, N_photon in enumerate(row):       # pix in photons count
                pix = N_photon/ (PHOTMJSR * gain * T_exp) 
                arr_out_1[i, j] = pix

    return arr_out_1

def model(xcen=0, ycen=0, flux=0, separation=0, position_angle=0, contrast=0):    
    # Make Primary PSF
    ix=int(xcen) # integer pixel position of model xcentroid+multinest_x_param
    iy=int(ycen)
	
    ix2 = int(xcen + separation*math.cos(position_angle*3.14159/180.0))
    iy2 = int(ycen + separation*math.sin(position_angle*3.14159/180.0))	

    fxu = xcen-ix # fractional pixel position (e.g. 1025.67 --> 0.67)
    fyu = ycen-iy

    fr = 10.**(contrast/(-2.5))
    #flux_both = flux * (1 + fr)        #flux here is flux of WD + companion (WD*fr)

    psfx = np.zeros((2*cutout_size+1,2*cutout_size+1))

    new_subarray=full_data_array[int(iy-cutout_size):int(iy+cutout_size)+1,int(ix-cutout_size):int(ix+cutout_size)+1]
    new_error_subarray=full_error_array[int(iy-cutout_size):int(iy+cutout_size)+1,int(ix-cutout_size):int(ix+cutout_size)+1]
	
    ftot=0	
    for aa in range(2*cutout_size+1):
        for bb in range(2*cutout_size+1):

            dx = (aa-cutout_size) - fxu #
            dy = (bb-cutout_size) - fyu
			
            dx2 = dx - separation*math.cos(position_angle*3.14159/180.0)
            dy2 = dy - separation*math.sin(position_angle*3.14159/180.0)			

            final_pixel_value1 = linearly_interpolate_nearest_psfs(dx, dy, ix, iy)
            final_pixel_value2 = linearly_interpolate_nearest_psfs(dx2,dy2,ix2,iy2)
			
            psfx[bb,aa] = final_pixel_value1 + fr*final_pixel_value2
            ftot = ftot + psfx[bb,aa]

    foundf = psfx * flux / ftot

    return foundf, new_subarray, new_error_subarray

# Define Priors
def prior(cube, ndim, nparams):
# Define primary parameters
    cube[0] = cube[0]*6.0 - 3.0      #xcen
    cube[1] = cube[1]*6.0 - 3.0      #ycen
    cube[2] = cube[2]*1e+5  #flux in photons
    cube[3] = cube[3]*5+0.01  # separation in pixels
    cube[4] = cube[4]*360.0  # position angle in degrees
    cube[5] = cube[5]*8.0   # contrast in magnitudes
    return

# Define likelihood function
def loglike_epsf(cube, ndim, nparams):
# Read variables
    xcen=newxcoor+cube[0]
    ycen=newycoor+cube[1]
    scaling=cube[2] # photons
    separation=cube[3]
    position_angle=cube[4]
    contrast=cube[5]
    output_model, new_subarray, new_error_subarray=model(xcen=xcen, ycen=ycen, flux=scaling, separation=separation, position_angle=position_angle, contrast=contrast)

# Flatten array without 0s to calculate the chi-squared
    norm_output_model = output_model.flatten()
    new_subarray_flat = new_subarray.flatten()
    new_error_subarray_flat = new_error_subarray.flatten()

    new_subarray_flat_red=new_subarray_flat[(new_subarray_flat+bkg_mean) != 0]
    new_error_subarray_flat=new_error_subarray_flat[(new_subarray_flat+bkg_mean) != 0]
    norm_output_model=norm_output_model[(new_subarray_flat+bkg_mean) != 0]

    loglikelihood = (-0.5 * ((new_subarray_flat_red[new_error_subarray_flat > 0.0] - norm_output_model[new_error_subarray_flat > 0.0]) / new_error_subarray_flat[new_error_subarray_flat > 0.0])**2).sum()
    return loglikelihood

def psf_fitting():
    # Define parameters in order as they will be used above
    parameters = ["xcen", "ycen", "scaling", "separation", "position_angle", "contrast"]
    n_params = len(parameters)

	# Define directory and files
    datafile=str(output_folder)+'/row_num_'+str(current_row)+'_doublerun'
    json.dump(parameters, open(datafile + '_params.json', 'w')) # save parameter names

	# Jay's epsf
    iterations=30000

    pymultinest.run(loglike_epsf, prior, n_params, importance_nested_sampling = True, resume = False, verbose = True, sampling_efficiency = 'model', max_iter=iterations, n_live_points = 400, outputfiles_basename=datafile + '_')		

    aa_epsf = pymultinest.Analyzer(outputfiles_basename=datafile + '_', n_params = n_params)
    a_lnZ_epsf = aa_epsf.get_stats()['global evidence']

    bestfit_epsf = aa_epsf.get_best_fit()
    bestfit_parameters_epsf = np.array(bestfit_epsf['parameters'])
    this_flux=bestfit_parameters_epsf[2]
    best_output_model_epsf, new_centered_data, new_centered_error=model(xcen=newxcoor+bestfit_parameters_epsf[0], ycen=newycoor+bestfit_parameters_epsf[1], flux=this_flux, separation=bestfit_parameters_epsf[3], position_angle=bestfit_parameters_epsf[4],contrast=bestfit_parameters_epsf[5])

    return best_output_model_epsf, bestfit_parameters_epsf, new_centered_data, new_centered_error


###############################################################################################################
# ##############################################################################################################
# ######### Read in data file for each target ##################################################################

# Define slice indices to run many scripts at the same time
import sys

start_ind = int(sys.argv[1])
#end_ind = int(sys.argv[2])
end_ind = np.copy(start_ind)
input_im_filt = 10

# Paths
base_folder = Path(f'/work/10875/andreavu/ls6/injection_and_recovery_F{input_im_filt}00W')
# skysub_folder = Path('/arc/projects/UdeM_whitedwarfs/mead')
psf_folder = base_folder/"MIRI_PSFs"
output_folder = base_folder/"Outputs"
sim_im_folder = output_folder/"simulated_images"

# Create folder to store psf results
psf_outputs_folder = output_folder/'doublepsf_fitting_outputs'
psf_outputs_folder.mkdir(parents=True, exist_ok=True)

# Define cutout size
cutout_size=10

# Get the x_cen and y_cen list and the corresponding image index
x_cen_use = []
y_cen_use = []
im_ind_list = []


with open(output_folder/"binaries_injected_parameters.csv", "r", newline="") as csvfile:
    csv_reader = csv.reader(csvfile)
    header = next(csv_reader)    
    rows = list(csv_reader)
    for row in rows:
        row_filt = row[0].split('_')[2]
        row_im_ind = row[0].split('_')[-1]
        if int(row_filt) == input_im_filt:
            row_x_cen = float(row[1])
            row_y_cen = float(row[2])
            x_cen_use.append(row_x_cen)
            y_cen_use.append(row_y_cen)
            im_ind_list.append(int(row_im_ind))

# Slice the lists from the given index => include the last index
x_cen_use = x_cen_use[start_ind : end_ind+1]
y_cen_use = y_cen_use[start_ind : end_ind+1]
im_ind_list = im_ind_list[start_ind : end_ind+1]

cen_len = len(x_cen_use)  

# Perform PSF fitting
current_row = start_ind + 1                           
loop_ind = 0

for _ in range(cen_len):
    # Define the center x and y for each interation
    x = x_cen_use[loop_ind]             
    y = y_cen_use[loop_ind]
    im_ind = im_ind_list[loop_ind]

    loop_ind += 1

    # Direct where the outputs will be
    output_name = f"row_num_{current_row}_im{str(input_im_filt)}_ind_{str(im_ind)}"
    output_file_folder = psf_outputs_folder/output_name
    output_file_folder.mkdir(parents=True, exist_ok=True)  

    # Open image and get relevant data
    x_im=fits.open(sim_im_folder/f'simulated_im_{input_im_filt}_index_{im_ind}.fits')
    
    filter_name=x_im[0].header['FILTER']
    
    # Get data in arrays
    global full_data_array
    global full_error_array
    
    data =x_im[1].data
    imh = x_im[1].header
    errordata_use = x_im[2].data
    area = x_im[4].data
    
    global PHOTMJSR
    global PIXAR_SR
    global EFFEXPTM
    PHOTMJSR = imh['PHOTMJSR']
    PIXAR_SR = imh['PIXAR_SR']
    EFFEXPTM = x_im[0].header['EFFEXPTM']
    
    data_conv = conv_to_N_photon(data)
    data_use = data_conv * area
    data_use[np.isnan(data_use)]=0
    full_data_array=data_use
    
    errordata_use = conv_to_N_photon(errordata_use)
    errordata_use = errordata_use * area
    errordata_use[np.isnan(errordata_use)]=0
    full_error_array=errordata_use
    
    # Read in ePSF library to use
    epsf_array=fits.open(str(psf_folder)+'/STDPSF_MIRI_'+filter_name+'.fits')
    epsf_library = epsf_array[0].data

    # Make subarrays
    new_sub=data_use[int(y-cutout_size):int(y+cutout_size)+1,int(x-cutout_size):int(x+cutout_size)+1]
    # plt.figure()
    # plt.imshow(new_sub)
    
    # Find brightest pixel and recenter
    tup=unravel_index(new_sub.argmax(), new_sub.shape)
    
    global newxcoor
    global newycoor

    newycoor, newxcoor=y-int(cutout_size)+tup[0], x-int(cutout_size)+tup[1]
    #print('THE TRUE POSITION: ', newxcoor, newycoor)
    
    new_sub2=data[int(newycoor-cutout_size):int(newycoor+cutout_size)+1,int(newxcoor-cutout_size):int(newxcoor+cutout_size)+1]
    
    # plt.figure()
    # plt.imshow(new_sub2)
    # plt.show()
    global bkg_mean
    # Remove Background
    rin, rout= 8, 12
    annulus = CircularAnnulus([newxcoor, newycoor], r_in=rin, r_out=rout)
    bkg = aperture_photometry(data_use, annulus)
    bkg_mean=bkg['aperture_sum']/(np.pi*(rout**2-rin**2))
    full_data_array=full_data_array-bkg_mean

    # Perform PSF fitting
    
    best_output_model_epsf, bestfit_parameters_epsf, new_centered_data, new_centered_error=psf_fitting()
    
    new_centered_data_flat=new_centered_data.flatten()
    new_centered_data_flat_red=new_centered_data_flat[new_centered_data_flat+bkg_mean!=0]
    
    new_centered_error=new_centered_error.flatten()
    new_centered_error=new_centered_error[new_centered_data_flat+bkg_mean!=0]
    
    best_output_model_epsf_flat=best_output_model_epsf.flatten()
    best_output_model_epsf_flat=best_output_model_epsf_flat[new_centered_data_flat+bkg_mean!=0]
    
    chi2_epsf = (((new_centered_data_flat_red[new_centered_error > 0.0]-best_output_model_epsf_flat[new_centered_error > 0.0])/new_centered_error[new_centered_error > 0.0] )**2).sum()/(len(new_centered_error)-6)
    # print(chi2_epsf)
    
    # print('Data: ', new_centered_data)
    # print('Model: ', best_output_model_epsf)
    
    fig, ((ax1, ax2,ax3)) = plt.subplots(1, 3, figsize=(18,6))
    im1 = ax1.imshow(new_centered_data, cmap=plt.get_cmap('Greys_r'),vmin=-10,vmax=np.max(new_centered_data))
    im2 = ax2.imshow(best_output_model_epsf, cmap=plt.get_cmap('Greys_r'),vmin=-10,vmax=np.max(new_centered_data))
    im3 = ax3.imshow(new_centered_data-best_output_model_epsf, cmap=plt.get_cmap('Greys_r'),vmin=-2,vmax=np.max(new_centered_data-best_output_model_epsf))
    
    ax1.set_ylim(ax1.get_ylim()[::-1])
    ax2.set_ylim(ax2.get_ylim()[::-1])
    ax3.set_ylim(ax3.get_ylim()[::-1])
    
    plt.colorbar(im1,ax=ax1)
    plt.colorbar(im1,ax=ax2)
    plt.colorbar(im3,ax=ax3)

    
    try:
        hdul=fits.PrimaryHDU(best_output_model_epsf)
        hdul.writeto(str(output_file_folder)+'/row_num_'+str(current_row)+'_bestmodel_double.fits', overwrite=True)
        
        hdul2=fits.PrimaryHDU(new_centered_data)
        hdul2.writeto(str(output_file_folder)+'/row_num_'+str(current_row)+'_data_double.fits', overwrite=True)
    except FileNotFoundError:
        pass
    
    plt.savefig(str(output_file_folder)+'/row_num_'+str(current_row)+'_output_double.png')
    plt.close()

    x_cen_psf = bestfit_parameters_epsf[0]
    y_cen_psf = bestfit_parameters_epsf[1]
    flux_psf = bestfit_parameters_epsf[2]
    sep_psf = bestfit_parameters_epsf[3]
    ang_psf = bestfit_parameters_epsf[4]
    contrast_psf = bestfit_parameters_epsf[5]

    x_cen_output = x_cen_psf + x
    y_cen_output = y_cen_psf + y

    pr=open(str(output_file_folder)+'/row_num_'+str(current_row)+'_output_parameters_double.txt','w')
    pr.write(str(x_cen_output)+'\t'+str(y_cen_output)+'\t'+str(flux_psf)+'\t'+str(sep_psf)+'\t'+str(ang_psf)+'\t'+str(contrast_psf)+'\t'+str(chi2_epsf)+'\t'+str(len(new_centered_error)))
    pr.close()

    # Save the index to know where to start again if the code stops running
    count=open(str(output_folder)+'/trial_counting.txt','a')
    count.write(f"Finished processing simulated at index {im_ind} for center x {x}, center y {y} at index of {current_row-1} in both the x_cen_use and y_cen_use lists.\n")
    count.close()

    current_row += 1



