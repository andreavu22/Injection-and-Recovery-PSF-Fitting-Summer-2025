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


##############################################################################################################
# ################# PSF Fitting functions
# ##############################################################################################################

# Bins the model
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
    
    wavel = float(filter_name[1:3])      # in microns

    if wavel == 10:
        gain = 4.44       # in e-/DN
    elif wavel == 15:
        gain = 4.77       # in e-/DN
    else:		
        print("Look for the appropriate gain on page 11 of https://www.stsci.edu/files/live/sites/www/files/home/jwst/documentation/technical-documents/_documents/JWST-STScI-008797.pdf.")

    T_exp = EFFEXPTM      #in seconds

    if arr_in.ndim == 1:
        for ind, pix in enumerate(arr_in):     # pix in MJy/sr
            N_photon =  pix * gain * T_exp / PHOTMJSR  
            arr_out[ind] = N_photon
    elif arr_in.ndim == 2:
        for i, row in enumerate(arr_in):
            for j, pix in enumerate(row):       # pix in MJy/sr
                N_photon =  pix * gain * T_exp / PHOTMJSR
                arr_out[i, j] = N_photon
    elif arr_in.ndim == 0:
        arr_out = arr_in * gain * T_exp / PHOTMJSR

    return arr_out
    
def conv_to_MJy_sr(arr_in_1):
    """Convert an array of each pixel value in number of photons to MJy/sr."""
    arr_in_1 = np.array(arr_in_1)
    arr_out_1 = np.zeros_like(arr_in_1)
    
    wavel = float(filter_name[1:3])      # in microns

    if wavel == 10:
        gain = 4.44       # in e-/DN
    elif wavel == 15:
        gain = 4.77       # in e-/DN
    else:		
        print("Look for the appropriate gain on page 11 of https://www.stsci.edu/files/live/sites/www/files/home/jwst/documentation/technical-documents/_documents/JWST-STScI-008797.pdf.")

    T_exp = EFFEXPTM      #in seconds

    if arr_in_1.ndim == 1:
        for ind, N_photon in enumerate(arr_in_1):     # pix in photons count
            pix = N_photon * PHOTMJSR/ (gain * T_exp) 
            arr_out_1[ind] = pix
    elif arr_in_1.ndim == 2:
        for i, row in enumerate(arr_in_1):
            for j, N_photon in enumerate(row):       # pix in photons count
                pix = N_photon * PHOTMJSR/ (gain * T_exp) 
                arr_out_1[i, j] = pix
    elif arr_in_1.ndim == 0:
        arr_out_1 = arr_in_1 * PHOTMJSR/ (gain * T_exp) 

    return arr_out_1

def poisson_noise(arr_input):
    arr_output = np.zeros_like(arr_input)
    for a, row in enumerate(arr_input):
        for b, pix in enumerate(row):    
            pix_poisson = np.random.poisson(lam=pix, size=1)[0]
            arr_output[a, b] = pix_poisson
    return arr_output

# Find WD with the closest SNR to the target SNR
def get_path_avgWD (target_snr, list_snr, list_path):
    current_diff = None
    current_ind = None
    dup_ind = []
    for i, snr in enumerate(list_snr):
        diff = abs(snr-target_snr)
        if current_diff == None:
            current_diff = diff
            current_ind = i
        elif diff < current_diff:
            current_diff = diff
            current_ind = i
        elif diff == current_diff:
            dup_ind.append(f'ind: {i} for val {snr} and diff {diff}')

    return list_path[current_ind]

# Copy fully fits file
def copy_fits_file(input_file):   # fits.open must already applied for the input_file => it's not the path
    copied_hdus = []
    for hdu in input_file:
        hdu_class = type(hdu)
        data_copy = hdu.data.copy() if hdu.data is not None else None
        header_copy = hdu.header.copy()
        copied_hdus.append(hdu_class(data=data_copy, header=header_copy))

    output_file = fits.HDUList(copied_hdus)
    return output_file

# Function that outputs the WD with the injected ePSF
def model_injection(xcen=0, ycen=0, flux=0, separation=0, position_angle=0, contrast=0):    #input flux is average WD flux and must be in unit of number of photons
	# Make Primary PSF
    ix=int(xcen) # integer pixel position of model xcentroid+multinest_x_param
    iy=int(ycen)

    ix2 = int(xcen + separation*math.cos(position_angle*3.14159/180.0))
    iy2 = int(ycen + separation*math.sin(position_angle*3.14159/180.0))	
	
    fxu = xcen-ix # fractional pixel position (e.g. 1025.67 --> 0.67)
    fyu = ycen-iy

    fr = 10.**(contrast/(-2.5))
    # flux_both = flux * (1 + fr)        #flux here is flux of WD + companion (WD*fr)

    psfx = np.zeros((2*cutout_size+1,2*cutout_size+1))

    ftot = 0
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

    # Add poisson noise
    foundf_N_photon = foundf 

    foundf_poisson_phot = poisson_noise(foundf_N_photon)
    # Poisson noise of injected model to add to error array
    noise = np.maximum(np.sqrt(foundf_poisson_phot), 1)
    foundf_poisson = conv_to_MJy_sr(foundf_poisson_phot)
    
    cutout_poisson_noise = conv_to_MJy_sr(noise)
    	
    return foundf_poisson, cutout_poisson_noise	


###############################################################################################################
# ##############################################################################################################
# ######### Read in data file for each target ##################################################################


# Paths
base_folder = Path.home()/"injection_and_recovery_F1000W"
skysub_folder = Path('/arc/projects/UdeM_whitedwarfs/mead')
psf_folder = base_folder/"MIRI_PSFs"
output_folder = base_folder/"Outputs"

# Define cutout size
cutout_size=10
cutout_size_flux = 5

# Generate a background with Gaussian noise and get input flux for model_injection
rms_1000 = []

path_1000 = []
snr_1000 = [] 

for file in skysub_folder.rglob("*_mirimage_skysub_cal.fits"):
    # Read data
    im_file = fits.open(file)
    filter_name = str(im_file[0].header['FILTER'])
    imh = im_file[1].header

    if filter_name == "F1000W":
        # Define parameters for conv_to_N_photon
        global PHOTMJSR
        global PIXAR_SR
        global EFFEXPTM
        PHOTMJSR = imh['PHOTMJSR']
        PIXAR_SR = imh['PIXAR_SR']
        EFFEXPTM = im_file[0].header['EFFEXPTM']
    
        # Get center x and y coordinates
        ra=im_file[0].header['TARG_RA']
        dec=im_file[0].header['TARG_DEC']
    
        w = WCS(im_file[1].header, fobj=im_file)
        x, y = w.all_world2pix(ra, dec, 0)
        
        # Get data in arrays
        global full_data_array_inj
        global full_data_array_WD
    
        data =im_file[1].data
        full_data_array_inj = data
    
        data_WD = np.copy(full_data_array_inj)
        data_WD[np.isnan(data_WD)]=0
        full_data_array_WD = data_WD
    
       # Crop image to only the WD, convert it to number of photons, and calculate the SNR
        cropped_im = full_data_array_WD[int(y-cutout_size_flux):int(y+cutout_size_flux)+1,int(x-cutout_size_flux):int(x+cutout_size_flux)+1]
        photons_im = conv_to_N_photon(cropped_im)
        photons_sum = np.sum(photons_im)
        if not np.isfinite(photons_sum) or photons_sum < 0:
            print(f"Photons sum is {photons_sum} for file {file}.")
        snr = np.sqrt(photons_sum)
        
        # Mask stars
        masked_im = np.copy(full_data_array_inj)
        masked_im[:,:365] = np.nan
        rms_1 = np.nanstd(full_data_array_inj)
        med_1 = np.nanmedian(full_data_array_inj)
        threshold = med_1 + (5*rms_1)
        ind = full_data_array_inj >= threshold
        masked_im[ind] = np.nan
        # Get background rms
        rms_2 = np.nanstd(masked_im)
    
        # Append to appropriate list
        rms_1000.append(rms_2)
        snr_1000.append(snr) 
        path_1000.append(file)



# Get the median SNR from list
snr_WD_1000 = np.median(snr_1000)

# Obtain parameters and data of the WD that has the closest SNR to the highest count value
WD_1000_path = get_path_avgWD(snr_WD_1000, snr_1000, path_1000)      # Gives MEAD ID 49 as the chosen image

# Average PHOTMJSR and PIXAR_SR
WD_chosen = fits.open(WD_1000_path)
imh_WD = WD_chosen[1].header
WD_chosen_data = WD_chosen[1].data
WD_chosen_data[np.isnan(WD_chosen_data)]=0
WD_chosen_array = WD_chosen_data
ra_WD = WD_chosen[0].header['TARG_RA']
dec_WD = WD_chosen[0].header['TARG_DEC']
w_WD = WCS(WD_chosen[1].header, fobj=WD_chosen)
x_WD, y_WD = w_WD.all_world2pix(ra_WD, dec_WD, 0)
WD_chosen_cropped = WD_chosen_array[int(y_WD-cutout_size_flux):int(y_WD+cutout_size_flux)+1,int(x_WD-cutout_size_flux):int(x_WD+cutout_size_flux)+1]
PHOTMJSR = imh_WD['PHOTMJSR']
PIXAR_SR = imh_WD['PIXAR_SR']
EFFEXPTM = WD_chosen[0].header['EFFEXPTM']


# Average rms for each of the filter
rms_1000_array = np.array(rms_1000)
rms_1000_avg = np.nanmean(rms_1000_array)

# Generate a Gaussian distribution for simulated background noise
global bkg_gauss_1000

# Make a complete copy of the WD_chosen
WD_chosen = fits.open(WD_1000_path)
sim_im = copy_fits_file(WD_chosen)

size_axis1 = 1032 - 365
size_axis2 = 1024
bkg_shape = (size_axis2, size_axis1)
bkg_gauss_1000 = np.random.normal(loc=0.0, scale=rms_1000_avg, size=bkg_shape)       # this is the background of 1024 rows and 667 columns for F1000W

# Replace the right side of sim_im array by the simulated background
sim_im_array = sim_im[1].data
sim_im_array[:, -bkg_gauss_1000.shape[1]:] = bkg_gauss_1000
sim_im_array[np.isnan(sim_im_array)]=0

# Generate the simulated error array
sim_error_array = sim_im[2].data
sim_error_array[:, -bkg_gauss_1000.shape[1]:] = rms_1000_avg
sim_error_array[np.isnan(sim_error_array)]=0

# Inject companion
## Access PSF of appropriate filter
filt = 'F1000W'
epsf_array=fits.open(str(psf_folder)+'/STDPSF_MIRI_'+filt+'.fits')
epsf_library = epsf_array[0].data

## Define the input flux for the WD 
flux_WD_1000 = np.sum(conv_to_N_photon(WD_chosen_cropped))

## Desired number of trials
trials = 10000

## Define the range of contrast tested
rms_1000_array_phot = conv_to_N_photon(rms_1000_array)     # Because flux 1000 is in photons count
rms_1000_phot_avg = np.nanmean(rms_1000_array_phot)
min_contrast = (-2.5) * np.log10(rms_1000_phot_avg / flux_WD_1000)

## Define the range of separation and position angle of WD and companion tested
max_sep = 5   # Defined as approximation of WD's airy disk radius
max_ang = 360

## Define the range of center of WD tested
r = cutout_size + max_sep 
# r = 20
n_rows, n_cols = np.shape(sim_im_array)   # n_rows is total number of y values and n_cols is total number of x values
x_start = 365   # approximate index that the x values start at 

x_cen_range = []    
y_cen_range = []

for i in range((n_cols-x_start)//(2*r)):
    x_cen_range.append((r + (i*2*r)) + x_start)
for j in range(n_rows//(2*r)):
    y_cen_range.append(r + (j*2*r))

x_cen_use = []  # inputs for model_injection
y_cen_use = []  # inputs for model_injection
for x_cen in x_cen_range:
    for y_cen in y_cen_range:
        x_cen_use.append(x_cen)
        y_cen_use.append(y_cen)

cen_len = len(x_cen_use)   # Max amount of interations for 1 background

# Count trials and arrays
trial_count = 0

# Make a copy of the simulated fits file to place models on
all_sim_im = []
current_sim_im = copy_fits_file(sim_im)
current_sim_arr = current_sim_im[1].data
current_error_arr = current_sim_im[2].data
im_ind = 0

# Make CSV file to save all input parameters
table_data = [['Simulated Image Name', 'Input x_cen', 'Input y_cen', 'Input Flux of Binary', 'Input Separation', 'Input Positional Angle', 'Input Contrast']] 

## Loop
for i in range(trials):
    # Define the input parameters for this trial
    sep = np.random.rand() * max_sep
    ang = np.random.rand() * max_ang
    contrast = np.random.rand() * min_contrast

    inj_binary_ind = trial_count % cen_len
    x_cen = x_cen_use[inj_binary_ind] + np.random.rand() - 0.5          
    y_cen = y_cen_use[inj_binary_ind] + np.random.rand() - 0.5

    # If the array is full then create a new one
    if trial_count > (cen_len-1) and trial_count % cen_len == 0:
        all_sim_im.append(current_sim_im)
        current_sim_im = copy_fits_file(sim_im)
        current_sim_arr = current_sim_im[1].data
        current_error_arr = current_sim_im[2].data
        im_ind+=1

    # Create the binary
    psf_1000, centered_poisson_error = model_injection(x_cen, y_cen, flux_WD_1000, sep, ang, contrast)

    # Add data row to csv file
    fraction = 10.**(contrast/(-2.5))
    flux_binary = flux_WD_1000 * (1 + fraction)
    row_data = ['simulated_im_10_index_'+str(im_ind), x_cen, y_cen, flux_binary, sep, ang, contrast]
    table_data.append(row_data)

    x_cen_int = math.ceil(x_cen)
    y_cen_int = math.ceil(y_cen)

    # Add the simulated binary onto the simulated background
    current_sim_arr[int(y_cen_int-cutout_size):int(y_cen_int+cutout_size)+1,int(x_cen_int-cutout_size):int(x_cen_int+cutout_size)+1] += psf_1000          
    current_error_arr[int(y_cen_int-cutout_size):int(y_cen_int+cutout_size)+1,int(x_cen_int-cutout_size):int(x_cen_int+cutout_size)+1] = np.sqrt((current_error_arr[int(y_cen_int-cutout_size):int(y_cen_int+cutout_size)+1,int(x_cen_int-cutout_size):int(x_cen_int+cutout_size)+1]**2) + (centered_poisson_error**2))            
                    
    trial_count += 1   

# Once the loop ends, save the current array even if it is not filled
if trial_count % cen_len != 0:
    all_sim_im.append(current_sim_im) 

# Creat and save all images into a folder
im_folder_name = 'simulated_images'
im_folder_path = output_folder/im_folder_name
im_folder_path.mkdir(parents=True, exist_ok=True) 
for ind, fits_file in enumerate(all_sim_im):
    fits_file.writeto(str(im_folder_path)+'/simulated_im_10_index_'+str(ind)+'.fits', overwrite=True)


filename = 'binaries_injected_parameters.csv'

with open(str(output_folder) + '/' + filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile) 
    writer.writerows(table_data)  

