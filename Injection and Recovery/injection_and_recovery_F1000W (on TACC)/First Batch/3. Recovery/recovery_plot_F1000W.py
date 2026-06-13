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
# import pymultinest
# from pymultinest import watch
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

# Paths
base_folder = Path('/work/10875/andreavu/ls6/injection_and_recovery_F1000W')           
input_path = base_folder/"Outputs"/"binaries_injected_parameters.csv"
doublepsf_output = base_folder/"Outputs"/"doublepsf_fitting_outputs"

# Make output folder
recovery_output = base_folder/"Outputs"/"Recovery Analysis"
recovery_output.mkdir(parents=True, exist_ok=True)

# Make CSV file to save results
table_data = [['Simulated Image Name', 'Row Number', 'Input x_cen', 'Input y_cen', 'Input Flux', 'Input Separation', 'Input Positional Angle', 'Input Contrast', 'Output x_cen', 'Output y_cen', 'Output Flux', 'Output Separation', 'Output Positional Angle', 'Output Contrast', 'Recovery?']] 

for folder in doublepsf_output.glob('row_num_*'):
    rownum = int(folder.name.split('_')[2])
    with open(input_path, "r", newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        rows = list(csv_reader)
    input_data = rows[rownum]
    im_name = input_data[0]
    input_x_cen = float(input_data[1])
    input_y_cen = float(input_data[2])
    input_flux = float(input_data[3])
    input_sep = float(input_data[4])
    input_PA = float(input_data[5])
    input_contr = float(input_data[6])

    if input_PA > 180.0:
        input_PA = input_PA - 180.0

    input_x2_cen = float(input_sep*math.cos(input_PA*math.pi/180.0))
    input_y2_cen = float(input_sep*math.sin(input_PA*math.pi/180.0))	

    output_data_file = folder/f'row_num_{rownum}_output_parameters_double.txt'
    if not output_data_file.exists():
        print(f"Skipping {output_data_file} (not found).")
        continue

    with open(output_data_file, 'r') as myfile:
        for line in myfile:
            output_data = line.split('\t')
            output_x_cen = float(output_data[0])
            output_y_cen = float(output_data[1])
            output_flux = float(output_data[2])
            output_sep = float(output_data[3])
            output_PA = float(output_data[4])
            output_contr = float(output_data[5])

            if output_PA > 180.0:
                output_PA = output_PA - 180.0

            output_x2_cen = float(output_sep*math.cos(output_PA*math.pi/180.0))
            output_y2_cen = float(output_sep*math.sin(output_PA*math.pi/180.0))

    diff_x = abs(input_x2_cen-output_x2_cen)
    diff_y = abs(input_y2_cen-output_y2_cen)
    diff_cen = math.sqrt(diff_x**2 + diff_y**2)

    diff_contr = abs(input_contr-output_contr)

    if diff_cen <= 0.2 and diff_contr <= 0.4:
        recov = 'Yes'
    else:
        recov = 'No'

    row_new = [im_name, rownum, input_x_cen, input_y_cen, input_flux, input_sep, input_PA, input_contr, output_x_cen, output_y_cen, output_flux, output_sep, output_PA, output_contr, recov]
    table_data.append(row_new)

filename = 'recovery_all_results.csv'

with open(str(recovery_output) + '/' + filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile) 
    writer.writerows(table_data)  

separation = []
contrast = []

sep_all = []
contr_all = []

with open(recovery_output/'recovery_all_results.csv', newline="") as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader) 
    for row in reader:
        sep = float(row[5])
        contr = float(row[7])
        sep_all.append(sep)
        contr_all.append(contr)
        if row[-1] == "Yes":
            separation.append(sep)
            contrast.append(contr)
      
separation = np.array(separation)
contrast = np.array(contrast)

sep_all = np.array(sep_all)
contr_all = np.array(contr_all)

# Define Bins
sep_bins = np.linspace(sep_all.min(), sep_all.max(), 20)
contrast_bins = np.linspace(contr_all.min(), contr_all.max(), 20)

all_hist, xedges, yedges = np.histogram2d(sep_all, contr_all, bins=[sep_bins, contrast_bins])
yes_hist, _, _ = np.histogram2d(separation, contrast, bins=[sep_bins, contrast_bins])
ratio = np.divide(yes_hist, all_hist, out=np.zeros_like(yes_hist), where=all_hist!=0)

# Plot
plt.figure(figsize=(8, 6))
X, Y = np.meshgrid(xedges, yedges)

plt.pcolormesh(X, Y, ratio.T, cmap='viridis') 
plt.xlabel("Separation (pix)")
plt.ylabel("Contrast (mag)")
plt.title("Colormap of Recovery Fraction Contrast vs Separation")
plt.gca().invert_yaxis()  
cbar = plt.colorbar()
cbar.set_label("Recovery Fraction")
plt.tight_layout()
plt.savefig(recovery_output/"Colormap of Recovered Binaries Contrast vs Separation")
plt.close()


plt.scatter(separation, contrast)
plt.xlabel("Separation (pix)")
plt.ylabel("Contrast (mag)")
plt.title("Recovered Binaries Contrast vs Separation")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(recovery_output/"Scatter Plot of Recovered Binaries Contrast vs Separation")
plt.close()

