#!/usr/bin/env python3
import numpy as np

import argparse
import os
import sys

# Allow running this file directly from Privitera_335 (as ./scripts/nonlinearity_studies/run_nonlinearity_studies.py) or from scripts (as ./nonlinearity_studies/run_nonlinearity_studies.py) 
# by adding the parent "scripts" directory to PYTHONPATH
# This block must come before any module imports from scripts directory
current_dir = os.path.dirname(os.path.abspath(__file__)) #nonlinearity_studies
parent_dir = os.path.dirname(current_dir) #scripts
project_root = os.path.dirname(parent_dir)  # Privitera_335
if parent_dir not in sys.path:
    sys.path.insert(0, project_root+parent_dir)

# Add scripts_dir to sys.path so sibling folders are importable
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tools.stitch_fits import stitch_fits
from nonlinearity_studies.nonlinearity_studies import get_fits, get_zero_one_peaks_ext, get_all_peaks_ext, get_nonlinearity_ext, get_nonlinearity_at_ext, \
                                                      plot_zero_one_peaks, plot_all_peaks, plot_nonlinearity

def main(args):
    """
    The main executable function of the script.
    
    Args:
        args: The Namespace object containing the parsed command-line arguments.
    """
    file_string = args.file_string

    if not args.stitch_fits:
        if not os.path.isabs(file_string):
            file_string = os.path.join(project_root, file_string)

    do_stitch_images = args.stitch_fits
    do_plot_zero_one_peaks = args.plot_zero_one_peaks
    do_plot_all_peaks = args.plot_all_peaks
    do_get_nonlinearity_at = args.get_nonlinearity_at
    do_plot_nonlinearity = args.plot_nonlinearity
    save_plots = args.save_plots

    if do_stitch_images:
        # Stitch images together by extension
        data_path = '/Users/abbychriss/Desktop/Privitera_335/data/test_chamber/Am241-Spectra-data/1x1-bin/'
        stitch_fits_image_string = file_string #f'{img_type}_img_CV_250x3500x500_bin1x1_125*'

        image_name = stitch_fits(data_path, directory='', image=stitch_fits_image_string, out_path='combined-fits/', print_header=False)
        image_name = image_name.split('/')[-1]
    else:
        image_name = file_string.split('/')[-1] #"avg_img_CV_250x3500x500_bin1x1_125_52_stitched.fits"
        file_path = '/'.join(s for s in file_string.split('/')[:-1]) #"/Users/abbychriss/Desktop/Privitera_335/data/test_chamber/Am241-Spectra-data/1x1-bin/combined-fits/"

    fig_path = "/Users/abbychriss/Desktop/Privitera_335/plots/nonlinearity_studies/"
    print(f'Analyzing image: {image_name}')
    # Get data from fits file
    data_ext = get_fits(image_name)

    # Fit zeroth and first electron peaks to double gaussians
    zero_one_counts_ext, zero_one_edges_ext, pedestals, gains, \
    double_gauss_popts, zero_one_ranges = get_zero_one_peaks_ext(data_ext, fit_bounds='default')

    # Apply scipy peak finder to find location of every electron peak
    counts_ext, edges_ext, peaks_ext, centers_ext, hist_ranges = get_all_peaks_ext(data_ext, 
                                                                                widths=[0.3,0.3,0.6,0.3], 
                                                                                buffers=[2.1,2.1,2.1,2.1], 
                                                                                pedestals=pedestals, 
                                                                                double_gauss_popts=double_gauss_popts,
                                                                                gains=gains,
                                                                                bins='default',
                                                                                flatten=True,
                                                                                do_convert_to_electrons=True, 
                                                                                range_left='default', 
                                                                                range_right=2500, 
                                                                                bin_factor=8)

    # Fit parabola to nonlinearity curve
    fit_range_right_ext= [600,800,500,1000]

    peak_charge_e_ext, charge_minus_npeak_ext, parabola_coeffs, parabola_pcovs, nonlinearity_at_500 = get_nonlinearity_ext(peaks_ext,
                                                                                                                           centers_ext, 
                                                                                                                           pedestals, 
                                                                                                                           gains, 
                                                                                                                           fit_range_right_ext, 
                                                                                                                           do_convert_to_electrons=False,
                                                                                                                           fit_bounds_low=-100, 
                                                                                                                           fit_bounds_high=100,
                                                                                                                           print_values=True)

    # Get nonlinearity at specified charge value(s)
    if do_get_nonlinearity_at:
        get_nonlinearity_at_ext([10, 500, 1000, 1500], parabola_coeffs, parabola_pcovs, fit_range_right_ext, print_values=True)

    # Fit a double gaussian to zero + 1 electron peak in each extension
    if do_plot_zero_one_peaks:
        plot_zero_one_peaks(data_ext, 
                            zero_one_counts_ext,
                            zero_one_edges_ext, 
                            pedestals, 
                            gains, 
                            double_gauss_popts, 
                            zero_one_ranges,
                            individual_figsize=(6,5), 
                            subplots_figsize=(9,7),
                            xlim='default',
                            #ylim=(0.00001,2e5),
                            yscale='linear',
                            fontsize=8,
                            n=200, 
                            do_convert_to_electrons=True,
                            plot_individual=False,
                            plot_together=True,
                            save_plots=save_plots,
                            fig_path=fig_path, 
                            file=image_name, 
                            dpi=350)

    if do_plot_all_peaks:
        plot_all_peaks_range_left=np.min(np.array(hist_ranges).flatten())
        plot_all_peaks_range_right=np.max(np.array(hist_ranges).flatten())

        plot_all_peaks(counts_ext, 
                    peaks_ext, 
                    centers_ext,  
                    xlim=(plot_all_peaks_range_left, plot_all_peaks_range_right),
                    ylim='none', 
                    yscale='linear',
                    plot_individual=False, 
                    plot_together=True, 
                    draw_lines=True, 
                    linecolor='r', 
                    linestyle='--',
                    individual_figsize=(7,6), 
                    subplots_figsize=(9,7),
                    suptitle='Peaks in Pixel Charge Distribution',
                    save_plots=save_plots,
                    fig_path=fig_path,
                    file=image_name, 
                    dpi=350,)

    if do_plot_nonlinearity:
        plot_nonlinearity(peaks_ext,
                        parabola_coeffs, 
                        peak_charge_e_ext, 
                        charge_minus_npeak_ext,
                        fit_range_right_ext,
                        xlim='default', 
                        ylim='default',
                        individual_figsize=(6,5), 
                        subplots_figsize=(9,7),
                        suptitle='Pixel Charge Nonlinearity Curve (Nimages = 10)',
                        line_color='r', 
                        scatter_color='b', 
                        s=2, 
                        alpha=0.5,
                        plot_individual=False, 
                        plot_together=True, 
                        save_plots=save_plots, 
                        fig_path=fig_path, 
                        file=image_name, 
                        dpi=350)
        

def init_argparse():
    """
    Initializes the ArgumentParser object and defines arguments.
    """
    parser = argparse.ArgumentParser(description="""Run nonlinearity analysis pipeline.

    This script can:
    - Stitch FITS images
    - Fit to zeroth/first electron peaks to compute pedestal, noise, gain
    - Fit and plot all electron peaks
    - Compute and plot nonlinearity
                                    
    You can enable any combination of steps using flags below.""")

    parser.add_argument('file_string', type=str, help='aboslute or relative path (from Privitera_335) to image file (.fz or .fits accepted)')
    parser.add_argument("-f","--stitch_fits", action="store_true", default=False, help="Stitch FITS files by extension")
    parser.add_argument("-z","--plot_zero_one_peaks", action="store_true", default=False, help="Plot fits to zero+one electron peaks")
    parser.add_argument("-a","--plot_all_peaks", action="store_true", default=False, help="Plot entire charge distribution with line at each peak")
    parser.add_argument("-g","--get_nonlinearity_at", action="store_true", default=False, help="Estimate nonlinearity at specified charge value(s) using parabolic fit")
    parser.add_argument("-n","--plot_nonlinearity", action="store_true", default=True, help="Plot nonlinearity curve with quadratic fit")
    parser.add_argument("-s","--save_plots", action="store_true", default=False, help="Save all plots as jpeg images")

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = init_argparse()

    main(args)