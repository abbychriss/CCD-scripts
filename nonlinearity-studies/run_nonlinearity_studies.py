#!/usr/bin/env python3
import argparse
import numpy as np
from ..stitch_fits import stitch_fits
from nonlinearity_studies import get_fits, get_zero_one_peaks_ext, get_all_peaks_ext, get_nonlinearity_ext, get_nonlinearity_at_ext, \
                                plot_zero_one_peaks, plot_all_peaks, plot_nonlinearity

def main(args):
    """
    The main executable function of the script.
    
    Args:
        args: The Namespace object containing the parsed command-line arguments.
    """
    do_stitch_images = args.stitch_fits
    do_plot_zero_one_peaks = args.plot_zero_one_peaks
    do_plot_all_peaks = args.plot_all_peaks
    do_get_nonlinearity = args.get_nonlinearity
    do_plot_nonlinearity = args.plot_nonlinearity

    if do_stitch_images:
        # Stitch images together by extension
        data_path = '/Users/abbychriss/Desktop/Privitera_335/data/test_chamber/Am241-Spectra-data/1x1-bin/'
        img_type='avg'
        image = f'{img_type}_img_CV_250x3500x500_bin1x1_125*'

        image_name = stitch_fits(data_path, directory='03*/', image=image, out_path='combined-fits/', print_header=False)
        print(image_name)

    else:
        image_name = "avg_img_CV_250x3500x500_bin1x1_125_52_stitched.fits"
        file_path = "/Users/abbychriss/Desktop/Privitera_335/data/test_chamber/Am241-Spectra-data/1x1-bin/combined-fits/"

    fig_path = "/Users/abbychriss/Desktop/Privitera_335/plots/"

    # Get data from fits file
    data_ext = get_fits(image_name)

    # Fit zeroth and first electron peaks to double gaussians
    zero_one_counts_ext, zero_one_edges_ext, pedestals, gains, \
    double_gauss_popts, zero_one_ranges = get_zero_one_peaks_ext(data_ext, 
                                                                fit_bounds='default')

    # Apply scipy peak finder to find location of every electron peak
    counts_ext, edges_ext, peaks_ext, centers_ext, hist_ranges = get_all_peaks_ext(data_ext, 
                                                                                widths=[0.9,0.8,1,1], 
                                                                                buffers=[2,1,1,2], 
                                                                                pedestals=pedestals, 
                                                                                double_gauss_popts=double_gauss_popts,
                                                                                gains=gains,
                                                                                bins='default',
                                                                                flatten=True,
                                                                                do_convert_to_electrons=True, 
                                                                                range_left=0, 
                                                                                range_right=2500, 
                                                                                bin_factor=8)

    # Fit parabola to nonlinearity curve
    fit_range_right_ext=[600,1500,1000,600]
    peak_charge_e_ext, charge_minus_npeak_ext, parabola_coeffs, parabola_pcovs, nonlinearity_at_500 = get_nonlinearity_ext(peaks_ext,
                                                                                                            centers_ext, 
                                                                                                            pedestals, 
                                                                                                            gains, 
                                                                                                            fit_range_right_ext, 
                                                                                                            fit_bounds_low=-100, 
                                                                                                            fit_bounds_high=100,
                                                                                                            print_values=True)

    # Get nonlinearity at specified charge value(s)
    if do_get_nonlinearity:
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
                            save_plots=False,
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
                    save_plots=False,
                    fig_path=fig_path,
                    file=image_name, 
                    dpi=350,)

    if do_plot_nonlinearity:
        plot_nonlinearity(peaks_ext,
                        parabola_coeffs, 
                        peak_charge_e_ext, 
                        charge_minus_npeak_ext,
                        xlim='default', 
                        ylim='default',
                        individual_figsize=(6,5), 
                        subplots_figsize=(9,7),
                        suptitle='Pixel Charge Nonlinearity Curve',
                        line_color='r', 
                        scatter_color='b', 
                        s=2, 
                        alpha=0.5,
                        plot_individual=False, 
                        plot_together=True, 
                        save_plots=False, 
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

    parser.add_argument('file_string', type=int, help='aboslute or relative path to image file (.fz or .fits accepted)')
    parser.add_argument("-s","--stitch_fits", action="store_true", default=False, help="Stitch FITS files by extension")
    parser.add_argument("-z","--plot_zero_one_peaks", action="store_true", default=False, help="Plot fits to zero+one electron peaks")
    parser.add_argument("-a","--plot_all_peaks", action="store_true", default=False, help="Plot entire charge distribution with line at each peak")
    parser.add_argument("-g","--get_nonlinearity", action="store_true", default=False, help="Estimate nonlinearity at specified charge value(s) using parabolic fit")
    parser.add_argument("-n","--plot_nonlinearity", action="store_true", default=True, help="Plot nonlinearity curve with quadratic fit")

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = init_argparse()
    #args = parser.parse_args()

    main(args)