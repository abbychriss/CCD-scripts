import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.signal import find_peaks as scipy_find_peaks
import math

from glob import glob

#plt.rcParams['text.usetex'] = True

#---------------- ANALYSIS FUNCTIONS ----------------------------

#---------------- (0) Convert to electrons ----------------------
def convert_to_electrons(data, pedestal, gain, flatten=True):
    if flatten:
        data = np.array(data).flatten()
    data_electrons = (data - pedestal) / gain  # Subtract pedestal (mean ADU of zero electron peak) and divide by gain
    return data_electrons

#---------------- (1) Calculate noise/gain ----------------------
# Function finds noise and gain from input pixel charge data
# zero_one_range is range of charge (in ADU) we want to restrict to for finding the zero and one electron peaks
def calculate_noise_gain(data, zero_one_test_range=[8,15], n=200, fit_bounds='default'):

    data = np.array(data).flatten()
    data_test_range = data[(data > zero_one_test_range[0]) & (data < zero_one_test_range[1])]

    nbins=min(2000, int(n*(zero_one_test_range[1]-zero_one_test_range[0])))
    counts_test, edges_test = np.histogram(data_test_range,bins=nbins,
                                           range=(zero_one_test_range[0],zero_one_test_range[1]))

    # Find index of maximum of counts, which corresponds to the mean ADU of the zero electron peak
    zero_peak_index = np.argmax(counts_test)
    centers_test = 0.5 * (edges_test[:-1] + edges_test[1:])
    zero_peak_charge = centers_test[zero_peak_index]

    # Restrict data range to only include the zero and one electron peaks
    # Nominal gain is around 1.3 and noise less than 1 electron 
    zero_one_left = zero_peak_charge - 1
    zero_one_right = zero_peak_charge + 2
    zero_one_range = [zero_one_left, zero_one_right]
    data_window = data[(data > zero_one_left) & (data < zero_one_right)]

    # Fit double gaussian to range [zero_peak_charge - 1, zero_peak_charge + 2.5]
    zero_one_counts, zero_one_edges = np.histogram(data_window,bins=nbins,range=zero_one_range)
    zero_one_centers = 0.5 * (zero_one_edges[:-1] + zero_one_edges[1:])

    if fit_bounds == 'default':
        fit_bounds = ([0.001, zero_one_left, 0.0001, zero_one_left+1, 1e4, 1e3], [10, zero_one_left+1, 10, zero_one_right, 1e7,1e7])
    popt, pcov = curve_fit(double_gauss, zero_one_centers, zero_one_counts, maxfev=2000, bounds=fit_bounds)
    
    # Extract pedestal, noise, gain, and rest of double gaussian coefficients from curve fit
    pedestal=tuple(popt)[1] # Pedestal is mean of zero electron peak
    noise=tuple(popt)[0] # Noise is standard deviation of zero electron peak 
    gain=tuple(popt)[3]-tuple(popt)[1] # Gain is difference between mean of one and zero electron peaks

    return zero_one_counts, zero_one_edges, pedestal, noise, gain, popt, zero_one_range


#---------------- (2) Find peaks ----------------------------
# Input is charge data (in ADU or electrons) from one extension
# bins is the number of bins given for initial charge histogram
# bin_factor is the multiple of the length of range used in fitting all peaks (number of bins per peak essentially)
# bin_factor is also used to define the distance parameter given to scipy_find_peaks 
# for the min distance between peaks (with buffer given by buffer) 
def find_electron_peaks(data, 
                        width, 
                        buffer, 
                        bins='default',
                        flatten=True,
                        do_convert_to_electrons=True,
                        range_left=0, 
                        range_right=2500, 
                        bin_factor=8):
    
    if flatten:
        data=np.array(data).flatten()

    #zero_one_counts, zero_one_edges, pedestal, noise, gain, popt, zero_one_range = calculate_noise_gain(data)
    
    """if do_convert_to_electrons:
        data = convert_to_electrons(data, pedestal, gain, flatten=True)"""

    """if range_left=='left_of_zero':
        range_left=pedestal-3*noise - 1 #make left end of range at the left side of the zero electron peak """

    hist_range = (range_left, range_right)

    if bins=='default':
       bins=math.floor((hist_range[1]-hist_range[0])*bin_factor)

    counts, edges = np.histogram(data, bins=bins,range=hist_range)
    centers = 0.5 * (edges[1:] + edges[:-1])
    peaks, properties = scipy_find_peaks(counts, height=0, width=width, distance=bin_factor-buffer)

    return counts, edges, peaks, centers, properties, hist_range


#---------------- (3) Fit nonlinearity ----------------------------
def fit_nonlinearity(peaks, centers, pedestal, gain, fit_range_right, fit_bounds_low=-100, fit_bounds_high=100):
    # Convert to electrons: subtract the pedestal (mean of zero electron peak) from all charge values, 
    # Then divide by the gain (difference between zero and 1 electron peak)
    peak_charge_e = np.array([(centers[p]-pedestal)/gain for p in peaks])
    charge_minus_npeak = [(peak_charge_e[i] - i) for i in range(len(peaks))]
    parabola_coeff, pcov = curve_fit(parabola, peak_charge_e[:fit_range_right], charge_minus_npeak[:fit_range_right],
                           maxfev=2000, bounds=(fit_bounds_low, fit_bounds_high))
    return parabola_coeff, peak_charge_e, charge_minus_npeak

#---------------- PLOTTING FUNCTIONS ----------------------------

#---------------- Plot zero-one peaks  -----------------------
# Usage: for plotting zero-one electron peaks from each extension on same subplot or individually by extension.
# Input data is list of 2D pixel charge arrays from all 4 extensions.
# xlim can be 'default', 'none', or tuple(left, right)
# ylim can be 'none' or tuple(bottom, top)
def plot_zero_one_peaks(data_ext, 
                        zero_one_counts_ext,
                        zero_one_edges_ext,
                        pedestals, 
                        gains, 
                        double_gauss_popts, 
                        zero_one_ranges,
                        individual_figsize=(6,5), 
                        subplots_figsize=(9,7),
                        xlim='default', ylim='none',
                        fontsize=7.5,
                        yscale='linear',
                        n=200, 
                        do_convert_to_electrons=False,
                        plot_individual=False,
                        plot_together=True,
                        save_plots=False,
                        fig_path='./', file='zero_one_peaks', 
                        dpi=350):


    if file !='zero_one_peaks':
        fig_name=fig_path+file[:-5]+'_zero_one_peaks'
    else:
        fig_name=fig_path+file

    if plot_individual:
        for ext, data in enumerate(data_ext):
            data = np.array(data).flatten()

            zero_one_counts=zero_one_counts_ext[ext]
            zero_one_edges=zero_one_edges_ext[ext]
            pedestal=pedestals[ext] 
            gain=gains[ext]
            double_gauss_popt=double_gauss_popts[ext]
            zero_one_range=zero_one_ranges[ext]

            fig, ax = plt.subplots(1, 1, figsize=individual_figsize, constrained_layout=True)

            double_gauss_coeff = tuple(double_gauss_popt)+(gain,)
            data_window = data[(data > zero_one_range[0]) & (data < zero_one_range[1])]
            nbins=int(n*(zero_one_range[1] - zero_one_range[0]))

            bin_width = zero_one_edges[1] - zero_one_edges[0]
            zero_one_centers = 0.5 * (zero_one_edges[:-1] + zero_one_edges[1:])

            if yscale=='log':
                zero_one_counts = np.maximum(zero_one_counts, 1) #need in order to prevent empty bars in histogram if there are any bins that have 0 counts
                ax.set_yscale('log')
            elif yscale!='linear':
                ax.set_yscale(yscale)
            ax.bar(zero_one_edges[:-1], zero_one_counts, edgecolor='none', align='edge', width=np.diff(zero_one_edges))

            ax.set_xlabel('Charge (ADU)')
            ax.set_ylabel('N')
            ax.set_title(f'EXT {ext}')
            
            ax.plot(zero_one_centers, double_gauss(zero_one_centers, *double_gauss_popt), 'r',
                label=r'$\sigma_0$ = %5.3f, $\mu_0$ = %5.3f, $\sigma_1$ = %5.3f, $\mu_1$ = %5.3f,'%double_gauss_coeff[0:4]
                +'\n'+'$N_0$ = %5.3f, $N_1$ = %5.3f, gain = %5.3f ADU/$e^{–}$'%double_gauss_coeff[4:])
            ax.legend(loc="upper right", fontsize=fontsize)
            
            if xlim=='default':
                ax.set_xlim(zero_one_range[0],zero_one_range[1])
            elif xlim!='none':
                ax.set_xlim(xlim)
            
            if ylim!='none':
                ax.set_ylim(ylim)

            if save_plots:
                plt.savefig(fig_name+f'_EXT{ext}.jpeg',dpi=dpi)
            plt.show()

        if do_convert_to_electrons:
            fig, axs = plt.subplots(2, 2, figsize=individual_figsize, constrained_layout=True)
            axs = axs.flatten()
            
            for ext, data in enumerate(data_ext):
                data=np.array(data).flatten()
                ax = axs[ext]
                zero_one_range = zero_one_ranges[ext]
                pedestal = pedestals[ext]
                gain = gains[ext] 

                data_window=data[(data > zero_one_range[0]) & (data < zero_one_range[1])]

                data_window_e = convert_to_electrons(data_window, pedestal, gain)
                zero_one_range_e = convert_to_electrons(zero_one_range, pedestal, gain) 
                nbins = int(n * (zero_one_range_e[1] - zero_one_range_e[0]))

                zero_one_counts_e, zero_one_edges_e = np.histogram(data_window_e, bins=nbins, range=zero_one_range_e)
                zero_one_centers_e = 0.5 * (zero_one_edges_e[:-1] + zero_one_edges_e[1:])
                bin_width_e = zero_one_edges_e[1] - zero_one_edges_e[0]
            
                double_gauss_popt_e, double_gauss_pcov_e = curve_fit(double_gauss, zero_one_centers_e, zero_one_counts_e, maxfev=2000, 
                                                                     bounds=([0.0001, -1, 0.0001, 0.5, 0, 0], 
                                                                             [1.0,  1,  1.0,  2.0, np.inf, np.inf]))
                
                if yscale=='log':
                    zero_one_counts_e = np.maximum(zero_one_counts_e, 1) #need in order to prevent empty bars in histogram if there are any bins that have 0 counts
                    ax.set_yscale('log')
                elif yscale!='linear':
                    ax.set_yscale(yscale)

                ax.bar(zero_one_edges_e[:-1], zero_one_counts_e, align='edge', edgecolor='none', width=np.diff(zero_one_edges_e))
                ax.set_xlabel(r'Charge ($e^–$)')
                ax.set_ylabel('N')
                ax.set_yscale(yscale)
                ax.set_title(f'EXT {ext}')
                ax.plot(zero_one_centers_e, double_gauss(zero_one_centers_e, *double_gauss_popt_e), 'r',
                    label=r'$\sigma_0$ = %5.3f $e^{–}$, $\mu_0$ = %5.3f $e^{–}$, $\sigma_1$ = %5.3f $e^{–}$, $\mu_1$ = %5.3f $e^{–}$'%tuple(double_gauss_popt_e)[0:4])
                ax.legend(loc="upper right", fontsize=fontsize)
                
                if xlim=='default':
                    ax.set_xlim(zero_one_range_e[0], zero_one_range_e[1])
                elif xlim!='none':
                    ax.set_xlim(xlim)

                if ylim!='none':
                    ax.set_ylim(ylim)

                if save_plots:
                    plt.savefig(fig_name+f'_electrons_EXT{ext}.jpeg',dpi=dpi)
                plt.show()

    if plot_together:

        fig, axs = plt.subplots(2, 2, figsize=subplots_figsize, constrained_layout=True)
        axs = axs.flatten()
        for ext, data in enumerate(data_ext):
            data = np.array(data).flatten()
            zero_one_counts=zero_one_counts_ext[ext]
            zero_one_edges=zero_one_edges_ext[ext]
            pedestal=pedestals[ext] 
            gain=gains[ext]
            double_gauss_popt=double_gauss_popts[ext]
            zero_one_range=zero_one_ranges[ext]

            ax = axs[ext]
            double_gauss_coeff = tuple(double_gauss_popt)+(gain,)
            data_window = data[(data > zero_one_range[0]) & (data < zero_one_range[1])]
            nbins=int(n*(zero_one_range[1]-zero_one_range[0]))

            zero_one_centers = 0.5 * (zero_one_edges[:-1] + zero_one_edges[1:])
            bin_width = zero_one_edges[1] - zero_one_edges[0]

            if yscale=='log':
                zero_one_counts = np.maximum(zero_one_counts, 1) #need in order to prevent empty bars in histogram if there are any bins that have 0 counts
                ax.set_yscale('log')
            elif yscale!='linear':
                    ax.set_yscale(yscale)

            ax.bar(zero_one_centers, zero_one_counts, align='center', edgecolor='none', width=bin_width)
            
            ax.plot(zero_one_centers, double_gauss(zero_one_centers, *double_gauss_popt), 'r',
                label=r'$\sigma_0$ = %5.3f, $\mu_0$ = %5.3f, $\sigma_1$ = %5.3f, $\mu_1$ = %5.3f,'%double_gauss_coeff[0:4]
                +'\n'+'$N_0$ = %5.3f, $N_1$ = %5.3f, gain = %5.3f ADU/$e^{–}$'%double_gauss_coeff[4:])
            
            ax.set_xlabel('Charge (ADU)')
            ax.set_ylabel('N')
            ax.set_title(f'EXT {ext}')
            ax.legend(loc="upper right", fontsize=fontsize)

            if xlim=='default':
                ax.set_xlim(zero_one_range[0],zero_one_range[1])
            elif xlim!='none':
                ax.set_xlim(xlim)

            if ylim!='none':
                ax.set_ylim(ylim)
        if save_plots:
            plt.savefig(fig_name+'.jpeg',dpi=dpi)
        plt.show()

        if do_convert_to_electrons:
            fig, axs = plt.subplots(2, 2, figsize=subplots_figsize, constrained_layout=True)
            axs = axs.flatten()

            for ext, data in enumerate(data_ext):
                ax = axs[ext]
                data = np.array(data).flatten()
                zero_one_range = zero_one_ranges[ext]
                pedestal = pedestals[ext]
                gain = gains[ext]

                data_window = data[(data > zero_one_range[0]) & (data < zero_one_range[1])]

                data_window_e = convert_to_electrons(data_window, pedestal, gain)
                zero_one_range_e = convert_to_electrons(zero_one_range, pedestal, gain) 
                nbins = int(n * (zero_one_range_e[1] - zero_one_range_e[0]))

                zero_one_counts_e, zero_one_edges_e = np.histogram(data_window_e, bins=nbins, range=zero_one_range_e)
                zero_one_centers_e = 0.5 * (zero_one_edges_e[:-1] + zero_one_edges_e[1:])
                bin_width_e = zero_one_edges_e[1] - zero_one_edges_e[0]
                
                double_gauss_popt_e, double_gauss_pcov_e = curve_fit(double_gauss, zero_one_centers_e, zero_one_counts_e, maxfev=2000, bounds=([0.0, -1, 0.0, 0.5, 0.0, 0.0], 
                                                                                                                                               [1.0,  1,  1.0,  2.0, 1e7, 1e7]))
                
                if yscale=='log':
                    zero_one_counts_e = np.maximum(zero_one_counts_e, 1) #need in order to prevent empty bars in histogram if there are any bins that have 0 counts
                    ax.set_yscale('log')

                elif yscale!='linear':
                    ax.set_yscale(yscale)

                ax.bar(zero_one_centers_e, zero_one_counts_e, align='center', edgecolor='none', width=bin_width_e)

                ax.set_title(f'EXT {ext}')
                ax.plot(zero_one_centers_e, double_gauss(zero_one_centers_e, *double_gauss_popt_e), 'r',
                    label=r'$\sigma_0$ = %5.3f $e^{–}$, $\mu_0$ = %5.3f $e^{–}$, $\sigma_1$ = %5.3f $e^{–}$, $\mu_1$ = %5.3f $e^{–}$'%tuple(double_gauss_popt_e)[0:4])
                ax.legend(loc="upper right", fontsize=fontsize)
                ax.set_xlabel(r'Charge ($e^–$)')
                ax.set_ylabel('N')

                ax.set_title(f'EXT {ext}')
                
                if xlim=='default':
                    ax.set_xlim(zero_one_range_e[0], zero_one_range_e[1])
                elif xlim!='none':
                    ax.set_xlim(xlim)

                if ylim!='none':
                    ax.set_ylim(ylim)

            if save_plots:
                plt.savefig(fig_name+f'_electrons.jpeg',dpi=dpi)
            plt.show()


#---------------- Plot all electron peaks ----------------------------
# Input is list of data from each of four extensions
# ylim can be 'none' or tuple=(ylim_bottom, ylim_top)
def plot_all_peaks(counts_ext, 
                   peaks_ext, 
                   centers_ext, 
                   xlim, ylim='none', 
                   yscale='log', 
                   plot_individual=True, plot_together=False, 
                   draw_lines=True, linecolor='r', linestyle='--',
                   individual_figsize=(6,5), subplots_figsize=(9,7),
                   suptitle='Peaks in Pixel Charge Distribution',
                   save_plots=False,
                   fig_path='./', file='peak_finder', 
                   dpi=350):


    if file !='peak_finder':
        fig_name=fig_path+file[:-5]+'_peak_finder'
    else:
        fig_name=fig_path+file

    if plot_individual:
        for ext, counts in enumerate(counts_ext):
            peaks=peaks_ext[ext]
            centers=centers_ext[ext]
            bin_width = centers[1] - centers[0]
            
            fig, ax = plt.subplots(1, 1, figsize=individual_figsize, constrained_layout=True)
            fig.suptitle(suptitle+f': EXT {ext}')
            ax.bar(centers, counts, align='center', edgecolor='none', width=bin_width)
            ax.set_xlabel(r'Charge ($e^-$)')
            ax.set_ylabel('N')
            if yscale!='linear':
                ax.set_yscale(yscale)
            ax.set_xlim(xlim)
            if ylim!='none':
                ax.set_ylim(ylim)

            # draw vertical lines and labels at each peak
            if draw_lines:
                for i,p in enumerate(peaks):
                    peak_x = centers[p]
                    peak_y = counts[p]

                    ax.axvline(peak_x, linestyle=linestyle, color=linecolor)
                    ax.text(peak_x,
                        peak_y,
                        f"{i}",
                        verticalalignment='bottom',
                        horizontalalignment='center',
                        color=linecolor,
                        fontsize=10)
                    
            if save_plots:
                plt.savefig(fig_name+f'_EXT{ext}.jpeg',dpi=dpi)
            plt.show()

    if plot_together:
        fig, axs = plt.subplots(2,2,figsize=subplots_figsize,constrained_layout=True)
        axs=axs.flatten()
        fig.suptitle(suptitle)

        for ext, data in enumerate(data_ext):
            counts=counts_ext[ext]
            peaks=peaks_ext[ext]
            centers=centers_ext[ext]
            bin_width = centers[1] - centers[0]
            ax = axs[ext]

            ax.bar(centers, counts, align='center', edgecolor='none', width=bin_width)
            ax.set_xlabel(r'Charge ($e^-$)')
            ax.set_ylabel('N')
            if yscale!='linear':
                ax.set_yscale(yscale)
            ax.set_xlim(xlim)
            if ylim!='none':
                ax.set_ylim(ylim)
            ax.set_title(f'EXT {ext}')

            # draw vertical lines and labels at each peak
            if draw_lines:
                for i,p in enumerate(peaks):
                    peak_x = centers[p]
                    peak_y = counts[p]

                    ax.axvline(peak_x, linestyle=linestyle, color=linecolor)
                    ax.text(peak_x,
                        peak_y,
                        f"{i}",
                        verticalalignment='bottom',
                        horizontalalignment='center',
                        color=linecolor,
                        fontsize=10)
        
        if save_plots:
            plt.savefig(fig_name+'.jpeg',dpi=dpi)
        plt.show()


#---------------- Plot nonlinearity ----------------------------
# xlim and ylim can be 'default', 'none', or tuple(ylim_bottom, ylim_top)
def plot_nonlinearity(peaks_ext,
                      parabola_coeffs, 
                      peak_charge_e_ext, 
                      charge_minus_npeak_ext,
                      xlim='default', ylim='default',
                      individual_figsize=(6,5), subplots_figsize=(9,7),
                      suptitle='Pixel Charge Nonlinearity Curve',
                      line_color='r', scatter_color='b', s=2, alpha=0.5,
                      plot_individual=False, 
                      plot_together=True, 
                      save_plots=False, 
                      fig_path='./', file='nonlinearity_curve', 
                      dpi=350):

    if file !='nonlinearity_curve':
        fig_name=fig_path+file[:-5]+'_nonlinearity'
    else:
        fig_name=fig_path+file

    if plot_individual:
        for ext, peaks in enumerate(peaks_ext):
            fig, ax = plt.subplots(1, 1, figsize=individual_figsize, constrained_layout=True)
            fig.suptitle(suptitle+f': EXT {ext}')
            ax.grid()

            parabola_coeff=parabola_coeffs[ext]
            peak_charge_e=peak_charge_e_ext[ext]
            charge_minus_npeak=charge_minus_npeak_ext[ext]

            ax.plot(peak_charge_e, parabola(peak_charge_e, *parabola_coeff), color=line_color,
                        label=r'$%5.6f x^2 + %5.3f x + %5.3f$' %tuple(parabola_coeff))
            ax.scatter(peak_charge_e, charge_minus_npeak, c=scatter_color, s=s, alpha=alpha)
            ax.legend(loc="upper right", fontsize=8)
            ax.set_xlabel(r'Measured Pixel Charge ($e^-$)')
            ax.set_ylabel(r'Measured Pixel Charge - Peak n. ($e^-$) ')

            if ylim=='default':
                ax.set_ylim(min(charge_minus_npeak)-10, max(charge_minus_npeak)+15)
            elif ylim!='none':
                ax.set_ylim(ylim)

            if xlim=='default':
                ax.set_xlim(-100, peak_charge_e[-1])
            elif xlim!='none':
                ax.set_xlim(xlim)

            if save_plots:
                plt.savefig(fig_name+f'_EXT{ext}.jpeg',dpi=dpi)
            plt.show()
            
    if plot_together:
        fig, axs = plt.subplots(2, 2, figsize=subplots_figsize, constrained_layout=True)
        axs=axs.flatten()
        fig.suptitle(suptitle)
        for ext, peak_charge_e in enumerate(peak_charge_e_ext):
            ax = axs[ext]
            ax.grid()

            parabola_coeff=parabola_coeffs[ext]
            charge_minus_npeak=charge_minus_npeak_ext[ext]

            ax.plot(peak_charge_e, parabola(peak_charge_e, *parabola_coeff), color=line_color,
                        label=r'$%5.6f x^2 + %5.3f x + %5.3f$' %tuple(parabola_coeff))
            ax.scatter(peak_charge_e, charge_minus_npeak, c=scatter_color, s=s, alpha=alpha)
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title(f'EXT {ext}')
            ax.set_xlabel(r'Measured Pixel Charge ($e^-$)')
            ax.set_ylabel(r'Measured Pixel Charge - Peak n. ($e^-$) ')

            if ylim=='default':
                ax.set_ylim(min(charge_minus_npeak)-10, max(charge_minus_npeak)+15)
            elif ylim!='none':
                ax.set_ylim(ylim)
  
            if xlim=='default':
                ax.set_xlim(-100, peak_charge_e[-1])
            elif xlim!='none':
                ax.set_xlim(xlim)

    if save_plots:
        plt.savefig(fig_name+'.jpeg',dpi=dpi)
    plt.show()


#---------------- UTILITY FUNCTIONS ----------------------------

#---------------- Get Fits ----------------------------
def get_fits(file_name, path="/Users/abbychriss/Desktop/Privitera_335/"):
    file=glob(path+'**/'+file_name,recursive=True)[0]
    hdu_list = fits.open(file)
    ext_charge=[hdu_list[i].data for i in range(1,5)]
    return ext_charge

#---------------- Return data for each extensions in a list from pixel charge data for all extensions
def get_zero_one_peaks_ext(data_ext, fit_bounds='default'):
    zero_one_counts_ext = []
    zero_one_edges_ext = []
    pedestals = []
    gains = []
    double_gauss_popts = []
    zero_one_ranges = []
    for data in data_ext:
        data=np.array(data).flatten()

        zero_one_counts, zero_one_edges, pedestal, noise, gain, double_gauss_popt, zero_one_range = calculate_noise_gain(data, fit_bounds=fit_bounds)
        zero_one_counts_ext.append(zero_one_counts)
        zero_one_edges_ext.append(zero_one_edges)
        pedestals.append(pedestal)
        gains.append(gain)
        double_gauss_popts.append(double_gauss_popt)
        zero_one_ranges.append(zero_one_range)

    return zero_one_counts_ext, zero_one_edges_ext, pedestals, gains, double_gauss_popts, zero_one_ranges
        

def get_all_peaks_ext(data_ext, widths, buffers, bins='default', flatten=True, do_convert_to_electrons=True, range_left='left_of_zero', range_right=2000, bin_factor=8):
    counts_ext = []
    edges_ext = []
    peaks_ext = []
    centers_ext = []
    hist_ranges = []
    for ext, data in enumerate(data_ext):
        data=np.array(data).flatten()
        width = widths[ext]
        buffer = buffers[ext]

        counts, edges, peaks, centers, properties, hist_range = find_electron_peaks(data, 
                                                                                    width, 
                                                                                    buffer, 
                                                                                    bins=bins,
                                                                                    flatten=flatten,
                                                                                    do_convert_to_electrons=do_convert_to_electrons,
                                                                                    range_left=range_left, 
                                                                                    range_right=range_right, 
                                                                                    bin_factor=bin_factor)
    
        counts_ext.append(counts)
        edges_ext.append(edges)
        peaks_ext.append(peaks)
        centers_ext.append(centers)
        hist_ranges.append(hist_range)

    return counts_ext, edges_ext, peaks_ext, centers_ext, hist_ranges


def get_nonlinearity_ext(data_ext, peaks_ext, centers_ext, pedestals, gains, fit_range_right_ext, fit_bounds_low=-100, fit_bounds_high=100):
    peak_charge_e_ext = []
    charge_minus_npeak_ext = []
    parabola_coeffs = []
    for ext, data in enumerate(data_ext):
        data=np.array(data).flatten()

        peaks=peaks_ext[ext]
        centers=centers_ext[ext]
        pedestal=pedestals[ext]
        gain=gains[ext]
        fit_range_right=fit_range_right_ext[ext]

        parabola_coeff, peak_charge_e, charge_minus_npeak = fit_nonlinearity(peaks,
                                                                             centers,
                                                                             pedestal, 
                                                                             gain, 
                                                                             fit_range_right, 
                                                                             fit_bounds_low, 
                                                                             fit_bounds_high)
        peak_charge_e_ext.append(peak_charge_e)
        charge_minus_npeak_ext.append(charge_minus_npeak)
        parabola_coeffs.append(parabola_coeff)

    return peak_charge_e_ext, charge_minus_npeak_ext, parabola_coeffs

#---------------- Curves ----------------------------
def double_gauss(x, s0, m0, s1, m1, N0, N1):
    return N0 * np.exp(-(x-m0)**2/(2*s0**2)) + N1 * np.exp(-(x-m1)**2/(2*s1**2))

def parabola(x, a, b, c):
    return a*x**2 + b*x + c


#------------------ RUN ANALYSIS AND MAKE PLOTS ------------------------------

image_name = "avg_img_CV_250x3500x500_bin1x1_125_52_stitched.fits"
file_path = "/Users/abbychriss/Desktop/Privitera_335/data/test_chamber/Am241-Spectra-data/1x1-bin/combined-fits/"
fig_path = "/Users/abbychriss/Desktop/Privitera_335/plots/"

do_plot_zero_one_peaks=True
do_plot_all_peaks=True
do_plot_nonlinearity=True

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
                                                                               bins='default',
                                                                               flatten=True,
                                                                               do_convert_to_electrons=True, 
                                                                               range_left=0, 
                                                                               range_right=2500, 
                                                                               bin_factor=8)

# Fit parabola to nonlinearity curve
nonlinearity_fit_ranges=[600,1500,1000,600]
peak_charge_e_ext, charge_minus_npeak_ext, parabola_coeffs = get_nonlinearity_ext(data_ext, 
                                                                                  peaks_ext,
                                                                                  centers_ext, 
                                                                                  pedestals, 
                                                                                  gains, 
                                                                                  nonlinearity_fit_ranges, 
                                                                                  fit_bounds_low=-100, 
                                                                                  fit_bounds_high=100)

#fit a double gaussian to zero + 1 electron peak in each extension
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