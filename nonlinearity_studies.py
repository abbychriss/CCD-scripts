import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import math

from .clustering import get_fits

#plt.rcParams['text.usetex'] = True


#---------------- ANALYSIS FUNCTIONS ----------------------------

#---------------- (0) Convert to electrons ----------------------
def convert_to_electrons(data, pedestal, gain, flatten=True):
    if flatten:
        data = np.array(data).flatten()
    data_electrons = [(q - pedestal)/gain for q in data]  # Subtract pedestal (mean ADU of zero electron peak) and divide by gain
    return data_electrons

#---------------- (1) Calculate noise/gain ----------------------
# Function finds noise and gain from input pixel charge data
# zero_one_range is range of charge (in ADU) we want to restrict to for finding the zero and one electron peaks
def calculate_noise_gain(data, zero_one_test_range=[8,15], n=200):

    data = np.array(data).flatten()
    data_test_range = data[(data > zero_one_test_range[0]) & (charge < zero_one_test_range[1])]

    nbins=int(n*len(zero_one_test_range))
    counts_test, edges_test = np.histogram(data_test_range,bins=nbins,range=(zero_one_test_range[0],zero_one_test_range[1]))

    # Find index of maximum of counts, which corresponds to the mean ADU of the zero electron peak
    zero_peak_index = np.where(counts_test == max(counts_test))
    zero_one_nbins = np.linspace(zero_one_test_range[0], zero_one_test_range[1], nbins)
    zero_peak_charge = zero_one_nbins[zero_peak_index]

    # Restrict data range to only include the zero and one electron peaks
    # Nominal gain is around 1.3 and noise less than 1 electron 
    zero_one_left = zero_peak_charge - 1
    zero_one_right = zero_peak_charge + 2.5
    zero_one_range = [zero_one_left, zero_one_right]
    data_window = data[(data > zero_one_left) & (charge < zero_one_right)]

    # Fit double gaussian to range [zero_peak_charge - 1, zero_peak_charge + 2.5]
    counts, edges = np.histogram(data_window,bins=nbins,range=zero_one_range)
    xdata = np.linspace(zero_one_left, zero_one_right, nbins)
    popt, pcov = curve_fit(double_gauss, xdata, counts, bounds=([0.02, zero_one_left, 0.005, zero_one_left+2,1e4,1e3], 
                                                                  [0.5, zero_one_right, 0.05, zero_one_right,1e7,1e7]))
    
    # Extract pedestal, noise, gain, and rest of double gaussian coefficients from curve fit
    pedestal=tuple(popt)[1] # Pedestal is mean of zero electron peak
    noise=tuple(popt)[0] # Noise is standard deviation of zero electron peak 
    gain=tuple(popt)[3]-tuple(popt)[1] # Gain is difference between mean of one and zero electron peaks

    return pedestal, noise, gain, popt, zero_one_range


#---------------- (2) Find peaks ----------------------------
# Input is charge data (in ADU or electrons) from one extension
def find_peaks(data, width, buffer=1, 
               convert_to_electrons=True,
               range_left='left_of_zero', 
               range_right=2500, 
               bin_factor=8):
    
    pedestal, noise, gain, popt, zero_one_range = calculate_noise_gain(data)
    
    if convert_to_electrons:
        data = convert_to_electrons(data, pedestal, gain, flatten=True)

    if range_left=='left_of_zero':
        range_left=pedestal-3*noise - 1 #make left end of range at the left side of the zero electron peak 

    hist_range = (range_left, range_right)

    counts, edges = np.histogram(data, bins=math.floor((hist_range[1]-hist_range[0])*bin_factor),range=hist_range)
    centers = 0.5 * (edges[1:] + edges[:-1])
    peaks, properties = find_peaks(counts, height=0, width=width, distance=bin_factor-buffer)

    return counts, peaks, centers, properties, hist_range


#---------------- (3) Fit nonlinearity ----------------------------
def fit_nonlinearity(centers, pedestal, fit_range, fit_bounds_low=-100, fit_bounds_high=100):
    # Convert to electrons: subtract the pedestal (mean of zero electron peak) from all charge values, 
    # Then divide by the gain (difference between zero and 1 electron peak)
    peak_charge_e = np.array([(centers[p]-pedestal)/gain for p in peaks])
    charge_minus_npeak = [(peak_charge_e[i] - i) for i in range(len(peaks))]
    parabola_coeff, pcov = curve_fit(parabola, peak_charge_e[:fit_range], charge_minus_npeak[:fit_range],
                           bounds=(fit_bounds_low, fit_bounds_high))
    return parabola_coeff, peak_charge_e, charge_minus_npeak


#---------------- PLOTTING FUNCTIONS ----------------------------

#---------------- Plot zero-one peaks subplot -----------------------
# Usage: for plotting zero-one electron peaks from each extension on same subplot. Input data is list of 2D pixel charge arrays from all 4 extensions.
def plot_zero_one_peaks_subplots(data_ext, zero_one_test_range=[8,15], 
                                 n=200, subplots=False, convert_to_electrons=False):
    
    fig, ax = plt.subplot(2, 2, figsize=(9,7), constrained_layout=True)
    for ext, data in enumerate(data_ext):
        data = np.array(data).flatten()

        pedestal, noise, gain, popt, zero_one_range = calculate_noise_gain(data, zero_one_test_range, n)
        coeff = tuple(popt)+(gain,)
        data_window = data[(data > zero_one_range[0]) & (charge < zero_one_range[1])]
        nbins=int(n*len(zero_one_range))
        xdata = np.linspace(zero_one_range[0], zero_one_range[1], nbins)

        ax[ext].hist(data_window,bins=nbins,range=tuple(zero_one_range))
        ax[ext].set_xlabel('Charge (ADU)')
        ax[ext].set_ylabel('N')
        ax[ext].set_title(f'EXT {ext}')
        
        ax1[ext].plot(xdata, double_gauss(xdata, *popt), 'r',
            label=r'$\sigma_0$ = %5.3f, $\mu_0$ = %5.3f, $\sigma_1$ = %5.3f, $\mu_1$ = %5.3f,'%coeff[0:4]
            +'\n'+'$N_0$ = %5.3f, $N_1$ = %5.3f, gain = %5.3f ADU/$e^{–}$'%coeff[4:])
        ax1[ext].legend(loc="upper right", fontsize=6)
        ax1[ext].set_ylim(0,max(counts1)+2e5)
        ax1[ext].set_xlim(tuple(zero_one_range))
        ax1[ext].legend()
    plt.show()

    if convert_to_electrons:
        fig, ax = plt.subplot(2, 2, figsize=(9,7), constrained_layout=True)
        data_window = convert_to_electrons(charge_window, coeff[1], gain)
        zero_one_range = convert_to_electrons(zero_one_range, coeff[1], gain) 
        counts, edges = np.histogram(data_window, bins=nbins, range=zero_one_range)
        xdata = np.linspace(zero_one_range[0], zero_one_range[1], nbins)
        popt, pcov = curve_fit(double_gauss, xdata, counts, bounds=([0.02, data_window[0], 0.005, data_window[0]+1.5,1e4,1e3], 
                                                                [0.5, data_window[1], 0.05, data_window[1],1e7,1e7]))
       
        ax[ext].hist(data_window,bins=nbins,range=tuple(zero_one_range))
        ax[ext].set_xlabel(r'Charge ($e^–$)')
        ax[ext].set_ylabel('N')

        ax[ext].set_title(f'EXT {ext}')
        
        ax[ext].plot(xdata, double_gauss(xdata, *popt), 'r',
            label=r'$\sigma_0$ = %5.3f $e^{–}$, $\mu_0$ = %5.3f $e^{–}$, $\sigma_1$ = %5.3f $e^{–}$, $\mu_1$ = %5.3f $e^{–}$'%coeff[0:4])
        ax[ext].legend(loc="upper right", fontsize=6)
        ax[ext].set_ylim(0,max(counts1)+2e5)
        ax[ext].set_xlim(tuple(zero_one_range))
        ax[ext].legend()
        plt.show()


#---------------- Plot zero-one peaks by extension ----------------------------
def plot_zero_one_peaks_individual(data, ext, zero_one_test_range=[8,15], n=200, convert_to_electrons=False):

    data = np.array(data).flatten()

    pedestal, noise, gain, popt, zero_one_range = calculate_noise_gain(data, zero_one_test_range, n)
    coeff = tuple(popt)+(gain,)
    data_window = data[(data > zero_one_range[0]) & (charge < zero_one_range[1])]
    nbins=int(n*len(zero_one_range))
    xdata = np.linspace(zero_one_range[0], zero_one_range[1], nbins)

    plt.hist(data_window, bins=nbins, range=tuple(zero_one_range))
    plt.xlabel('Charge (ADU)')
    plt.ylabel('N')
    plt.title(f'Combined Am-241 Pixel Charge Distribution, EXT {ext}')
    
    plt.plot(xdata, double_gauss(xdata, *popt), 'r',
            label=r'$\sigma_0$ = %5.3f, $\mu_0$ = %5.3f, $\sigma_1$ = %5.3f, $\mu_1$ = %5.3f,'%coeff[0:4]+'\n'
            +'$N_0$ = %5.3f, $N_1$ = %5.3f, gain = %5.3f ADU/$e^{–}$'%coeff[4:])
    plt.legend(loc="upper right", fontsize=6)
    plt.ylim(0,max(counts1)+2e5)
    plt.xlim(tuple(zero_one_range))
    plt.legend()
    plt.show()

    if convert_to_electrons:
        data_window = convert_to_electrons(charge_window, coeff[1], gain)
        zero_one_range = convert_to_electrons(zero_one_range, coeff[1], gain)
        counts, edges = np.histogram(data_window, bins=nbins, range=zero_one_range)
        xdata = np.linspace(zero_one_range[0], zero_one_range[1], nbins)
        popt, pcov = curve_fit(double_gauss, xdata, counts, bounds=([0.02, data_window[0], 0.005, data_window[0]+1.5,1e4,1e3], 
                                                                [0.5, data_window[1], 0.05, data_window[1],1e7,1e7]))
       
        plt.hist(data_window,bins=nbins,range=tuple(zero_one_range))
        plt.xlabel(r'Charge ($e^–$)')
        plt.ylabel('N')

        plt.title(f'Combined Am-241 Pixel Charge Distribution, EXT {ext}')
        
        plt.plot(xdata, double_gauss(xdata, *popt), 'r',
            label=r'$\sigma_0$ = %5.3f, $\mu_0$ = %5.3f, $\sigma_1$ = %5.3f, $\mu_1$ = %5.3f,'%coeff[0:4])
        plt.legend(loc="upper right", fontsize=6)
        plt.ylim(0,max(counts1)+2e5)
        plt.xlim(tuple(zero_one_range))
        plt.legend()
        plt.show()


#---------------- Plot all electron peaks ----------------------------
# Input is list of data from each of four extensions
def plot_all_peaks(data_ext, 
                   widths, buffers, convert_to_electrons, range_left, range_right, bin_factor,
                   fig_path, file, dpi=350,
                   plot_individual=True, plot_together=False, 
                   draw_lines=True, linecolor='r', linestyle='--',
                   subplots_figsize=(9,7), individual_figsize=(6,5),
                   xlim=hist_range, ylim=(0,4e5),
                   bins=math.floor(hist_range[1]-hist_range[0])*20,
                   suptitle='Peaks in Pixel Charge Distribution'):

    fig_name=fig_path+file[:-5]+'_peak_finder'

    if plot_individual:
        for ext, data in enumerate(data_ext):
            width=widths[ext]
            buffer=buffers[ext]
            # Get pedestal, noise, gain, etc. from individual extension data
            counts, peaks, centers, properties, hist_range = find_peaks(data, width=width, buffer=buffer, 
                                                                        convert_to_electrons=convert_to_electrons,
                                                                        range_left=range_left, 
                                                                        range_right=range_right, 
                                                                        bin_factor=bin_factor)
            
            fig, ax = plt.subplots(1, 1, figsize=individual_figsize, constrained_layout=True)
            fig.suptitle(suptitle+f': EXT {ext}')
            ax.hist(charge, bins=bins, range=hist_range)
            ax.set_xlabel(r'Charge ($e^-$)')
            ax.set_ylabel('N')
            ax.set_yscale('log')
            ax.set_xlim(xlim)
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
            width=widths[ext]
            buffer=buffers[ext]
            # Get pedestal, noise, gain, etc. from individual extension data
            counts, peaks, centers, properties, hist_range = find_peaks(data, width=width, buffer=buffer, 
                                                                        convert_to_electrons=convert_to_electrons,
                                                                        range_left=range_left, 
                                                                        range_right=range_right, 
                                                                        bin_factor=bin_factor)
            ax = axs[ext]
            ax.hist(charge, bins=bins, range=hist_range)
            ax.set_yscale('log')
            ax.set_xlim(xlim)
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
def plot_nonlinearity(data_ext,
                    fit_range_left, fit_range_right,
                    fig_path, file, 
                    widths=[0.9,0.8,1,1], 
                    buffers=[2,1,1,2], 
                    bin_factor=8,
                    subplots_figsize=(9,7), individual_figsize=(6,5), 
                    suptitle='Pixel Charge Nonlinearity Curve',
                    xlim=(-100, hist_range[1]-400), 
                    ylim=(min(charge_minus_npeak)-10, max(charge_minus_npeak)+15),
                    line_color='r', scatter_color='b', s=2, alpha=0.5,
                    plot_individual=False, plot_together=True, save_plots=False, dpi=350):

    fig_name=fig_path+file[:-5]+'_nonlinearity'

    if plot_individual:
        for ext, data in enumerate(data_ext):
            fig, ax = plt.subplots(1, 1, figsize=individual_figsize, constrained_layout=True)
            axs=axs.flatten()
            fig.suptitle(suptitle+f': EXT {ext}')
            
            pedestal, noise, gain, double_gauss_coeff, zero_one_range = calculate_noise_gain(data)

            width=widths[ext]
            buffer=buffers[ext]
            counts, peaks, centers, properties, hist_range = find_peaks(data, width=width, buffer=buffer, 
                                                                        convert_to_electrons=True,
                                                                        range_left=fit_range_left, 
                                                                        range_right=fit_range_right, 
                                                                        bin_factor=bin_factor)
            
            parabola_coeff, peak_charge_e, charge_minus_npeak = fit_nonlinearity(centers, pedestal, fit_range=(range))

            ax.plot(peak_charge_e, parabola(peak_charge_e, *parabola_coeff), color=line_color,
                        label=r'$%5.6f x^2 + %5.3f x + %5.3f$' %tuple(parabola_coeff))
            ax.scatter(peak_charge_e, charge_minus_npeak, c=scatter_color, s=s, alpha=alpha)
            ax.legend(loc="upper right", fontsize=8)
            ax.set_xlabel(r'Measured Pixel Charge ($e^-$)')
            ax.set_ylabel(r'Measured Pixel Charge - Peak n. ($e^-$) ')
            ax.set_ylim(ylim)
            ax.set_xlim(xlim)
            ax.grid()
            if save_plots:
                plt.savefig(fig_name+f'_EXT{ext}.jpeg',dpi=dpi)
            plt.show()
            
    if plot_together:
        fig, axs = plt.subplots(2, 2, figsize=subplots_figsize, constrained_layout=True)
        axs=axs.flatten()
        fig.suptitle(suptitle)
        for ext, data in enumerate(data_ext):
            ax = axs[ext]
            fig, ax = plt.subplots(1, 1, figsize=individual_figsize, constrained_layout=True)
            axs=axs.flatten()
            fig.suptitle(suptitle+f': EXT {ext}')
            
            pedestal, noise, gain, double_gauss_coeff, zero_one_range = calculate_noise_gain(data)

            width=widths[ext]
            buffer=buffers[ext]
            counts, peaks, centers, properties, hist_range = find_peaks(data, width=width, buffer=buffer, 
                                                                        convert_to_electrons=True,
                                                                        range_left=fit_range_left, 
                                                                        range_right=fit_range_right, 
                                                                        bin_factor=bin_factor)
            
            parabola_coeff, peak_charge_e, charge_minus_npeak = fit_nonlinearity(centers, pedestal, fit_range=(range))

            ax.plot(peak_charge_e, parabola(peak_charge_e, *parabola_coeff), color=line_color,
                        label=r'$%5.6f x^2 + %5.3f x + %5.3f$' %tuple(parabola_coeff))
            ax.scatter(peak_charge_e, charge_minus_npeak, c=scatter_color, s=s, alpha=alpha)
            ax.legend(loc="upper right", fontsize=8)
            ax.set_xlabel(r'Measured Pixel Charge ($e^-$)')
            ax.set_ylabel(r'Measured Pixel Charge - Peak n. ($e^-$) ')
            ax.set_ylim(ylim)
            ax.set_xlim(xlim)
            ax.grid()

            if save_plots:
                plt.savefig(fig_name+'.jpeg',dpi=dpi)
            plt.show()


#---------------- UTILITY FUNCTIONS ----------------------------

#---------------- Curves ----------------------------
def double_gauss(x, a, b, c, d, e, f):
    return (e/(np.sqrt(a*2*np.pi))) * np.exp(-(x-b)**2/(2*a)) + (f/(np.sqrt(c*2*np.pi))) * np.exp(-(x-d)**2/(2*c))

def parabola(x, a, b, c):
    return a*x**2 + b*x + c


#------------------ RUN ANALYSIS AND MAKE PLOTS ------------------------------

file = "avg_img_CV_250x3500x500_bin1x1_125_52_stitched.fits"
file_path = "/Users/abbychriss/Desktop/Privitera_335/data/test_chamber/Am241-Spectra-data/1x1-bin/combined-fits/"
fig_path = "/Users/abbychriss/Desktop/Privitera_335/plots/"
plot_zero_one_peaks=True
plot_all_peaks=True
plot_nonlinearity=True

widths=[0.9,0.8,1,1]
buffers=[2,1,1,2]
bin_factor=8
find_peaks_range_right=2000
nonlinearity_fit_range=[600,1500,1000,600]

# Get data from fits file
data_ext = get_fits(file)

hist_ranges = []
peak_charges_e = []
charge_minus_npeaks = []
for ext, data in enumerate(data_ext):
    width = widths[ext]
    buffer = buffers[ext]

    pedestal, noise, gain, popt, zero_one_range = calculate_noise_gain(data)
    hist_range = (pedestal-3*noise-1, find_peaks_range_right)
    hist_ranges.append(hist_range)

    counts, peaks, centers, properties, hist_range = find_peaks(data, width=width, buffer=buffer, 
               convert_to_electrons=True,
               range_left='left_of_zero', 
               range_right=find_peaks_range_right, 
               bin_factor=bin_factor)

    fit_range=nonlinearity_fit_range[ext]
    parabola_coeff, peak_charge_e, charge_minus_npeak = fit_nonlinearity(centers, 
                                                                         pedestal, 
                                                                         fit_range, 
                                                                         fit_bounds_low=-100, 
                                                                         fit_bounds_high=100):
    peak_charges_e.append(peak_charge_e)
    charge_minus_npeaks.append(charge_minus_npeak)


#fit a double gaussian to zero + 1 electron peak in each extension
if plot_zero_one_peaks:
    plot_zero_one_peaks_subplots(data_ext, zero_one_test_range=[8,15], 
                                    n=200, subplots=False, convert_to_electrons=False)
    
    for ext, data in enumerate(data):
        plot_zero_one_peaks_individual(data, ext, zero_one_test_range=[8,15], n=200, convert_to_electrons=False)

plot_all_peaks(data_ext, 
                   widths, buffers, convert_to_electrons, range_left=0, range_right=1000, bin_factor=bin_factor,
                   fig_path=fig_path, file=file, dpi=350,
                   plot_individual=True, plot_together=False, 
                   draw_lines=True, linecolor='r', linestyle='--',
                   subplots_figsize=(9,7), individual_figsize=(6,5),
                   xlim=(range_left, range_right), ylim=(0,4e5),
                   bins=math.floor(range_right-range_left)*20,
                   suptitle='Peaks in Pixel Charge Distribution')

plot_nonlinearity(data_ext,
                  fit_range_left=0, fit_range_right=2000,
                    fig_path=fig_path, file=file, 
                    widths=widths, 
                    buffers=buffers, 
                    bin_factor=bin_factor,
                    subplots_figsize=(9,7), individual_figsize=(6,5), 
                    suptitle='Pixel Charge Nonlinearity Curve',
                    xlim=(-100, hist_range[1]-400), 
                    ylim=(min(charge_minus_npeak)-10, max(charge_minus_npeak)+15),
                    line_color='r', scatter_color='b', s=2, alpha=0.5,
                    plot_individual=False, plot_together=True, save_plots=False, dpi=350)
