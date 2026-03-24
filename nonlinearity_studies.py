import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import math

from .clustering import get_fits

#plt.rcParams['text.usetex'] = True

file = "avg_img_CV_250x3500x500_bin1x1_125_52_stitched.fits"
file_path = "/Users/abbychriss/Desktop/Privitera_335/data/test_chamber/Am241-Spectra-data/1x1-bin/combined-fits/"
fig_path = "/Users/abbychriss/Desktop/Privitera_335/"
plot_zero_one_peaks=True
plot_all_peaks=True
plot_vert_lines=False
plot_nonlinearity=True
save_plots=True
subplots=False

alphabet=['A','B','C','D']

ext_charge = get_fits(file)

#fit a double gaussian to zero + 1 electron peak in each extension
zero_one_peak_range=[[10,13],[11.5,14.5],[8,11.5],[8.5,11.5]]
n=200

def double_gauss(x, a, b, c, d, e, f):
    return (e/(np.sqrt(a*2*np.pi))) * np.exp(-(x-b)**2/(2*a)) + (f/(np.sqrt(c*2*np.pi))) * np.exp(-(x-d)**2/(2*c))

def parabola(x, a, b, c):
    return a*x**2 + b*x + c

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
    
    # Extract gain, noise and rest of double gaussian coefficients from curve fit
    gain=tuple(popt)[3]-tuple(popt)[1] # Gain is difference between mean of one and zero electron peaks
    noise=tuple(popt)[0] # Noise is standard deviation of zero electron peak 

    return gain, noise, popt, zero_one_range

# Usage: for plotting zero-one electron peaks from each extension on same subplot. Input data is list of 2D pixel charge arrays from all 4 extensions.
def plot_zero_one_peaks_subplots(data_ext, zero_one_test_range=[8,15], n=200, subplots=False, convert_to_electrons=False):
    
    fig, ax = plt.subplot(2, 2, figsize=(9,7), constrained_layout=True, convert_to_electrons=False)
    for ext, data in enumerate(data_ext):
        data = np.array(data).flatten()

        gain, noise, popt, zero_one_range = calculate_noise_gain(data, zero_one_test_range, n)
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
        data_window = [(charge - coeff[1])/gain for charge in charge_window] # Subtract pedestal (mean ADU of zero electron peak) and divide by gain
        zero_one_range = [(charge - coeff[1])/gain for charge in zero_one_range] 
        counts, edges = np.histogram(data_window, bins=nbins, range=zero_one_range)
        xdata = np.linspace(zero_one_range[0], zero_one_range[1], nbins)
        popt, pcov = curve_fit(double_gauss, xdata, counts, bounds=([0.02, data_window[0], 0.005, data_window[0]+1.5,1e4,1e3], 
                                                                [0.5, data_window[1], 0.05, data_window[1],1e7,1e7]))
       
        ax[ext].hist(data_window,bins=nbins,range=tuple(zero_one_range))
        ax[ext].set_xlabel(r'Charge ($e^–$)')
        ax[ext].set_ylabel('N')

        ax[ext].set_title(f'EXT {ext}')
        
        ax[ext].plot(xdata, double_gauss(xdata, *popt), 'r',
            label=r'$\sigma_0$ = %5.3f, $\mu_0$ = %5.3f, $\sigma_1$ = %5.3f, $\mu_1$ = %5.3f,'%coeff[0:4])
        ax[ext].legend(loc="upper right", fontsize=6)
        ax[ext].set_ylim(0,max(counts1)+2e5)
        ax[ext].set_xlim(tuple(zero_one_range))
        ax[ext].legend()
    plt.show()

def plot_zero_one_peaks_ext(data, ext, zero_one_test_range=[8,15], n=200, subplots=False, convert_to_electrons=False):

    data = np.array(data).flatten()

    gain, noise, popt, zero_one_range = calculate_noise_gain(data, zero_one_test_range, n)
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
        data_window = [(charge - coeff[1])/gain for charge in charge_window] # Subtract pedestal (mean ADU of zero electron peak) and divide by gain
        zero_one_range = [(charge - coeff[1])/gain for charge in zero_one_range] 
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


#open two different subplots outside of loop to be filled
if plot_zero_one_peaks and subplots:
    fig1, ax1 = plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    ax1=ax1.flatten()
    fig1.suptitle('Combined Am-241 Pixel Charge Distribution')

if plot_all_peaks and subplots:
    fig2, ax2 = plt.subplots(2,2,figsize=(9,7),constrained_layout=True)
    fig2.suptitle('Peaks in Pixel Charge Distribution')
    ax2=ax2.flatten()

if plot_nonlinearity and subplots:
    fig3, ax3 = plt.subplots(2,2,figsize=(9,7),constrained_layout=True)
    fig3.suptitle('Nonlinearity of Pixel Charge Fit')
    ax3=ax3.flatten()

for ext, charge in enumerate(ext_charge):

    #----------------- PLOT 1: CALCULATE NOISE AND GAIN -------------------------------------------
    nbins=int(n*len(zero_one_peak_range[ext]))
    charge = np.array(charge).flatten()
    charge_window = charge[(charge > zero_one_peak_range[ext][0]) & (charge < zero_one_peak_range[ext][1])]
    counts1, edges1 = np.histogram(charge_window,bins=nbins,range=(zero_one_peak_range[ext][0],zero_one_peak_range[ext][1]))
    xdata = np.linspace(zero_one_peak_range[ext][0],zero_one_peak_range[ext][1],nbins)
    popt, pcov = curve_fit(double_gauss, xdata, counts1, bounds=([0.02, zero_one_peak_range[ext][0], 0.005, zero_one_peak_range[ext][0]+2,1e4,1e3], 
                                                                  [0.5, zero_one_peak_range[ext][1], 0.05, zero_one_peak_range[ext][1],1e7,1e7]))
    gain=tuple(popt)[3]-tuple(popt)[1]
    print(f'Gain for CCD {alphabet[ext]} = {gain}')
    coeff = tuple(popt)+(gain,)

    if plot_zero_one_peaks:
        #Fill first plot
        if subplots:
            ax1[ext].hist(charge_window,bins=nbins,range=tuple(zero_one_peak_range[ext]))
            ax1[ext].set_xlabel('Charge (ADU)')
            ax1[ext].set_ylabel('N')
            ax1[ext].set_title(f'CCD {alphabet[ext]}')
            
            
            ax1[ext].plot(xdata, double_gauss(xdata, *popt), 'r',
                    label=r'$\sigma_0$ = %5.3f, $\mu_0$ = %5.3f, $\sigma_1$ = %5.3f, $\mu_1$ = %5.3f,'%coeff[0:4]
                    +'\n'+'$N_0$ = %5.3f, $N_1$ = %5.3f, gain = %5.3f ADU/$e^{–}$'%coeff[4:])
            ax1[ext].legend(loc="upper right", fontsize=6)
            ax1[ext].set_ylim(0,max(counts1)+2e5)
            ax1[ext].set_xlim(tuple(zero_one_peak_range[ext]))
            ax1[ext].legend()
        else:
            plt.hist(charge_window,bins=nbins,range=tuple(zero_one_peak_range[ext]))
            plt.xlabel('Charge (ADU)')
            plt.ylabel('N')
            plt.title(f'Combined Am-241 Pixel Charge Distribution, CCD {alphabet[ext]}')
            
            plt.plot(xdata, double_gauss(xdata, *popt), 'r',
                    label=r'$\sigma_0$ = %5.3f, $\mu_0$ = %5.3f, $\sigma_1$ = %5.3f, $\mu_1$ = %5.3f,'%coeff[0:4]+'\n'
                    +'$N_0$ = %5.3f, $N_1$ = %5.3f, gain = %5.3f ADU/$e^{–}$'%coeff[4:])
            plt.legend(loc="upper right", fontsize=6)
            plt.ylim(0,max(counts1)+2e5)
            plt.xlim(tuple(zero_one_peak_range[ext]))
            plt.legend()
            plt.show()

    #-----------------PLOT 2: PEAKFINDER-------------------------------------------

    #use peakfinder to plot all charge in each extension with labels on each peak
    hist_range = (coeff[1]-3*coeff[0]-1, 2500) #left end of range should be on the left side of the zero electron peak 
                                                    #--> was fitting negative (cross talk) peaks before!
    print(hist_range)
    widths=[0.9,0.8,1,1]
    distances=[-2,-1,-1,-2]
    bin_factor=8
    counts2, edges2 = np.histogram(charge,bins=math.floor((hist_range[1]-hist_range[0])*bin_factor),range=hist_range)
    centers = 0.5 * (edges2[1:] + edges2[:-1])
    peaks, properties = find_peaks(counts2, height=0,width=widths[ext],distance=bin_factor+distances[ext])

    if plot_all_peaks:
        if subplots:
            ax2[ext].hist(charge, bins=math.floor(hist_range[1]-hist_range[0])*20, range=hist_range)
            ax2[ext].set_yscale('log')
            ax2[ext].set_xlim((hist_range[0],20))
            ax2[ext].set_title(f'CCD {alphabet[ext]}')
        else:
            plt.hist(charge, bins=math.floor(hist_range[1]-hist_range[0])*20, range=hist_range)
            plt.xlabel(r'Charge (ADU)')
            plt.ylabel('N')
            plt.yscale('log')
            plt.xlim((hist_range[0],20))
            plt.title(f'Peaks in Pixel Charge Distribution, CCD {alphabet[ext]}')

        # draw vertical lines and labels at each peak
        if plot_vert_lines:
            for i,p in enumerate(peaks):
                peak_x = centers[p]
                peak_y = counts2[p]

                if subplots:
                    ax2[ext].axvline(peak_x, linestyle='--', color='r')
                    ax2[ext].text(
                        peak_x,
                        peak_y,
                        f"{i}",
                        verticalalignment='bottom',
                        horizontalalignment='center',
                        color='red',
                        fontsize=10
                    )
                    ax2[ext].set_xlabel(r'Charge ($e^-$)')
                    ax2[ext].set_ylabel('N')
                    ax2[ext].set_ylim(0,1000)
                else:
                    plt.axvline(peak_x, linestyle='--', color='r')
                    plt.text(
                        peak_x,
                        peak_y,
                        f"{i}",
                        verticalalignment='bottom',
                        horizontalalignment='center',
                        color='red',
                        fontsize=10
                    )
        if not subplots:
            plt.show()
    

    #-----------------PLOT 3: NONLINEARITY FIT-------------------------------------------
    #convert to electrons: subtract the pedestal (mean of zero electron peak) from all charge values, 
    #then divide by the gain (difference between zero and 1 electron peak)
    peak_charge_e = np.array([(centers[p]-coeff[1])/gain for p in peaks])
    charge_minus_npeak = [(peak_charge_e[i] - i) for i in range(len(peaks))]
    fit_range=[600,1500,1000,600]
    popt, pcov = curve_fit(parabola, peak_charge_e[:fit_range[ext]], charge_minus_npeak[:fit_range[ext]],
                           bounds=([-100,-100,-100], [100,100,100]))
    if plot_nonlinearity:
        if subplots:
            ax3[ext].plot(peak_charge_e, parabola(peak_charge_e, *popt), color='r',
                        label=r'$%5.6f x^2 + %5.3f x + %5.3f$' %tuple(popt))
            ax3[ext].scatter(peak_charge_e,charge_minus_npeak, c='blue',s=2,alpha=0.5)
            ax3[ext].legend(loc="upper right", fontsize=8)
            ax3[ext].set_xlabel(r'Measured Pixel Charge ($e^-$)')
            ax3[ext].set_ylabel(r'Measured Pixel Charge ($e^-$) - Peak n.')
            ax3[ext].set_ylim(min(charge_minus_npeak)-10,max(charge_minus_npeak)+15)
            ax3[ext].set_xlim(-100,hist_range[1]-400)
            ax3[ext].grid()
            ax3[ext].set_title(f'CCD {alphabet[ext]}')
        
        else:
            plt.plot(peak_charge_e, parabola(peak_charge_e, *popt), color='r',
                        label=r'$%5.6f x^2 + %5.3f x + %5.3f$' %tuple(popt))
            plt.scatter(peak_charge_e,charge_minus_npeak, c='blue',s=2,alpha=0.5)
            plt.legend(loc="upper right", fontsize=8)
            plt.xlabel(r'Measured Pixel Charge ($e^-$)')
            plt.ylabel(r'Measured Pixel Charge ($e^-$) - Peak n.')
            plt.title(f'Nonlinearity of Pixel Charge Fit, CCD {alphabet[ext]}')
            plt.ylim(min(charge_minus_npeak)-5,max(charge_minus_npeak)+15)
            plt.xlim(-100,hist_range[1]-400)
            plt.grid()
            plt.show()

if save_plots:
    if subplots:
        if plot_zero_one_peaks:
            fig1.savefig(fig_path+file[:-5]+'_noise_fit.jpeg',dpi=350)
        if plot_nonlinearity:
            fig3.savefig(fig_path+file[:-5]+'_nonlinearity.jpeg',dpi=350)

if subplots:
    plt.show()