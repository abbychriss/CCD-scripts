import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

#plt.rcParams['text.usetex'] = True

file = "avg_img_CV_250x3500x500_bin1x1_125_51_stitched.fits"
file_path = "/Users/abbychriss/Desktop/Privitera_335/data/test_chamber/Am241-Spectra-data/1x1-bin/combined-fits/"
fig_path = "/Users/abbychriss/Desktop/Privitera_335/"
plot_zero_one_peaks=True
plot_all_peaks=False
plot_nonlinearity=True
save_plots=True
subplots=False

hdu_list = fits.open(file_path+file)
#unpack each extension separately
ext_charge=[hdu_list[i].data.flatten() for i in range(1,5)]
alphabet=['A','B','C','D']

#fit a double gaussian to zero + 1 electron peak in each extension
zero_one_peak_range=[[10,13],[11.5,14.5],[8,11.5],[8.5,11.5]]
n=200

def double_gauss(x, a, b, c, d, e, f):
    return (e/(np.sqrt(a*2*np.pi))) * np.exp(-(x-b)**2/(2*a)) + (f/(np.sqrt(c*2*np.pi))) * np.exp(-(x-d)**2/(2*c))

def parabola(x, a, b, c):
    return a*x**2 + b*x + c

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

for ext,charge in enumerate(ext_charge):

    #-----------------PLOT 1: CALCULATE NOISE AND GAIN-------------------------------------------
    nbins=int(n*len(zero_one_peak_range[ext]))
    charge = np.array(charge).flatten()
    charge_window = charge[(charge > zero_one_peak_range[ext][0]) & (charge < zero_one_peak_range[ext][1])]
    counts1, edges1 = np.histogram(charge_window,bins=nbins,range=(zero_one_peak_range[ext][0],zero_one_peak_range[ext][1]))
    xdata = np.linspace(zero_one_peak_range[ext][0],zero_one_peak_range[ext][1],nbins)
    popt, pcov = curve_fit(double_gauss, xdata, counts1, bounds=([0.02, zero_one_peak_range[ext][0], 0.005, zero_one_peak_range[ext][0]+2,1e4,1e3], 
                                                                  [0.5, zero_one_peak_range[ext][1], 0.05, zero_one_peak_range[ext][1],1e7,1e7]))
    gain=tuple(popt)[3]-tuple(popt)[1]

    if plot_zero_one_peaks:
        #Fill first plot
        if subplots:
            ax1[ext].hist(charge_window,bins=nbins,range=tuple(zero_one_peak_range[ext]))
            ax1[ext].set_xlabel('Charge (ADU)')
            ax1[ext].set_ylabel('N')
            ax1[ext].set_title(f'CCD {alphabet[ext]}')
            
            coeff = tuple(popt)+(gain,)
            ax1[ext].plot(xdata, double_gauss(xdata, *popt), 'r',
                    label=r'$\sigma_0$ = %5.3f, $\mu_0$ = %5.3f, $\sigma_1$ =%5.3f, $\mu_1$ =%5.3f,'%coeff[0:4]
                    +'\n'+'$N_0$ =%5.3f, $N_1$=%5.3f, gain=%5.3f ADU/$e^{–}$'%coeff[4:])
            ax1[ext].legend(loc="upper right", fontsize=6)
            ax1[ext].set_ylim(0,max(counts1)+2e5)
            ax1[ext].set_xlim(tuple(zero_one_peak_range[ext]))
            ax1[ext].legend()
        else:
            plt.hist(charge_window,bins=nbins,range=tuple(zero_one_peak_range[ext]))
            plt.xlabel('Charge (ADU)')
            plt.ylabel('N')
            plt.title(f'Combined Am-241 Pixel Charge Distribution, CCD {alphabet[ext]}')
            
            coeff = tuple(popt)+(gain,)
            plt.plot(xdata, double_gauss(xdata, *popt), 'r',
                    label=r'$\sigma_0$ = %5.3f, $\mu_0$ = %5.3f, $\sigma_1$ =%5.3f, $\mu_1$ =%5.3f,'%coeff[0:4]+'\n'
                    +'$N_0$ =%5.3f, $N_1$=%5.3f, gain=%5.3f ADU/$e^{–}$'%coeff[4:])
            plt.legend(loc="upper right", fontsize='medium')
            plt.ylim(0,max(counts1)+2e5)
            plt.xlim(tuple(zero_one_peak_range[ext]))
            plt.legend()
            plt.show()

    #-----------------PLOT 2: PEAKFINDER-------------------------------------------

    #use peakfinder to plot all charge in each extension with labels on each peak
    hist_range = (0, 2500)
    widths=[0.9,0.8,1,1]
    distances=[-2,-1,-1,-2]
    bin_factor=8
    counts2, edges2 = np.histogram(charge,bins=(hist_range[1]-hist_range[0])*bin_factor,range=hist_range)
    centers = 0.5 * (edges2[1:] + edges2[:-1])
    peaks, properties = find_peaks(counts2, height=0,width=widths[ext],distance=bin_factor+distances[ext])

    #fill second plot
    if plot_all_peaks:
        if subplots:
            ax2[ext].hist(charge, bins=(hist_range[1]-hist_range[0])*20, range=hist_range)
            ax2[ext].set_title(f'CCD {alphabet[ext]}')
        else:
            plt.hist(charge, bins=(hist_range[1]-hist_range[0])*20, range=hist_range)
            plt.xlabel(r'Charge ($e^-$)')
            plt.ylabel('N')
            plt.ylim(0,50)
            plt.xlim(hist_range)
            plt.title(f'Peaks in Pixel Charge Distribution, CCD {alphabet[ext]}')

    # draw vertical lines and labels at each peak
    for i,p in enumerate(peaks):
        peak_x = centers[p]
        peak_y = counts2[p]

        if plot_all_peaks:
            if subplots:
                ax2[ext].axvline(peak_x, linestyle='--', color='r')
                ax2[ext].text(
                    peak_x,
                    peak_y,
                    f"{i}",
                    verticalalignment='bottom',
                    horizontalalignment='center',
                    color='red',
                    fontsize=6
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
                    fontsize=6
                )
    if not subplots:
        plt.show()
    

    #-----------------PLOT 3: NONLINEARITY FIT-------------------------------------------
    print(f'Gain for CCD {alphabet[ext]} = {gain}')
    peak_charge_e = np.array([centers[p]/gain for p in peaks])
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
            ax3[ext].set_xlabel('Measured Pixel Charge (e-)')
            ax3[ext].set_ylabel('Measured Pixel Charge - Peak N. (e-)')
            ax3[ext].set_ylim(min(charge_minus_npeak)-10,max(charge_minus_npeak)+15)
            ax3[ext].set_xlim(-100,hist_range[1]-400)
            ax3[ext].grid()
            ax3[ext].set_title(f'CCD {alphabet[ext]}')
        
        else:
            plt.plot(peak_charge_e, parabola(peak_charge_e, *popt), color='r',
                        label=r'$%5.6f x^2 + %5.3f x + %5.3f$' %tuple(popt))
            plt.scatter(peak_charge_e,charge_minus_npeak, c='blue',s=2,alpha=0.5)
            plt.legend(loc="upper right", fontsize=8)
            plt.xlabel('Measured Pixel Charge (e-)')
            plt.ylabel('Measured Pixel Charge - Peak N. (e-)')
            plt.title(f'Nonlinearity of Pixel Charge Fit, CCD {alphabet[ext]}')
            plt.ylim(min(charge_minus_npeak)-5,max(charge_minus_npeak)+15)
            plt.xlim(-100,hist_range[1]-400)
            plt.grid()
            plt.show()

if save_plots:
    if subplots:
        #fig3.savefig(fig_path+file[:-5]+'noise_fit.pdf')
        if plot_zero_one_peaks:
            fig1.savefig(fig_path+file[:-5]+'_noise_fit.jpeg',dpi=350)
        if plot_nonlinearity:
            fig3.savefig(fig_path+file[:-5]+'_nonlinearity.jpeg',dpi=350)

if subplots:
    plt.show()