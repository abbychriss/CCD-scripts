import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

#plt.rcParams['text.usetex'] = True

file = "avg_img_CV_250x3500x500_bin1x1_125_stitched_26.fits"
file_path = "/Users/abbychriss/Desktop/Privitera_335/data/test_chamber/Am241-Spectra-data/1x1-bin/combined-fits/"
fig_path = "/Users/abbychriss/Desktop/Privitera_335/"
save_plots=False

hdu_list = fits.open(file_path+file)
#for stitched fits files
ext_charge=[hdu_list[i].data.flatten() for i in range(1,5)]


#for combined fits files
"""
charge_values=[]
charge_values+=(list(hdu_list[i].data.flatten()) for i in range(1,4001))
charge_values=np.array(charge_values).flatten()
"""
"""
#plot first few electron peaks in each extension
range=(7,30)
n=500
fig, ax = plt.subplots(2,2,figsize=(8,6),constrained_layout=True)
ax=ax.flatten()
for ext,charge in enumerate(ext_charge):
    ax[ext].hist(charge,bins=n*len(range),range=range)
    ax[ext].set_xlabel('Charge (ADU)')
    ax[ext].set_ylabel('N')
    ax[ext].set_title(f'CCD {ext+1}')
plt.suptitle('Combined Am-241 Pixel Charge')
plt.show()"""

#fit a double gaussian to zero + 1 electron peak in each extension
ranges=[[10,13],[11.5,14.5],[8,11.5],[8.5,12.5]]
n=200

def func(x, a, b, c, d,e,f):
    return (e/(np.sqrt(a*2*np.pi))) * np.exp(-(x-b)**2/(2*a)) + (f/(np.sqrt(c*2*np.pi))) * np.exp(-(x-d)**2/(2*c))

#open two different subplots outside of loop to be filled
fig1, ax1 = plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
ax1=ax1.flatten()

fig2, ax2 = plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
ax2=ax2.flatten()
for ext,charge in enumerate(ext_charge):
    nbins=int(n*len(ranges[ext]))
    charge = np.array(charge).flatten()
    charge_window = charge[(charge > ranges[ext][0]) & (charge < ranges[ext][1])]
    counts1, edges1 = np.histogram(charge_window,bins=nbins,range=(ranges[ext][0],ranges[ext][1]))
    xdata = np.linspace(ranges[ext][0],ranges[ext][1],nbins)
    popt, pcov = curve_fit(func, xdata, counts1, bounds=([0.02, ranges[ext][0], 0.005, ranges[ext][0]+2,1e4,1e3], [0.5, ranges[ext][1], 0.05, ranges[ext][1],1e7,1e7]))
    
    #Fill first plot
    ax1[ext].hist(charge_window,bins=nbins,range=tuple(ranges[ext]))
    ax1[ext].set_xlabel('Charge (ADU)')
    ax1[ext].set_ylabel('N')
    ax1[ext].set_title(f'CCD {ext+1}')

    gain=tuple(popt)[3]-tuple(popt)[1]
    coeff = tuple(popt)+(gain,)
    ax1[ext].plot(xdata, func(xdata, *popt), 'r',
            label=r'$\sigma_0$ = %5.3f, $\mu_0$ = %5.3f, $\sigma_1$ =%5.3f, $\mu_1$ =%5.3f,\n $N_0$ =%5.3f, $N_1$=%5.3f, gain=%5.3f ADU/$e^{–}$' %coeff,
            loc="upper right", fontsize=8)
    ax1[ext].set_ylim(0,4e5)
    ax1[ext].legend()

    #use peakfinder to plot all charge in each extension with labels on each peak
    hist_range = (0, 800)
    
    counts2, edges2 = np.histogram(charge,bins=hist_range[1]*200)
    centers = 0.5 * (edges2[1:] + edges2[:-1])
    peaks, properties = find_peaks(counts2, height=0)

    #fill second plot
    ax2[ext].hist(charge, bins=100*nbins, range=hist_range)
    ax2[ext].set_title(f'CCD {ext+1}')
    # draw vertical lines and labels at each peak
    for p in peaks:
        peak_x = centers[p]
        peak_y = counts2[p]

        ax2[ext].axvline(peak_x, linestyle='--', color='r')
        ax2[ext].text(
            peak_x,
            peak_y,
            f"{peak_x:.2f}",
            rotation=90,
            verticalalignment='bottom',
            horizontalalignment='center',
            color='red'
        )
        ax2[ext].set_xlabel('Charge (ADU)')
        ax2[ext].set_ylabel('N')
fig1.suptitle('Combined Am-241 Pixel Charge Distribution')
if save_plots:
    fig1.savefig(fig_path+file[:-5]+'.pdf')
    fig1.savefig(fig_path+file[:-5]+'.jpeg',dpi=350)

fig2.suptitle('Peaks in Pixel Charge Distribution')
plt.show()