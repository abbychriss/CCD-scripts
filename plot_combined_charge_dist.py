import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit

plt.rcParams['text.usetex'] = True

file = "combined_1000_avg_img_CV_20x630x500_bin10x10_125.fits"
file_path = "/Users/abbychriss/Desktop/Privitera_335/data/Am241-Spectra-data/combined-fits/"
fig_path = "/Users/abbychriss/Desktop/Privitera_335/"
save_plots=True

hdu_list = fits.open(file_path+file)
charge_values=[]
charge_values+=(list(hdu_list[i].data.flatten()) for i in range(1,4001))
charge_values=np.array(charge_values).flatten()

#plot all charge
range=(0,30)
n=500
fig, ax = plt.subplots()
ax.hist(charge_values,bins=n*len(range),range=range)
ax.set_xlabel('Charge (ADU)')
ax.set_ylabel('N')
ax.set_title('Combined Am-241 Pixel Charge Distribution')

plt.show()

#fit a double gaussian to zero + 1 electron peak
min=6
max=9
range=(min,max)
n=200
nbins=int(n*len(range))
counts, edges = np.histogram(charge_values,bins=nbins,range=range)
xdata = np.linspace(min,max,nbins)
fig, ax = plt.subplots(figsize=(5,6),constrained_layout=True)
ax.hist(charge_values,bins=nbins,range=range)
ax.set_xlabel('Charge (ADU)')
ax.set_ylabel('N')
ax.set_title('Combined Am-241 Pixel Charge Distribution')

def func(x, a, b, c, d,e,f):
    return (e/(np.sqrt(a*2*np.pi))) * np.exp(-(x-b)**2/(2*a)) + (f/(np.sqrt(c*2*np.pi))) * np.exp(-(x-d)**2/(2*c))
popt, pcov = curve_fit(func, xdata, counts, bounds=([0.02, 7, 0.02, 8,1e4,1e4], [0.5, 8, 0.5, 9,1e7,1e7]))

gain=tuple(popt)[3]-tuple(popt)[1]
coeff = tuple(popt)+(gain,)
plt.plot(xdata, func(xdata, *popt), 'r',
         label=r'$\sigma_0$ = %5.3f, $\mu_0$ = %5.3f, $\sigma_1$ =%5.3f, $\mu_1$ =%5.3f, \\ $N_0$ =%5.3f, $N_1$=%5.3f, gain=%5.3f ADU/$e^{–}$' %coeff)
ax.set_ylim(0,4e5)
ax.legend()
if save_plots:
    plt.savefig(fig_path+file[:-5]+'.pdf')
    plt.savefig(fig_path+file[:-5]+'.jpeg',dpi=350)
plt.show()