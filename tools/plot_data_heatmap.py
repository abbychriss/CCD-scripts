import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as colors
from astropy.io import fits
from astropy import units, stats

images = ["itp_img_CV_10x630x500_bin1x10_124_20260214_013431_4.fz"]
file_path = "/Users/abbychriss/Desktop/Privitera_335/"
fig_path = "/Users/abbychriss/Desktop/Privitera_335/"
save_plots=True

for image_file in images:
    channel_data=[]
    energies=[]
    for i in [1,2,3,4]:
        hdu_list = fits.open(file_path+'/'+image_file)
        image_data = hdu_list[i].data
        channel_data.append(image_data)

        energies.append(image_data.flatten())


    cmap = matplotlib.colormaps['magma']
    fig, ax = plt.subplots(4,1,figsize=(12, 4),constrained_layout=True)
    for i in range(4):
        im = ax[i].imshow(channel_data[i], cmap=cmap,norm=colors.AsinhNorm(linear_width=0.2,vmin=min(energies[i]), vmax=max(energies[i])))
        ax[i].set_title('Channel '+str(i+1))
        ax[i].set_ylabel('PixelY')
        ax[i].set_yticks([5])
    ax[3].set_xlabel('PixelX')
    fig.colorbar(im, ax=ax[:], label='Charge (ADU)')

    if save_plots:
        plt.savefig('/Users/abbychriss/Desktop/Privitera_335/'+image_file[:-3]+'.pdf')
        plt.savefig('/Users/abbychriss/Desktop/Privitera_335/'+image_file[:-3]+'.jpeg',dpi=300)
    
    else:
        plt.show()