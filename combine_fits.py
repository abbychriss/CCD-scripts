from astropy.table import Table
from astropy import units as u
from astropy.io import fits
from astropy import units, stats

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as colors

import numpy as np

import glob

img_type='avg'
file_path = f'/Users/abbychriss/Desktop/Privitera_335/data/Am241-Spectra-data/{img_type}-img/'
image = f'{img_type}_img_CV_20x630x500_bin10x10_125*.fz'

image_names = glob.glob(file_path+image)
image_names.sort()

combined_data=[]

headers=[]
k=0
for image_file in image_names:
    hdu_list = fits.open(image_file)
    channel_data=[]
    for i in [1,2,3,4]:
        primary_hdr = hdu_list[0].header
        if k==0 and i==1:
            headers.append(primary_hdr)
        hdr = hdu_list[i].header
        headers.append(hdr)
        image_data = hdu_list[i].data
        channel_data.append(image_data)
    combined_data.append(channel_data)
    k+=1 

all_hdus = []
for im in range(len(combined_data)):
    for n in [0,1,2,3]:
        #print(np.shape(np.array(combined_data[im][n])))
        im_hdu_extn = fits.ImageHDU(name=image_names[im].split('/')[1][:-3]+'_ext'+str(n), 
                                  data=np.array(combined_data[im][n]))
        #print(im_col_extn)
        all_hdus.append(im_hdu_extn)
print('length of headers list: ',len(headers), f'expecting {1+4*len(image_names)}')
primary_hdr=headers[0]
primary_hdr['COMMENT'] = f'Combined fits file from {k} images'
primary_hdu = fits.PrimaryHDU(header=primary_hdr)

hdul = fits.HDUList([primary_hdu]+all_hdus)

output_name=f'/Users/abbychriss/Desktop/Privitera_335/data/Am241-Spectra-data/combined-fits/combined_{k}_'+image[:-4]+'.fits'
hdul.writeto(output_name,overwrite=True)
print(f'saving files to {output_name}')
