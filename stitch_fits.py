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
import os


img_type='avg'
file_path = '/Users/abbychriss/Desktop/Privitera_335/data/test_chamber/Am241-Spectra-data/1x1-bin/'
out_path = file_path+'combined-fits/'
os.makedirs(out_path, exist_ok=True)
image = f'{img_type}_img_CV_250x3500x500_bin1x1_125*'

files = sorted(glob.glob(file_path+'*/'+image))
nfiles = len(files)

# inspect first file to get dimensions
ext_headers = []
with fits.open(files[0], memmap=True) as f:
    ny, nx = f[1].data.shape
    nextensions = len(f) - 1

    primary_header = f[0].header.copy()

    for ext in [1,2,3,4]:
        ext_headers.append(f[ext].header.copy())
print(ext_headers)
outname = out_path+image[:-1]+'_stitched.fits'

primary_header['NROW'] = primary_header['NROW']*nfiles

# Create output file
primary_hdu = fits.PrimaryHDU(header=primary_header)

hdul = fits.HDUList([primary_hdu])

for ext in range(1, nextensions + 1):

    big_shape = (ny * nfiles, nx)

    # Preserve compression structure from original files
    hdr = ext_headers[ext-1]

    tile1 = hdr.get("ZTILE1", nx)
    tile2 = hdr.get("ZTILE2", 1)
    cmptype = hdr.get("ZCMPTYPE", "RICE_1")

    hdu = fits.CompImageHDU(
        data=np.zeros(big_shape, dtype=np.float32),
        header=hdr,
        compression_type=cmptype,
        tile_shape=(tile1, tile2)
    )

    # Preserve original extension name if present
    if "EXTNAME" in hdr:
        hdu.name = hdr["EXTNAME"]

    hdu.header["STITCHED"] = nfiles
    hdu.header["SRCFILE"] = files[0]

    hdul.append(hdu)

hdul.writeto(outname, overwrite=True)

# reopen with memmap
hdul = fits.open(outname, mode="update", memmap=True)

for i, f in enumerate(files):

    with fits.open(f, memmap=True) as infile:

        y0 = i * ny
        y1 = (i + 1) * ny

        for ext in range(1, nextensions + 1):

            data = infile[ext].data  # (630,20)

            hdul[ext].data[y0:y1, :] = data

    if i % 50 == 0:
        print(f"{i}/{nfiles}")

hdul.flush()
hdul.close()
if len(glob.glob(outname))>0:
    print(f'successfully saved stitched file to {outname}')
else:
    print('file not saved correctly')