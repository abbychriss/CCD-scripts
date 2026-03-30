import numpy as np

# Set up matplotlib
import matplotlib.pyplot as plt

from astropy.io import fits

image_file = "ccd_data/skp_img_conv_5x630x500_bin10x10_125_20251029_200106_21.fz"

for i in [1,2,3,4]:
	hdu_list = fits.open(image_file)
	hdu_list.info()

	image_data = hdu_list[i].data

	print(type(image_data))
	print(image_data.shape)

	hdu_list.close()

	#plt.imshow(image_data, cmap="gray")
	#plt.colorbar()

	print("Min:", np.min(image_data))
	print("Max:", np.max(image_data))
	print("Mean:", np.mean(image_data))
	print("Stdev:", np.std(image_data))

	print(type(image_data.flatten()))
	print(image_data.flatten().shape)
	
	if i in [1,4]:	
		r=[-5,0.5]
	if i in [2,3]:
		r=[-10,-5]
	histogram = plt.hist(image_data.flatten(), bins=70, range=(r[0],r[1]))
	plt.show()
