import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units, stats

import argparse, sys, os
images = [sys.argv[1]]

#file_path = "/Users/abbychriss/Desktop/Privitera_335/data/test_chamber/02-13-2026/1x10-image/"
#fig_path = "/Users/abbychriss/Desktop/Privitera_335/"
save_pdf=False
save_jpeg=False

col_start=int(sys.argv[2].split(',')[0])
col_end=int(sys.argv[2].split(',')[1])
cols_analyze=range(col_start,col_end)
print('Analyzing columns',[col_start,col_end])
nstd = 1.5

for image in images:
	#image_file = file_path+image+".fz"
	print('Analyzing image',image)
	fig_path = image.split('/')[:-1]
	print('Figures will be saved to directory: ',fig_path)
	median_charge_col_safe=[]
	safe_cols=[]
	median_charge_col_hot=[]
	hot_cols=[]

	print("Image "+image[-17:]+":")
	for i in [1,2,3,4]:
		hdu_list = fits.open(image) #image_file

		image_data = hdu_list[i].data
		#calculate median charge in each column before pedestal subtraction
		median_charge_col_i = []
		for j in cols_analyze[:-1]:
			median_charge = np.median([image_data[k][j] for k in range(len(image_data))])
			median_charge_col_i.append(median_charge)

		std = np.sqrt(stats.biweight.biweight_midvariance(image_data.flatten()))
		avg = stats.biweight.biweight_location(image_data.flatten())
		median_charge_col_safe_i=[]
		safe_cols_i=[]
		median_charge_col_hot_i=[]
		hot_cols_i=[]
		for j in range(len(median_charge_col_i)):
			if np.abs(median_charge_col_i[j] - avg) >= nstd*std:
				median_charge_col_hot_i.append(median_charge_col_i[j])
				hot_cols_i.append(j)
			else:
				median_charge_col_safe_i.append(median_charge_col_i[j])
				safe_cols_i.append(j)

		median_charge_col_safe.append(median_charge_col_safe_i)
		safe_cols.append(safe_cols_i)
		median_charge_col_hot.append(median_charge_col_hot_i)
		hot_cols.append(hot_cols_i)

		print("In CCD",i,"columns", hot_cols_i, "are hot (charge >="+str(nstd)+" SDs from the mean), and ", round(len(hot_cols_i)/len(median_charge_col_i)*100,1),"percent of columns are hot")
		hdu_list.close()
		"""
		plt.scatter(safe_cols_i, median_charge_col_safe_i, s=1)
		plt.scatter(hot_cols_i, median_charge_col_hot_i, s=2, color='red')
		plt.xlabel("Column")
		plt.ylabel("Median Charge (ADU)")
		plt.title("CCD "+str(i))
		plt.show()

		plt.scatter(safe_cols_i, median_charge_col_safe_i, s=1)
		plt.xlabel("Column")
		plt.ylabel("Median Charge (ADU)")
		plt.title("CCD "+str(i)+" masked")
		plt.show()
		"""
	#plot all columns
	fig, axes = plt.subplots(2, 2, figsize=(8,6))
	plt.suptitle("All Columns (No Masking), Image "+image[-17:])

	# Plot on each subplot
	axes[0, 0].scatter(safe_cols[0], median_charge_col_safe[0], s=1)
	axes[0, 0].scatter(hot_cols[0], median_charge_col_hot[0], s=1, color='red')
	#axes[0, 0].set_xlabel("Column Number")
	axes[0, 0].set_ylabel("Median Charge (ADU)")
	axes[0, 0].set_title("CCD 1")

	axes[0, 1].scatter(safe_cols[1], median_charge_col_safe[1], s=1)
	axes[0, 1].scatter(hot_cols[1], median_charge_col_hot[1], s=1, color='red')
	#axes[0, 1].set_xlabel("Column Number")
	#axes[0, 1].set_ylabel("Median Charge (ADU)")
	axes[0, 1].set_title("CCD 2")

	axes[1, 0].scatter(safe_cols[2], median_charge_col_safe[2], s=1)
	axes[1, 0].scatter(hot_cols[2], median_charge_col_hot[2], s=1, color='red')
	axes[1, 0].set_xlabel("Column Number")
	axes[1, 0].set_ylabel("Median Charge (ADU)")
	axes[1, 0].set_title("CCD 3")

	axes[1, 1].scatter(safe_cols[3], median_charge_col_safe[3], s=1)
	axes[1, 1].scatter(hot_cols[3], median_charge_col_hot[3], s=1, color='red')
	axes[1, 1].set_xlabel("Column Number")
	#axes[1, 1].set_ylabel("Median Charge (ADU)")
	axes[1, 1].set_title("CCD 4")

	plt.tight_layout()
	if (save_pdf==True) and (save_jpeg==True):
		plt.savefig(fig_path+'adu_column_no_mask_'+image+'.pdf')
		plt.savefig(fig_path+'adu_column_no_mask_'+image+'.jpeg',dpi=300)
	if (save_pdf==False) and (save_jpeg==True):
		plt.savefig(fig_path+'adu_column_no_mask_'+image+'.jpeg',dpi=300)
	if (save_pdf==True) and (save_jpeg==False):
		plt.savefig(fig_path+'adu_column_no_mask_'+image+'.pdf')
	elif (save_pdf==False) and (save_jpeg==False):
		plt.show()

	#plot columns with mask
	fig, axes = plt.subplots(2, 2, figsize=(8, 6))
	plt.suptitle("Safe Columns (Masking Applied: nstd="+str(nstd)+")"+", Image "+image[-2:])

	axes[0, 0].scatter(safe_cols[0], median_charge_col_safe[0], s=1)
	#axes[0, 0].set_xlabel("Column Number")
	axes[0, 0].set_ylabel("Median Charge (ADU)")
	axes[0, 0].set_title("CCD 1")

	axes[0, 1].scatter(safe_cols[1], median_charge_col_safe[1], s=1)
	#axes[0, 1].set_xlabel("Column Number")
	#axes[0, 1].set_ylabel("Median Charge (ADU)")
	axes[0, 1].set_title("CCD 2")

	axes[1, 0].scatter(safe_cols[2], median_charge_col_safe[2], s=1)
	axes[1, 0].set_xlabel("Column Number")
	axes[1, 0].set_ylabel("Median Charge (ADU)")
	axes[1, 0].set_title("CCD 3")

	axes[1, 1].scatter(safe_cols[3], median_charge_col_safe[3], s=1)
	axes[1, 1].set_xlabel("Column Number")
	#axes[1, 1].set_ylabel("Median Charge (ADU)")
	axes[1, 1].set_title("CCD 4")

	plt.tight_layout()
	if (save_pdf==True) and (save_jpeg==True):
		plt.savefig(fig_path+'adu_column_masked_'+image+'.pdf')
		plt.savefig(fig_path+'adu_column_masked_'+image+'.jpeg',dpi=300)
	if (save_pdf==False) and (save_jpeg==True):
		plt.savefig(fig_path+'adu_column_masked_'+image+'.jpeg',dpi=300)
	if (save_pdf==True) and (save_jpeg==False):
		plt.savefig(fig_path+'adu_column_masked_'+image+'.pdf')
	elif (save_pdf==False) and (save_jpeg==False):
		plt.show()
