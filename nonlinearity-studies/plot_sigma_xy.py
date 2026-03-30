from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.cluster import k_means, AffinityPropagation, OPTICS

subplots=True
plot_data=False

file_name = "itp_img_CV_250x3500x500_bin1x1_125_20260317_130159_36.fz"
path = "/Users/abbychriss/Desktop/Privitera_335/"
files=glob(path+'**/'+file_name,recursive=True)

charge_thresh=20
clustering_range=[0,250,0,250] #min_x,max_x,min_y,max_y
for i, file in enumerate(files):
    hdu_list = fits.open(file)
    ext_charge=[hdu_list[i].data for i in range(1,5)]
    for ext,charge in enumerate(ext_charge):
        x_coords,y_coords=np.where(charge>charge_thresh)
        # Filter to keep only pixels within the clustering range
        mask = (x_coords >= clustering_range[0]) & (x_coords < clustering_range[1]) & \
               (y_coords >= clustering_range[2]) & (y_coords < clustering_range[3])
        coords=np.column_stack((x_coords[mask], y_coords[mask]))
        charge_masked=[charge[coord[0]][coord[1]] for coord in coords]

        #Plot pixel charge heatmaps to visually inspect clusters and validate algorithm
        if plot_data:
            figd,axd=plt.subplots(1,1,figsize=(10,4),constrained_layout=True)
            caxd=axd.imshow(charge,cmap="viridis",alpha=0.99,norm=colors.LogNorm(vmin=min(charge[i]), vmax=max(charge[i])))
            figd.colorbar(caxd,label="Charge (ADU)",orientation='horizontal',ax=axd)
            axd.set_xlabel("Pixel X")
            axd.set_ylabel("Pixel Y")
            axd.set_title(f"DATA Pixel Charge >{charge_thresh} ADU, ext {ext}")
            plt.show()

        charge_stacked = np.column_stack((coords,charge_masked))

        #Other clustering algorithms for later
        """center, label, sigma = k_means(charge, n_clusters=150, init='k-means++')
        affinity_cluster = AffinityPropagation(random_state=5).fit(charge)"""
        optics_cluster = OPTICS(min_samples=10,metric='euclidean',xi=0.05,min_cluster_size=2,eps=10).fit(coords)
        print(vars(optics_cluster))

        # Plot clusters identified by optics as an image with noise gray and background white
        labels = optics_cluster.labels_
        print(f"Number of labels: {len(labels)}")
        # Build a 2D image of cluster labels for the clustering region
        cluster_image = np.full((clustering_range[1]-clustering_range[0], clustering_range[3]-clustering_range[2]), 0, dtype=int)
        for idx, (x, y) in enumerate(coords):
            cluster_image[x - clustering_range[0], y - clustering_range[2]] = labels[idx] if labels[idx] != -1 else -1

        # Create discrete colormap that includes background (white), noise (light gray), and cluster colors
        n_clusters = len(np.unique(labels[labels != -1]))
        
        # Build colormap: white (bg), light gray (noise), then cluster colors
        base_colors = ['white', 'lightgray']
        cluster_colors = list(plt.cm.tab20c(np.linspace(0, 1, n_clusters)))
        color_list = base_colors + cluster_colors
        cmap = colors.ListedColormap(color_list)
        
        # Remap cluster_image values to colormap indices
        # Original: 0=background, -1=noise, 1+=clusters
        # New: 0=white, 1=lightgray, 2+=cluster colors
        remapped_image = cluster_image.copy()
        remapped_image[cluster_image == -1] = 1  # noise -> index 1 (lightgray)
        for cluster_id in range(1, int(np.max(cluster_image)) + 1):
            remapped_image[cluster_image == cluster_id] = cluster_id + 1

        figc, axc = plt.subplots(1, 1, figsize=(10, 4), constrained_layout=True)
        caxc = axc.imshow(remapped_image,
                          cmap=cmap,
                          vmin=0,
                          vmax=len(color_list)-1,
                          interpolation='nearest')
        axc.set_xlabel("Pixel X")
        axc.set_ylabel("Pixel Y")
        
        # Create colorbar with automatic labels
        cbar = figc.colorbar(caxc, label="Cluster ID", orientation='horizontal', ax=axc)
        tick_positions = np.arange(len(color_list))
        tick_labels = ['bg', 'noise'] + [str(i) for i in range(1, n_clusters + 1)]
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(tick_labels, fontsize=8)
        figc.gca().invert_yaxis()
        axc.set_title(f"OPTICS Clusters (charge>{charge_thresh} ADU), ext {ext}")
        plt.show()