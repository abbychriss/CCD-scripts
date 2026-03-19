from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.cluster import k_means, AffinityPropagation, OPTICS

subplots=True
plot_data_only=False
plot_optics_only=False
plot_all=True

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
        if plot_data_only:
            figd,axd=plt.subplots(1,1,figsize=(10,4),constrained_layout=True)
            caxd=axd.imshow(charge,cmap="viridis",alpha=0.99,norm=colors.LogNorm(vmin=min(charge[i]), vmax=max(charge[i])))
            figd.colorbar(caxd,label="Charge (ADU)",orientation='horizontal',ax=axd)
            axd.set_xlabel("Pixel X")
            axd.set_ylabel("Pixel Y")
            axd.set_xlim(clustering_range[0],clustering_range[1])
            axd.set_ylim(clustering_range[2],clustering_range[3])
            axd.set_title(f"DATA Pixel Charge >{charge_thresh} ADU, ext {ext}")
            plt.show()

        #---------------------CLUSTERING ALGORITHMS--------------------------------------
        #def get_optics_cluster(file,)


        charge_stacked = np.column_stack((coords,charge_masked))

        #Other clustering algorithms for later
        """center, label, sigma = k_means(charge, n_clusters=150, init='k-means++')"""

        affinity_cluster = AffinityPropagation(random_state=5).fit(charge)

        optics_cluster = OPTICS(min_samples=10,metric='euclidean',xi=0.05,min_cluster_size=2,eps=10).fit(coords)
        print(vars(optics_cluster))

        # Plot clusters identified by optics as an image with noise gray and background white
        labels = optics_cluster.labels_
        
        # Build a 2D image of cluster labels for the clustering region
        cluster_image = np.full((clustering_range[1]-clustering_range[0], clustering_range[3]-clustering_range[2]), 0, dtype=int)
        for idx, (x, y) in enumerate(coords):
            cluster_image[x - clustering_range[0], y - clustering_range[2]] = labels[idx] if labels[idx] != -1 else -1

        # Create discrete colormap that includes background (white), noise (black), and cluster colors
        n_clusters = len(np.unique(labels[labels != -1]))
        print(f"Number of clusters: {n_clusters}")
        
        # Build colormap: white (bg), black (noise), then cluster colors
        # Combine multiple tab colormaps to get up to 60 distinct colors

        def make_discrete_cmap(n):
            # Anchor colors in RGB [0, 255] for easy comparison with color picker online
            anchor_colors_255 = np.array([
                (100, 200, 70),   # teal-green
                (155, 66, 194),   # purple
                (255, 151, 23),   # orange
                (114, 202, 224),   # cyan-blue
                (131, 230, 122),   # light green
                (252, 119, 212),   # pink
                (255, 255, 125),   # yellow-green
                (224, 13, 13),   # red
            ])

            # Normalize to [0,1] for matplotlib
            anchor_colors = anchor_colors_255 / 255.0

            # Continuous cmap
            cmap_cont = LinearSegmentedColormap.from_list("custom_1000", anchor_colors)

            # Resample to N discrete colors
            #show colormap for debugging
            colors = cmap_cont(np.linspace(0, 1, n))
            fig, ax = plt.subplots(1, 1, figsize=(6, 2))
            ax.imshow(np.vstack((np.linspace(0, 10, n), np.linspace(0, 10, n))), cmap=ListedColormap(colors))
            ax.set_axis_off()
            plt.show()
            return ListedColormap(colors)
        
        base_colors = np.array([[.255, .255, .255, 0.25], [0., 0., 0., 1.]]) #white, #black
        cmap_n_colors = make_discrete_cmap(n_clusters)
        print(cmap_n_colors.colors)
        color_list = np.vstack((base_colors, cmap_n_colors.colors))
        print(color_list)
        cmap = colors.ListedColormap(color_list)
        
        # Remap cluster_image values to colormap indices
        # Original: 0=background, -1=noise, 1+=clusters
        # New: 0=white, 1=black, 2+=cluster colors
        remapped_image = cluster_image.copy()
        remapped_image[cluster_image == -1] = 1  # noise -> index 1 (black)

        if plot_optics_only:
            
            #plot optics clustering algorithm results
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
            cbar.set_ticklabels(tick_labels, fontsize=4)
            figc.gca().invert_yaxis()
            axc.set_title(f"OPTICS Clusters (charge>{charge_thresh} ADU), ext {ext}")
            plt.show()

        if plot_all:

            # Plot data and clustering algorithms in same subplot to compare directly
            fig, ax = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
            print(ax.shape)
            ax=ax.flatten()
            cax1 = ax[0].imshow(charge,
                                cmap="viridis",
                                alpha=0.99,
                                norm=colors.LogNorm(vmin=min(charge[i]), vmax=max(charge[i]))
                                )
            fig.colorbar(cax1,label="Charge (ADU)",orientation='horizontal',ax=ax[0])
            ax[0].set_xlim(clustering_range[0])
            ax[0].set_xlabel("Pixel X")
            ax[0].set_ylabel("Pixel Y")
            ax[0].set_xlim(clustering_range[0],clustering_range[1])
            ax[0].set_ylim(clustering_range[2],clustering_range[3])
            ax[0].set_title(f"DATA Pixel Charge >{charge_thresh} ADU, ext {ext}")

            cax2 = ax[1].imshow(remapped_image,
                            cmap=cmap,
                            vmin=0,
                            vmax=len(color_list)-1,
                            interpolation='nearest'
                            )
            ax[1].set_xlabel("Pixel X")
            ax[1].set_ylabel("Pixel Y")
            ax[1].set_title(f"OPTICS Clusters (charge>{charge_thresh} ADU), ext {ext}")
            cbar = fig.colorbar(cax2, label="Cluster ID", orientation='horizontal', ax=ax[1])
            cbar.set_ticks(tick_positions)
            cbar.set_ticklabels(tick_labels, fontsize=8)
            tick_positions = np.arange(len(color_list))
            tick_labels = ['bg', 'noise'] + [str(i) for i in range(1, n_clusters + 1)]
            fig.gca().invert_yaxis()
            plt.show()