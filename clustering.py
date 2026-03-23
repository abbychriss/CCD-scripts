from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from astropy.io import fits

from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.cluster import k_means, AffinityPropagation, OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# --------------------- FUNCTION DEFINITIONS ------------------------------

#--------------------GET DATA FROM FITS--------------------------------------
    # function get_fits inputs a string that can be a relative or full path to a .fz or .fits file
    # expects data to be multiextension fits file with header as extension 0 and data in extensions 1-4
    # opens fits file with astropy fits io and returns list with shape (4, nrows,ncols) containing the data 
    # data are (nrows,ncols) pixel arrays where each entry is charge in ADU
def get_fits(file):
    hdu_list = fits.open(file)
    ext_charge=[hdu_list[i].data for i in range(1,5)]
    return ext_charge

#--------------------CREATE CHARGE MASK--------------------------------------
    # Function create_mask inputs a float representing the minimum charge we want to fit clusters to 
    # and array of single extension charge data in hdu format 
    # Returns flattened charge array containing charge data above threshold and (x,y) coordinates corresponding to masked charge
    # and stacked array of masked coordinates and charge
def create_mask(charge_thresh, charge, charge_weight=0):
    x_coords,y_coords=np.where(charge>charge_thresh)
    # Filter to keep only pixels within the clustering range
    mask = (x_coords >= coord_range[0]) & (x_coords < coord_range[1]) & \
            (y_coords >= coord_range[2]) & (y_coords < coord_range[3])
    coords=np.column_stack((x_coords[mask], y_coords[mask]))
    charge_masked=[charge[coord[0]][coord[1]] for coord in coords]
    charge_stacked = np.column_stack((coords,charge_masked))

    # Normalize and scale charge to have less weight
    # Use scikit standard scaler to scale all columns unit variance and remove the
    # mean so that when we downweight charge, the scaling is applied consistently
    scaler = StandardScaler()
    charge_stacked_scaled = scaler.fit_transform(charge_stacked)
    charge_stacked_scaled[:, 2] *= charge_weight
    return coords, charge_stacked_scaled

#---------------------CLUSTERING ALGORITHMS--------------------------------------

#------ Find nearest neighbor for OPTICS DBSCAN
def find_optimal_eps(data, k=10):
    #Plot k-distance graph to find optimal eps for DBSCAN
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(data)
    distances, indices = neighbors_fit.kneighbors(data)
    
    # Sort distances and plot
    distances = np.sort(distances[:, k-1], axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.ylabel('k-distance')
    plt.xlabel('Data Points sorted by distance')
    plt.title(f'k-distance graph (k={k})')
    plt.axhline(y=0.5, color='r', linestyle='--', label='eps=0.5')
    plt.axhline(y=1.0, color='g', linestyle='--', label='eps=1.0')
    plt.legend()
    plt.show()
    
    return distances

#------- OPTICS
def get_optics_cluster(data, min_samples=10, metric='euclidean',
                       cluster_method='xi', 
                       eps=0.1, xi=0.05, 
                       min_cluster_size=5):
        
    optics_cluster = OPTICS(min_samples=min_samples, metric=metric, 
                            cluster_method=cluster_method, xi=xi,
                            min_cluster_size=min_cluster_size, eps=eps).fit(data)
    labels = optics_cluster.labels_
    algorithms.append('OPTICS')
    return labels

#------ AffinityPropogation
def get_affinity_cluster(data,damping=0.5,convergence_iter=15,max_iter=200,random_state=5):
    affinity_cluster = AffinityPropagation(damping=damping, 
                                           max_iter=max_iter, 
                                           convergence_iter=convergence_iter, 
                                           affinity='euclidean', 
                                           random_state=random_state).fit(data)
    center_indices = affinity_cluster.cluster_centers_indices_
    labels = affinity_cluster.labels_
    algorithms.append('Affinity Propogation')
    return center_indices, labels

#------ k-means
def get_kmeans_cluster(data,n_clusters=100,init='k-means++'):
    centers, labels, sigmas = k_means(data, n_clusters=n_clusters, init=init)
    algorithms.append('k-means')
    return centers, labels, sigmas

#------

#------- Build cluster image from labels
# Build a 2D pixel array with values given by cluster labels in the clustering region defined by coord_range
def get_cluster_image(labels):
    cluster_image = np.full((coord_range[1]-coord_range[0], coord_range[3]-coord_range[2]), 0, dtype=int)
    for idx, (x, y) in enumerate(coords):
        cluster_image[x - coord_range[0], y - coord_range[2]] = labels[idx]
    return cluster_image


#---- Remap cluster image so that first two indices can become the background and noise colors
    # Remap cluster_image values to colormap indices
    # Original: 0=background, -1=noise, 1+=clusters
    # New: 0=white, 1=black, 2+=cluster colors
def remap_image(cluster_image):
    remapped_image = cluster_image.copy()
    remapped_image[cluster_image == -1] = 1  # noise -> index 1 (black)
    for cluster_id in range(1, int(np.max(cluster_image)) + 1):
        remapped_image[cluster_image == cluster_id] = cluster_id + 1
    return remapped_image

#---------------------COLOR MAPS--------------------------------------

#----------- Make discrete color map from n colors----------
def make_discrete_cmap(n,display_colors=False):
    # Anchor colors in RGB [0, 255] for easy comparison with online color picker
    anchor_colors_255 = np.array([
        (100, 200, 70),    # teal-green
        (155, 66, 194),    # purple
        (255, 151, 23),    # orange
        (114, 202, 224),   # cyan-blue
        (131, 230, 122),   # light green
        (252, 119, 212),   # pink
        (255, 255, 125),   # yellow-green
        (224, 13, 13),     # red
    ])

    # Normalize to [0,1] for matplotlib
    anchor_colors = anchor_colors_255 / 255.0

    # Continuous cmap
    cmap_cont = LinearSegmentedColormap.from_list("custom_1000", anchor_colors)

    # Resample to N discrete colors
    colors = cmap_cont(np.linspace(0, 1, n))

    # Show colormap for debugging
    if display_colors:
        fig, ax = plt.subplots(1, 1, figsize=(6, 2))
        ax.imshow(np.vstack((np.linspace(0, 10, n), np.linspace(0, 10, n))), cmap=ListedColormap(colors))
        ax.set_axis_off()
        plt.show()

    discrete_cmap = ListedColormap(colors)
    return discrete_cmap

#-------- Combine discrete color map with custom background and noise colors
def get_full_bg_noise_cmap(n, base_colors=np.array([[.255, .255, .255, 0.25], [0., 0., 0., 1.]])):
    cmap_n_colors = make_discrete_cmap(n)
    color_list = np.vstack((base_colors, cmap_n_colors.colors))
    cmap = colors.ListedColormap(color_list)
    return cmap

#---------------------PLOTTING FUNCTIONS--------------------------------------
def plot_data(charge,coord_range):
    figd,axd=plt.subplots(1,1,figsize=(10,4),constrained_layout=True)
    caxd=axd.imshow(charge,cmap="viridis",alpha=0.99,norm=colors.LogNorm(vmin=min(charge[i]), vmax=max(charge[i])))
    figd.colorbar(caxd,label="Charge (ADU)",orientation='horizontal',ax=axd)
    axd.set_xlabel("Pixel X")
    axd.set_ylabel("Pixel Y")
    axd.set_xlim(coord_range[0],coord_range[1])
    axd.set_ylim(coord_range[2],coord_range[3])
    axd.set_title(f"Data Pixel Charge >{charge_thresh} ADU, EXT {ext}")
    plt.show()

def plot_clusters(cluster_imgs, cmaps, n_clusters, nrows, ncols, charge_thresh, ext, algorithms, figsize=(8,4), tick_labels=False):
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    for i, ax in enumerate(axs):
        n=n_clusters[i]
        cmap=cmaps[i]
        cluster_img = cluster_imgs[i]
        aluster_alg = algorithms[i]
        caxc = ax.imshow(cluster_img,
                        cmap=cmap,
                        vmin=0,
                        vmax=n+2,
                        interpolation='nearest')
        ax.set_xlabel("Pixel X")
        ax.set_ylabel("Pixel Y")

        # Create colorbar with automatic labels
        cbar = fig.colorbar(caxc, label="Cluster ID", orientation='horizontal', ax=ax)
        if tick_labels:
            tick_positions = np.arange(n+2)
            tick_labels = ['bg', 'noise'] + [str(i) for i in range(1, n + 1)]
            cbar.set_ticks(tick_positions)
            cbar.set_ticklabels(tick_labels, fontsize=4)
        else:
            cbar.set_ticks([])
        ax.invert_yaxis()
        ax.set_title(f"{algorithms[i]}")
    plt.suptitle(f'Clusters of charge > {charge_thresh} ADU, ext {ext}\n'
                 +f"{algorithms[i]} found {n_clusters[i]}" for i in range(len(algorithms)))
    plt.show()

def plot_data_clusters(data, cluster_imgs, cmaps, ext, coord_range, 
                       algorithms, alg_rows, alg_cols=1, 
                       figsize=(10, 7), alg_tick_labels=False):

    # Create figure with GridSpec for 1x2 layout where column 2 has nested subplots
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])
    
    # Column 1: single large subplot for data
    ax_data = fig.add_subplot(gs[0, 0])
    
    # Column 2: nested subplots for cluster algorithms
    gs_nested = gs[0, 1].subgridspec(alg_rows, alg_cols)
    alg_axs = [fig.add_subplot(gs_nested[i, 0]) for i in range(alg_rows)]

    # Plot data in column 1
    cax = ax_data.imshow(data,
                        cmap="viridis",
                        alpha=0.99,
                        norm=colors.LogNorm(vmin=min(data[ext]), vmax=max(data[ext]))
                        )
    fig.colorbar(cax,label="Charge (ADU)",orientation='horizontal',ax=ax_data)
    ax_data.set_xlabel("Pixel X")
    ax_data.set_ylabel("Pixel Y")
    ax_data.set_xlim(coord_range[0],coord_range[1])
    ax_data.set_ylim(coord_range[2],coord_range[3])
    ax_data.set_title(f"Data")

    # Plot cluster algorithms in column 2
    for i, alg_ax in enumerate(alg_axs):
        n=n_clusters[i]
        cmap=cmaps[i]
        cluster_img=cluster_imgs[i]
        cluster_alg=algorithms[i]
        alg_cax = alg_ax.imshow(cluster_img,
                    cmap=cmap,
                    vmin=0,
                    vmax=n+2,
                    interpolation='nearest'
                    )
        alg_ax.set_xlabel("Pixel X")
        alg_ax.set_ylabel("Pixel Y")
        alg_ax.set_title(f"{cluster_alg}")
        alg_ax.invert_yaxis()

        # Make colorbars from discrete colormap for each cluster
        cbar = fig.colorbar(alg_cax, label="Cluster ID", orientation='horizontal', ax=alg_ax)
        
        if alg_tick_labels==True:
            tick_positions = np.arange(n+2)
            tick_labels_list = ['bg', 'noise'] + [str(i) for i in range(1, n + 1)]
            cbar.set_ticks(tick_positions)
            cbar.set_ticklabels(tick_labels_list, fontsize=8)
        else:
            cbar.set_ticks([])

    fig.suptitle(f"Data and Clustering Algorithms\n(Pixel charge > {charge_thresh} ADU), EXT {ext}")

    plt.show()


# --------------------- RUN CODE AND MAKE PLOTS ------------------------------

do_optics=True
do_kmeans=False
do_affinity=False

plot_data_only=False
plot_cluster_algs=False
plot_all=True

file_name = "itp_img_CV_250x3500x500_bin1x1_125_20260317_130159_36.fz"
path = "/Users/abbychriss/Desktop/Privitera_335/"
files=glob(path+'**/'+file_name,recursive=True)

charge_thresh=40 #minimum charge in ADU to pass clustering algorithm
coord_range=[0,250,0,250] #[min_x, max_x, min_y, max_y]

for i, file in enumerate(files):

    ext_charge = get_fits(file)

    for ext,charge in enumerate(ext_charge):

        coords, charge_stacked_scaled = create_mask(charge_thresh,
                                                    charge,
                                                    charge_weight=0.01
                                                    )

        # Plot pixel charge heatmaps to visually inspect clusters and validate algorithm
        if plot_data_only:
            plot_data(charge,coord_range)

        #find_optimal_eps(charge_stacked_scaled)
            
        labels = []
        algorithms = []

        # Generate clusters using optics algorithm
        if do_optics:
            optics_labels = get_optics_cluster(charge_stacked_scaled,
                                                        min_samples=10,
                                                        metric='euclidean',
                                                        cluster_method='xi',
                                                        xi=0.2,
                                                        min_cluster_size=5,
                                                        eps=0.01
                                                        )
            labels.append(optics_labels)
        
        # Generate clusters using k-means algorithm
        if do_kmeans:
            centers, kmeans_labels, sigmas = get_kmeans_cluster(charge_stacked_scaled,
                                                        n_clusters=20
                                                        )
            labels.append(kmeans_labels)

        if do_affinity:
            affinity_centers, affinity_labels = get_affinity_cluster(charge_stacked_scaled,
                                                        damping=0.97, #higher values = less clusters
                                                        convergence_iter=15,
                                                        max_iter=100,
                                                        random_state=5)
            labels.append(affinity_labels)

        cluster_images = [get_cluster_image(label) for label in labels]
        remapped_images = [remap_image(image) for image in cluster_images]

        n_clusters = [len(np.unique(label[label != -1])) for label in labels]

        print(f"EXT {ext}: " + ", ".join(f"{algorithms[i]} found {n_clusters[i]}" for i in range(len(algorithms))))

        # Create discrete colormap that includes background (white), noise (black), and cluster colors
        cmaps = [get_full_bg_noise_cmap(n) for n in n_clusters]

        # Compare clustering algorithm results
        if plot_cluster_algs:
            plot_clusters(remapped_images,
                          cmaps,
                          n_clusters,
                          nrows=1,
                          ncols=len(algorithms),
                          figsize=(8,4),
                          charge_thresh=charge_thresh,
                          ext=ext
                          )
        
        # Plot data and clustering algorithms in same subplot to compare directly
        if plot_all:
            plot_data_clusters(charge,
                               remapped_images,
                               cmaps=cmaps,
                               ext=ext,
                               coord_range=coord_range,
                               algorithms=algorithms,
                               alg_rows = len(algorithms)
                               )
