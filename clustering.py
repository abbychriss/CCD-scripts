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

plot_data_only=False
plot_cluster_algs=True
plot_all=False

file_name = "itp_img_CV_250x3500x500_bin1x1_125_20260317_130159_36.fz"
path = "/Users/abbychriss/Desktop/Privitera_335/"
files=glob(path+'**/'+file_name,recursive=True)

charge_thresh=60
coord_range=[0,250,0,250] #[min_x, max_x, min_y, max_y]
algorithms = []
alphabet=['A','B','C','D']

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
    axd.set_title(f"Data Pixel Charge >{charge_thresh} ADU, ext {ext}")
    plt.show()

def plot_clusters(images, cmaps, n_clusters, nrows, ncols, charge_thresh, ext, figsize=(8,4), tick_labels=False):
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    for i, ax in enumerate(axs):
        n=n_clusters[i]
        cmap=cmaps[i]
        cluster_img = images[i]
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
            tick_labels = ['bg', 'noise'] + [str(i) for i in range(1, n_clusters + 1)]
            cbar.set_ticks(tick_positions)
            cbar.set_ticklabels(tick_labels, fontsize=4)
        else:
            cbar.set_ticks([])
        ax.invert_yaxis()
        ax.set_title(f"{algorithms[i]}")
    plt.suptitle(f'Clusters of charge > {charge_thresh} ADU, ext {ext}\n'
                 +f"{algorithms[i]} found {n_clusters[i]}" for i in range(len(algorithms)))
    plt.show()

def plot_data_clusters(charge,cluster_img,cmap,cluster_alg):
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    ax=ax.flatten()
    cax1 = ax[0].imshow(charge,
                        cmap="viridis",
                        alpha=0.99,
                        norm=colors.LogNorm(vmin=min(charge[i]), vmax=max(charge[i]))
                        )
    fig.colorbar(cax1,label="Charge (ADU)",orientation='horizontal',ax=ax[0])
    ax[0].set_xlim(coord_range[0])
    ax[0].set_xlabel("Pixel X")
    ax[0].set_ylabel("Pixel Y")
    ax[0].set_xlim(coord_range[0],coord_range[1])
    ax[0].set_ylim(coord_range[2],coord_range[3])
    ax[0].set_title(f"Data Pixel Charge >{charge_thresh} ADU, ext {ext}")

    cax2 = ax[1].imshow(cluster_img,
                    cmap=cmap,
                    vmin=0,
                    vmax=n_clusters+2,
                    interpolation='nearest'
                    )
    ax[1].set_xlabel("Pixel X")
    ax[1].set_ylabel("Pixel Y")
    ax[1].set_title(f"{cluster_alg} Clusters (charge>{charge_thresh} ADU), ext {ext}")
    cbar = fig.colorbar(cax2, label="Cluster ID", orientation='horizontal', ax=ax[1])
    tick_positions = np.arange(n_clusters+2)
    tick_labels = ['bg', 'noise'] + [str(i) for i in range(1, n_clusters + 1)]
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels, fontsize=8)
    fig.gca().invert_yaxis()
    plt.show()


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
            
        # Generate clusters using optics algorithm
        labels = []

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
        centers, kmeans_labels, sigmas = get_kmeans_cluster(charge_stacked_scaled,
                                                     n_clusters=20
                                                     )
        labels.append(kmeans_labels)

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
                               cmaps,
                               cluster_alg='OPTICS'
                               )
