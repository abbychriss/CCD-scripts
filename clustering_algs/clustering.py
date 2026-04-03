from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from astropy.io import fits

from sklearn.cluster import k_means, AffinityPropagation, OPTICS, MeanShift, HDBSCAN
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
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
def get_optics_cluster(data, algorithms_list,
                       min_samples=10, 
                       metric='euclidean',
                       cluster_method='xi', 
                       eps=0.1, xi=0.05, 
                       min_cluster_size=5):
        
    optics_cluster = OPTICS(min_samples=min_samples, metric=metric, 
                            cluster_method=cluster_method, xi=xi,
                            min_cluster_size=min_cluster_size, eps=eps).fit(data)
    labels = optics_cluster.labels_
    algorithms_list.append('OPTICS')
    return labels

#------ Affinity Propogation
def get_affinity_cluster(data, algorithms_list,
                         damping=0.5, convergence_iter=15, max_iter=200, random_state=5):
    affinity_cluster = AffinityPropagation(damping=damping, 
                                           max_iter=max_iter, 
                                           convergence_iter=convergence_iter, 
                                           affinity='euclidean', 
                                           random_state=random_state).fit(data)
    center_indices = affinity_cluster.cluster_centers_indices_
    labels = affinity_cluster.labels_
    algorithms_list.append('Affinity Propogation')
    return center_indices, labels

#------ k-means
def get_kmeans_cluster(data, algorithms_list,
                       n_clusters=100, init='k-means++'):
    centers, labels, sigmas = k_means(data, n_clusters=n_clusters, init=init)
    algorithms_list.append('k-means')
    return centers, labels, sigmas

#------ HDBSCAN
def get_hdbscan_cluster(data, algorithms_list,
                        min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.0, 
                        max_cluster_size=None, metric='euclidean', alpha=1.0, algorithm='auto',
                        leaf_size=40, cluster_selection_method='eom', 
                        allow_single_cluster=False, store_centers=None):
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=cluster_selection_epsilon, 
                        max_cluster_size=max_cluster_size, metric=metric, alpha=alpha, algorithm=algorithm,
                        leaf_size=leaf_size, cluster_selection_method=cluster_selection_method, allow_single_cluster=allow_single_cluster,
                        store_centers=store_centers).fit(data)
    labels = hdb.labels_
    probabilities = hdb.probabilities_

    algorithms_list.append('HDBSCAN')

    if store_centers=='centroid':
        centroids = hdb.centroids_
        return labels, probabilities, centroids
    elif store_centers=='medoid':
        medoids = hdb.medoids_
        return labels, probabilities, medoids
    elif store_centers=='both':
        return labels, probabilities, centroids, medoids
    else:
        return labels, probabilities
    
    
#----- Gaussian Mixture
def get_gaussian_mixture_cluster(data, algorithms_list,
                                 n_components=20, covariance_type='full', tol=0.001, 
                                 reg_covar=1e-06, max_iter=100, 
                                 n_init=10, init_params='kmeans', weights_init=None, 
                                 means_init=None, random_state=None, 
                                 warm_start=False):
    gm = GaussianMixture(n_components=n_components, 
                        covariance_type=covariance_type, tol=tol, 
                        reg_covar=reg_covar, max_iter=max_iter, 
                        n_init=n_init, init_params='kmeans', 
                        weights_init=weights_init, means_init=means_init, 
                        random_state=random_state, warm_start=warm_start)
    labels = gm.fit_predict(data)
    log_likelihood = gm.score(data)
    weights = gm.weights_
    algorithms_list.append('Gaussian Mixture')
    return labels, log_likelihood, weights

#----- Bayesian Gaussian Mixture
def get_bayesian_gaussian_mixture_cluster(data, algorithms_list,
                                        n_components=20, covariance_type='full', tol=0.001, 
                                        reg_covar=1e-06, max_iter=100, n_init=10, init_params='kmeans', 
                                        weight_concentration_prior_type='dirichlet_process', weight_concentration_prior=None, 
                                        mean_precision_prior=None, mean_prior=None, degrees_of_freedom_prior=None, 
                                        covariance_prior=None, random_state=None, warm_start=False):
    bgm = BayesianGaussianMixture(n_components=n_components, 
                        covariance_type=covariance_type, tol=tol, 
                        reg_covar=reg_covar, max_iter=max_iter, 
                        n_init=n_init, init_params=init_params, 
                        weight_concentration_prior_type=weight_concentration_prior_type, 
                        weight_concentration_prior=weight_concentration_prior, 
                        mean_precision_prior=mean_precision_prior, mean_prior=mean_prior,
                        degrees_of_freedom_prior=degrees_of_freedom_prior, covariance_prior=covariance_prior,
                        random_state=random_state, warm_start=warm_start)
    labels = bgm.fit_predict(data)
    log_likelihood = bgm.score(data)
    weights = bgm.weights_
    algorithms_list.append('Bayesian Gaussian Mixture')
    return labels, log_likelihood, weights
 

#------ Mean Shift
def get_mean_shift(data, algorithms_list,
                   bandwidth=None, seeds=None, bin_seeding=False, 
                   min_bin_freq=1, cluster_all=True, max_iter=300):
    ms_clusters = MeanShift(bandwidth=bandwidth, seeds=seeds, bin_seeding=bin_seeding, min_bin_freq=min_bin_freq, 
                            cluster_all=cluster_all, max_iter=max_iter).fit(data)
    labels = ms_clusters.labels_
    centers = ms_clusters.cluster_centers_
    algorithms_list.append('Mean Shift')
    return labels, centers

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

#--------------------- COLOR MAPS FUNCTIONS --------------------------------------

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

#--------------------- PLOTTING FUNCTIONS --------------------------------------

def plot_data(charge,coord_range, save_plot=False, figname='data'):
    figd,axd=plt.subplots(1,1,figsize=(10,4),constrained_layout=True)
    caxd=axd.imshow(charge,cmap="viridis",alpha=0.99,norm=colors.LogNorm(vmin=min(charge[i]), vmax=max(charge[i])))
    figd.colorbar(caxd,label="Charge (ADU)",orientation='horizontal',ax=axd)
    axd.set_xlabel("Pixel X")
    axd.set_ylabel("Pixel Y")
    axd.set_xlim(coord_range[0],coord_range[1])
    axd.set_ylim(coord_range[2],coord_range[3])
    axd.set_title(f"Data (Pixel Charge > {charge_thresh} ADU), EXT {ext}")
    if save_plot:
            plt.savefig(str(figname)+f'{coord_range[0]}-{coord_range[1]}x{coord_range[2]}-{coord_range[3]}_EXT{ext}.jpg',dpi=350)
    plt.show()

def plot_clusters_i(fig, ax, cluster_img, cluster_alg, cmap, n, tick_labels=False):
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
    ax.set_title(f"{cluster_alg}: {n} clusters")

def plot_clusters(cluster_imgs, cmaps, n_clusters, nrows_subplot, ncols_subplot, charge_thresh, 
                  ext, algorithms, subplots=True, figsize_subplot=(8,4), figsize_i=(5.5,6), tick_labels=False,
                  save_plot=False, figname='clusters'):
    if subplots:
        fig, axs = plt.subplots(nrows_subplot, ncols_subplot, figsize=figsize_subplot, constrained_layout=True)
        for i, ax in enumerate(axs):
            n=n_clusters[i]
            cmap=cmaps[i]
            cluster_img = cluster_imgs[i]
            cluster_alg = algorithms[i]
            plot_clusters_i(fig,ax,cluster_img,cluster_alg,cmap,n,tick_labels=tick_labels)
        plt.suptitle(f'Clusters (Pixel Charge > {charge_thresh} ADU), EXT {ext}')
        if save_plot:
            plt.savefig(str(figname)+''.join(f'_{alg[:1]}' for alg in algorithms)+'_EXT{ext}.jpg', dpi=350)
        plt.show()
    else:
        for i in range(len(algorithms)):
            n=n_clusters[i]
            cmap=cmaps[i]
            cluster_img = cluster_imgs[i]
            cluster_alg = algorithms[i]
            fig, ax = plt.subplots(1, 1, figsize=figsize_i, constrained_layout=True)
            plot_clusters_i(fig,ax,cluster_img,cluster_alg,cmap,n,tick_labels=tick_labels)
            plt.suptitle(f'Clusters (Pixel Charge > {charge_thresh} ADU), EXT {ext}')
            if save_plot:
                plt.savefig(figname+f'_{cluster_alg}_EXT{ext}.jpg',dpi=350)
            plt.show()

def plot_data_clusters(data, cluster_imgs, cmaps, ext, coord_range, 
                       algorithms, nrows_cluster, ncols_cluster=1, 
                       figsize=(11, 7), alg_tick_labels=False,
                       save_plot=False,figname=''):

    # Create figure with GridSpec for 1x2 layout where column 2 has nested subplots
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5])
    
    # Column 1: single large subplot for data
    ax_data = fig.add_subplot(gs[0, 0])
    
    # Column 2: nested subplots for cluster algorithms
    gs_nested = gs[0, 1].subgridspec(nrows_cluster, ncols_cluster)
    alg_axs = [['' for j in range(ncols_cluster)] for i in range(nrows_cluster)]
    for i in range(nrows_cluster):
        for j in range(ncols_cluster):
            alg_axs[i][j] = fig.add_subplot(gs_nested[i,j])

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
    k=0
    if k<= len(algorithms)-1:
        for i in range(nrows_cluster):
            for j in range(ncols_cluster):
                if k<= len(algorithms)-1:
                    alg_ax = alg_axs[i][j]
                    n=n_clusters[k]
                    cmap=cmaps[k]
                    cluster_img=cluster_imgs[k]
                    cluster_alg=algorithms[k]
                    alg_cax = alg_ax.imshow(cluster_img,
                                cmap=cmap,
                                vmin=0,
                                vmax=n+2,
                                interpolation='nearest'
                                )
                    alg_ax.set_xlabel("Pixel X")
                    alg_ax.set_ylabel("Pixel Y")
                    alg_ax.set_title(f"{cluster_alg}:\n{n} clusters")
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
                    k+=1
                else:
                    break
        k+=1

    fig.suptitle(f"Data and Clustering Algorithms\n(Pixel charge > {charge_thresh} ADU), EXT {ext}")

    if save_plot:
        plt.savefig(str(figname)+'_data+clusters_'+'_'.join(f'{alg[:2]}' for alg in algorithms)+f'_EXT{ext}.jpg', dpi=350)

    plt.show()


# --------------------- RUN CODE AND MAKE PLOTS ------------------------------

do_optics=True
do_hdbscan=True
do_kmeans=False
do_affinity=False
do_mean_shift=False
do_gaussian_mixture=True
do_bayesian_gaussian_mixture=True

plot_data_only=False
plot_clusters_only=False
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
                                                    charge_weight=0.01)
        labels = []
        algorithms = []

        # Generate clusters using specified algorithm

        if do_optics:
            #optionally, can find the optimal epsilon manually
            #find_optimal_eps(charge_stacked_scaled)
            optics_labels = get_optics_cluster(charge_stacked_scaled,
                                               algorithms,
                                               min_samples=10,
                                               metric='euclidean',
                                               cluster_method='xi',
                                               xi=0.2,
                                               min_cluster_size=5,
                                               eps=0.01)
            labels.append(optics_labels)
        
        if do_kmeans:
            centers, kmeans_labels, sigmas = get_kmeans_cluster(charge_stacked_scaled,
                                                                algorithms,
                                                                n_clusters=20)
            labels.append(kmeans_labels)

        if do_affinity:
            affinity_centers, affinity_labels = get_affinity_cluster(charge_stacked_scaled,
                                                                     algorithms,
                                                                     damping=0.97, #higher values = less clusters
                                                                     convergence_iter=15,
                                                                     max_iter=100,
                                                                     random_state=5)
            labels.append(affinity_labels)

        if do_hdbscan:
            hdb_labels, hdb_probabilities, hdb_centers = get_hdbscan_cluster(charge_stacked_scaled, 
                                                                             algorithms,
                                                                             min_cluster_size=10,
                                                                             min_samples = 10,
                                                                             store_centers='medoid')
            labels.append(hdb_labels)

        if do_gaussian_mixture:
            gm_labels, gm_log_likelihood, gm_weights = get_gaussian_mixture_cluster(charge_stacked_scaled, 
                                                                                    algorithms,
                                                                                    n_components=22, tol=0.001, 
                                                                                    max_iter=300, 
                                                                                    n_init=10, init_params='kmeans',
                                                                                    random_state=None)
            labels.append(gm_labels)

        if do_bayesian_gaussian_mixture:
            bgm_labels, bgm_log_likelihood, bgm_weights = get_bayesian_gaussian_mixture_cluster(charge_stacked_scaled, algorithms,
                                            n_components=30, covariance_type='full', tol=0.001, 
                                            reg_covar=1e-06, max_iter=300, n_init=10, init_params='kmeans', 
                                            weight_concentration_prior_type='dirichlet_process')
            labels.append(bgm_labels)
            
        if do_mean_shift:
            ms_labels, ms_centers = get_mean_shift(charge_stacked_scaled,
                                                   algorithms,
                                                   min_bin_freq=1, 
                                                   cluster_all=True, 
                                                   max_iter=300)
            labels.append(ms_labels)
            
        cluster_images = [get_cluster_image(label) for label in labels]
        remapped_images = [remap_image(image) for image in cluster_images]

        n_clusters = [len(np.unique(label[label != -1])) for label in labels]

        print(f"EXT {ext}: " + ", ".join(f"{algorithms[i]} found {n_clusters[i]}" for i in range(len(algorithms))))

        # Create discrete colormap that includes background (white), noise (black), and cluster colors
        cmaps = [get_full_bg_noise_cmap(n) for n in n_clusters]

        #--------- PLOT RESULTS -----------------------

        # Plot pixel charge heatmaps to visually inspect clusters and validate algorithm
        if plot_data_only:
            plot_data(charge,coord_range)
            
        # Compare clustering algorithm results
        if plot_clusters_only:
            plot_clusters(remapped_images,
                          cmaps,
                          n_clusters,
                          nrows_subplot=1,
                          ncols_subplot=len(algorithms),
                          figsize_subplot=(8,4),
                          figsize_i=(5.5,6),
                          algorithms=algorithms,
                          subplots=False,
                          charge_thresh=charge_thresh,
                          ext=ext,
                          tick_labels=False
                          )

        
        # Plot data and clustering algorithms in same subplot to compare directly
        if plot_all:
            plot_data_clusters(charge,
                               remapped_images,
                               cmaps=cmaps,
                               ext=ext,
                               coord_range=coord_range,
                               algorithms=algorithms,
                               figsize=(11,7),
                               nrows_cluster=2,#1
                               ncols_cluster = 2,#len(algorithms)
                               save_plot=True,
                               figname='/Users/abbychriss/Desktop/Privitera_335/plots/'+'_'.join(file.split('/')[-1].split('_')[i] \
                                                                                                 for i in range(len(file.split('/')[-1].split('_'))-1))
                               )
