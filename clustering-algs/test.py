#test
import numpy as np
nrows_cluster = 2
ncols_cluster = 2
alg_axs = np.empty((nrows_cluster,ncols_cluster),dtype=str)
alg_axs = [['' for i in range(nrows_cluster)] for j in range(ncols_cluster)]
for i,j in range(nrows_cluster),range(ncols_cluster):
        print(i,j)
        alg_axs[i][j] = f'quest {i-j}'
print(np.shape(alg_axs))
alg_axs = alg_axs