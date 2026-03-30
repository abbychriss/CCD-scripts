from astropy.table import Table
from astropy import units as u
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np

save_plots=False

data=[]
energy = []
erange = np.zeros((3,2))
shields = ['Al','Cu','Mylar']

#------- EXTRACT DATA FROM CSV ------------------------
for i in range(len(shields)):
    shield = shields[i]
    path='/Users/abbychriss/Desktop/Privitera_335/'
    build_path='/Users/abbychriss/Desktop/Privitera_335/geant4_sims/Am241_Sim/build'
    table = Table.read(path+'/geant4_sims/Am241_Sim/Am241_csv/Am241_Data_nt_Hits_'+shield+'.csv',data_start=9,
    names=['EventID', 'PixelX', 'PixelY', 'Edep_keV', 'ElectronHolePairs'])
    table['Edep_keV'] = table['Edep_keV'] * u.keV
    table.sort(['PixelX', 'PixelY'])

    x,y,energy_i = table['PixelX'], table['PixelY'], table['Edep_keV']
    nrows = max(x)+1
    ncols = max(y)+1
    data_i = np.zeros((nrows,ncols))

    for j in range(len(energy_i)):
        data_i[int(x[j]),int(y[j])] = energy_i[j]

    data.append(data_i)

    energy.append(energy_i)
    erange[i,0] = min(energy_i)
    erange[i,1] = max(energy_i)

#------- PLOT DETECTOR HIT MAPS ------------------------
cmap = matplotlib.colormaps['PuRd']
fig, ax = plt.subplots(1,3,figsize=(9, 6),constrained_layout=True)

for i in range(len(shields)):
    im = ax[i].imshow(data[i], cmap=cmap, norm=colors.LogNorm(vmin=erange[i,0], vmax=erange[i,1]))
    ax[i].set_title(shields[i])
    ax[i].set_xlabel('X')
    ax[0].set_ylabel('Y')

plt.suptitle('Energy Deposition per Pixel')

fig.colorbar(im, ax=ax[2], label='Energy (keV)')

if save_plots==True:
    plt.savefig(path+'Am241_hits_all.pdf')
    plt.savefig(path+'Am241_hits_all.jpeg',dpi=300)
    plt.close()
else:
    plt.show()

#------- PLOT ENERGY SPECTRA ------------------------
fig, ax = plt.subplots(1,3,figsize=(12, 4),constrained_layout=True)

for i in range(len(shields)):
    im = ax[i].hist(energy[i],range=(0.2,60),bins=75)
    ax[i].set_title('2.5mm '+shields[i]+' shielding')
    ax[i].set_xlabel('Energy (keV)')
    ax[0].set_ylabel('N')

plt.suptitle('Am241 Energy Spectrum (10^6 events)')

if save_plots==True:
    plt.savefig(path+'Am241_energy_all.pdf')
    plt.savefig(path+'Am241_energy_all.jpeg',dpi=300)
    plt.close()
else:
    plt.show()
