from astropy.table import Table
from astropy import units as u
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np

#------- USER INPUTS ------------------------
plot_emitted_xray_energy = False
plot_1d_hit_hist = True
plot_2d_hitmap = True
calculate_percent_hit = True
calculate_percent_blocked = False
plot_energy = True
save_plots = False
path = '/Users/abbychriss/Desktop/Privitera_335/'
ext_path = 'geant4_sims/geant4_sims_output/Fe55_sim_csv/'
run = 'Fe55' #name of overall simulation
names = ['Fe55_name']
nevts = 1000000
 #Below lists must have same number of elements (= number of sims we are comparing)
parameters_dict = {'shielding': ['mylar foil'], #shielding material
            'thickness': ['15 micron'], #shielding thickness
            'distance':['4 cm'] #distance of source from detector
 } 
parameters = list(parameters_dict.values()) 

# ------- EXTRACT DATA FROM CSV'S ------------------------

# ------- DETECTOR HITS TABLE ------------------------
data=[]
energy = []
erange = np.zeros((len(names),2))
for i in range(len(names)):
    name = names[i]
    hits_table = Table.read(path+name+'_Data_nt_Hits.csv',data_start=9,
    names=['EventID', 'PixelX', 'PixelY', 'Edep_keV', 'ElectronHolePairs','ParticleName'])
    hits_table['Edep_keV'] = hits_table['Edep_keV'] * u.keV
    hits_table.sort(['PixelX','PixelY'])

    x,y,energy_i,name = hits_table['PixelX'], hits_table['PixelY'], hits_table['Edep_keV'], hits_table['ParticleName']
    nrows = 6000
    ncols = 1500
    data_i = np.zeros((nrows,ncols))
    for w in range(len(x)):
        data_i[int(x[w]),int(y[w])] = energy_i[w]
    data.append(data_i)

    energy.append(energy_i)
    erange[i,0] = min(energy_i)
    erange[i,1] = max(energy_i)

    percent_pix_hit = (len(x)/(6000*1500))*100
    print(percent_pix_hit)

#Figure out which parameters we are varing across simulations to make title/subtitle more sensible
param_different = [False, False, False]
for i in range(len(parameters)): 
    for j in range(len(names)):
        for k in range(len(names)):
            if parameters[i][j] != parameters[i][k]:
                param_different[i] = True
                break
            else:
                continue

#------- PLOT DETECTOR HIT MAPS ------------------------

if plot_2d_hitmap:
    cmap = matplotlib.colormaps['PuRd']
    fig, ax = plt.subplots(1,len(names),figsize=(3*len(names), 6),constrained_layout=True)

    hits_big_title = 'Energy Deposition per Pixel'
    for i in range(len(names)):
        if param_different[i]==False:
            hits_big_title+=', '+parameters[i][0]

    hits_subtitle = []
    for l in range(len(param_different)):
        hits_subtitle.append('')
    for m in range(len(param_different)):
        for j in range(len(names)):
            if param_different[m]==True:
                hits_subtitle[j]+=' '+parameters[m][j]+' '

    if len(names)>1:
        for k in range(len(names)):
            im = ax[k].imshow(data[k], cmap=cmap,norm=colors.LogNorm(vmin=erange[k,0], vmax=erange[k,1]))
            ax[k].set_title('d = '+hits_subtitle[k])
            ax[k].set_xlabel('PixelY')
            ax[0].set_ylabel('PixelX')
        fig.colorbar(im, ax=ax[len(names)-1], label='Energy (keV)')

    if len(names)==1:
        im = ax.imshow(data[0], cmap=cmap, norm=colors.LogNorm(vmin=erange[0,0], vmax=erange[0,1]))
        ax.set_title(hits_subtitle[0])
        ax.set_xlabel('PixelY')
        ax.set_ylabel('PixelX')
        fig.colorbar(im, ax=ax, label='Energy (keV)')

    plt.suptitle('Fe-55 Energy Deposition per Pixel \n (1000000 events, 15 micron mylar foil shielding)')#hits_big_title)

    if save_plots==True:
        plt.savefig(path+run+'_hits.pdf')
        plt.savefig(path+run+'_hits.jpeg',dpi=300)
        plt.close()
    else:
        plt.show()

#------- PLOT ENERGY HITS SPECTRA ------------------------
if plot_energy: 
    fig, ax = plt.subplots(1,len(names),figsize=(len(names)*4, 4.5),constrained_layout=True)
    energy_big_title = run+' Energy Spectrum '+'('+np.format_float_scientific(nevts,precision=3)+ ' events)'
    for i in range(len(names)):
        if param_different[i]==False:
            energy_big_title+=(', '+parameters[i][0])

    energy_subtitle = []
    for i in range(len(param_different)):
        energy_subtitle.append('')
    for i in range(len(param_different)):
        for j in range(len(names)):
            if param_different[i]==True:
                energy_subtitle[j]+=(' '+parameters[i][j]+' ')

    if len(names)>1:
        for i in range(len(names)):
            im = ax[i].hist(energy[i],range=(0.2,erange[i,1]+5),bins=75)
            ax[i].set_title('d = '+energy_subtitle[i])
            ax[i].set_xlabel('Energy (keV)')
            ax[0].set_ylabel('N')

    if len(names)==1:
        im = ax.hist(energy[0],range=(0.2,erange[0,1]+5),bins=75)
        ax.set_title(energy_subtitle[i])
        ax.set_xlabel('Energy (keV)')
        ax.set_ylabel('N')

    plt.suptitle(energy_big_title)#'Fe-55 Energy Spectrum (1000000 events, 15 micron mylar foil shielding)')#

    if save_plots==True:
        plt.savefig(path+run+'_energy.pdf')
        plt.savefig(path+run+'_energy.jpeg',dpi=300)
        plt.close()
    else:
        plt.show()

