from astropy.table import Table
from astropy import units as u
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import glob
import math

#------- USER INPUTS ------------------------
plot_secondary_energy = False
plot_1d_hit_hist = False
plot_2d_hitmap = True
calculate_percent_hit = False
calculate_percent_blocked = False
plot_hits_energy = True
save_plots = True
path = '/Users/abbychriss/Desktop/Privitera_335/'
run = 'Fe55' #name of overall simulation
ext_path = 'geant4_sims/geant4_sims_output/'+run+'_sim_csv/'
names = ['16cm_DoubleMylarFoil','13cm_DoubleMylarFoil','10cm_DoubleMylarFoil',
         '8cm_DoubleMylarFoil','6cm_DoubleMylarFoil','4cm_DoubleMylarFoil',
         '3cm_DoubleMylarFoil','2cm_DoubleMylarFoil']

 #Below lists must have same number of elements (= number of sims we are comparing)
parameters_dict = {'shielding': ['30 micron double aluminized mylar foil' for j in range(len(names))], #shielding material
            #'thickness': ['2.5 mm'], #shielding thickness
            'distance':['16 cm','13 cm','10 cm', '8 cm','6 cm', '4 cm', '3 cm', '2 cm'], #distance of source from detector
            'nEvents': [4000000 for j in range(len(names))]
 } 
parameters = list(parameters_dict.values()) 

#Figure out which parameters we are varing across simulations to make title/subtitle more sensible
param_different = []
for i in range(len(parameters)): 
    param_different.append(False)
    for j in range(len(names)):
        for k in range(len(names)):
            if parameters[i][j] != parameters[i][k]:
                param_different[i] = True
                break
            else:
                continue

title_ext_names=[]
title_ext=''
for j,param in enumerate(parameters):
    if param_different[j]==False:
        title_ext_names.append(list(parameters_dict.keys())[j]+': '+str(param[0]))
q=0
for j, name in enumerate(title_ext_names):
    if j==0:
        title_ext+=name+', '
    else:
        if len(title_ext.split('\n')[q])>20:
            title_ext+='\n'+name+', '
            q+=1
        else:
            title_ext+=name+', '
title_ext=title_ext[:-2]

print(title_ext)


#create subtitles for each subplot!
subtitle = ['' for j in range(len(names))]
q=0
for i, name in enumerate(parameters):
    for j in range(len(names)):
        if param_different[i]==True and subtitle[j]=='':
            subtitle[j]+=list(parameters_dict.keys())[i]+': '+parameters[i][j]+', '
        elif param_different[i]==True:
            if len(subtitle[j].split('\n')[q])>20:
                subtitle[j]+='\n'+list(parameters_dict.keys())[i]+': '+parameters[i][j]+', '
                q+=1
            else:
                subtitle[j]+=list(parameters_dict.keys())[i]+': '+parameters[i][j]+', '
subtitle=[name[:-2] for name in subtitle]


# ------- EXTRACT DATA FROM CSV'S ------------------------

# ------- DETECTOR HITS TABLE ------------------------
data=[]
energy = []
erange = np.zeros((len(names),2))
for i,name in enumerate(names):
    files = glob.glob(f"{path}{ext_path}{run}_{name}_Data_nt_Hits*.csv")
    if len(files)==0:
        print(f"No files found with name {path}{ext_path}{run}_{name}_Data_nt_Hits*.csv !!!")
    df = pd.concat([pd.read_csv(f, comment='#', header=None) for f in files])
    df.to_csv(f"{path}{ext_path}{run}_{name}_Data_nt_Hits.csv", index=False)
    hits_table = Table.read(f'{path}{ext_path}{run}_{name}_Data_nt_Hits.csv',data_start=9,
    names=['EventID', 'PixelX', 'PixelY', 'Edep_keV', 'ElectronHolePairs','ParticleName'])
    hits_table['Edep_keV'] = hits_table['Edep_keV'] * u.keV
    hits_table.sort(['PixelX','PixelY'])

    x,y,energy_i,part_name  = hits_table['PixelX'], hits_table['PixelY'], hits_table['Edep_keV'],hits_table['ParticleName']
    nrows = 6000
    ncols = 1500
    data_i = np.zeros((nrows,ncols))
    data_i[x, y] = energy_i
    data.append(data_i)

    energy.append(energy_i)
    erange[i,0] = min(energy_i)
    erange[i,1] = max(energy_i)

    # ------- PERCENTAGE OF PIXELS WITH ENERGY > 7keV ------------------------

    n_greater_6 = len([x for x in energy_i if x > 6.5])

    percent_greater_6 = (n_greater_6/len(energy_i))*100
    #print(f'Percentage of pixels with energy > 6.5keV: ({names[i]})',percent_greater_6)


    """percent_gammas = (list(part_name).count('gamma')/len(x))*100
    percent_electrons = (list(part_name).count('e-')/len(x))*100
    percent_remaining = 100 - percent_gammas - percent_electrons
    print(name, ': % gammas: ',percent_gammas,' % e-: ',percent_electrons,' % remaining: ',percent_remaining)
    """
    # ------- PERCENTAGE OF PIXELS HIT ------------------------
    if calculate_percent_hit:
    
        npix_tot = 6000*1500 #nrows * ncols
        pix = []
        for l in range(6000):
            for j in range(1500):
                pix.append([l,j])

        pix_hit=[]
        for j in range(len(x)):
            pix_hit.append([int(x[j]), int(y[j])])
        print('npix_hit ('+names[i]+'): '+str(len(x)))

        npix_repeated=0
        for p in range(len(pix_hit)-1):
            if sorted(pix_hit)[p] == sorted(pix_hit)[p+1]:
                npix_repeated+=1
                #print('repeat found: ',pix_hit[p])
        percent_hit = ((len(x)-npix_repeated)/npix_tot)*100
        print('% pixels hit: ('+names[i]+'): '+str(round(percent_hit,4)))

    # ------- 1d pixel hit histogram ------------------------  
    if plot_1d_hit_hist:
        plt.hist(x,bins=len(x),range=(0,nrows))
        plt.xticks(np.arange(0,nrows,step=500,dtype=int))
        plt.xlabel('Pixel X')
        plt.ylabel('NHits')
        plt.title('Fe-55 Pixel Row Hits \n'+title_ext)
        
        if save_plots:
            plt.savefig(path+'hits_per_row_'+names[i]+'.pdf')
            plt.savefig(path+'hits_per_row_'+names[i]+'.jpeg',dpi=300)
        plt.show()

        plt.hist(y,bins=len(y),range=(0,ncols),color='purple')
        plt.xticks(np.arange(0,ncols,step=150,dtype=int))
        plt.xlabel('Pixel Y')
        plt.ylabel('NHits')
        plt.title('Fe-55 Pixel Col Hits \n'+title_ext)

        if save_plots:
            plt.savefig(path+'hits_per_col_'+names[i]+'.pdf')
            plt.savefig(path+'hits_per_col_'+names[i]+'.jpeg',dpi=300)
        plt.show()
        
    # ------- PERCENTAGE OF X-RAYS BLOCKED BY MYLAR -------
    if calculate_percent_blocked:
        # ----------- open secondary particle table -----------------
        secondaries_table = Table.read(path+name+'_Data_nt_Secondaries.csv',data_start=9,
        names=['EventID', 'Ekin','ParticleName'])
        secondaries_table['Ekin'] = secondaries_table['Ekin'] * u.keV

        secondary_energy_i = secondaries_table['Ekin']
        secondary_eventID_i = secondaries_table['EventID']
        secondary_partname_i = secondaries_table['ParticleName']
        print('npix_emit: ',len(secondaries_table))

        # ------- number of xrays detected -------
        #Primary x-ray line: 5.9 keV, secondary x-ray line: 6.5 keV
        gammas_hit = []
        for l in range(len(energy_i)):
            if part_name[l]=='gamma':
                gammas_hit.append(energy_i[l])
        xray_hit=np.histogram(gammas_hit, bins=1, range=(5.45,6.5))
        nxray_hit = xray_hit[0][0]

        # -------- calculate number of xrays emitted ----------
        gammas_emit = []
        for l in range(len(energy_i)):
            if secondary_partname_i[l]=='gamma':
                gammas_emit.append(secondary_energy_i[l])
        xray_emit=np.histogram(gammas_emit, bins=1, range=(5.45,6.5))
        nxray_emit = xray_emit[0][0]

        # -------- calculate percentage of xrays blocked ----------
        # % blocked = (1 - n_detected / n_generated)*100
        percent_blocked = 100*(1 - nxray_hit/nxray_emit)
        print('% blocked:',percent_blocked)

    # --->>> these numbers are not making sense so plot actual emitted energy spectrum
    if plot_secondary_energy:
            # ----------- open secondary particle table -----------------
        secondaries_table = Table.read(path+name+'_Data_nt_Secondaries.csv',data_start=9,
        names=['EventID', 'Ekin','ParticleName'])
        secondaries_table['Ekin'] = secondaries_table['Ekin'] * u.keV

        secondary_energy_i = secondaries_table['Ekin']
        secondary_eventID_i = secondaries_table['EventID']
        secondary_partname_i = secondaries_table['ParticleName']
        print('npix_emit: ',len(secondaries_table))

        plt.hist(secondary_energy_i, bins=75, range=(4.5,12))
        plt.xlabel('Energy (keV)')
        plt.ylabel('N')
        plt.title('Fe-55 Emitted Energy Spectrum, '+title_ext)

        if save_plots:
            plt.savefig(path+run+'_'+name+'_secondary_spectrum.pdf')
            plt.savefig(path+run+'_'+name+'_secondary_spectrum.jpeg',dpi=300)
        plt.show()


#------- PLOT DETECTOR HIT MAPS ------------------------
if plot_2d_hitmap:

    hits_big_title = 'Energy Deposition per Pixel\n'

    cmap = matplotlib.colormaps['PuRd']

    #formatting rows and columns of subplots
    n = len(names)
    ncols = 4#math.ceil(math.sqrt(n))
    nrows = 2#math.ceil(n / ncols)
    fig, ax = plt.subplots(nrows, ncols, figsize=(2.5*ncols, 4*nrows), constrained_layout=True)
    ax = np.atleast_1d(ax).ravel()

    # turn off any unused subplot axes
    for k in range(len(names), len(ax)):
        ax[k].axis('off')


    if len(names)>1:
        for q in range(len(names)):
            im = ax[q].imshow(data[q],cmap=cmap, 
                              norm=colors.SymLogNorm(linthresh=0.01,vmin=erange[q,0], vmax=erange[q,1]))
            ax[q].set_title(subtitle[q])
            ax[q].set_xlabel('PixelY')
            ax[0].set_ylabel('PixelX')
            ax[4].set_ylabel('PixelX')
            fig.colorbar(im, ax=ax[q], label='Energy (keV)')

    if len(names)==1:
        im = ax.imshow(data[0],cmap=cmap,
                       norm=colors.SymLogNorm(linthresh=0.01,vmin=erange[0,0], vmax=erange[0,1]))
        ax.set_title(subtitle[0])
        ax.set_xlabel('PixelY')
        ax.set_ylabel('PixelX')
        fig.colorbar(im, ax=ax, label='Energy (keV)')

    plt.suptitle(hits_big_title+title_ext)

    if save_plots:
        #plt.savefig(path+run+'_'+name+'_hits.pdf')
        plt.savefig(path+run+'_hits.jpeg',dpi=300)
    plt.show()


#------- PLOT ENERGY HITS SPECTRA ------------------------

if plot_hits_energy: 
    fig, ax = plt.subplots(nrows,ncols,figsize=(3*ncols, 3*nrows),constrained_layout=True)
    ax = np.atleast_1d(ax).ravel()
    # turn off any unused subplot axes
    for k in range(len(names), len(ax)):
        ax[k].axis('off')

    energy_big_title = run+' Energy Spectrum \n'+title_ext
    en_range=(erange[i,0],erange[i,1])
    if len(names)>1:
        for i in range(len(names)):
            im = ax[i].hist(energy[i],range=en_range,bins=75)
            ax[i].set_title(subtitle[i])
            ax[i].set_xlabel('Energy (keV)')
            ax[0].set_ylabel('N')
            ax[4].set_ylabel('N')

    if len(names)==1:
        im = ax.hist(energy[0],range=en_range,bins=75)
        ax.set_title(subtitle[0])
        ax.set_xlabel('Energy (keV)')
        ax.set_ylabel('N')

    plt.suptitle(energy_big_title)

    if save_plots==True:
        #plt.savefig(path+run+'_'+name+'_energy.pdf')
        plt.savefig(path+run+'_energy.jpeg',dpi=300)
    plt.show()

