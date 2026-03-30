import pysimdamicm as dam

from matplotlib import pyplot as plt
from glob import glob
import numpy as np
import pandas as pd
from array import array
import ROOT

def main(calibration=5.2,Nbins=20000,Npeaks=500):

    pcd, rawdata = get_pcd("img_2000skips/new_config/Image_Am241_Source_*fits", 'panaSKImg_recon_config_Am241_2000skips.json', Npeaks=100 )

    data = pd.DataFrame.from_dict({'pcd':pcd})
    data.to_csv("PCD_2000skips_images_new_config.csv",index=False)
    
    # fit the gaussians
    h,r = l.get_th1(pcd, calibration, Nbins, Npeaks, xmax=Npeaks+10)


    # DISTANCE BETWEEN PEAKS
    mu  = np.array(r['mu'])*calibration
    emu = np.array(r['emu'])*calibration
    peak_distance = (mu-mu[0])/r['n_e']

    dg = ROOT.TGraphErrors(
            len(r['n_e'])-1, 
            array('d',r['n_e'][1:]), 
            array('d',peak_distance), 
            array('d',[1]*len(r['n_e'])), 
            array('d',emu)
            )
    cg.GetXaxis().SetTitle("number of electrons")
    cg.GetYaxis().SetTitle("distance between peaks (calibration) [ADU]")
    c = ROOT.TCanvas("cdg")
    dg.Draw("A EP1")
    c.Draw()
    input("press enter ...")

    # CALIBRATION 
    cg = ROOT.TGraphErrors(
            len(r['n_e']), 
            array('d',r['n_e']), 
            array('d',mu), 
            array('d',[1]*len(r['n_e'])), 
            array('d',emu)
            )
    
    fitfunc = fitfunc = ROOT.TF1("fitfunc","pol1")
    fitfunc.SetLineColor(2)
    cg.Fit(fitfunc)
    cg.GetXaxis().SetRangeUser(-1,Npeaks+10)
    cg.GetXaxis().SetTitle("number of electrons")
    cg.GetYaxis().SetTitle("fitted $\mu{center}$ [ADU]")
    c = ROOT.TCanvas("ccg")
    cg.Draw("A EP1")
    c.Draw()
    input("press enter ...")

    # RESIDUALS
    y_fit = np.zeros_like(mu)
    for i,ne in enumerate(r['n_e']):
        y_fit[i] = fitfunc.Eval(ne)
    residuals = y_fit - mu
    rg = ROOT.TGraph(
            len(r['n_e'])-1,
            array('d',r['n_e'][1:]),
            array('d',residuals)
            )
    c = ROOT.TCanvas("crg")
    rg.Draw("A P")
    c.Draw()
    input("press enter ...")




def get_pcd(lof, json, Npeaks=10, skip_start=3, skip_end=-1):
    
    c = dam.utils.config.Config(json, False)

    lof = sorted(glob(lof))

    # compress process
    comp = dam.processes.skipper_analysis.CompressSkipperProcess()
    comp.id_skip_end = skip_end
    comp.id_skip_start = skip_start

    # pedestal process
    ped = dam.processes.skipper_analysis.PedestalSubtractionProcess()
    ped.n_sigma_to_mask = 10
    ped.in_overscan = True
    ped.axis = "row"
    ped.method = "gauss_fit"
    
    pixel_charge = []
    for f in lof:
        # create rawdata object
        rawdata = dam.io.rawdata.BuilderRawData(f, c.configuration['input'])
        rawdata.prepare_data()
        # execute process
        comp.execute_process(rawdata)
        ped.execute_process(rawdata)
        
        pixel_charge.extend( np.ma.array( rawdata.image_mean_compressed_pedestal_subtracted,
            mask=rawdata.mask_image_active_region).compressed() )

    return pixel_charge, rawdata

def get_th1(pcd,calibration,Nbins,Npeaks,xmin=None,xmax=None, draw=True):

    pcd = np.array(pcd)
    pcd = pcd[pcd>-5]

    if xmin is None:
        xmin = min(pcd)
    if xmax is None:
        xmax = max(pcd)
    


    hpcd = ROOT.TH1F("PCD","Pixel Charge Distribution",Nbins,xmin,xmax)
    for qij in pcd:
        hpcd.Fill(qij/calibration)
    
    fgaus = []
    results = { 'mu':[],'emu':[], 'sigma':[], 'esigma':[] , 'norm':[], 'n_e':[]}
    
    mu_center = 0
    for p in range(Npeaks+1):
        xL,xR = mu_center-0.4, mu_center+0.4
        fgaus.append( ROOT.TF1("f_{}e".format(p), "gaus",xL,xR) )
        hpcd.Fit(fgaus[-1],"ER")
        
        results['n_e'].append(p)
        results['norm'].append(fgaus[-1].GetParameter(0))
        results['mu'].append(fgaus[-1].GetParameter(1))
        results['emu'].append(fgaus[-1].GetParError(1))
        results['sigma'].append(fgaus[-1].GetParameter(2))
        results['esigma'].append(fgaus[-1].GetParError(2))

        # next center
        mu_center = results['mu'][-1] + 1
    
    # draw
    if draw:
        c = ROOT.TCanvas()
        hpcd.Draw("HIST")
        for f in fgaus:
            f.SetLineColor(2)
            f.Draw("SAME")

        c.SetLogy()
        c.Draw()
        input("Press enter ... ")

    return hpcd, results

def run_lineality(lof, json, Npeaks=10, skip_start=3, skip_end=-1):

    pcd, rawdata = get_pcd(lof, json, Npeaks, skip_start, skip_end)
    
    # lineality process
    lineality = dam.processes.skipper_comissioning.FitCalibrationConstant()
    lineality.n_peaks = Npeaks
    lineality.__verbose__ = True
    lineality.calibration = 5.18

    lineality.use_pcd = np.array(pcd)
    lineality.execute_process(rawdata)
    

    plt.figure(999, figsize=(12,6))
    distance = np.diff(np.array(lineality.results['mu_e']) * lineality.results['calibration'] )
    plt.scatter(lineality.results['n_e'][1:], distance, s=15, edgecolors='black', color='red')
    plt.xlabel("number of electrons")
    plt.ylabel("distance between consecutive peaks")

    plt.show(block=True)



    return


#####################################
def energy_resolution(lof, json, calibration=5.12,e2eV=3.77):

    c = dam.utils.config.Config(json, False)

    lof = sorted(glob(lof))

    # compress process
    comp = dam.processes.skipper_analysis.CompressSkipperProcess()
    comp.id_skip_start = 3

    # pedestal process
    ped = dam.processes.skipper_analysis.PedestalSubtractionProcess()
    ped.n_sigma_to_mask = 10
    ped.in_overscan = True
    ped.axis = "row"
    ped.method = "gauss_fit"
 
    rdata = []
    rdata64 = []
    for f in lof:
        # create rawdata object
        rawdata = dam.io.rawdata.BuilderRawData(f, c.configuration['input'])
        rawdata.prepare_data()
        # create rawdata object
        rawdata64 = dam.io.rawdata.BuilderRawData(f, c.configuration['input'])
        rawdata64.prepare_data()
        
        # execute for 2000-skips
        comp.id_skip_end = 1999
        comp.execute_process(rawdata)
        ped.execute_process(rawdata)
        rdata.extend( np.ma.array( rawdata.image_mean_compressed_pedestal_subtracted,
            mask=rawdata.mask_image_active_region).compressed() )

        # execute for 64-skips
        comp.id_skip_end = 64
        comp.execute_process(rawdata64)
        ped.execute_process(rawdata64)
        rdata64.extend( np.ma.array( rawdata64.image_mean_compressed_pedestal_subtracted,
            mask=rawdata64.mask_image_active_region).compressed() )
    
    qtrue = np.array(rdata)/calibration*e2eV/1000.
    qrec = np.array(rdata64)/calibration*e2eV/1000.
    
    th2 = ROOT.TH2D("Erecon","recon",1000,qrec.min(),qrec.max(),500,qtrue.min(),qtrue.max())
    for x,y in zip(qrec,qtrue):
        th2.Fill(x,y)

    return th2




