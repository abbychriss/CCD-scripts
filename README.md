# Privitera_335 scripts

Analysis and DAQ helper scripts for the test-chamber CCD setup.

## Layout

```
scripts/
├── tools/                          # general-purpose CCD / DAQ utilities
├── nonlinearity_studies/           # nonlinearity analysis package
├── radioactive_source_experiment/  # Fe-55 / Am-241 / Geant4 plotting
└── clustering_algs/                # clustering algorithm
```

## tools/

CCD operations, raw-trace decoding, FITS handling, and one-off shell helpers.

| Script | Purpose |
|---|---|
| `trace_plotter.py` | Decode `.RJ45ADCDATA*` binary traces to CSV and plot full / per-row views with pedestal+signal overlays. NROW is read from the `.meta` file or the filename or command line argument. Works with the new `acmdecoder` package or the older `acmpy`. |
| `adc_plotter.py` | Older trace plotter (CSV input only). Includes PSD / CDS calculations. |
| `oscilloscope.py` | Take data from Rohde & Schwarz oscilloscope over VISA and write csv output. |
| `plot_oscilloscope.py` | Plot saved oscilloscope traces. |
| `set_voltages.py` | Push voltages to the ACM/CDAQ. |
| `set_VR_take_image.py` | Sweep VR voltages and take images. |
| `take_n_images.py` | Take N images via `ccd_cdaq`. |
| `stitch_fits.py` | Combine multi-amp FITS frames into one image. |
| `combine_fits.py` | Merge / coadd FITS files. |
| `plot_data_heatmap.py` | 2-D heatmap of image data. |
| `plot_charge_per_column.py` | Mean charge vs column number. |
| `astropy-fits-tutorial.py` | Reference snippets for FITS I/O. |
| `rsync_acm.sh` | Pull data off the ACM machine for a given YYYY-MM-DD run. |
| `pdf-to-jpeg.sh` | Convert panaSKImg PDF output to JPEG. |
| `run_panaSKImg*.sh` | Wrappers for the panaSKImg processing chain (single / combo / v2 variants). |

### Most-used: `trace_plotter.py`

```bash
# full trace
python trace_plotter.py file.RJ45ADCDATA0

# single row, ±1000 µs around the integration region
python trace_plotter.py file.RJ45ADCDATA0 --row 3 --window 1000

# arbitrary time window
python trace_plotter.py file.RJ45ADCDATA0 --tmin 200000 --tmax 220000

# also save CSV / JPEG outputs
python trace_plotter.py file.RJ45ADCDATA0 --save_csv --save_plot

# list detected row boundaries
python trace_plotter.py file.RJ45ADCDATA0 --list_rows
```

## nonlinearity_studies/

Package for the CCD nonlinearity analysis. See its own `PACKAGE_SETUP.md`.

## radioactive_source_experiment/

Plotting scripts for Fe-55, Am-241, and Geant4-simulated source spectra.

## clustering_algs/

Cluster finding algorithms for FITS images. Tested multiple algorirthms from scikit-learn.

## Environments


Scripts that require `acmpy` such as `trace_plotter.py` and `adc_plotter.py` must be run in the `acm-env` conda environment (which has `acmdecoder`, `astropy`, etc.):

```bash
conda activate acm-env
```

Scripts that require WADERS need the `waders-env` conda environment:

```bash
conda activate waders-env
```

All other scripts can be run from the `uchicago-env` conda environment:

```bash
conda activate uchicago-env
```