#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from scipy.optimize import curve_fit

def exponential(x, a, b, c):
    return a * np.exp(- x / b) + c

def plot_csv(fname, plot_type='line', title=None, labels=None, save_plot=True, outname=None):

    channel_data = pd.read_csv(fname) # Pandas dataframe of csv file
    time = channel_data['time']

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.grid(alpha=0.7)

    data_cols = [col for col in channel_data.columns if col != 'time']

    for i, col in enumerate(data_cols):
        label = labels[i] if labels and i < len(labels) else col
        if plot_type=='scatter':
            ax.scatter(time, channel_data[col],  s=2, label=label)
        else:
            ax.plot(time, channel_data[col], label=label)

    # Find by eye time window (in seconds) that V2 or H2 is changing from plot
    t_start, t_end = 0.00176, 0.003317
    # Find indices of closest times in time column
    idx_start = np.argmin(np.abs(channel_data['time'] - t_start))
    idx_end   = np.argmin(np.abs(channel_data['time'] - t_end))

    voltage_change_time_window = channel_data['time'].iloc[idx_start:idx_end] - t_start
    clock_voltage_change_window = channel_data['ch2'][idx_start:idx_end]

    # Make exponential fit to clock voltage change curve in time window
    popt, pcov = curve_fit(exponential, voltage_change_time_window, clock_voltage_change_window, maxfev = 2000, bounds=([-10, 0, 0], [10, 0.1, 10]))

    t = np.linspace(t_start, t_end, 10000)
    plt.plot(t, exponential((t - t_start), *popt), 'r--',
            label='y(t) = %5.3fexp(- t / %5.6f) + %5.3f' % tuple(popt))

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage (V)')
    
    """ticks = ax.get_xticks()
    ax.set_xticks(ticks)  # fixes the locator first
    ax.set_xticklabels([f'{tick - t_start:.3f}' for tick in ax.get_xticks()]) # Artificially shift t=0 to t_start"""
    ax.set_xlim(0,0.0036)
    ax.set_ylim(5.9,11.2)
    ax.legend(loc='lower right', bbox_to_anchor=(0.84, 0.8))

    if title is not None:
        fig.suptitle(title)

    if save_plot:
        if outname==None:
            fname = Path(fname).expanduser()
            outname = Path(fname).parent.parent / 'plots' / Path(fname).with_suffix('.jpeg').name
            plt.savefig(f'{outname}', dpi=350)
        else:
            outname = Path(outname).expanduser()
            plt.savefig(f'{outname}', dpi=350)

    plt.show()

    return fig, ax


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Script to plot csv data for oscilloscope measurements. 
                                 Usage: ./plot_oscilloscope.py '~/Privitera_335/oscilloscope_tests/csv/clock_ccd_sw_h2_h3_defaultV_H2_C_L_6.7_7.85_100skip_0.01sec.csv' -t 'clock_ccd: H2_C_L = (6.7, 7.85), all other voltages default, 100 skips' -l 'SW_2' 'H2_A' 'H3_B' -s
                                 ''')
    parser.add_argument("fname", type=str,
                        help="file name (include .csv)")
    parser.add_argument("--plot_type", type=str,
                        help="options: 'line' or 'scatter'", default='line')
    parser.add_argument("-t", "--title", type=str,
                        help="title of figure")
    parser.add_argument("-l", "--labels", type=str, nargs="+",
                        help="labels for legend")
    parser.add_argument("-s", "--save_plot", action="store_true",
                        help="save plot as jpeg")
    parser.add_argument("-o", "--outname", type=str,
                        help="name of outfile for jpeg plot, if None uses FNAME")

    args = parser.parse_args()

    plot_csv(args.fname, args.plot_type, args.title, args.labels, args.save_plot, args.outname)