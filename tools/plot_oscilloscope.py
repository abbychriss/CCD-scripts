#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pathlib import Path

def plot_csv(fname, title=None, labels=None, save_plot=True, outname=None):

    df = pd.read_csv(fname)
    time = df['time']

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.grid(alpha=0.7)

    data_cols = [col for col in df.columns if col != 'time']

    for i, col in enumerate(data_cols):
        label = labels[i] if labels and i < len(labels) else col
        ax.plot(time, df[col], label=label)  #s=2, 

    #ax.set_xlim(0, 0.005)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage (V)')
    ax.legend()
    if title is not None:
        fig.suptitle(title)

    if save_plot:
        if outname==None:
            outname = Path(fname).stem
            plt.savefig(f'{outname}.jpeg', dpi=350)
        else:
            plt.savefig(f'{outname}.jpeg', dpi=350)

    plt.show()
    return fig, ax


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fname", type=str, required=True,
                        help="file name (include .csv)")
    parser.add_argument("-t", "--title", type=str,
                        help="title of figure")
    parser.add_argument("-l", "--labels", type=str, nargs="+",
                        help="labels for legend")
    parser.add_argument("-s", "--save_plot", action="store_true",
                        help="save plot as jpeg")
    parser.add_argument("-o", "--outname", type=str,
                        help="name of outfile for plot")

    args = parser.parse_args()

    plot_csv(args.fname, args.title, args.labels, args.save_plot, args.outname)