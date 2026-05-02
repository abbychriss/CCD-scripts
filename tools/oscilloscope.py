#!/usr/bin/env python3
from RsInstrument import *
from time import time, sleep
import numpy as np
import pandas as pd
from tqdm import tqdm
import threading
from plot_oscilloscope import plot_csv
import argparse

#!/usr/bin/env python3
from RsInstrument import *
from time import time, sleep
import numpy as np
import pandas as pd
from tqdm import tqdm
import threading
from plot_oscilloscope import plot_csv
import argparse
import json
import os

# Script for acquiring multi channel data from Rohde & Schwarz RTB2004 oscilloscope and writing out data as csv file

DEFAULT_CONFIG = 'oscilloscope_config.json'

def load_json(path):
    with open(path) as f:
        return json.load(f)

def connect_instrument(resource_str):
    instrument_name = RsInstrument(resource_str, True, False)
    print(f'IDN: {instrument_name.idn_string}')
    return instrument_name

def set_visa_timeout(instrument_name, acquisition_time_s):
    instrument_name.visa_timeout = acquisition_time_s * 2000 + 10000
    instrument_name.opc_timeout = acquisition_time_s * 2000 + 10000
    print(f"Setting visa and opc timeout to {acquisition_time_s} seconds")
    instrument_name.instrument_status_checking = True
    instrument_name.clear_status()
    instrument_name.reset()

def set_acquisition_time(instrument_name, acquisition_time_s):
    instrument_name.write_str(f"TIM:ACQT {acquisition_time_s}")

# Horizontal (time) scale in seconds/div
def set_horizontal_scale(instrument_name, hscale):
    instrument_name.write_str(f'TIM:SCAL {hscale}')

def set_channel_settings(instrument_name, channel_number, range=2.0, offset=0.0, position=0.0):
    instrument_name.write_str(f"CHAN{channel_number}:RANG {range*10}")
    instrument_name.write_str(f"CHAN{channel_number}:OFFS {offset}")
    instrument_name.write_str(f"CHAN{channel_number}:POS {position}")
    instrument_name.write_str(f"CHAN{channel_number}:COUP DCL")
    instrument_name.write_str(f"CHAN{channel_number}:STAT ON")
    instrument_name.query_opc()

# Slope can be NEG or POS depending on whether you want falling or rising trigger edge
def set_trigger(instrument_name, source_channel, trigger_level, trigger_slope='NEG'):
    instrument_name.write_str("TRIG:A:MODE AUTO")
    instrument_name.write_str(f"TRIG:A:TYPE EDGE;:TRIG:A:EDGE:SLOP {trigger_slope}")
    instrument_name.write_str(f"TRIG:A:SOUR CH{source_channel}")
    instrument_name.write_str(f"TRIG:A:LEV{source_channel} {trigger_level}")
    instrument_name.query_opc()

def initiate_single(instrument_name, acquisition_time_s):
    """Arm for a single acquisition, blocking until complete, with a progress bar."""
    timeout_ms = int(acquisition_time_s * 1000) + 10000

    # Progress bar runs in a background thread, ticking every 0.1s
    stop_event = threading.Event()

    def _progress():
        with tqdm(total=acquisition_time_s, desc="Acquiring", unit="s", ncols=70,
                  bar_format="{l_bar}{bar}| {n:.1f}/{total:.0f}s [{elapsed}]") as pbar:
            t0 = time()
            last = 0.0
            while not stop_event.is_set():
                elapsed = min(time() - t0, acquisition_time_s)
                pbar.update(elapsed - last)
                last = elapsed
                sleep(0.1)
            # Fill to 100% on completion
            pbar.update(acquisition_time_s - last)

    t = threading.Thread(target=_progress, daemon=True)
    t.start()

    try:
        instrument_name.write_str_with_opc("SING", timeout_ms)
    finally:
        stop_event.set()
        t.join()

def get_time_axis(instrument_name, num_points, acquisition_time_s):
    """Build a time axis using the scope's actual sample rate."""
    try:
        sample_rate = float(instrument_name.query_str("ACQ:SRAT?"))
        print(f"Sample rate: {sample_rate:.3e} Sa/s")
        dt = 1.0 / sample_rate
        t = np.arange(num_points) * dt
    except Exception:
        # Fallback: evenly space points across the acquisition window
        print("Warning: could not query sample rate, using linspace fallback")
        t = np.linspace(0, acquisition_time_s, num_points)
    return t

def query_data(instrument_name, channel_number):
    t = time()
    instrument_name.bin_float_numbers_format = BinFloatFormat.Single_4bytes_swapped
    instrument_name.write_str("FORM REAL,32")
    instrument_name.query_opc()
    trace = instrument_name.query_bin_or_ascii_float_list(f'CHAN{channel_number}:DATA?')
    print(f'Channel {channel_number}: {len(trace)} points in {time() - t:.3f}s')
    return trace

def write_csv(time_axis, traces, channel_numbers, outname):
    """Save time + channel traces as columns using pandas."""
    # Build dictionary for DataFrame
    data = {'time': time_axis}
    for ch, trace in zip(channel_numbers, traces):
        data[f'ch{ch}'] = trace

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(outname, index=False)

    print(f'Saved to {outname} ({df.shape[0]} rows, {df.shape[1]} cols)')

def take_screenshot(instrument_name):
    instrument_name.write_str("MMEM:CDIR '/INT/'")
    instrument_name.InstrumentStatusChecking = False
    instrument_name.write_str("MMEM:DEL 'Dev_Screenshot.png'")
    instrument_name.query_opc()
    instrument_name.clear_status()
    instrument_name.InstrumentStatusChecking = True
    instrument_name.write_str("HCOP:LANG PNG;:MMEM:NAME 'Dev_Screenshot'")
    instrument_name.write_str("HCOP:IMM")
    instrument_name.query_opc()
    instrument_name.read_file_from_instrument_to_pc(r'Dev_Screenshot.png', r'Screenshot.png')
    print("Screenshot saved to Screenshot.png")

def close_session(instrument_name):
    instrument_name.close()


def main(args):

    resource_str = args.resource_str
    channels = args.channels
    voltage_ranges = args.voltage_ranges
    offsets = args.offsets
    positions = args.positions
    acquisition_time_s = args.acquisition_time_s
    fname = args.fname

    rtb = connect_instrument(resource_str)

    set_visa_timeout(rtb, acquisition_time_s)
    set_acquisition_time(rtb, acquisition_time_s)
    #set_horizontal_scale(rtb, 1e-3)

    # Configure all channel setting before triggering acquisition.
    # Enabling a channel mid-acquisition causes the scope to restart its sweep which drops the connection.
    for i, chann in enumerate(channels):
        set_channel_settings(rtb, chann, voltage_ranges[i], offsets[i], positions[i])

    # Trigger for epurge: source horizontal clock (ch3), level 8V, falling slope (NEG)
    # Trigger for erase: source vsub (ch2), level 42, falling slope (NEG)
    set_trigger(rtb, 3, 6, 'NEG')

    # Arm for a single acquisition and block until complete (with progress bar)
    initiate_single(rtb, acquisition_time_s)

    # Read each channel only after acquisition is finished
    traces = []
    ranges=[]
    for ch in channels:
        trace = query_data(rtb, ch)
        range = rtb.query_str(f"CHAN{ch}:RANG?")
        traces.append(trace)
        ranges.append(range)

    # Build time axis from actual sample rate
    time_axis = get_time_axis(rtb, len(traces[0]), acquisition_time_s)

    write_csv(time_axis, traces, channels, fname)

    close_session(rtb)

    if args.plot:
        plot_csv(args.fname, args.title, args.labels, args.save_plot, args.plot_name)

    #plot_csv(outname, title='clock_ccd: default voltages, 100 skips', labels=['SW_2', 'H2_A', 'H3_B'], save_plot=True)
    return range


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # --- Config file args ---
    parser.add_argument("-j", "--json", type=str, default=None,
                        help=f"Path to a JSON config file to override defaults in {DEFAULT_CONFIG}")

    # --- All original args (all default=None so they don't interfere with JSON values) ---
    parser.add_argument("-f", "--fname", type=str, default=None,
                        help="output csv file name (include .csv)")
    parser.add_argument("--resource_str", type=str, default=None)
    parser.add_argument("--channels", type=int, nargs="+", default=None,
                        help="oscilloscope channels to acquire data from (must be 1, 2, 3, or 4, e.g: 1 2 3)")
    parser.add_argument("--voltage_ranges", type=float, nargs="+", default=None,
                        help="voltage range for each channel in V/div, nargs must be same as number of channels")
    parser.add_argument("--offsets", type=float, nargs="+", default=None,
                        help="voltage offsets for each channel, nargs must be same as number of channels")
    parser.add_argument("--positions", type=float, nargs="+", default=None,
                        help="voltage positions for each channel, nargs must be same as number of channels, must be within voltage range")
    parser.add_argument("--acquisition_time_s", type=float, default=None,
                        help="time in seconds for data acquisition")
    parser.add_argument("--plot", action="store_true", default=None,
                        help="plot oscilloscope data")
    parser.add_argument("-t", "--title", type=str, default=None,
                        help="title of figure")
    parser.add_argument("-l", "--labels", type=str, nargs="+", default=None,
                        help="labels for legend")
    parser.add_argument("-s", "--save_plot", action="store_true", default=None,
                        help="save plot as jpeg")
    parser.add_argument("-o", "--plot_name", type=str, default=None,
                        help="name of outfile for plot")

    cli_args = parser.parse_args()

    # --- Build final config with 3-layer priority ---

    # Layer 1: hardcoded fallback defaults
    fallback_defaults = {
        "resource_str": "TCPIP::localhost::5025::SOCKET",
        "channels": [1, 2, 3, 4],
        "voltage_ranges": [5, 5, 5, 5],
        "offsets": [0, 0, 0, 0],
        "positions": [0, 0, 0, 0],
        "acquisition_time_s": 1,
        "plot": False,
        "title": None,
        "labels": None,
        "save_plot": False,
        "plot_name": None,
        "fname": None,
    }

    # Layer 2: default JSON config file (if it exists)
    config = fallback_defaults.copy()
    if os.path.exists(DEFAULT_CONFIG):
        config.update(load_json(DEFAULT_CONFIG))
    else:
        print(f"Note: no {DEFAULT_CONFIG} found, using hardcoded defaults")

    # Layer 3: user-supplied --config file
    if cli_args.config:
        config.update(load_json(cli_args.config))

    # Layer 4: explicit CLI flags override everything
    cli_overrides = {k: v for k, v in vars(cli_args).items()
                     if k != 'config' and v is not None}
    config.update(cli_overrides)

    # Validate required arg
    if config.get('fname') is None:
        parser.error("argument -f/--fname is required (either via CLI or config file)")

    # Convert to namespace so main() works unchanged
    args = argparse.Namespace(**config)
    main(args)


"""Notes: 
erase_ccd/epurge_ccd probe channels: ch2 (green) = VSUB (pin 60), ch3 (orange) = H2_A (pin 38), ch4 (blue) = V2_B (pin 28)
clock_ccd('AcquireImage') probe channels: ch1 (yellow) = SW_2 (pin 47), ch2 (green) = H2_A, ch3 (orange) = H3_B (pin 35)
clock_ccd('AcquireImage') HORIZONTAL probe channels: ch2 (green) = H2_A, ch3 (orange) = H3_A, ch4 (blue) = H1_A
clock_ccd('AcquireImage') VERTICAL probe channels: ch2 (green) = V2_A, ch3 (orange) = V3_A, ch4 (blue) = V1_B
"""
