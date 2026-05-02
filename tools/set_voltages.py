import argparse
from ccd_cdaq import add2queue, expconfig
import libABCD

libABCD.init('set_voltages_script', expconfig=expconfig)

parser = argparse.ArgumentParser(description='Script to change voltages ')
parser.add_argument('acmid',
                    help='e.g. acm124 or acm125')
args = parser.parse_args()

voltages = {"SW_1_H": 1.0,
            "SW_1_L": -9.0,
            "OG_1_H": 0.0,
            "OG_1_L": -8.0,
            "RG_1_H": 9.0,
            "RG_1_L": 3.0,
            "DG_1_H": 0.0,
            "DG_1_L": -7.0,
            "SW_2_H": 1.0,
            "SW_2_L": -9.0,
            "OG_2_H": 0.0,
            "OG_2_L": -8.0,
            "RG_2_H": 9.0,
            "RG_2_L": 3.0,
            "DG_2_H": 0.0,
            "DG_2_L": -7.0,
            "V1_B_H": 9.9,
            "V1_B_L": 7.7,
            "V2_C_H": 9.9,
            "V2_C_L": 7.9,
            "V3_B_H": 9.9,
            "V3_B_L": 7.7,
            "TG_B_H": 7.4,
            "TG_B_L": 7.4,
            "V1_A_H": 9.9,
            "V1_A_L": 7.9,
            "V3_A_H": 9.9,
            "V3_A_L": 7.9,
            "TG_A_H": 7.4,
            "TG_A_L": 7.4,
            "H1_B_H": 9.5,
            "H1_B_L": 6.5,
            "H2_C_H": 9.0,
            "H2_C_L": 6.7,
            "H3_B_H": 9.5,
            "H3_B_L": 6.5,
            "H1_A_H": 9.5,
            "H1_A_L": 6.7,
            "H3_A_H": 9.5,
            "H3_A_L": 6.7,
            "VR1": -6.0,
            "VR2": -6.0
            }

for key, value in voltages.items():
    add2queue(f'daq[{key}] = {value}', target=args.acmid)

# end the communication
libABCD.disconnect()