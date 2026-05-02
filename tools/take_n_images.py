from scriptutils import get_image_params
import argparse
from ccd_cdaq import add2queue, expconfig
import libABCD


parser = argparse.ArgumentParser(description='''Script to take n images, n specified by user \n
                                 Usage: python take_n_images.py 'VR_img_4hrs' -t acm125 -n 10 --erase --epurge
                                 ''')
parser.add_argument('img_name',
                    help='The name of the image as defined in image_params.toml')
parser.add_argument('-t', '--to', default='acmpool',
                    help='acmid, e.g. acm124 or acm125')
parser.add_argument('-n', '--nimages',
                    help='number of images to take, defaults to 1')
parser.add_argument('--erase', action='store_true', default=False,
                    help='if flag present, erase procedure is done between each image')
parser.add_argument('--epurge', action='store_true', default=False,
                    help='if flag present, epurge procedure is done within erase')

args = parser.parse_args()


p = get_image_params(args.img_name)
img_kw = ', '.join(f"{k}={repr(v)}" if type(v) is str else f"{k}={v}" for k, v in p.items())

FNAME = f"img_L{p['level']}"
if p['subroutine'] == 'AcquireImage':
    pass
elif p['subroutine'] == 'AcquireRegisterImage':
    FNAME += "_SR"
elif p['subroutine'] == 'AcquireImageDumpless':
    FNAME += "_DL"


# initialize communication
libABCD.init('take_n_images_script', expconfig=expconfig)

# Take n images
for i in range(args.nimages):

    # only do erase + epurge process if taking more than one image
    if args.nimages > 1:
        if args.erase:
            # This erase and epurge loop is copied exactly from erase.py
            for j in range(2):
                # Erase
                add2queue('daq.erase_ccd(highP=9.9, t_0V = 2)', target=args.acmid, queue_type='simple')
                add2queue('daq.flush()', target=args.acmid, queue_type='simple')
                add2queue('daq.flush()', target=args.acmid, queue_type='simple')

                # Epurge
                if args.epurge:
                    add2queue('daq.epurge_ccd(lowP=-9., t_0V = 1)', target=args.acmid, queue_type='simple')
                    add2queue('daq.flush()', target=args.acmid, queue_type='simple')
                    add2queue('daq.flush()', target=args.acmid, queue_type='simple')
        
        # Do another 10 flushes to clear all charge
        for j in range(10):
            add2queue('daq.flush()', target=args.acmid, queue_type='simple')
    
    add2queue(f'daq.take_image(fname="{FNAME}", {img_kw})', target=args.to, queue_type='simple')

# end the communication
libABCD.disconnect()