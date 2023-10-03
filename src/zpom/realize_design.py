
import torch as t
import numpy as np
from zpom.optic_design import *
import argparse
import os

def main():

    parser = argparse.ArgumentParser(
        prog='realize_rzp_design',
        description='Converts a base RZP design file into a vectorized gdsii file.')


    parser.add_argument('base_file', type=str, help='The base rzp design file')
    parser.add_argument('grating_level', type=float, help='The fraction of the max amplitude to saturate at. Setting this to 1 provides best quality, at the expense of efficiency. 0.6 is usually a reasonable middle ground, and 0 produces a zone plate without any amplitude variation.')
    parser.add_argument('buttress_width', type=float, help='The width of the buttresses, in nm. 0 will produce no buttresses')
    parser.add_argument('--n_processes', '-n', type=int, default=1, help='The number of simultaneous processes to run, default=1')
    parser.add_argument('--chunk_size', type=int, default=4096, help='The chunk size for loading and processing the files, default is 4096')
    parser.add_argument('--output', '-o', type=str, help='The output filename base (without the .gds or .h5 extension) to be used for the output gdsii and low-resolution mask file. There is a sensible default')
    args = parser.parse_args()

    if args.output is None:
        output = ('.'.join(args.base_file.split('.')[:-1])
                  + '_GL=%0.2f_BW=%0.2fnm' % (args.grating_level,
                                              args.buttress_width))
    else:
        output=args.output
    gds_file = output + '.gds'


    base_file_size = os.path.getsize(args.base_file) * 1e-9 # in GB
    estimated_LR_size = base_file_size* 4 / (3 * args.reduction_factor**2)
    
    with h5py.File(args.base_file,'r') as f:
        dr = float(f['dr'][()])
        step = float(f['step'][()])

    buttress_width = args.buttress_width*1e-9

    print('Outer zone width %0.3f nm' % (1e9 * dr))    
    print('Reduced resolution pixel size: %0.3f nm'
          % (step * args.reduction_factor * 1e9))
    print('Estimated uncompressed reduced resolution file size: %0.3f GB' %
          estimated_LR_size)
    print('Buttress width: %0.3f nm' % (buttress_width * 1e9))
    print('Grating Level: %0.3f' % args.grating_level)
    print('')
    print('Starting Calculation')


    realize_design(args.base_file, gds_file,
                   buttress_width=buttress_width,
                   grating_max=args.grating_level,
                   n_processes=args.n_processes,
                   chunk_size=args.chunk_size,
                   verbose=True, view=False)


if __name__ == '__main__':
    main()
