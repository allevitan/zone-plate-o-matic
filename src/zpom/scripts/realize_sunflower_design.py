
import torch as t
import numpy as np
from zpom.optic_design import *
import argparse
import os
import shutil
import gdstk

def main():

    parser = argparse.ArgumentParser(
        prog='realize_rzp_design',
        description='Converts a base sunflowe RZP design directory into a vectorized gdsii file.')


    parser.add_argument('base_folder', type=str, help='The base rzp design folder')
    parser.add_argument('grating_level', type=float, help='The fraction of the max amplitude to saturate at. Setting this to 1 provides best quality, at the expense of efficiency. 0.6 is usually a reasonable middle ground, and 0 produces a zone plate without any amplitude variation.')
    parser.add_argument('buttress_width', type=float, help='The width of the buttresses, in nm. 0 will produce no buttresses')
    parser.add_argument('--zone_plate_index', '-n', type=int, default=None, help='The index of the zone plate to design within the full array. Default is all zone plates.')
    parser.add_argument('--n_processes', '-np', type=int, default=1, help='The number of simultaneous processes to run, default=1')
    parser.add_argument('--chunk_size', type=int, default=4096, help='The chunk size for loading and processing the files, default is 4096')
    parser.add_argument('--output', '-o', type=str, help='The output folder nameto be used for the output gdsii and low-resolution mask files. There is a sensible default')
    args = parser.parse_args()

    args.base_folder = args.base_folder.rstrip('/ ')
    if args.output is None:
        output = (args.base_folder[:-20] + '_GL=%0.2f_BW=%0.2fnm'
                  % (args.grating_level, args.buttress_width))
    else:
        output=args.output

    if not os.path.exists(output):
        # It appears that in a HPC environment, if many jobs are launched at
        # the same time, sometimes os.path.exists(output) will return false
        # when in reality the folder exists. So, we also do a try/except here
        try:
            os.mkdir(output)
        except FileExistsError():
            pass

    if not os.path.isdir(output):
        raise FileExistsError('Output folder already exists but is not a directory')

    full_gds_file = output + '/full_mask.gds'
    gds_folder = output + '/masks/'
    if not os.path.exists(gds_folder):
        os.mkdir(gds_folder)

    if not os.path.isdir(gds_folder):
        raise FileExistsError('Mask folder already exists but is not a directory')

    file_list = [f for f in os.listdir(args.base_folder)
                 if f[-3:] == '.h5']
    file_list = sorted(file_list)

    for id, base_file in enumerate(file_list):
        if args.zone_plate_index is not None:
            if id != args.zone_plate_index:
                continue

        print('Working on mini ZP %d of %d (%s)'
              % (id+1, len(file_list), base_file))
        
        base_file_path = args.base_folder + '/' + base_file
        buttress_width = args.buttress_width*1e-9

        print('Buttress width: %0.3f nm' % (buttress_width * 1e9))
        print('Grating Level: %0.3f' % args.grating_level)
        print('')

        gds_file = gds_folder + '.'.join(base_file.split('.')[:-1]) + '.gds'

        realize_design(base_file_path,
                       gds_file,
                       buttress_width=buttress_width,
                       grating_max=args.grating_level,
                       n_processes=args.n_processes,
                       chunk_size=args.chunk_size,
                       verbose=True, view=False)

        

if __name__ == '__main__':
    main()
