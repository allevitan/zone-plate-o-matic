
import torch as t
import numpy as np
from zpom.optic_design import *
import argparse
import os
import gdstk

def main():

    parser = argparse.ArgumentParser(
        prog='realize_rzp_design',
        description='Converts a base RZP design file into a vectorized gdsii file and a low-resolution mask file for later simulation')


    parser.add_argument('base_folder', type=str, help='The base rzp design folder')
    parser.add_argument('grating_level', type=float, help='The fraction of the max amplitude to saturate at. Setting this to 1 provides best quality, at the expense of efficiency. 0.6 is usually a reasonable middle ground, and 0 produces a zone plate without any amplitude variation.')
    parser.add_argument('buttress_width', type=float, help='The width of the buttresses, in nm. 0 will produce no buttresses')
    parser.add_argument('reduction_factor', type=int, help='The number of pixels to bin into a single pixel for the low resolution mask output')
    parser.add_argument('--n_processes', '-n', type=int, default=1, help='The number of simultaneous processes to run, default=1')
    parser.add_argument('--chunk_size', type=int, default=4096, help='The chunk size for loading and processing the files, default is 4096')
    parser.add_argument('--output', '-o', type=str, help='The output folder nameto be used for the output gdsii and low-resolution mask files. There is a sensible default')
    args = parser.parse_args()

    if args.output is None:
        output = (args.base_folder + '_GL=%0.2f_BW=%0.2fnm'
                  % (args.grating_level, args.buttress_width))
    else:
        output=args.output

    os.mkdir(output)
    lr_folder = output + '/lr_images/'
    os.mkdir(lr_folder)
    full_gds_file = output + '/full_mask.gds'
    gds_folder = output + '/masks/'
    os.mkdir(gds_folder)

    file_list = [f for f in os.listdir(args.base_folder)
                 if f[-3:] == '.h5']
    file_list = sorted(file_list)

    for id, base_file in enumerate(file_list):

        print('Working on mini ZP %d of %d (%s)'
              % (id+1, len(file_list), base_file))
        
        base_file_path = args.base_folder + '/' + base_file
        base_file_size = os.path.getsize(base_file_path) * 1e-9 # in GB
        estimated_LR_size = base_file_size * 4 / (3 * args.reduction_factor**2)

        buttress_width = args.buttress_width*1e-9

        print('Estimated uncompressed reduced resolution file size: %0.3f GB' %
              estimated_LR_size)
        print('Buttress width: %0.3f nm' % (buttress_width * 1e9))
        print('Grating Level: %0.3f' % args.grating_level)
        print('')

        lr_file = lr_folder + '.'.join(base_file.split('.')[:-1]) + '_lr.h5'
        gds_file = gds_folder + '.'.join(base_file.split('.')[:-1]) + '.gds'

        realize_design(base_file_path,
                       lr_file,
                       gds_file,
                       buttress_width=buttress_width,
                       grating_max=args.grating_level,
                       n_processes=args.n_processes,
                       chunk_size=args.chunk_size,
                       reduction_factor=args.reduction_factor,
                       verbose=True, view=False)

    print('Combining gds files into full mask')
    for id, base_file in enumerate(file_list):
        gds_file = gds_folder + '.'.join(base_file.split('.')[:-1]) + '.gds'

        lib = gdstk.read_gds(gds_file)
        if id==0:
            # We do this here to get access to the precision
            full_lib = gdstk.Library(unit=1e-6,precision=lib.precision)
            full_cell = full_lib.new_cell('RZP')

        for polygon in lib.cells[0].get_polygons():
            polygon.layer = id
            full_cell.add(polygon)
    full_lib.write_gds(full_gds_file)
        

if __name__ == '__main__':
    main()
