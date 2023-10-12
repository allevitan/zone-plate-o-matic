import torch as t
import numpy as np
import argparse
import os
import shutil
import gdstk

def main():

    parser = argparse.ArgumentParser(
        prog='collate_sunflower_design',
        description='Packs all the lenslets of a sunflower style zone plate into a single .gds file.')


    parser.add_argument('input_folder', type=str, help='The base rzp design folder, which should contain the mask folder inside it.')
    parser.add_argument('--output', '-o', type=str, help='The output folder nameto be used for the output gdsii and low-resolution mask files. Default is "full_mask.gds".')
    args = parser.parse_args()

    args.input_folder = args.input_folder.rstrip('/ ')
    if args.output is None:
        output = args.input_folder + '/full_mask.gds'
    else:
        output = args.output

    gds_folder = args.input_folder + '/masks/'

    file_list = [f for f in os.listdir(gds_folder)
                 if f[-4:] == '.gds']
    file_list = sorted(file_list)

    print('Combining gds files into full mask')
    for id, gds_file in enumerate(file_list):
        print('Working on ZP file', id, 'of', len(file_list), flush=True)
        print(gds_file)
        lib = gdstk.read_gds(gds_folder+gds_file)
        if id==0:
            # We do this here to get access to the precision
            full_lib = gdstk.Library(unit=1e-6,precision=lib.precision)
            full_cell = full_lib.new_cell('RZP')

        for polygon in lib.cells[0].get_polygons():
            polygon.layer = id
            full_cell.add(polygon)

    full_lib.write_gds(output)


if __name__ == '__main__':
    main()
