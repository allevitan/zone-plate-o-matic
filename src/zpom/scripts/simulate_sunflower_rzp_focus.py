import numpy as np
from zpom.optic_simulation import *
import os

def main():

    parser = argparse.ArgumentParser(
        prog='simulate-sunflower-rzp-focus',
        description='Simulates the monochromatic focal spot from a single zone plate in a sunflower array')

    parser.add_argument('mask_file', type=str, help='The base rzp design mask')
    parser.add_argument('wavelength', type=float, help='The wavelength of light to simulate, in nm')
    parser.add_argument('focal_distance', type=float, help='The focal distance to simulate at, in mm')
    parser.add_argument('step', type=float, help='The pixel step size to simulate, in nm')
    parser.add_argument('n_pix', )
    parser.add_argument('--zone_plate_index', '-n', type=int, default=None, help='The index of the zone plate to design within the full array. Default is all zone plates.')
    parser.add_argument('--tile_size', type=int, default=4096, help='The tile size for loading and processing the files, default is 4096')
    parser.add_argument('--output', '-o', type=str, default=None, help='The output folder nameto be used for the output gdsii and low-resolution mask files. There is a sensible default')
    args = parser.parse_args()

    wavelength = args.wavelength * 1e-9 # nm
    focal_distance = args.focal_distance * 1e-3 # mm
    step = args.step * 1e-9 # nm 
    output_shape = [args.n_pix, args.n_pix]

    if args.output is None:
        output = (args.mask_file[:4] + '_focus_sim.h5')
    else:
        output=args.output
    
    rasterized_zp, input_offset = rasterize_zp(args.mask_file, step)
    focus = simulate_focus(
        rasterized_zp,
        input_offset,
        focal_distance,
        wavelength,
        step,
        output_shape,
        tile_shape=[2048,2048])

    with h5py.File(output, 'w') as f:
        f.create_dataset('sim_focus', data=focus)
        f.create_dataset('wavelength', data=[wavelength])
        f.create_dataset('focal_distance', data=[focal_distance])
        f.create_dataset('step', data=[step])


if __name__ == '__main__':
    main()
