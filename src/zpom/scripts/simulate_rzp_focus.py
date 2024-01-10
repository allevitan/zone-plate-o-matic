import numpy as np
from zpom.optic_simulation import *
import os
import argparse
import h5py
# This fixes a bug when the Qt backed is installed badly. Needed because
# matplotlib is used to do the rasterizing.
# Using a non-interactive backend gets rid of the dependence on X forwarding
import matplotlib
matplotlib.use('Agg')


def main():

    parser = argparse.ArgumentParser(
        prog='simulate-rzp-focus',
        description='Simulates the focus of a randomized zone plate design file')

    parser.add_argument('mask_file', type=str, help='The folder containing the base rzp design masks')
    parser.add_argument('wavelength', type=float, help='The wavelength of light to simulate, in nm')
    parser.add_argument('focal_distance', type=float, help='The focal distance to simulate at, in mm')
    parser.add_argument('step', type=float, help='The pixel step size to simulate, in nm')
    parser.add_argument('n_pix', type=int, help='The diameter of the simulated focal spot window, in pixels')
    parser.add_argument('--tile_size', type=int, default=4096, help='The tile size for loading and processing the files, default is 4096')
    parser.add_argument('--output', '-o', type=str, default=None, help='The filename to be used for the output simulated focus.')
    parser.add_argument('--device', type=str, default='cpu', help='The device to perform the light propagation step on, default is cpu')
    args = parser.parse_args()

    wavelength = args.wavelength * 1e-9 # nm
    focal_distance = args.focal_distance * 1e-3 # mm
    step = args.step * 1e-9 # nm 
    output_shape = (args.n_pix, args.n_pix)
    output_fov = (step * args.n_pix, step * args.n_pix)

    print('Simulating a %0.2f nm pixel size.' % (step * 1e9))
    print('Focal distance: %0.2f mm.' % (focal_distance * 1e3))
    print('Wavelength: %0.2f nm.' % (wavelength * 1e9))
    print('Output shape: %d x %d.' % output_shape)
    print('Output FOV: %0.2f x %0.2f um.' % tuple(d * 1e6 for d in output_fov))
    print('Calculation device:', args.device, flush=True)

    if args.output is None:
        output_folder = ('.'.join(args.mask_file.split('.')[:-1])
                         + '_focal_spots/')
        output = output_folder + 'focaldist%0.3fum.h5' % (focal_distance * 1e6)
        if not os.path.exists(output_folder):
            # It appears that in a HPC environment, if many jobs are launched at
            # the same time, sometimes os.path.exists(output) will return false
            # but the folder gets created before this job tries to create it.
            # So, we also do a try/except
            try:
                os.mkdir(output_folder)
            except:
                pass            

    else:
        output=args.output
    
    mask_file = args.mask_file

    print('Rasterizing .gds file', flush=True)
    rasterized_zp, input_offset = rasterize_zp(mask_file, step, verbose=True)
    print('Rasterized, simulating the focus', flush=True)
    focus = simulate_focus(
        rasterized_zp,
        input_offset,
        focal_distance,
        wavelength,
        step,
        output_shape,
        tile_shape=[args.tile_size, args.tile_size],
        verbose=True,
        calculation_device=args.device)
    print('Focus simulation complete, saving', flush=True)

    with h5py.File(output, 'w') as f:
        f.create_dataset('sim_focus', data=focus)
        f.create_dataset('wavelength', data=[wavelength])
        f.create_dataset('focal_distance', data=[focal_distance])
        f.create_dataset('step', data=[step])
    print('Saved')

if __name__ == '__main__':
    main()

