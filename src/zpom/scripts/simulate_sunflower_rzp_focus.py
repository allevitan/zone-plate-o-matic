import numpy as np
from zpom.optic_simulation import *
import os
import argparse
import h5py

import matplotlib
matplotlib.use('Agg')

def main():

    parser = argparse.ArgumentParser(
        prog='simulate-sunflower-rzp-focus',
        description='Simulates the monochromatic focal spot from a single zone plate in a sunflower array')

    parser.add_argument('mask_folder', type=str, help='The folder containing the base rzp design masks')
    parser.add_argument('wavelength', type=float, help='The wavelength of light to simulate, in nm')
    parser.add_argument('focal_distance', type=float, help='The focal distance to simulate at, in mm')
    parser.add_argument('step', type=float, help='The pixel step size to simulate, in nm')
    parser.add_argument('n_pix', type=int, help='The diameter of the simulated focal spot window, in pixels')
    parser.add_argument('--zone_plate_index', '-n', type=int, default=None, help='The index of the zone plate to design within the full array. Default is all zone plates.')
    parser.add_argument('--tile_size', type=int, default=4096, help='The tile size for loading and processing the files, default is 4096')
    parser.add_argument('--output', '-o', type=str, default=None, help='The folder name to be used for the output simulated focal spot.')
    parser.add_argument('--cache-raster', action='store_true', help='If set, the rasterized optic will be saved')
    parser.add_argument('--ignore-cache', action='store_true', help='If set, will force a re-rasterization of the optic, even if a cached version exists')
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

    if args.zone_plate_index is None:
        raise NotImplementedError()

    if args.output is None:
        output = (args.mask_folder + '/focal_spots/'
                  + ('ZP%03d.h5' % args.zone_plate_index))
        if not os.path.exists(args.mask_folder + '/focal_spots'):
            # It appears that in a HPC environment, if many jobs are launched at
            # the same time, sometimes os.path.exists(output) will return false
            # when in reality the folder exists. So, we also do a try/except
            try:
                os.mkdir(args.mask_folder + '/focal_spots')
            except:
                pass            

    else:
        output=args.output

        
    mask_file = args.mask_folder + ('/masks/ZP%03d.gds' % args.zone_plate_index)

    raster_file = ('.'.join(mask_file.split('.')[:-1])
                   + ('_raster%0.3fnm.h5' % (step * 1e9)))

    must_rasterize = True

    if os.path.exists(raster_file) and not args.ignore_cache:
        print('Loading rasterized optic from', raster_file)
        try:
            with h5py.File(raster_file, 'r') as f:
                rasterized_zp = np.array(f['rasterized_zp'])
                input_offset = np.array(f['input_offset'])
            must_rasterize = False
        except:
            print('Cached optic file was malformed, rerasterizing')

    if must_rasterize:
        print('Rasterizing .gds file', flush=True)
        rasterized_zp, input_offset = rasterize_zp(
            mask_file, step, verbose=True)
        print('Rasterized', flush=True)
        if args.cache_raster:
            print('Caching the rasterized optic', flush=True)
            with h5py.File(raster_file, 'w') as f:
                f.create_dataset('rasterized_zp', data=rasterized_zp)
                f.create_dataset('input_offset', data=input_offset)
                f.create_dataset('step', data=step)

    print('Simulating the focus', flush=True)
    focus = simulate_focus(
        rasterized_zp,
        input_offset,
        focal_distance,
        wavelength,
        step,
        output_shape,
        tile_shape=[2048,2048],
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
