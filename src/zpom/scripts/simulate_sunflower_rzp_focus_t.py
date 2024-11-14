import numpy as np
from zpom.optic_simulation import *
import os
import argparse
import h5py

from zpom.helper_functions import linterp_1d

import matplotlib
matplotlib.use('Agg')

def main():

    parser = argparse.ArgumentParser(
        prog='simulate-sunflower-rzp-focus-t',
        description='Simulates one or more points in time from the time-dependent focal spot of a single zone plate in a sunflower array')

    parser.add_argument('mask_folder', type=str, help='The folder containing the base rzp design masks')
    parser.add_argument('pulse_file', type=str, help='The file containing information about the time-dependence of the pulse envelope.')
    parser.add_argument('wavelength', type=float, help='The wavelength of light to simulate, in nm')
    parser.add_argument('focal_distance', type=float, help='The focal distance to simulate at, in mm')
    parser.add_argument('step', type=float, help='The pixel step size to simulate, in nm')
    parser.add_argument('n_pix', type=int, help='The diameter of the simulated focal spot window, in pixels')
    parser.add_argument('--zone_plate_index', '-n', type=int, default=None, help='The index of the zone plate to design within the full array. Default is all zone plates.')
    parser.add_argument('--time', type=float, help='If set, the single time point to calculate a result for, in ps')
    parser.add_argument('--start-time', '-st', type=float, help='If set, the start of the time range to simulate, in ps')
    parser.add_argument('--end-time', '-et', type=float, help='If set, the end of the time range to simulate, in ps')
    parser.add_argument('--time-step', '-dt', type=float, help='If set, the time step for the time range to simulate, in ps')
    parser.add_argument('--tile_size', type=int, default=4096, help='The tile size for loading and processing the files, default is 4096')
    parser.add_argument('--z-step', type=float, help='If set, the step along the propagation direction to use to simulate a focus volume, in um')
    parser.add_argument('--z-range', type=float, help='If set, simulates a range of z positions out to +/- z-range, in um')
    parser.add_argument('--cache-raster', action='store_true', help='If set, the rasterized optic will be saved')
    parser.add_argument('--ignore-cache', action='store_true', help='If set, will force a re-rasterization of the optic, even if a cached version exists')
    parser.add_argument('--device', type=str, default='cpu', help='The device to perform the light propagation step on, default is cpu')
    parser.add_argument('--time-offset', type=float, help='An explicit time offset. The default is the time taken for light to travel from the input plane to the output plane')

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

    output_folder = args.mask_folder + '/focal_spots_t/'
    if not os.path.exists(output_folder):
        # It appears that in a HPC environment, if many jobs are launched at
        # the same time, sometimes os.path.exists(output) will return false
        # but the folder gets created before this job tries to create it.
        # So, we also do a try/except
        try:
            os.mkdir(output_folder)
        except:
            pass            

    if args.zone_plate_index is None:
        raise NotImplementedError('Doing all ZPs at once is not yet supported')
        
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


    if args.time is not None:
        print('Simulating a single time step')
        times = [args.time * 1e-12]
    else:
        if ((args.start_time is None)
            or (args.end_time is None)
            or (args.time_step is None)):
            raise ValueError('Either define --time, or all three of --start-time, --end-time, and --time-step.')
        times = np.arange(args.start_time, args.end_time, args.time_step)
        times *= 1e-12

    c = 299792458 # in m/s
    if args.time_offset is None:
        args.time_offset = focal_distance / c

    with h5py.File(args.pulse_file, 'r') as f:
        pulse_t0 = np.array(f['t0']).ravel()[0]
        pulse_dt = np.array(f['dt']).ravel()[0]
        A_array = t.as_tensor(np.array(f['A']), device=args.device)

    def A_func(time):
        return linterp_1d(A_array, time, pulse_t0, pulse_dt)
    
    if args.z_step is not None or args.z_range is not None:
        if args.z_step is None or args.z_range is None:
            raise ValueError('Either both or neither of --z_step and --z-range should be set')

        focus_offsets = np.arange(0, args.z_range, args.z_step) * 1e-6
        focus_offsets = np.concatenate([-focus_offsets[-2::-1],
                                        focus_offsets])
        distances = focal_distance + focus_offsets
    else:
        distances = [focal_distance]
        
    for distance in distances:
        print('Working on distance', distance, flush=True)
        for time in times:
            print('Working on time', time, flush=True)            
            # So, the times sometimes wind up being not quite, but just barely
            # less than zero. Without this correction, it gets printed as
            # negative zero, which then is annoying when I try to
            # algorithmically read all the results. There is a fix for this
            # which was introduced in python 3.11 (PEP 682), but I don't want\
            # to introduce a dependency on 3.11
            rounded_time = round(time*1e15, 1) + 0
            output = (output_folder
                      + ('ZP%03d_focaldist%0.3fum_t=%0.1ffs_step%0.3fnm.h5'
                         % (args.zone_plate_index,
                            distance * 1e6,
                            rounded_time,
                            step*1e9)))
            
            print('Simulating the focus at time', time, flush=True)
            focus = simulate_focus_t(
                rasterized_zp,
                input_offset,
                distance,
                wavelength,
                step,
                A_func,
                time,
                output_shape,
                t_0=args.time_offset,
                tile_shape=[args.tile_size, args.tile_size],
                verbose=True,
                calculation_device=args.device)
            print('Focus simulation complete, saving', flush=True)

            with h5py.File(output, 'w') as f:
                f.create_dataset('sim_focus', data=focus)
                f.create_dataset('wavelength', data=[wavelength])
                f.create_dataset('distance', data=[distance])
                f.create_dataset('step', data=[step])
                f.create_dataset('time', data=[time])
                f.create_dataset('time_offset', data=[args.time_offset])
            print('Saved')

if __name__ == '__main__':
    main()
