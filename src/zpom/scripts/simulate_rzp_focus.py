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

    parser.add_argument('mask_file', type=str, help='The base rzp design mask file (.gds)')
    parser.add_argument('wavelength', type=float, help='The wavelength of light to simulate, in nm')
    parser.add_argument('focal_distance', type=float, help='The focal distance to simulate at, in mm')
    parser.add_argument('step', type=float, help='The pixel step size to simulate, in nm')
    parser.add_argument('n_pix', type=int, help='The diameter of the simulated focal spot window, in pixels')
    parser.add_argument('--tile_size', type=int, default=4096, help='The tile size for loading and processing the files, default is 4096')
    parser.add_argument('--output', '-o', type=str, default=None, help='The filename to be used for the output simulated focus.')
    parser.add_argument('--device', type=str, default='cpu', help='The device to perform the light propagation step on, default is cpu')
    parser.add_argument('--cache-raster', action='store_true', help='If set, the rasterized optic will be saved')
    parser.add_argument('--ignore-cache', action='store_true', help='If set, will force a re-rasterization of the optic, even if a cached version exists')
    parser.add_argument('--zone-double-width', type=float, default=None, help='If set, will roughly simulate a zone-doubled optic with the specified thickness of material, in nanometers, deposited')
    parser.add_argument('--efficiency_calc_radius', type=float, default=None, help='In nanometers, the radius of the region at the sample plane to consider when calculating the optic efficiency')
    parser.add_argument('--efficiency_calc_optic_diameter', type=float, default=None, help='In micrometers, the diameter of the optic being simulated, to help with the efficiency calculation')
    parser.add_argument('--efficiency_calc_beamstop_ratio', type=float, default=None, help='The beamstop ratio of the optic being simulated, to help with the efficiency calculation')
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

    raster_file = ('.'.join(args.mask_file.split('.')[:-1])
                   + ('_raster%0.3fnm.h5' % (step * 1e9)))
    raster_fabsim_file = ('.'.join(args.mask_file.split('.')[:-1])
                              + ('_raster_fabsim_%0.3fnm.h5' % (step * 1e9)))

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
        rasterized_zp, input_offset = rasterize_zp(mask_file, step,
                                                   verbose=True)
        print('Rasterized', flush=True)
        if args.cache_raster:
            print('Caching the rasterized optic', flush=True)
            with h5py.File(raster_file, 'w') as f:
                f.create_dataset('rasterized_zp', data=rasterized_zp)
                f.create_dataset('input_offset', data=input_offset)
                f.create_dataset('step', data=step)

    if args.zone_double_width is not None:
        print('Simulating a fab process with zone double width')
        zdw = 1e-9 * args.zone_double_width
        rasterized_zp = simulate_fab_process(
            rasterized_zp,
            zone_material=0,
            dilation_radii=[zdw / step],
            dilation_materials=[1],
            background_material=0,
            invert=True, #assuming the buttresses are part of the fab structure
        )
        if args.cache_raster:
            print('Caching the fab simulation optic', flush=True)
            with h5py.File(raster_fabsim_file, 'w') as f:
                f.create_dataset('rasterized_zp', data=rasterized_zp)
                f.create_dataset('input_offset', data=input_offset)
                f.create_dataset('step', data=step)

    if args.efficiency_calc_radius is not None:
        print('Calculating the initial illumination intensity for efficiency '
              'comparison')
        abs_input = np.abs(rasterized_zp)

        illuminated_area = \
            ( np.pi * (1000*args.efficiency_calc_optic_diameter / 2)**2 -
              np.pi * (args.efficiency_calc_beamstop_ratio * 
                       1000*args.efficiency_calc_optic_diameter / 2)**2
             )

        illuminated_npix = illuminated_area / (1e9*step)**2

        init_intensity = \
            illuminated_npix * np.amax(abs_input).astype(np.float64)**2

        
    print('Simulating the focus', flush=True)
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

    if args.efficiency_calc_radius is not None:
        print('Calculating the efficiency ratio')
        xs = (t.arange(focus.shape[1]) - focus.shape[1]//2) * step*1e9
        ys = (t.arange(focus.shape[1]) - focus.shape[1]//2) * step*1e9
        Xs, Ys = t.meshgrid(xs, ys, indexing='ij')
        calc_region = (Xs**2 + Ys**2) <= args.efficiency_calc_radius**2
        
        output_intensity = t.sum(t.abs(calc_region * focus)**2)

        # Okay, what have we actually calculated here?

        # init_intensity is an estimate of the total power which illuminated
        # the optic, based on provided geometry parameters and the maximum
        # value of the input, rasterized optic.

        # output_intensity is a measurement of the total power within the
        # provided radius at the focus plane

        # Here, all the optics we simulate are defined as pure-amplitude
        # optics which are generated from design files. After fabrication,
        # the efficiency may vary dramatically, but we can at least compare
        # the measured efficiency of the perfectly-fabricated amplitude-only
        # zone plate to that of an ideal amplitude-only zone plate, and get
        # a ratio of how much penalty in efficiency is baked into the optic.

        # The efficiency of such a perfect optic is 1/pi**2, hence the
        # multiplication by pi**2
        
        efficiency_ratio = np.pi**2 * output_intensity.numpy() / init_intensity
        print('Calculated efficiency ratio:', efficiency_ratio)
        
    else:
        efficiency_ratio = 0
        
    print('Focus simulation complete, saving', flush=True)

    
    with h5py.File(output, 'w') as f:
        f.create_dataset('sim_focus', data=focus)
        f.create_dataset('wavelength', data=[wavelength])
        f.create_dataset('focal_distance', data=[focal_distance])
        f.create_dataset('step', data=[step])
        f.create_dataset('efficiency_ratio', data=[efficiency_ratio])
    print('Saved')

if __name__ == '__main__':
    main()

