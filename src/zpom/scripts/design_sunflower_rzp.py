import torch as t
import numpy as np
from zpom.optic_design import *
import argparse
from time import time

def main():

    start = time()
    parser = argparse.ArgumentParser(
        prog='design_sunflower_rzp',
        description='Makes the large, raster, base design file for a specified sunflower-style randomized zone plate. Please contact Abraham Levitan (allevitan@gmail.com) for help.')


    parser.add_argument('plan_file', type=str, help='The plan file to use.')
    parser.add_argument('--zone_plate_index', '-n', type=int, default=None, help='The index of the zone plate to design within the full array. Default is all zone plates.')
    parser.add_argument('--device', type=str, default='cpu', help='The device to perform the light propagation step on, default is cpu')
    parser.add_argument('--tile_size', '-ts', type=int, default=4098, help='The size of the tiles to use for the various computation steps, default=4098. Larger tiles are more efficient in many cases, provided there is sufficient memory available.')
    parser.add_argument('--yes', '-y', action='store_true', help='Automatically confirms the parameters, good for HPC environments or batch computing.')
    parser.add_argument('--output', '-o', type=str, help='The output filename for the base raster design file. A sensible default is used if not specified')

    args = parser.parse_args()
    
    print('N GPUs:', t.cuda.device_count(), flush=True)

    with h5py.File(args.plan_file, 'r') as plan:
        U_0 = t.as_tensor(np.array(plan['design_focus']))
        dtype = U_0.dtype
        f = np.array(plan['design_focal_distance']).ravel()[0]
        dr = np.array(plan['outer_zone_width']).ravel()[0]
        focus_diameter = np.array(plan['focus_diameter']).ravel()[0]
        n_frames = np.array(plan['n_frames']).ravel()[0]
        mini_zps_per_frame = np.array(plan['n_zps_per_frame']).ravel()[0]
        mini_zp_spacing = np.array(plan['zp_spacing']).ravel()[0]
        mini_zp_width = np.array(plan['zp_width']).ravel()[0]
        inner_zone_index = np.array(plan['inner_zone_index']).ravel()[0]
        wavelength = np.array(plan['design_wavelength']).ravel()[0]
        step = np.array(plan['pix_size']).ravel()[0]
        buttress_spacing = np.array(plan['buttress_spacing']).ravel()[0]
        buttress_deviation = np.array(plan['buttress_deviation']).ravel()[0]
        apodization_ratio = np.array(plan['apodization_ratio']).ravel()[0]
        tiling_style = np.array(plan['tiling_style']).ravel()[0]


    NZ = inner_zone_index + (n_frames-1) * mini_zp_spacing + mini_zp_width
    hc = 1.23984e-6 # in m*eV
    optic_diameter = 4 * NZ * dr

    if args.output is None:
        output_filename = args.plan_file[:-8:] + '_intermediate_design'
    else:
        output_filename = args.output


    print('\nDesigning a zone plate to the following specifications:')
    print('-------------------------------------------------------')
    print('Outer Zone Width (dr): %0.3f nm' % (dr*1e9))
    print('Buttress Spacing: %0.3f' % (buttress_spacing*1e9))
    print('Number of Frames:', n_frames)
    print('Mini ZPs per frame:', mini_zps_per_frame)
    print('Inner Zone Index:', inner_zone_index)
    print('Mini ZP Spacing:', mini_zp_spacing)
    print('Total Number of Zones:', NZ)
    print('Design Wavelength: %0.3f nm' % (wavelength*1e9))
    print('Design Energy: %0.3f eV' % (hc / wavelength))
    print('Focal Distance at Design Wavelength: %0.3f mm' % (f* 1e3))
    print('Focal Distance per Energy (A1): %0.3f um/eV' % (1e6*f * wavelength / hc))
    print('Focal Spot Diameter: %0.3f um' % (focus_diameter * 1e6))
    print('Optic Diameter: %0.3f um' % (optic_diameter*1e6))
    print('Max Buttress Deviation: %0.1f%%' % (buttress_deviation * 100))
    print('Apodization Ratio:', apodization_ratio)
    print('Calculation Device:',  args.device)
    print('')
    if args.zone_plate_index is None:
        raise NotImplementedError('Looping over all frames not yet implemented. Specify a zone plate with -n')
    print('Designing Zone Plate # %03d' % args.zone_plate_index)
    print('', flush=True)


    if not args.yes:
        answer = input('Would you like to begin the calculation? (Y/N): ')
        if answer.lower().strip() != 'y':
            print('Calculation Aborted')
            exit()

    print('Starting Calculation')

    design_sunflower_array(args.plan_file, args.zone_plate_index,
                           output_file = output_filename,
                           tile_size = args.tile_size,
                           device = args.device,
                           verbose=True)
    end = time()
    print('Total wall clock time:', end-start, 'seconds', flush=True)

            
if __name__ == '__main__':
    main()
