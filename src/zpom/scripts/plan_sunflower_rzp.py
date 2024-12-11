
import torch as t
import numpy as np
from zpom.optic_design import *
import argparse

def main():

    parser = argparse.ArgumentParser(
        prog='plan_sunflower_rzp',
        description='Makes a design plan for all the randomzed zone plates in the specified sunflower-style array. Please contact Abraham Levitan (allevitan@gmail.com) for help.')


    parser.add_argument('dr', type=float, help='The outer zone width of the zone plate, in nanometers')
    parser.add_argument('buttress_spacing', type=float, help='The azimuthal spacing of zone plate buttresses, in nm. Usually this is close to 4*dr. Lower numbers lead to a larger period and require tighter OSAs for the final optic.')
    parser.add_argument('n_frames', type=float, help='The number of frames')
    parser.add_argument('focus_diameter', type=float, help='The diameter (in nm) of the designed focal spot') 
    parser.add_argument('inner_zone', type=int, help='The index of the innermost zone to place for the first frame')
    parser.add_argument('mini_zp_spacing', type=int, help='The number of zones separating the inner ring of subsequent frames')
    parser.add_argument('wavelength', type=float, help='The wavelength of light to design for, in nanometers')
    parser.add_argument('step', type=float, help='The step size of the output array in real space, in nanometers. Typically, 1/10 of the outer zone width is the minimum for good performance')
    parser.add_argument('--apodization_ratio', '-ar', type=float, default=2, help='The apodization ratio used to apodize the output ZP. 2 is the default, and is good for most scenarios')
    parser.add_argument('--buttress_deviation', '-bd', type=float, default=15, help='The maximum deviation allowed (in %%) from the true buttress spacing before a new zone is created.')
    parser.add_argument('--mini_zp_width', '-mzw', type=int, default=None, help='The width of each mini-zp, in zones')
    parser.add_argument('--mini_zp_length_factor', '-mzl', type=float, default=1, help='The ratio of the zone plate length to a sensible default. Defaults to 1.')
    parser.add_argument('--mini_zps_per_frame', '-mpf', default=3, type=int, help='Number of mini-zps per frame')
    parser.add_argument('--special_order', action='store_true', help='Whether to use the special order for 2 mini-zps per frame')
    parser.add_argument('--vary_width', '-vw', action='store_true', help='If set, allows the inner zone plates to be wider than the outer ones')
    parser.add_argument('--yes', '-y', action='store_true', help='Automatically confirms the parameters, good for HPC environments or batch computing.')
    parser.add_argument('--output', '-o', type=str, help='The output filename for the base raster design file. A sensible default is used if not specified')

    args = parser.parse_args()

    dr = args.dr * 1e-9
    buttress_spacing = args.buttress_spacing * 1e-9
    focus_diameter = args.focus_diameter * 1e-9
    wavelength = args.wavelength * 1e-9
    step = args.step * 1e-9
    apodization_ratio = args.apodization_ratio

    if args.mini_zp_width is None:
        args.mini_zp_width = args.mini_zp_spacing
    NZ = NZ = args.inner_zone + (
        (args.n_frames-1) * args.mini_zp_spacing + args.mini_zp_width )
    f = 4 * NZ * dr**2 / wavelength # focal length

    hc = 1.23984e-6 # in m*eV

    print((f * wavelength / hc))

    if args.output is None:
        output_filename = 'Sunflower_RZP_dr=%0.2fnm_NF=%d_focdiam=%0.2fum_plan.h5' % (dr*1e9, args.n_frames, focus_diameter*1e6)
    else:
        output_filename = args.output


    
    optic_diameter = 4 * NZ * dr

    
    print('\nDesigning a zone plate to the following specifications:')
    print('-------------------------------------------------------')
    print('Outer Zone Width (dr): %0.3f nm' % (dr*1e9))
    print('Buttress Spacing: %0.3f nm' % (buttress_spacing*1e9))
    print('Number of Frames:', args.n_frames)
    print('Mini ZPs per frame:', args.mini_zps_per_frame)
    print('Inner Zone Index:', args.inner_zone)
    print('Mini ZP Spacing:', args.mini_zp_spacing)
    print('Total Number of Zones:', NZ)
    print('Design Wavelength: %0.3f nm' % (wavelength*1e9))
    print('Design Energy: %0.3f eV' % (hc / wavelength))
    print('Focal Distance at Design Wavelength: %0.3f mm' % (f* 1e3))
    print('Focal Distance per Energy (A1): %0.3f um/eV' % (1e6*f * wavelength / hc))
    print('Focal Spot Diameter: %0.3f um' % (focus_diameter * 1e6))
    print('Optic Diameter: %0.3f um' % (optic_diameter*1e6))
    print('Max Buttress Deviation: %0.1f%%' % (args.buttress_deviation))
    print('Apodization Ratio:', apodization_ratio)
    print(f'Pixel Size in Optic Design: {step*1e9 : 0.3f} nm')
    print('Output File:', output_filename)
    print('', flush=True)


    if not args.yes:
        answer = input('Would you like to save the plan? (Y/N): ')
        if answer.lower().strip() != 'y':
            print('Plan not saved.')
            exit()

    print('Saving plan...')
    
    plan_sunflower_array(dr,
                         args.n_frames,
                         wavelength,
                         focus_diameter,
                         args.inner_zone,
                         args.mini_zp_spacing,
                         step,
                         output_filename,
                         buttress_spacing=buttress_spacing,
                         mini_zp_width=args.mini_zp_width,
                         mini_zp_length_factor=args.mini_zp_length_factor,
                         mini_zps_per_frame=args.mini_zps_per_frame,
                         equal_width=~args.vary_width,
                         tiling_style='alternating',
                         buttress_deviation=0.01*args.buttress_deviation,
                         apodization_ratio=apodization_ratio,
                         special_order=args.special_order)
            
if __name__ == '__main__':
    main()
