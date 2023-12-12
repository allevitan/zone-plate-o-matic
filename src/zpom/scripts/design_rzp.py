
import torch as t
import numpy as np
from zpom.optic_design import *
import argparse

def main():

    parser = argparse.ArgumentParser(
        prog='design_rzp',
        description='Makes the large, raster, base design file for a specified randomized zone plate. Please contact Abraham Levitan (allevitan@gmail.com) for help.')


    parser.add_argument('dr', type=float, help='The outer zone width of the zone plate, in nanometers')
    parser.add_argument('buttress_spacing', type=float, help='The azimuthal spacing of zone plate buttresses, in nm. Usually this is close to 4*dr. Lower numbers lead to a larger period and require tighter OSAs for the final optic.')
    parser.add_argument('NZ', type=float, help='The number of zones NZ')
    parser.add_argument('NS', type=float, help='The number of speckles across the focal spot NS (focal spot diameter = NS * OZW)') 
    parser.add_argument('wavelength', type=float, help='The wavelength of light to design for, in nanometers')
    parser.add_argument('step', type=float, help='The step size of the output array in real space, in nanometers. Typically, 1/10 of the outer zone width is the minimum for good performance')
    parser.add_argument('--apodization_ratio', '-ar', type=float, default=2, help='The apodization ratio used to apodize the output ZP. 2 is the default, and is good for most scenarios')
    parser.add_argument('--device', type=str, default='cpu', help='The device to perform the light propagation step on, default is cpu')
    parser.add_argument('--tile_size', '-ts', type=int, default=4096, help='The size of the tiles to use for the various computation steps, default=4096. Larger tiles are more efficient in many cases, provided there is sufficient memory available.')
    parser.add_argument('--buttress_deviation', '-bd', type=float, default=15, help='The maximum deviation allowed (in %%) from the true buttress spacing before a new zone is created.')
    parser.add_argument('--beamstop_ratio', '-bsr', type=float, default=0.5, help='The beamstop diameter, as a fraction of the overall diameter. 0.5 is the default')
    parser.add_argument('--yes', '-y', action='store_true', help='Automatically confirms the parameters, good for HPC environments or batch computing.')
    parser.add_argument('--output', '-o', type=str, help='The output filename for the base raster design file. A sensible default is used if not specified')

    args = parser.parse_args()

    dr = args.dr * 1e-9
    buttress_spacing = args.buttress_spacing * 1e-9
    NZ = args.NZ
    NS = args.NS
    wavelength = args.wavelength * 1e-9
    step = args.step * 1e-9
    apodization_ratio = args.apodization_ratio
    tile_size = args.tile_size
    bs_ratio = args.beamstop_ratio

    f = 4 * NZ * dr**2 / wavelength # focal length

    hc = 1.23984e-6 # in m*eV

    if args.output is None:
        output_filename = 'RZP_dr=%0.2fnm_NZ=%d_NS=%d.h5' % (dr*1e9, NZ, NS)
    else:
        output_filename = args.output

    sample_r = NS * dr / 2 # Illumination spot radius
    input_shape = [int((2 * sample_r) // step) + 1]*2
    optic_r = 2 * NZ * dr # overall ZP radius
    output_shape = [int((2 * optic_r) // step) + 1]*2
    estimated_output_size = 3 * output_shape[0] * output_shape[1] * 1e-9

    print('\nDesigning a zone plate to the following specifications:')
    print('-------------------------------------------------------')
    print('Outer Zone Width (dr): %0.3f nm' % (dr*1e9))
    print('Buttress Spacing: %0.3f' % (buttress_spacing*1e9))
    print('Number of Zones:', NZ)
    print('Number of Speckles:', NS)
    print('Design Wavelength: %0.3f nm' % (wavelength*1e9))
    print('Design Energy: %0.3f eV' % (hc / wavelength))
    print('Focal Distance at Design Wavelength: %0.3f mm' % (f* 1e3))
    print('Focal Distance per Energy (A1): %0.3f um/eV' % (1e6*f * wavelength / hc))
    print('Depth of Focus at Design Wavelength: +-%0.3f um' \
          % (1e6*2*dr**2 / wavelength)) 
    print('Focal Spot Diameter: %0.3f um' % (2*sample_r*1e6))
    print('Optic Diameter: %0.3f um' % (2*optic_r*1e6))
    print('Beamstop Diameter: %0.3f um' % (bs_ratio * 2*optic_r*1e6))
    print('Max Buttress Deviation: %0.1f%%' % (args.buttress_deviation))
    print('Apodization Ratio:', apodization_ratio)
    print('Optic Design Array Shape: [ %d x %d ]' % tuple(output_shape))
    print('Focal Spot Design Array Shape: [ %d x %d ]' % tuple(input_shape))
    print('Tile Size for Calculation: [ %d x %d ]' % (tile_size, tile_size))
    print('Estimated Optic Design File Size: %0.3f GB' % estimated_output_size)
    print('Calculation Device:',  args.device)
    print('Output File:', output_filename)
    print('', flush=True)


    if not args.yes:
        answer = input('Would you like to begin the calculation? (Y/N): ')
        if answer.lower().strip() != 'y':
            print('Calculation Aborted')
            exit()

    print('Starting Calculation')

    design_rzp(dr, NZ, NS, wavelength, step, output_filename,
               tile_size=tile_size, verbose=True,
               buttress_spacing=buttress_spacing,
               tiling_style='alternating',
               device=args.device,
               buttress_deviation=0.01*args.buttress_deviation,
               apodization_ratio=apodization_ratio)

if __name__ == '__main__':
    main()
