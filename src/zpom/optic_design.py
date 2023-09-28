"""
This file contains functions for generating randomized zone plate designs. 

The basic zone plate design process used here proceeds in two steps. The
first step is the "design" step, and it's output is a .h5 file containing
two raster images, one which defines the grating positions and one which
contains information about the desired wavefield amplitude.

The design step writes directly to a file because the design files can be
very large, and this avoids creating a memory bottleneck. The files can
also often be compressed very successfully once written to disk.

This "design file" forms the input for the second step, which is to
"realize" the zone plate design. In this step, a vectorized .gdsii
file is produced with a specific choice regarding the level of amplitude
control and buttress width. At the same time, a lower resolution raster
image is generated as a .png, to be used for later simulation of the
optic's focal spot.

Author: Abraham Levitan
Dates: January 2022 to September 2023
"""

import torch as t
import numpy as np
import itertools as it
import h5py
from zpom import propagation
from multiprocessing import Pool
from skimage.measure import find_contours
import gdstk
import tqdm
import os
from matplotlib import pyplot as plt

# TODO: In the future I could also add an "abberation" function or something
# like that so that this code could be used to design regular zone plates
# with a fixed aberration function in Fourier space, like what they use at
# SLS.

def design_rzp(dr,
               NZ,
               NS,
               wavelength,
               step,
               output_filename,
               bs_ratio=0.5,
               dtype=t.complex128,
               tile_size=None,
               device='cpu',
               buttress_spacing=None,
               buttress_deviation=0.15,
               tiling_style='alternating',
               apodization_ratio=0,
               verbose=False,):
    
    if buttress_spacing is None:
        buttress_spacing = 4 * dr

    
    # The real-valued dtype corresponding to the given complex-valued dtype.
    real_dtype = t.real(t.ones(1,dtype=dtype)).dtype

    # We create the speckle texture for the target focal spot, at a resolution
    # which matches the final ozw
    sample_r = NS * dr / 2 
    base_input_shape = [int((2 * sample_r) // dr) + 1]*2
    U_0 = t.exp(2j*np.pi*t.rand(*base_input_shape,
                                dtype=real_dtype))

    if verbose:
        print('Focal Spot Diameter',2*sample_r*1e6,'um', flush=True)

    # And we upsample it to the full resolution of our final design file
    input_shape = [int((2 * sample_r) // step) + 1]*2
    U_0 = propagation.fourier_pad_to_shape(U_0, input_shape)

    # Finally, we crop the design focal spot to the desired size
    xs = t.arange(0, U_0.shape[0], dtype=t.float32) * step
    ys = t.arange(0, U_0.shape[1], dtype=t.float32) * step
    xs = (xs - t.mean(xs)).to(dtype=t.float32)
    ys = (ys - t.mean(ys)).to(dtype=t.float32)
    Xs, Ys = t.meshgrid(xs, ys, indexing='ij')
    Rs2 = (xs**2)[:,None] + (ys**2)[None,:]
    U_0[Rs2>sample_r**2] = 0
    del Xs, Ys, Rs2

    apodization_width = apodization_ratio * 8 * NZ * dr / NS
    
    # Calculate the focal distance & radius of the optic
    f = 4 * NZ * dr**2 / wavelength
    optic_r = 2 * NZ * dr
    window = ((-optic_r,optic_r), (-optic_r, optic_r))

    def window_function(X,Y, Angle, R):
        # TODO: This function needs to also include the apodization
        return  t.logical_and(R > bs_ratio * optic_r, R < optic_r)

        # if apodize!=0:
        #     outer_edge = t.logical_and(Rs > (optic_r - apodize),
        #                                Rs < optic_r).to(device=out_tile.device)
        #     inner_edge = t.logical_and(Rs < (bs_r + apodize),
        #                                Rs > bs_r)
        #     # This is a Tukey window
        #     out_tile[outer_edge] *= (1 - t.cos(
        #         np.pi * (Rs[outer_edge] - optic_r) / apodize))/2
        #     out_tile[inner_edge] *= (1 - t.cos(
        #     np.pi * (Rs[inner_edge] - bs_r) / apodize))/2

    # This populates the design file
    design_grating_hologram(U_0,
                            f,
                            window,
                            window_function,
                            wavelength,
                            step,
                            output_filename,
                            dtype=dtype,
                            device=device,
                            tile_size=tile_size,
                            buttress_spacing=buttress_spacing,
                            buttress_deviation=buttress_deviation,
                            outer_r=optic_r,
                            tiling_style=tiling_style,
                            verbose=verbose)

    # As a final convenience, we add a few extra bits of metadata to the design
    # file that the general grating hologram function didn't know about
    with h5py.File(output_filename, 'r+') as f:
        apodization_ratio = apodization_ratio
        f.create_dataset('dr', data=[dr])
        f['dr'].attrs['units'] = 'm'
        f.create_dataset('NZ', data=[NZ])
        f.create_dataset('NS', data=[NS])
        f.create_dataset('focus_diameter', data=[2*sample_r])
        f['focus_diameter'].attrs['units'] = 'm'
        f.create_dataset('buttress_spacing', data=[buttress_spacing])
        f.create_dataset('buttress_deviation', data=[buttress_deviation])
        f.create_dataset('apodization_ratio', data=[apodization_ratio])
        f.create_dataset('beamstop_ratio', data=[bs_ratio])


# An important constant for the sunflower array    
golden_angle = (np.pi * (3 - np.sqrt(5)))

def place_sunflower_zps(dr, n_frames, wavelength,
                        inner_zone_index, mini_zp_spacing, mini_zp_width=None,
                        mini_zp_length_factor=1, mini_zps_per_frame=3,
                        equal_width=True, phi_0=0):
    """Defines the properties of the mini zone plates in a multi-frame RZP

    Parameters
    ----------
    dr : : float
        The outer zone width of the entire multi-frame zone plate, in meters
    n_frames : int
        The number of frames (distinct mini-zone plate radii) to design for
    wavelength : float
        The design wavelength, in meters
    inner_zone_index : int
        The index of the innermost zone of the first frame.
    mini_zp_spacing : int
        The number of zones separating each frame from the subsequent frame
    mini_zp_width : int, optional
        The radial width (in number of zones) of each mini-zone plate. Default
        is equal to mini_zp_spacing
    mini_zp_length_factor : float, optional
        The ratio of the azimuthal length of the mini-zone plates to half the
        average nearest-neighbor distance. Default is 1
    mini_zps_per_frame : int, optional
        The number of mini-zps to place at each radius. Default is 3
    equal_width : bool, optional
        Default is True. Whether to restrict the width of the inner mini-ZPs
        to match the width of the outer one
    phi_0 : float, optional
        The azimuthal angle of the first mini-zp, in radians. Default is 0.
    verbose : bool, optional
        Whether to print the zone plate parameters. Default is false.

    Returns
    -------
    design: tuple
        A tuple of dictionaries describing the parameters of each individual
        mini zone plate
    """
    
    if mini_zp_width is None:
        mini_zp_width = mini_zp_spacing

    ns = np.arange(n_frames * mini_zps_per_frame)
    phis = phi_0 + ns * golden_angle

    NZ = inner_zone_index + (n_frames-1) * mini_zp_spacing + mini_zp_width

    # Calculation of the focal distance, including the n**2 * lambda**2 term in the zone plate equation
    a = wavelength**2
    b = 2 * NZ * wavelength**3 - 4 * dr**2 * NZ * wavelength
    c = NZ**2 * wavelength**2 * ( wavelength**2 - 4 * dr**2 / 2 )
    f = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    # f =  dr**2 * 4 * NZ / wavelength ## This is the simplified calculation that's usually used

    frame_idx = np.repeat(np.arange(0,n_frames), mini_zps_per_frame)
    starting_zones = inner_zone_index + frame_idx * mini_zp_spacing
    ending_zones = starting_zones + mini_zp_width
    inner_rs = np.sqrt(starting_zones * f * wavelength + starting_zones**2 * wavelength**2 / 4)
    outer_rs = np.sqrt(ending_zones * f * wavelength + ending_zones**2 * wavelength**2 / 4)

    if equal_width:
        width = outer_rs[-1] - inner_rs[-1]
        outer_rs = inner_rs + width

    # This is (roughly) the area of the optic occupied by region associated with each frame
    frame_area = np.pi * (mini_zp_spacing * f * wavelength)
    # And this is a rough measure of a good mini-zp radius
    base_radius = np.sqrt(frame_area / mini_zps_per_frame) / 2

    xs = (inner_rs + outer_rs)/2 * np.cos(phis)
    ys = (inner_rs + outer_rs)/2 * np.sin(phis)

    design = [
        { 'phi': phi,
          'x': x,
          'y': y,
          'start_zone': sz,
          'end_zone': ez,
          'f': f,
          'frame_id': frame_id,
          'dr': dr,
          'wavelength': wavelength,
          'inner_r': inner_r,
          'outer_r': outer_r,
          'radius': base_radius * mini_zp_length_factor,
        } for (phi, x, y, sz, ez, frame_id, inner_r, outer_r)
        in zip(
            phis,
            xs,
            ys,
            starting_zones,
            ending_zones,
            frame_idx,
            inner_rs,
            outer_rs
        )
    ]

    return design


def inspect_sunflower_placement(design, pix_size=1e-6):
    plt.figure()
    max_r = max([mini_zp['outer_r'] for mini_zp in design])
    xs = np.arange(-max_r*1.1, max_r*1.1, pix_size)
    Xs, Ys = np.meshgrid(xs, xs, indexing='xy')
    Rs = np.sqrt(Xs**2+Ys**2)
    Angles = np.arctan2(Xs,Ys)
    zp_mask = np.zeros_like(Xs)
    for mini_zp in design:
        mask = np.sqrt(((Xs - mini_zp['x'])**2 
                        + (Ys - mini_zp['y'])**2)) < mini_zp['radius']
        mask[Rs < mini_zp['inner_r']] = 0
        mask[Rs > mini_zp['outer_r']] = 0
        zp_mask += mask
    plt.imshow(zp_mask)
    plt.colorbar()


def design_sunflower_array(dr,
                           n_frames,
                           wavelength,
                           focus_diameter,
                           inner_zone_index,
                           mini_zp_spacing,
                           step,
                           output_filename,
                           dtype=t.complex128,
                           tile_size=None,
                           device='cpu',
                           mini_zp_width=None,
                           mini_zp_length_factor=1,
                           mini_zps_per_frame=3,
                           equal_width=True,
                           phi_0=0,
                           buttress_spacing=None,
                           buttress_deviation=0.15,
                           tiling_style='alternating',
                           apodization_ratio=0,
                           verbose=False,):

    if mini_zp_width is None:
        mini_zp_width = mini_zp_spacing

    if buttress_spacing is None:
        buttress_spacing = 4 * dr

    zp_locations = place_sunflower_zps(
        dr, n_frames, wavelength,
        inner_zone_index, mini_zp_spacing,
        mini_zp_width=mini_zp_width,
        mini_zp_length_factor=mini_zp_length_factor,
        mini_zps_per_frame=mini_zps_per_frame,
        equal_width=equal_width,
        phi_0=phi_0)
    
    os.mkdir(output_filename)

    # We make one design focal spot which corresponds to the focal spot of the
    # full zp array, if it wasn't broken into separate bits. We'll then use
    # this same focal spot for each mini-ZP, but all the mini-ZPs will get
    # different sections of the ZP in Fourier space.

    # The real-valued dtype corresponding to the given complex-valued dtype.
    real_dtype = t.real(t.ones(1,dtype=dtype)).dtype

    # We create the speckle texture for the target focal spot, at a resolution
    # which matches the final ozw
    base_input_shape = [int(focus_diameter // dr) + 1]*2
    U_0 = t.exp(2j*np.pi*t.rand(*base_input_shape,
                                dtype=real_dtype))

    if verbose:
        print('Focal Spot Diameter',focus_diameter*1e6,'um', flush=True)

    # And we upsample it to the full resolution of our final design file
    input_shape = [int(focus_diameter // step) + 1]*2
    U_0 = propagation.fourier_pad_to_shape(U_0, input_shape)

    # Finally, we crop the design focal spot to the desired size
    xs = t.arange(0, U_0.shape[0], dtype=t.float32) * step
    ys = t.arange(0, U_0.shape[1], dtype=t.float32) * step
    xs = (xs - t.mean(xs)).to(dtype=t.float32)
    ys = (ys - t.mean(ys)).to(dtype=t.float32)
    Xs, Ys = t.meshgrid(xs, ys, indexing='ij')
    Rs2 = (xs**2)[:,None] + (ys**2)[None,:]
    U_0[Rs2>(focus_diameter/2)**2] = 0
    del Xs, Ys, Rs2

    # Calculation of the focal distance, including the n**2 * lambda**2 term in the zone plate equation
    NZ = inner_zone_index + (n_frames-1) * mini_zp_spacing + mini_zp_width
    print('NZ', NZ)
    a = wavelength**2
    b = 2 * NZ * wavelength**3 - 4 * dr**2 * NZ * wavelength
    c = NZ**2 * wavelength**2 * ( wavelength**2 - 4 * dr**2 / 2 )
    f = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    # f =  dr**2 * 4 * NZ / wavelength ## This is the simplified calculation that's usually used

    for id, mini_zp in enumerate(zp_locations):
        if verbose:
            print('Working on Mini ZP %d of %d (ZP%03d).' % (id+1, len(zp_locations), id))

        mini_zp_filename = output_filename + ('/ZP%03d.h5' % id)
        # First we have to set up the window and window function
        x = mini_zp['x']
        y = mini_zp['y']
        r = mini_zp['radius']
        window = ((x - r, x + r), (y - r, y + r))

        def window_function(X, Y, Angle, R):
            radial_band = t.logical_and(R > mini_zp['inner_r'], R < mini_zp['outer_r'])
            mini_R = t.sqrt((X-x)**2 + (Y-y)**2)
            disk = mini_R < r
            return t.logical_and(radial_band, disk)
        
        
        design_grating_hologram(U_0,
                                f,
                                window,
                                window_function,
                                wavelength,
                                step,
                                mini_zp_filename,
                                dtype=dtype,
                                device=device,
                                tile_size=tile_size,
                                buttress_spacing=buttress_spacing,
                                buttress_deviation=buttress_deviation,
                                outer_r=mini_zp['outer_r'],
                                tiling_style=tiling_style,
                                verbose=verbose)
        
        with h5py.File(mini_zp_filename, 'r+') as output:
            apodization_ratio = apodization_ratio
            output.create_dataset('dr', data=[dr])
            output['dr'].attrs['units'] = 'm'
            output.create_dataset('focus_diameter', data=[focus_diameter])
            output['focus_diameter'].attrs['units'] = 'm'
            output.create_dataset('buttress_spacing', data=[buttress_spacing])
            output.create_dataset('buttress_deviation', data=[buttress_deviation])
            output.create_dataset('apodization_ratio', data=[apodization_ratio])

    
    
            
def design_grating_hologram(U_0,
                            f,
                            window,
                            window_function,
                            wavelength,
                            step,
                            output_filename,
                            dtype=t.complex128,
                            device='cpu',
                            tile_size=None, 
                            buttress_spacing=None,
                            buttress_deviation=0.15, # max deviation allowed from the defined buttress spacing
                            outer_r=None, # Definition of the outer position where the buttress spacing is correct, default is the edge of the window.
                            tiling_style='alternating',
                            compression='lzf',
                            verbose=False):

    # The COM of the input window is defined as (0,0), and the output window is defined
    # with respect to that.
    input_shape = list(U_0.shape)
    
    full_xs = t.arange(int(window[0][0]/step), int(window[0][1]/step)) * step
    full_ys = t.arange(int(window[1][0]/step), int(window[1][1]/step)) * step
    output_shape = [len(full_xs), len(full_ys)]
    base_offset = [-(input_shape[0] / 2 ) - int(window[0][0]/step),
                   -(input_shape[1] / 2 ) - int(window[1][0]/step)]

    if tile_size is None:
        tile_shape = list(np.minimum(np.array(output_shape),
                                     np.array(input_shape)))
    else:
        tile_shape= [tile_size, tile_size]

    # It will be more convenient from here on out to have a separate x & y step,
    # but it was unweildy to take both in as input.
    step = [step, step]

    # We will use this to store the maximum amplitude we find, to store with the
    # design files at the end.
    max_abs = 0

    with h5py.File(output_filename, 'w') as output_store:
        # First, we populate some metadata with info about the optic
        output_store.create_dataset('focal_distance', data=[f])
        output_store['focal_distance'].attrs['units'] = 'm'
        # This is with respect to the center of the input array, to be used
        # to arrange the tiles in the final design
        output_store.create_dataset('offset', data=[full_xs[0], full_ys[0]])
        hc = 1.23984e-6 # in m*eV
        A1 = f * wavelength / hc # focal distance per photon energy
        output_store.create_dataset('A1', data=[A1])
        output_store['A1'].attrs['units'] = 'm/eV'
        output_store.create_dataset('wavelength', data=[wavelength])
        output_store['wavelength'].attrs['units'] = 'm'
        output_store.create_dataset('step', data=[step[0]])
        output_store['step'].attrs['units'] = 'm'


        chunk_shape = np.minimum(tile_shape, output_shape)
        # This output will include the zones and information for designing
        # the variable width buttresses
        grating_output = output_store.create_dataset('grating', dtype=np.int8,
                                                     shape=tuple(output_shape),
                                                     chunks=tuple(chunk_shape),
                                                     compression=compression)
        
        # This output will include the wavefield amplitudes
        amplitude_output = output_store.create_dataset(
            'amplitude', dtype=np.int16,
            shape=tuple(output_shape),
            chunks=tuple(chunk_shape),
            compression=compression)
        
        
        output_is = t.arange(propagation.get_num_tiles(output_shape[0], tile_shape[0]))
        output_js = t.arange(propagation.get_num_tiles(output_shape[1], tile_shape[1]))
        input_is = t.arange(propagation.get_num_tiles(U_0.shape[0], tile_shape[0]))
        input_js = t.arange(propagation.get_num_tiles(U_0.shape[1], tile_shape[1]))
        combos = it.product(input_is, input_js, output_is, output_js)
        n = len(output_is)*len(output_js)*len(input_is)*len(input_js)
        
        for idx, (in_i, in_j, out_i, out_j) in enumerate(combos):

            if verbose:
                print('Working on tile',idx+1,'of',n, flush=True)

            # We calculate the window function first, because if it's all zeroes we don't
            # even need to do the full calculation
            xs = full_xs[0] + t.arange(out_i*tile_shape[0],
                                       (out_i + 1) * tile_shape[0]) * step[0]
            ys = full_ys[0] + t.arange(out_j * tile_shape[1],
                                       (out_j + 1) * tile_shape[1]) * step[1]
            Xs,Ys = t.meshgrid(xs,ys, indexing='ij')
            Angles = t.atan2(Xs, Ys)
            Rs2 = Xs**2 + Ys**2
            Rs = t.sqrt(Rs2)            
            window_fn_output = window_function(Xs,Ys,Angles,Rs)

            in_tile = t.zeros(tile_shape, dtype=dtype, device=U_0.device)
            in_selection = U_0[in_i*tile_shape[0]:(in_i+1) * tile_shape[0],
                               in_j*tile_shape[1]:(in_j+1) * tile_shape[1]]
            # What this does is ensure a standard size, even when the 
            # selection overlaps the edge.
            in_tile[:in_selection.shape[0],
                    :in_selection.shape[1]] = in_selection
            tile_offset = [(in_i - out_i) * tile_shape[0] + base_offset[0],
                           (in_j - out_j) * tile_shape[1] + base_offset[1]]
            if t.all(t.eq(window_fn_output,0)):
                # No point in doing the expensive calculation if it's all just
                # going to be masked off
                out_tile = in_tile * 0
            else:
                out_tile = propagation.FFT_DI(in_tile.to(device=device),
                                              -f, wavelength, step,
                                              offset=tile_offset).cpu()
           
                out_tile[window_fn_output == 0] = 0
            
                
            # Now we set up the grating
            # The following line is formally correct:
            # 
            # perfect_zp_phase = np.sqrt(f**2 + Rs2) * (2*np.pi/wavelength)
            # 
            # But, it's actually better to use the taylor series because f is
            # usually far larger than R, and so we run into issues with 
            # numerical precision pretty quickly, even for 64-bit floats

            order = 4
            coefficients = [1/2,-1/8,1/16,-5/128,7/256,-21/1024,
                            33/2048, -429/32768, 715/65536, -2431/262144]
            coefficients = [c * 2 * np.pi/wavelength for c in coefficients]
            perfect_zp_phase = sum(coefficients[n] * Rs2**(n+1) / (f**(2*n+1))
                                   for n in range(order))
            
            zone_phase = 0.5 * perfect_zp_phase + 0.5 * t.angle(out_tile)
            
            new_zone_phase = (perfect_zp_phase + t.angle(out_tile))
            
            modified_radius = t.sqrt(new_zone_phase/coefficients[0] * f)
            
            buttress_regions = t.floor((t.log(Rs)-np.log(outer_r)) / np.log(1-buttress_deviation))
            nominal_Rs = t.exp(buttress_regions * np.log(1-buttress_deviation) + np.log(outer_r))
            grating_phase = Angles * 2 * np.pi * nominal_Rs / buttress_spacing

            if tiling_style.lower() == 'simple':
                grating = t.remainder(grating_phase/np.pi, 2) - 1
            elif tiling_style.lower() == 'alternating':
                grating = t.remainder((grating_phase + zone_phase)
                                      / np.pi, 2) - 1
            else:
                raise KeyError('Invalid tiling_style, '+
                               'use "simple" or "alternating"')

            # This converts the grating to a form which we can use to calculate
            # the final zone plate once whe know the maximum intensity
            grating = t.abs(grating)
            numpy_grating = (grating*127).to(dtype=t.int8, device='cpu')
            
            numpy_grating[window_fn_output==0] = -128
            numpy_grating[(t.angle(out_tile) > 0)] = -128
            
            result = t.abs(window_fn_output * out_tile).to(dtype=t.bfloat16)
            max_abs = max(max_abs, float(t.max(result)))
            
            # This just serves to make sure that when the tile output 
            # is large for the section of the output array, that it just
            # saves the section that's appropriate.
            out_slice = np.s_[out_i * tile_shape[0]:(out_i+1) * tile_shape[0],
                              out_j * tile_shape[1]:(out_j+1) * tile_shape[1]]
            out_selection = grating_output[out_slice]
            
            grating_output[out_slice] = numpy_grating[:out_selection.shape[0],
                                                      :out_selection.shape[1]]
            
            amplitude_output[out_slice] = result[:out_selection.shape[0],
                                                 :out_selection.shape[1]].view(
                                                    dtype=t.int16).cpu().numpy()
    

        for s in amplitude_output.iter_chunks():
            loaded_array = t.as_tensor(np.array(amplitude_output[s])).view(
                t.bfloat16)
            amplitude_output[s] = (loaded_array / max_abs).view(t.int16).numpy()


def realize_design(filename, lr_filename, gds_filename,
                   buttress_width=15e-9, grating_max=0.9,
                   reduction_factor=8, chunk_size=2048, n_processes=6,
                   verbose=False, overlap=512, view=False):
    """Makes contours and the low resolution version"""
    with h5py.File(filename,'r') as f:
        step = float(f['step'][()])
        buttress_spacing = float(f['buttress_spacing'][()])
        buttress_fraction = buttress_width / (buttress_spacing)
        shape = f['amplitude'].shape
        offset = np.array(f['offset'])

        chunk_size = (chunk_size // reduction_factor) * reduction_factor
        
        if verbose:
            print('Starting on the low res version')
        # First, we generate the low resolution image
        with h5py.File(lr_filename,'w') as lr_f:
            lr_f.create_dataset('lr_rzp',
                                dtype=np.float32,
                                shape=(s//reduction_factor for s in shape),
                                compression='lzf')
            lr_f.create_dataset('step', data=[step*reduction_factor])
            i_list = t.arange(propagation.get_num_tiles(shape[0], chunk_size))
            j_list = t.arange(propagation.get_num_tiles(shape[1], chunk_size))
            for i, j in it.product(i_list, j_list):
                if verbose:
                    print('Tile',float(i),float(j))
                start_i = i * chunk_size
                end_i = (i + 1) * chunk_size 
                start_j = j * chunk_size
                end_j = (j + 1) * chunk_size
                
                amplitude = np.array(f['amplitude']\
                                     [start_i:end_i, start_j:end_j])
                amplitude = t.as_tensor(amplitude).view(dtype=t.bfloat16)
                grating = np.array(f['grating'][start_i:end_i, start_j:end_j])
                grating = t.as_tensor(grating) / 127

                # make the buttresses                
                mask = ((amplitude >= grating_max * grating)
                        * (grating >= 0)
                        * (grating <= (1-buttress_fraction)))
                
                mask = mask.to(dtype=t.float32)
                lr_mask = t.nn.functional.avg_pool2d(
                    mask.unsqueeze(0).unsqueeze(0),
                    reduction_factor)[0,0]

                start_i = i * chunk_size//reduction_factor
                end_i = (i + 1) * chunk_size//reduction_factor
                start_j = j * chunk_size//reduction_factor
                end_j = (j + 1) * chunk_size//reduction_factor
                
                lr_f['lr_rzp'][start_i:end_i, start_j:end_j] = lr_mask.numpy()


        def get_padded_chunk(i, j):
            pad_i = (overlap if i != 0 else 0)
            pad_j = (overlap if j != 0 else 0)
            start_i = i * chunk_size - pad_i
            end_i = (i + 1) * chunk_size + overlap
            start_j = j * chunk_size - pad_j
            end_j = (j + 1) * chunk_size + overlap
            amplitude = np.array(f['amplitude'][start_i:end_i, start_j:end_j])
            amplitude = t.as_tensor(amplitude).view(dtype=t.bfloat16)
            grating = np.array(f['grating'][start_i:end_i, start_j:end_j])
            grating = t.as_tensor(grating) / 127

            mask = ((amplitude >= grating_max * grating)
                    * (grating >= 0)
                    * (grating <= (1-buttress_fraction)))
            mask = mask.to(dtype=t.float32).numpy()
            return mask, pad_i, pad_j, start_i, start_j, chunk_size

        print('Working on making contours')
        chunks = (get_padded_chunk(i,j) for i,j in it.product(i_list, j_list))
        with Pool(processes=n_processes) as pool:
            if verbose:
                contour_lists = tqdm.tqdm(
                    pool.starmap(process_contours, chunks, chunksize=1),
                    total=(len(i_list)*len(j_list)), miniters=1)
            else:
                contour_lists = pool.starmap(process_contours, chunks,
                                             chunksize=1)

        contours = [c for contour_list in contour_lists for c in contour_list]
        
        print('Now cleaning the contours')
        final_contours = clean_contours_multiprocess(
            contours, n_processes=n_processes, show_progress=verbose)
        
        conversion_factor = step * 1e6 # um since this is the default gdsii unit
        converted_contours = [conversion_factor * c + offset * 1e6
                              for c in final_contours]

        # The GDSII file is called a library, which contains multiple cells.
        # We tie the precision to the step size, because if the step size is
        # too small, the field size can be limited because positions are stored
        # as int32s I believe
        lib = gdstk.Library(unit=1e-6,precision=step/10)#1e-12)
        
        # Geometry must be placed in cells.
        cell = lib.new_cell('RZP')
        
        # Create the geometry (a single rectangle) and add it to the cell.
        
        for contour in converted_contours:
            cell.add(gdstk.Polygon(contour))
            
        # Save the library in a file called 'first.gds'.
        lib.write_gds(gds_filename)

        if verbose:
            print('Result Saved')

        if view:
            # Display all cells using the internal viewer.
            gdstk.LayoutViewer(lib)

def process_contours(chunk, pad_i, pad_j, start_i, start_j, chunk_size):
    # Close contours at the edge, so this can work with non-buttressed ZPs
    chunk[:1,:] = 0
    chunk[-1:,:] = 0
    chunk[:,:1] = 0
    chunk[:,-1:] = 0
    
    contours = find_contours(chunk, level=0.5,
                             fully_connected='high')
    # Here I'm closing over the start_<i,j> and end_<i,j> variables
    def check_contour(contour):
        # we do 1: because the fist and last point are the same
        center = np.mean(contour[1:], axis=0)
        check_i = center[0] >= pad_i and center[0] < pad_i + chunk_size
        check_j = center[1] >= pad_j and center[1] < pad_j + chunk_size
        return check_i and check_j
    
    offset = np.array([start_i, start_j])

    return [offset + c for c in contours if check_contour(c)]

def max_offset(points):
    dists = np.linalg.norm(points - points[0], axis=1)
    if not np.allclose(points[0], points[-1], rtol=1e-7):
        dists = np.minimum(dists, np.linalg.norm(points - points[-1], axis=1))
        
        vec = points[-1] - points[0]
        point_dists = points[0] - points
        denom = np.linalg.norm(vec)
        num = np.abs(vec[0]*point_dists[:,1] - vec[1]*point_dists[:,0])
        line_dists = num / denom
        dists = np.minimum(dists, line_dists)

    max_idx = np.argmax(dists)
    return max_idx, dists[max_idx]


def rdp(points, epsilon=1):
    to_search = [(0, len(points)-1)]
    indices = [0, len(points)-1]
    while to_search:
        start, end = to_search.pop()
        max_idx, dist = max_offset(points[start:end+1])
        max_idx += start
        if dist > epsilon:
            indices.append(max_idx)
            if (max_idx-start) > 1:
                to_search.append((start, max_idx))
            if (end - max_idx) > 1:
                to_search.append((max_idx, end))

    return points[sorted(indices)]

    

def clean_contours(contours, epsilon=1.5, verbose=False):

    n_contours = len(contours)
    contours = (rdp(c, epsilon=epsilon) for c in contours)
    # Must be rectangles
    contours_out = []
    for i, c in enumerate(contours):
        if verbose and i % 100 == 0:
            print('Working on contour',i,'of',n_contours, flush=True)

        if len(c)>=5 and np.allclose(c[0],c[-1]):
            contours_out.append(c)

    return contours_out

def clean_contours_multiprocess(contours, n_processes=4, show_progress=False,
                                miniters=1, remove_small=True):
    
    with Pool(processes=n_processes) as pool:
        if show_progress:
            contours = list(tqdm.tqdm(pool.imap(rdp, contours, chunksize=100),
                                      total=len(contours), miniters=miniters))
        else:
            contours = list(pool.imap(rdp, contours, chunksize=100))

    if remove_small:
        contours = [c for c in contours if len(c) >=5]

    return contours
