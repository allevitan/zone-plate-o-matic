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
import multiprocessing as mp
from skimage.measure import find_contours
from scipy import optimize, spatial
import gdstk
import tqdm
import os
from matplotlib import pyplot as plt
import functools
import itertools

# TODO: In the future I could also add an "abberation" function or something
# like that so that this code could be used to design regular zone plates
# with a fixed aberration function in Fourier space, like what they use at
# SLS.


def calc_f(NZ, dr, wavelength):
    """Calculates the focal distance including wavelength-dependent corrections

    This function calculates the focal distance of a zone plate with a specified
    number of zones, design wavelength, and outer zone width.

    The function includes a wavelength-dependent correction - in other words, it
    assumes that the zone plate has been designed to work properly at high
    numerical apertures at the specified wavelength, and calculates the focal
    distance at that design wavelength.
    
    Parameters
    ----------
    NZ : int
        The number of zones in the zone plate (with the starting zone defined to be on the optical axis)
    dr : float
        The outer zone width of the zone plate, in meters
    wavelength : float
        The design wavelength, in meters

    Returns
    -------
    f : float
        The focal distance of the optic at the design wavelength, in meters
    """
    # This is the simplified calculation that's usually used
    # f =  dr**2 * 4 * NZ / wavelength

    # And this is a corrected one
    a = wavelength**2    
    b = NZ * wavelength**3 - 4 * dr**2 * NZ * wavelength
    c = NZ**2 * wavelength**2 * ( wavelength**2 / 4 - dr**2 )
    f = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

    return f


def calc_diameter(NZ, dr, wavelength=None):
    """Calculates the diameter of an optic including a wavelength-dependent correction

    If no wavelength is given, it simply returns 4 * NZ * dr, the limit for an
    infinitessimally small wavelength.

    If a wavelength is given, it calculates the radius of an optic which has
    been corrected to work properly at high numerical apertures at the defined
    wavelength.
    
    Parameters
    ----------
    NZ : int
        The number of zones in the zone plate (with the starting zone defined to be on the optical axis)
    dr : float
        The outer zone width of the zone plate, in meters
    wavelength : float
        The design wavelength, in meters. If not specified, the result is calculated for the limit as wavelength -> 0

    Returns
    -------
    diameter : float
        The diameter of the optic at the specified zone number
    """

    if wavelength is None:
        return 4 * NZ * dr
    else:
        f = calc_f(NZ, dr, wavelength)
        r = np.sqrt(NZ*f*wavelength + NZ**2 * wavelength**2 / 4)
        return 2 * r

def optimize_off_axis_z_fixed_r(f, center_zone, wavelength, target_r):

    def calc_r(zone):
        return np.sqrt(zone*f*wavelength + zone**2 * wavelength**2 / 4)

    def calc_off_axis_r(num_zones):
        inner_rs = calc_r(center_zone - num_zones/2)
        outer_rs = calc_r(center_zone + num_zones/2)
        return outer_rs - inner_rs

    def optimization_target(num_zones):
        return 1e9 * (calc_off_axis_r(num_zones) - target_r)**2

    res = optimize.minimize(optimization_target, 1)
    return res['x']
        

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
    f = calc_f(NZ, dr, wavelength)
    optic_r = calc_diameter(NZ, dr, wavelength=wavelength) / 2
    window = ((-optic_r,optic_r), (-optic_r, optic_r))

    def window_function(X,Y, Angle, R):
        # TODO: This function needs to also include the apodization
        window = t.logical_and(R > bs_ratio * optic_r, R < optic_r)

        if apodization_ratio!=0:
            window = window.to(dtype=R.dtype)
            
            aw = apodization_width # just a shorthand
            bs_r = bs_ratio * optic_r

            outer_edge = t.logical_and(R > (optic_r - aw), R < optic_r)
            inner_edge = t.logical_and(R < (bs_r + aw), R > bs_r)
            
            # This is a Tukey window
            window[outer_edge] *= \
                (1 - t.cos(np.pi * (R[outer_edge] - optic_r) / aw) ) / 2
            window[inner_edge] *= \
                (1 - t.cos(np.pi * (R[inner_edge] - bs_r) / aw) ) / 2

        return window

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

def design_custom_focus_rzp(
        U_0,
        dr,
        NZ,
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
        verbose=False,
):
    
    if buttress_spacing is None:
        buttress_spacing = 4 * dr

    NS = U_0.shape[0] * step / dr # Approximate
    apodization_width = apodization_ratio * 8 * NZ * dr / NS
    
    # Calculate the focal distance & radius of the optic
    f = calc_f(NZ, dr, wavelength)
    optic_r = calc_diameter(NZ, dr, wavelength=wavelength) / 2
    window = ((-optic_r,optic_r), (-optic_r, optic_r))

    def window_function(X,Y, Angle, R):
        window = t.logical_and(R > bs_ratio * optic_r, R < optic_r)

        if apodization_ratio!=0:
            window = window.to(dtype=R.dtype)
            
            aw = apodization_width # just a shorthand
            bs_r = bs_ratio * optic_r

            outer_edge = t.logical_and(R > (optic_r - aw), R < optic_r)
            inner_edge = t.logical_and(R < (bs_r + aw), R > bs_r)
            
            # This is a Tukey window
            window[outer_edge] *= \
                (1 - t.cos(np.pi * (R[outer_edge] - optic_r) / aw) ) / 2
            window[inner_edge] *= \
                (1 - t.cos(np.pi * (R[inner_edge] - bs_r) / aw) ) / 2

        return window

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
        f.create_dataset('buttress_spacing', data=[buttress_spacing])
        f.create_dataset('buttress_deviation', data=[buttress_deviation])
        f.create_dataset('apodization_ratio', data=[apodization_ratio])
        f.create_dataset('beamstop_ratio', data=[bs_ratio])


# An important constant for the sunflower array    
golden_angle = (np.pi * (3 - np.sqrt(5)))
golden_ratio = 0.618033988749

def place_sunflower_zps(dr, n_frames, wavelength,
                        inner_zone_index, mini_zp_spacing, mini_zp_width=None,
                        mini_zp_length_factor=1, mini_zps_per_frame=3,
                        special_order=False,
                        use_multi_spiral=True,
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
    special_order : bool, optional
        If True, and mini_zps_per_frame=2, it swaps the 2nd and 3rd mini zp, 6th and 7th, and so on.
    use_multi_spiral : bool, optional
        If True, it uses multiple spirals to place the ZPs such that the set of n_zps_per_frame are equally spaced

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

    if use_multi_spiral:
        for n in range(mini_zps_per_frame):
            phis[n::mini_zps_per_frame] = ((2 * np.pi / mini_zps_per_frame)
                    * (n  + golden_ratio * np.arange(n_frames)))

    NZ = inner_zone_index + (n_frames-1) * mini_zp_spacing + mini_zp_width

    f = calc_f(NZ, dr, wavelength)

    frame_idx = np.repeat(np.arange(0,n_frames), mini_zps_per_frame)

    if mini_zps_per_frame == 2 and special_order==True:
        for idx in range(1, len(frame_idx), 4):
            frame_idx[idx], frame_idx[idx+1] = frame_idx[idx+1], frame_idx[idx]

    
    starting_zones = inner_zone_index + frame_idx * mini_zp_spacing
    ending_zones = starting_zones + mini_zp_width
    center_zones = starting_zones + mini_zp_width // 2
     
    inner_rs = np.sqrt(starting_zones * f * wavelength + starting_zones**2 * wavelength**2 / 4)
    outer_rs = np.sqrt(ending_zones * f * wavelength + ending_zones**2 * wavelength**2 / 4)
    center_rs = np.sqrt(center_zones * f * wavelength + center_zones**2 * wavelength**2 / 4)

    if equal_width:
        # In this case, we rerun all the calculations to ensure all the ZPs
        # have the same width defined by the outer frame's natural width
        width = outer_rs[-1] - inner_rs[-1]
        off_axis_zp_nzones = [
            optimize_off_axis_z_fixed_r(f, cz, wavelength, width)
            for cz in center_zones
        ]
        # Just clean up the number of zones so it's an even integer
        # This still has higher precision than we'll need
        off_axis_zp_nzones = np.array([
            int(np.round(num_zones/2))*2
            for num_zones in off_axis_zp_nzones
        ])
        outer_rs = np.sqrt(center_zones * f * wavelength + center_zones**2 * wavelength**2 / 4)

        starting_zones = center_zones - off_axis_zp_nzones // 2
        ending_zones = center_zones + off_axis_zp_nzones // 2
        
        inner_rs = np.sqrt(starting_zones * f * wavelength + starting_zones**2 * wavelength**2 / 4)
        outer_rs = np.sqrt(ending_zones * f * wavelength + ending_zones**2 * wavelength**2 / 4)
        center_rs = np.sqrt(center_zones * f * wavelength + center_zones**2 * wavelength**2 / 4)

    # This is (roughly) the area of the optic occupied by a region associated
    # with each frame
    frame_area = np.pi * (mini_zp_spacing * f * wavelength)
    # And this is a rough measure of a good mini-zp radius
    base_radius = np.sqrt(frame_area / mini_zps_per_frame) / 2

    xs = (center_rs) * np.cos(phis)
    ys = (center_rs) * np.sin(phis)

    design = [
        { 'phi': phi,
          'x': x,
          'y': y,
          'start_zone': sz,
          'end_zone': ez,
          'center_zone' : cz,
          'f': f,
          'frame_id': frame_id,
          'dr': dr,
          'wavelength': wavelength,
          'inner_r': inner_r,
          'outer_r': outer_r,
          'radius': base_radius * mini_zp_length_factor,
        } for (phi, x, y, sz, ez, cz, frame_id, inner_r, outer_r)
        in zip(
            phis,
            xs,
            ys,
            starting_zones,
            ending_zones,
            center_zones,
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
    

def plan_sunflower_array(
        dr,
        n_frames,
        wavelength,
        focus_diameter,
        inner_zone_index,
        mini_zp_spacing,
        step,
        output_filename,
        dtype=t.complex128,
        mini_zp_width=None,
        mini_zp_length_factor=1,
        mini_zps_per_frame=3,
        equal_width=True,
        phi_0=0,
        buttress_spacing=None,
        buttress_deviation=0.15,
        special_order=False,
        use_multi_spiral=True,
        tiling_style='alternating',
        apodization_ratio=0):
    """Saves a sunflower zone plate array design plan to a file"""

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
        phi_0=phi_0,
        special_order=special_order,
        use_multi_spiral=use_multi_spiral,
    )


    # The real-valued dtype corresponding to the given complex-valued dtype.
    real_dtype = t.real(t.ones(1,dtype=dtype)).dtype

    # We create the speckle texture for the target focal spot, at a resolution
    # which matches the final ozw
    base_input_shape = [int(focus_diameter // dr) + 1]*2
    U_0 = t.exp(2j*np.pi*t.rand(*base_input_shape,
                                dtype=real_dtype))

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

    # Calculation of the focal distance, including the
    # n**2 * lambda**2 term in the zone plate equation
    NZ = inner_zone_index + (n_frames-1) * mini_zp_spacing + mini_zp_width

    a = wavelength**2    
    b = NZ * wavelength**3 - 4 * dr**2 * NZ * wavelength
    c = NZ**2 * wavelength**2 * ( wavelength**2 / 4 - dr**2 )
    f = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    
    hc = 1.23984e-6 # in m*eV
    
    design_energy = hc / wavelength
    A1 = f * wavelength / hc
    optic_diameter = 4 * NZ * dr
    
    with h5py.File(output_filename, 'w') as output:
        output.create_dataset('design_focus', data=U_0.numpy())
        output.create_dataset('pix_size', data=[step])
        output.create_dataset('outer_zone_width', data=[dr])
        output.create_dataset('n_frames', data=[n_frames])
        output.create_dataset('n_zps_per_frame', data=[mini_zps_per_frame])
        output.create_dataset('design_wavelength', data=[wavelength])
        output.create_dataset('design_energy', data=[design_energy])
        output.create_dataset('focus_diameter', data=[focus_diameter])
        output.create_dataset('inner_zone_index', data=[inner_zone_index])
        output.create_dataset('zp_spacing', data=[mini_zp_spacing])
        output.create_dataset('zp_width', data=[mini_zp_width])
        output.create_dataset('zp_length_factor', data=[mini_zp_length_factor])
        output.create_dataset('equal_width', data=[equal_width])
        output.create_dataset('phi_0', data=[phi_0])
        output.create_dataset('buttress_spacing', data=[buttress_spacing])
        output.create_dataset('buttress_deviation', data=[buttress_deviation])
        output.create_dataset('tiling_style', data=tiling_style)
        output.create_dataset('apodization_ratio', data=[apodization_ratio])
        output.create_dataset('design_focal_distance', data=[f])
        output.create_dataset('A1', data=[A1])
        output.create_dataset('optic_diameter', data=[optic_diameter])
        output.create_dataset('total_n_zones', data=[NZ])
        
        for idx, zp in enumerate(zp_locations):
            grp = output.create_group('ZP%03d' % idx)
            for key in zp.keys():
                grp.create_dataset(key, data=[zp[key]])


def design_sunflower_array(plan_file, zone_plate_index,
                           output_file,
                           tile_size=None,
                           device='cpu',
                           verbose=False):

    # TODO: allow a few overrides, for example the buttress spacing or
    # deviation

    with h5py.File(plan_file, 'r') as plan:
        U_0 = t.as_tensor(np.array(plan['design_focus']))
        dtype = U_0.dtype
        f = float(np.array(plan['design_focal_distance']).ravel()[0])
        dr = float(np.array(plan['outer_zone_width']).ravel()[0])
        focus_diameter = float(np.array(plan['focus_diameter']).ravel()[0])
        wavelength = float(np.array(plan['design_wavelength']).ravel()[0])
        step = float(np.array(plan['pix_size']).ravel()[0])
        buttress_spacing = float(np.array(plan['buttress_spacing']).ravel()[0])
        buttress_deviation = \
            float(np.array(plan['buttress_deviation']).ravel()[0])
        apodization_ratio = \
            float(np.array(plan['apodization_ratio']).ravel()[0])
        
        tiling_style = plan['tiling_style'][()].decode()

        mini_zp = plan['ZP%03d/' % zone_plate_index]
        x = float(np.array(mini_zp['x']).ravel()[0])
        y = float(np.array(mini_zp['y']).ravel()[0])
        r = float(np.array(mini_zp['radius']).ravel()[0])
        window = ((x - r, x + r), (y - r, y + r))
        inner_r = float(np.array(mini_zp['inner_r']).ravel()[0])
        outer_r = float(np.array(mini_zp['outer_r']).ravel()[0])

    if not os.path.exists(output_file):
        # It appears that in a HPC environment, if many jobs are launched at
        # the same time, sometimes os.path.exists(output) will return false
        # when in reality the folder exists. So, we also do a try/except here
        try:
            os.mkdir(output_file)
        except:
            print('Caught an exception trying to make the folder')
            pass

    if not os.path.isdir(output_file):
        raise FileExistsError('Output design folder already exists but is not a directory')
    
    zp_filename = output_file + ('/ZP%03d.h5' % zone_plate_index)

    # This should be equivalent to the form given in the "normal" design
    # code, under the paraxial approximation. But, this one only relies on
    # parameters that still have clear meaning for the sunflower RZP
    # It's just the apodization_ratio multiplied by an estimate of the
    # speckle size in the optic plane
    apodization_width = apodization_ratio * 2 * (
        f * wavelength / focus_diameter)
    
    def window_function(X, Y, Angle, R):
        radial_band = t.logical_and(R > inner_r, R < outer_r)
        mini_R = t.sqrt((X-x)**2 + (Y-y)**2)
        disk = mini_R < r
        window =  t.logical_and(radial_band, disk)

        if apodization_ratio!=0:
            window = window.to(dtype=R.dtype)
            
            aw = apodization_width # just a shorthand
            
            outer_edge = t.logical_and(R > (outer_r - aw), R < outer_r)
            inner_edge = t.logical_and(R < (inner_r + aw), R > inner_r)
            radial_band = t.logical_and(mini_R > (r - aw), mini_R < r)
            
            # This is a Tukey window
            window[outer_edge] *= \
                (1 - t.cos(np.pi * (R[outer_edge] - outer_r) / aw) ) / 2
            window[inner_edge] *= \
                (1 - t.cos(np.pi * (R[inner_edge] - inner_r) / aw) ) / 2
            window[radial_band] *= \
                (1 - t.cos(np.pi * (mini_R[radial_band] - r) / aw) ) / 2

            # There will be overlaps between the radial band and both the
            # inner and outer edges. I think that just multiplying the
            # relevant Tukey windows will create something that is at
            # least still has a continuous derivative everywhere
            
        return window

    
    design_grating_hologram(U_0,
                            f,
                            window,
                            window_function,
                            wavelength,
                            step,
                            zp_filename,
                            dtype=dtype,
                            device=device,
                            tile_size=tile_size,
                            buttress_spacing=buttress_spacing,
                            buttress_deviation=buttress_deviation,
                            outer_r=outer_r,
                            tiling_style=tiling_style,
                            verbose=verbose)
        
    with h5py.File(zp_filename, 'r+') as output:
        apodization_ratio = apodization_ratio
        output.create_dataset('outer_zone_width', data=[dr])
        output.create_dataset('focus_diameter', data=[focus_diameter])
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
                            outer_r=None, # Definition of the outer radius where the buttress spacing is correct, default is the edge of the window.
                            tiling_style='alternating',
                            compression='lzf',
                            verbose=False):

    # The COM of the input window is defined as (0,0), and the output window
    # is defined with respect to that.
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
        output_store.create_dataset('design_wavelength', data=[wavelength])
        output_store['design_wavelength'].attrs['units'] = 'm'
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
        # For input, it has to be a list, because we're going to iterate
        # over it repeatedly
        input_combos = list(it.product(input_is, input_js))
        output_combos = list(it.product(output_is, output_js))
        n_input = len(input_is)*len(input_js)
        n_output = len(output_is)*len(output_js)
        
        for idx, (out_i, out_j) in enumerate(output_combos):
            if verbose:
                print('Working on output tile', idx+1,'of',n_output, flush=True)

            # We calculate the window function first,
            # because if it's all zeroes we don't
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

            # We allocate this outside of the inner, "in_i/in_j" loop, so
            # that we can always be adding to it.
            out_tile = t.zeros(tile_shape, dtype=dtype, device=U_0.device)

            if t.all(t.eq(window_fn_output,0)):
                # No point in doing the expensive calculation if it's all just
                # going to be masked off
                if verbose:
                    print('Output tile fully masked; skipping', flush=True)

            else:

                # Now we have to iterate through all the input tiles to get
                # the full output wavefield
                for in_idx, (in_i, in_j) in enumerate(input_combos):
                    if verbose:
                        print('Working on input tile', in_idx+1,
                              'of', n_input, flush=True)
            

                    # We allocate a tile for the input and then fill it
                    # What this does is ensure a standard size, even when the 
                    # selection overlaps the edge.
                    in_tile = t.zeros(tile_shape, dtype=dtype,
                                      device=U_0.device)
                    in_selection = U_0[in_i*tile_shape[0]:
                                       (in_i+1) * tile_shape[0],
                                       in_j*tile_shape[1]:
                                       (in_j+1) * tile_shape[1]]

                    in_tile[:in_selection.shape[0],
                            :in_selection.shape[1]] = in_selection
                    
                    tile_offset = [(in_i - out_i) * tile_shape[0]
                                   + base_offset[0],
                                   (in_j - out_j) * tile_shape[1]
                                   + base_offset[1]]
                    
                    out_tile += propagation.FFT_DI(in_tile.to(device=device),
                                                   -f, wavelength, step,
                                                   offset=tile_offset).cpu()
           
                out_tile[window_fn_output == 0] = 0
            
                
            # Now we set up the grating
            # The following line is formally correct:
            # 
            # perfect_zp_phase_base = ((np.sqrt(f**2 + Rs2) - f) 
            #     * (2*np.pi/wavelength))

            # But, it's better to use the form below, because the form above
            # will lead to numerical stability issues when f >> R, commonly
            # the case. They are equivalent for calculations with reals.

            perfect_zp_phase = (2*np.pi/wavelength) * (
                Rs2 / (np.sqrt(f**2 + Rs**2) + f))

            # This creates an "effective zone phase" which is constant within
            # each zone, and jumps sharply at the zone transition. This is
            # useful for constructing a globally consistent pattern of
            # buttresses out of only locally available information.
            zone_phase = 0.5 * perfect_zp_phase + 0.5 * t.angle(out_tile)
            
            # Another use for it is constructing a quantity which is always
            # very close to the radius, but only jumps at zone transitions.
            # This lets us do nice stuff like dice the ZP up into radial
            # zones, making sure that the transitions always occur away from
            # patterned zones
            effective_Rs =  ( np.sqrt(zone_phase * wavelength / np.pi) *
                     np.sqrt((zone_phase * wavelength / np.pi) + 2*f))
            
            buttress_regions = t.floor((t.log(effective_Rs)-np.log(outer_r))
                                       / np.log(1-buttress_deviation))
            nominal_Rs = t.exp(buttress_regions * np.log(1-buttress_deviation)
                               + np.log(outer_r))
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


def realize_design(filename, gds_filename,
                   buttress_width=15e-9, grating_max=0.9,
                   chunk_size=2048, n_processes=6,
                   verbose=False, overlap=512, view=False,
                   min_dimension=None,
                   use_rectangles=False):
    """Makes contours and the low resolution version"""
    with h5py.File(filename,'r') as f:
        step = float(f['step'][()])
        buttress_spacing = float(f['buttress_spacing'][()])
        buttress_fraction = buttress_width / (buttress_spacing)
        offset = np.array(f['offset'])        

        shape = f['amplitude'].shape
        offset = np.array(f['offset'])

        i_list = t.arange(propagation.get_num_tiles(shape[0], chunk_size))
        j_list = t.arange(propagation.get_num_tiles(shape[1], chunk_size))

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

        if verbose:
            print('Working on making contours. Note that progress bar is for inputs, so the last\nchunks may still be processing once it reaches 100%.')
        chunks = (get_padded_chunk(i,j) for i,j in it.product(i_list, j_list))
        with mp.get_context('spawn').Pool(processes=n_processes) as pool:
            if verbose:
                # We use tqdm on the inputs because starmap waits until
                # the end to return anything
                tqdm_chunks = tqdm.tqdm(chunks, 
                                        total=(len(i_list)*len(j_list)),
                                        miniters=1)
                contour_lists = \
                    pool.starmap(process_contours, tqdm_chunks, chunksize=1)
                
            else:
                contour_lists = pool.starmap(process_contours, chunks,
                                             chunksize=1)
        contours = [c for contour_list in contour_lists for c in contour_list]
        
        if verbose:
            print('Now cleaning the contours')
        if use_rectangles:
            print('Output contours will be defined as rectangles')
            final_contours = clean_contours_multiprocess(
                contours, n_processes=n_processes, show_progress=verbose,
                max_points=8, epsilon=0.8, use_rectangles=use_rectangles,
                min_dimension=(min_dimension / step))
        else:
            print('Output contours will be defined as arbitrary polygons')
            final_contours = clean_contours_multiprocess(
                contours, n_processes=n_processes, show_progress=verbose,
                use_rectangles=use_rectangles)
            
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


def rdp(points, epsilon=1, max_points=None):
    # Note that the whole max_points thing isn't really part of the classic
    # RDP algorithm. So, I don't love how this one handles it, but this
    # implementation is so much faster than the rdp_old implementation
    # that I think it's better to go with this one
    indices = [0, len(points)-1]
    while max_points is None or len(indices) < max_points:
        max_dist = epsilon

        indices_to_add = []
        for idx in range(len(indices)-1):
            start = indices[idx]
            end = indices[idx + 1]
            max_idx, dist = max_offset(points[start:end+1])
            if dist > epsilon:
                indices_to_add.append((dist, max_idx + start))

        if len(indices_to_add) == 0:
            break
        elif max_points is None:
            indices.extend(
                [ index for dist, index in sorted(indices_to_add) ]
            )
            indices = sorted(indices)
        else:
            # we will add up to n_to_add
            n_to_add = max_points - len(indices)
             # sorts lexicographically on tuple
            indices.extend(
                [ index for dist, index in sorted(indices_to_add)[:n_to_add] ]
            )
            indices = sorted(indices)
            if len(indices) >= max_points:
                break
            
    return points[indices]
    

# NOTE: this function was generated by ChatGPT 4o on June 02, 2025
# It was provided under a CC0 public domain dedication waiver, but
# be aware of this in case any licensing issues appear related to AI
# generated code in the future
def minimum_bounding_rectangle(points):
    hull = spatial.ConvexHull(points)
    hull_points = points[hull.vertices]

    # Calculate edge angles
    edges = np.diff(hull_points, axis=0, append=hull_points[:1])
    angles = np.arctan2(edges[:,1], edges[:,0])
    unique_angles = np.unique(np.abs(angles % (np.pi / 2)))

    min_area = np.inf
    best_rect = None

    for angle in unique_angles:
        # Rotation matrix
        R = np.array([
            [np.cos(-angle), -np.sin(-angle)],
            [np.sin(-angle),  np.cos(-angle)]
        ])
        rot_points = hull_points @ R.T

        min_x, min_y = np.min(rot_points, axis=0)
        max_x, max_y = np.max(rot_points, axis=0)
        area = (max_x - min_x) * (max_y - min_y)

        if area < min_area:
            min_area = area
            # Rectangle corners in rotated space
            rect = np.array([
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y],
                [min_x, min_y]
            ])
            # Rotate back to original space
            best_rect = rect @ R

    return best_rect


# NOTE: this function was generated by ChatGPT 4o on June 02, 2025
# It was provided under a CC0 public domain dedication waiver, but
# be aware of this in case any licensing issues appear related to AI
# generated code in the future
def polygon_area(points):
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


# NOTE: original_contour shouldn't be closed! i.e. the first and last
# points should not be equal
#
# Shrinks the rectangle by an equal amount along all sides until
# it's area matches that of the original contour
#
def shrink_rect(rect_points, original_contour):
    original_area = polygon_area(original_contour)
    rect_area = polygon_area(rect_points)
    ax1 = rect_points[1] - rect_points[0]
    ax2 = rect_points[2] - rect_points[1]

    l1 = np.linalg.norm(ax1)
    l2 = np.linalg.norm(ax2)

    ax1 = ax1 / l1
    ax2 = ax2 / l2

    a = 4
    b = -2 * (l1 + l2)
    c = rect_area - original_area
    shrink_amount = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)

    
    # I'm sure a smarter person or AI model than me could do this more elegantly
    output_points = np.copy(np.array(rect_points))
    output_points[0] += ax1 * shrink_amount
    output_points[1] -= ax1 * shrink_amount
    output_points[2] -= ax1 * shrink_amount
    output_points[3] += ax1 * shrink_amount
    output_points[0] += ax2 * shrink_amount
    output_points[1] += ax2 * shrink_amount
    output_points[2] -= ax2 * shrink_amount
    output_points[3] -= ax2 * shrink_amount

    return output_points


def calc_min_dimension(rect_points):
    l1 = np.linalg.norm(rect_points[1] - rect_points[0])
    l2 = np.linalg.norm(rect_points[2] - rect_points[1])
    return np.minimum(l1, l2)


def single_clean_step(contour,epsilon=1,max_points=None,use_rectangles=False):
    if use_rectangles:
        if len(contour) >=4:
            rect_points = minimum_bounding_rectangle(contour)
            rect_points = shrink_rect(rect_points, contour[:-1])
            #rect_area = polygon_area(rect_points)
            #original_area = polygon_area(contour[:-1])
            #rect_points = scale_polygon(
            #    rect_points, np.sqrt(original_area/rect_area)
            #)
            points = np.concatenate([rect_points, rect_points[:1]], axis=0)
        else:
            points = contour
    else:
        points = rdp(contour, epsilon=epsilon, max_points=max_points)

    return points


def clean_contours_multiprocess(contours, n_processes=4, show_progress=False,
                                miniters=1, remove_small=True, epsilon=1,
                                max_points=None, use_rectangles=False,
                                min_dimension=None):
    print('max points', max_points)
    print('epsilon', epsilon)
    print('use rectangles?', use_rectangles)
    
    single_step = functools.partial(
        single_clean_step,
        epsilon=epsilon,
        max_points=max_points,
        use_rectangles=use_rectangles)

    
    
    with mp.get_context('spawn').Pool(processes=n_processes) as pool:
        if show_progress:
            contours = list(tqdm.tqdm(pool.imap(single_step,
                                                contours, chunksize=100),
                                      total=len(contours), miniters=miniters))
        else:
            contours = list(pool.imap(single_step, contours, chunksize=100))

    print('Removing contours with fewer than 4 points')
    if remove_small:
        contours = [c for c in contours if len(c) >=5]

    print('Removing contours with a critical dimensions that is too small')
    if min_dimension is not None and use_rectangles:
        contours = [c for c in contours
                    if calc_min_dimension(c) >= min_dimension]

    return contours
