import numpy as np
from matplotlib import pyplot as plt

golden_angle = (np.pi * (3 - np.sqrt(5)))

def define_zp_locations(n_frames, ozw, wavelength,
                        inner_zone_index, mini_zp_spacing, mini_zp_width=None,
                        mini_zp_length_factor=1, mini_zps_per_frame=3,
                        equal_width=True, phi_0=0):
    """Defines the properties of the mini zone plates in a multi-frame RZP

    Parameters
    ----------
    n_frames : int
        The number of frames (distinct mini-zone plate radii) to design for
    ozw : float
        The outer zone width of the entire multi-frame zone plate, in meters
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

    NZ = inner_zone_index + n_frames * mini_zp_spacing

    # Calculation of the focal distance, including the n**2 * lambda**2 term in the zone plate equation
    a = wavelength**2
    b = 2 * NZ * wavelength**3 - 4 * ozw**2 * NZ * wavelength
    c = NZ**2 * wavelength**2 * ( wavelength**2 - 4 * ozw**2 / 2 )
    f = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    # f =  ozw **2 * 4 * NZ / wavelength ## This is the simplified calculation that's usually used

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
          'ozw': ozw,
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


def inspect_zp_design(design, pix_size=1e-6):
    plt.figure()
    max_r = max([mini_zp['outer_r'] for mini_zp in design])
    xs = np.arange(-max_r, max_r, pix_size)
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

design = define_zp_locations(6, 40e-9, 2.066e-10, 2000, 1000, mini_zp_width=1000)
inspect_zp_design(design)
design = define_zp_locations(12, 40e-9, 2.066e-10, 2000, 500, mini_zp_width=500)
inspect_zp_design(design)
plt.show()