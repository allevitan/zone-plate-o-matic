"""
This file contains functions for light propagation using
the FFT-DI method due to Shen and Wang (2006), doi:10.1364/AO.45.001102.

This method is advantageous because it accomplishes light propagation with
open boundary conditions in one step. The two major downsides are that:

1) It fails below a minimum distance, dependent on the sampling rate
2) It requires operating on arrays which are double the size of the wavefield

The first condition is rarely an issue for ZP design. The second issue is
more troublesome, and as a result a method is used which splits the propagation
calculation into many individual tiles. This cuts the memory requirements down
significantly to (if needed) less than what is needed to hold the
final result in memory.

Author: Abraham Levitan
Dates: January 2022 through September 2023
"""

import torch as t
import numpy as np
import gc
import itertools as it

__all__ = ['FFT_DI', 'FFT_DI_Tiled', 'create_G', 'get_num_tiles',
           'far_field', 'inverse_far_field', 'pad_to_shape']


def create_G(U_0, z, wavelength, step, offset=[0,0]):
    """Creates the G array for a specific propagation problem.

    The output is the array G, expressed in real space, as defined
    in Shen and Wang (2006), doi:10.1364/AO.45.001102.

    It can incorporate an optional offset between the
    input and output planes, which helps with tiled light propagation.

    This function is used internally by FFT_DI, but in some cases it can
    be useful to only generate G once and reuse it many times.

    Parameters
    ----------
    U_0 : torch.tensor
        An example light field to ensure compatibility with. The output will
        match it's shape, precision, and device (CPU or GPU).
    z : float
        The distance separating the input and output planes, in meters
    wavelength : float
        The wavelength of light, in meters
    step : float
        The pitch of the discretized light field array, in meters.
    offset : array
        Optional, the (i,j) vector in units of pixels which maps the corner
        of the input array to corner of the output array.

    Returns
    -------
    G : torch.tensor
        A G array for the FFT_DI method

    """

    # This is the real-valued dtype which matches the precision of U_0
    real_dtype = t.real(t.ones(1,dtype=U_0.dtype)).dtype

    # This is the final output shape we'd like to achieve
    padto = (U_0.shape[0]*2-1, U_0.shape[1]*2-1)

    # Including the sign(z) allows us to deal with reverse propagation,
    # while still attenuating frequencies above k_0
    ik = np.sign(z) * 2j * np.pi/wavelength

    # Now we calculate the distance to each pixel
    xs = (t.arange(padto[0]) - U_0.shape[0] + 1 - offset[0]) * step[0]
    xs = xs.to(dtype=real_dtype, device=U_0.device)
    ys = (t.arange(padto[1]) - U_0.shape[1] + 1 - offset[1]) * step[1]
    ys = ys.to(dtype=real_dtype, device=U_0.device)
    X,Y = t.meshgrid(xs, ys, indexing='ij')
    R = t.sqrt(X**2 + Y**2 + z**2)

    # Probably overkill but maybe helpful for memory
    del X, Y
    gc.collect()

    # Now we finally calculate G
    prefactor = (step[0] * step[1] / (2 * np.pi)) * np.abs(z)
    G = prefactor * t.exp(ik * R) * (1/R - ik) / R**2

    del R
    gc.collect()

    return G


def FFT_DI(U_0, z, wavelength, step, offset=[0,0], verbose=False):
    """Uses FFT-DI to propagate a light field

    This function performs light propagation under open boundary conditions
    using FFT-DI (Shen and Wang (2006), doi:10.1364/AO.45.001102).

    It can be quite memory intensive, so be warned!

    Parameters
    ----------
    U_0 : torch.tensor
        The input light field
    z : float
        The distance separating the input and output planes, in meters
    wavelength : float
        The wavelength of light, in meters
    step : float
        The pitch of the discretized light field array, in meters.
    offset : array
        Optional, the (i,j) vector in units of pixels which maps the corner
        of the input array to corner of the output array.
    verbose : bool
        Optional, default is False. Whether to print out progress reports

    Returns
    -------
    U_z : torch.tensor
        The light field at the output plane

    """

    if verbose:
        print('H being created', flush=True)

    # We have to embed the input array within an array twice it's size (H)
    padto = (U_0.shape[0]*2-1, U_0.shape[1]*2-1)
    H = t.zeros(padto, dtype=U_0.dtype, device=U_0.device)
    H[:U_0.shape[0],:U_0.shape[1]] = U_0

    if verbose:
        print('H created, performing FFT')

    H = t.fft.fft2(H, out=H)

    if verbose:
        print('FFT complete on H, creating G')

    # The magic of this method that allows for open boundary conditions is the
    # fact that we create G in real space, and then calculate it's FFT, instead
    # of using e.g. the angular spectrum propagator directly in Fourier space.
    G = create_G(U_0, z, wavelength, step, offset=offset)

    if verbose:
        print('G created, performing FFT')

    G = t.fft.fft2(G, out=G)

    if verbose:
        print('FFT complete on G, multiplying G and H')

    G *= H

    del H
    gc.collect()

    if verbose:
        print('Multiplication complete, performing ifft')

    G = t.fft.ifft2(G, out=G)

    if verbose:
        print('IFFT Complete, copying out result to correctly sized array')

    # This particular code pattern explicitly copies the data to a new array to
    # ensure that the reference to larger array is dropped so it can be deleted
    Q = t.empty_like(U_0)
    Q[:,:] = G[U_0.shape[0]-1:2*U_0.shape[0]-1,
               U_0.shape[1]-1:2*U_0.shape[1]-1]

    del G
    gc.collect()

    return Q


def get_num_tiles(size, tile_size):
    """Returns the number of tiles needed to cover a large region

    Parameters
    ----------
    size : int or array(int)
        The size of the array which needs to be tiled
    tile_size : int or array(int)
        The size of the tiles

    Returns
    -------
    n_tiles : int or array(int)
        The number of tiles needed along each dimension
    """
    return (size - 1) // tile_size + 1


def FFT_DI_tiled(U_0, z, wavelength, step,
                 offset=[0,0],
                 output_shape=None,
                 tile_shape=None,
                 calculation_device=None,
                 output_dtype=None,
                 output_device=None,
                 output_function=lambda x:x,
                 verbose=False,
                ):
    """Uses FFT-DI to propagate a light field, breaking the problem into tiles.

    Because FFT-DI uses open boundary conditions, it's easy to break a large
    problem into many subproblems by tiling both the input and output planes.

    This has several consequences:
    - It can have a substantially reduced memory impact
    - It is much faster for unequally sized arrays

    To take full advantage of these benefits, a few additional options are
    available when compared to FFT_DI.

    In most cases, choose a tile size which is
    - Equal to or smaller than the smaller of the two arrays
    - A multiple of two, or a highly composite number (for FFTs)
    - As large as your memory limits allow

    Parameters
    ----------
    U_0 : torch.tensor
        The input light field
    z : float
        The distance separating the input and output planes, in meters
    wavelength : float
        The wavelength of light, in meters
    step : float
        The pitch of the discretized light field array, in meters.
    offset : array(int)
        Optional, the (i,j) vector in units of pixels which maps the corner
        of the input array to corner of the output array.
    output_shape : array(int)
        Optional, the shape of the output field. Default is equal to the shape of U_0
    tile_shape : array(int)
        Optional, the shape of the tiles to use. Default is the minimum of the input
        and output shapes.
    verbose : bool
        Optional, default is False. Whether to print out progress reports

    Returns
    -------
    U_z : torch.tensor
        The light field at the output plane, or a derived quantity

    """

    # First, we set all the defaults
    if output_shape is None:
        output_shape = U_0.shape

    if tile_shape is None:
        tile_shape = list(np.minimum(np.array(output_shape),
                                     np.array(U_0.shape)))
        print('Tile shape:', tile_shape, flush=True)

    if calculation_device is None:
        calculation_device = U_0.device

    if output_dtype is None:
        output_dtype=U_0.dtype

    if output_device is None:
        output_device = U_0.device

    # Next we create the output array. It's good to do this first, because sometimes it's
    # too much to hold in memory, and if so we want it to crash quickly!
    output = t.zeros(output_shape, dtype=output_dtype, device=output_device)

    # Now we create the list of all the tile combinations (input/output)
    output_is = t.arange(get_num_tiles(output_shape[0], tile_shape[0]))
    output_js = t.arange(get_num_tiles(output_shape[1], tile_shape[1]))
    input_is = t.arange(get_num_tiles(U_0.shape[0], tile_shape[0]))
    input_js = t.arange(get_num_tiles(U_0.shape[1], tile_shape[1]))
    combos = it.product(input_is, input_js, output_is, output_js)
    n = len(output_is)*len(output_js)*len(input_is)*len(input_js)

    for idx, (in_i, in_j, out_i, out_j) in enumerate(combos):
        if verbose:
            print('Working on tile',idx+1,'of',n, flush=True)
        
        # We get a standard input tile in way which will still give us a
        # uniform size when the tile selection overlaps the edge.
        in_tile = t.zeros(tile_shape, dtype=U_0.dtype, device=U_0.device)
        in_selection = U_0[in_i * tile_shape[0]:(in_i+1) * tile_shape[0],
                           in_j * tile_shape[1]:(in_j+1) * tile_shape[1]]
        in_tile[:in_selection.shape[0],
                :in_selection.shape[1]] = in_selection

        # This gets the offset betweent the input and output tile
        tile_offset = [(in_i - out_i) * tile_shape[0] + offset[0],
                       (in_j - out_j) * tile_shape[1] + offset[1]]

        # And we perform the actual calculation
        out_tile = FFT_DI(in_tile, z, wavelength, step, offset=tile_offset)

        # Finally, we extract the output tile in a way which gets rid of any
        # section which might extend beyond the output region.
        out_slice = np.s_[out_i * tile_shape[0]:(out_i+1) * tile_shape[0],
                           out_j * tile_shape[1]:(out_j+1) * tile_shape[1]]
        out_selection = output[out_slice]
        output[out_slice] += output_function(out_tile[:out_selection.shape[0],
                                                      :out_selection.shape[1]])

    return output


def far_field(wavefront):
    """Propagates a wavefront to the far field using an FFT and FFTshift.
    """
    shifted = t.fft.ifftshift(wavefront, dim=(-1,-2))
    propagated = t.fft.fft2(shifted, norm='ortho')
    return t.fft.fftshift(propagated, dim=(-1,-2))


def inverse_far_field(wavefront):
    """Propagates a wavefront from the far field using an IFFT and IFFTshift
    """
    shifted = t.fft.ifftshift(wavefront, dim=(-1,-2))
    propagated = t.fft.ifft2(shifted, norm='ortho')
    return t.fft.fftshift(propagated, dim=(-1,-2))


def fourier_pad_to_shape(arr, shape):
    """Pads an array in Fourier space up to a given shape

    Fundamentally, this operation upsamples or downsamples an image to a given
    shape, preserving it's real-space width but changing the pixel size.

    - This function assumes periodic boundary conditions
    - The function preserves the zero-frequency pixel location during the crop
    - It will happily crop (downsample) if given a shape smaller than the input
    
    Parameters
    ----------
    arr : torch.tensor
        The array to pad
    shape : array(int)
        The shape to pad up to
    """

    pad0l = shape[-2]//2 - arr.shape[-2]//2
    pad0r = shape[-2] - arr.shape[-2] - pad0l
    pad1l = shape[-1]//2 - arr.shape[-1]//2
    pad1r = shape[-1] - arr.shape[-1] - pad1l

    fftwf = far_field(arr)
    fftwf = t.nn.functional.pad(fftwf, (pad1l, pad1r, pad0l, pad0r))
    return inverse_far_field(fftwf)
