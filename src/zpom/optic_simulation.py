"""
This file contains functions for simulating randomized zone plate designs. 

The base function here uses the FFT_DI light propagation method to simulate
the focal spot of a zone plate using that optic's gdsii file as input.

Author: Abraham Levitan
Dates: January 2022 to September 2023
"""
import gdstk
import numpy as np
import torch as t
from matplotlib import pyplot as plt
from matplotlib import patches, transforms
import io
from scipy.ndimage import binary_erosion, binary_dilation
import PIL
# May need to be updated if the ZP is too large
PIL.Image.MAX_IMAGE_PIXELS = 10000000000 
from zpom import propagation
from tqdm import tqdm

def rasterize_zp(gds_file, pix_size, cell=None, layer=None, verbose=False):
    #
    # The procedure for rasterizing was inspired by:
    # https://github.com/HelgeGehring/gdshelpers/blob/master/gdshelpers/geometry/chip.py
    #
    # The problem is that matplotlib's rastering engine is not exactly precise,
    # so there might be pixel-level issues. I did my best to avoid them.
    # 

    lib = gdstk.read_gds(gds_file)

    if cell is None:
        cells = lib.cells
    else:
        cells = [c for c in lib.cells if c.name == cell]
    fig, ax = plt.subplots()

    print('Populating the polygons')
    for cell in cells:
        print('Working on', cell)
        if verbose:
            to_iter = tqdm(cell.get_polygons(layer=layer))
        else:
            to_iter = cell.get_polygons(layer=layer)
        for polygon in to_iter:
            patch = patches.Polygon(polygon.points, antialiased=True,
                                    facecolor='k')
            ax.add_patch(patch)

    print('Polygons populated')

    ax.set_aspect(1)
    ax.autoscale(True, tight=True)  
    ax.axis('off')
    ylim, xlim = ax.get_ylim(), ax.get_xlim()

    if ((xlim[1]-xlim[0]) >= 2**15) or ((ylim[1] - ylim[0]) >= 2**15):
        raise ValueError('Matplotlib is being used to rasterize the .gds file. It has an issue with outputting figures with either dimension greater than 2**15 pixels. Your options are (1) increase the pixel size for the simulation, so that the rasterized optic is smaller than that, or (2) email Abe at abraham.levita@psi.ch to get him to finally fix this. If you are Abe, tough luck.')

    # The natural units of the gdsii file are um, so we follow that
    # convention here and plot in microns
    pix_size_um = pix_size*1e6
    xlim = ((np.floor(xlim[0]/pix_size_um)-0.5)*pix_size_um,
            (np.ceil(xlim[1]/pix_size_um)+0.51)*pix_size_um)
    ylim = ((np.floor(ylim[0]/pix_size_um)-0.5)*pix_size_um,
            (np.ceil(ylim[1]/pix_size_um)+0.51)*pix_size_um)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    dpi = 300
    fig.set_dpi(dpi)
    size_um = np.asarray((xlim[1] - xlim[0], ylim[1] - ylim[0]))
    size_inch = size_um * 1e-6 / (pix_size * dpi)
    fig.set_size_inches(size_inch)
    ax.set_position([0,0,1,1])

    
    bbox_inches = transforms.Bbox.from_extents([0,0,size_inch[0], size_inch[1]])
    with io.BytesIO() as temp_file:
        print('Rasterizing plot')
        plt.savefig(temp_file, transparent=True, bbox_inches=bbox_inches,
                    dpi=dpi, format='png')
        print('Plot rasterized')
        plt.close()
        with PIL.Image.open(temp_file) as im:
            # The alpha layer contains the info we need
            offset = np.array([-ylim[1], xlim[0]]) * 1e-6
            return (np.array(im)[:,:,3], offset)


def make_dilation_element(radius):
    I, J = np.mgrid[:2*int(radius)+1,:2*int(radius)+1]
    I = I - int(radius)
    J = J - int(radius)
    R = np.sqrt(I**2 + J**2)
    return R <= radius

def simulate_fab_process(
        rasterized_zp,
        erosion_radius=0,
        dilation_radii=[],
        dilation_materials=[],
        zone_material=0,
        background_material=1,
):
    # First erode the rasterized ZP design by the specified amount

    rasterized_zp = np.logical_not(np.isclose(rasterized_zp,0)).astype(np.int8)
    if erosion_radius != 0:
        previous_design = binary_erosion(
            rasterized_zp,
            make_dilation_element(erosion_radius)
        )
    else:
        previous_design = rasterized_zp

    final_design = (zone_material * previous_design).astype(np.complex128)
    # Then do a series of dilations, in each case tracking the difference
    # and finally setting the
    for dilation_radius, dilation_material in \
            zip(dilation_radii, dilation_materials):
        dilated_design = binary_dilation(
            previous_design,
            make_dilation_element(dilation_radius)
        )
        final_design += (dilated_design - previous_design) * dilation_material
        previous_design = dilated_design

    final_design += (1-dilated_design) * background_material

    return final_design


def simulate_focus(rasterized_zp,
                   input_offset,
                   focal_distance,
                   wavelength,
                   pix_size,
                   output_shape,
                   tile_shape=None,
                   calculation_device=None,
                   verbose=False):
    
    output_offset = - (np.array(output_shape)-1)/2
    input_offset = input_offset / pix_size
    offset = input_offset - output_offset
    torch_zp = t.as_tensor(rasterized_zp).to(dtype=t.complex128)
    focus = propagation.FFT_DI_tiled(torch_zp, focal_distance,
                                     wavelength, [pix_size]*2,
                                     offset=offset,
                                     output_shape=output_shape,
                                     calculation_device=calculation_device,
                                     tile_shape=tile_shape, verbose=verbose)
    return focus.cpu()

    
def simulate_focus_t(rasterized_zp,
                     input_offset,
                     focal_distance,
                     wavelength,
                     pix_size,
                     A,
                     time,
                     output_shape,
                     t_0=None,
                     tile_shape=None,
                     calculation_device=None,
                     verbose=False):
    
    output_offset = - (np.array(output_shape)-1)/2
    input_offset = input_offset / pix_size
    offset = input_offset - output_offset
    torch_zp = t.as_tensor(rasterized_zp).to(dtype=t.complex128)
    focus = propagation.FFT_DI_tiled_t(torch_zp, focal_distance,
                                       wavelength, [pix_size]*2,
                                       A, time,
                                       offset=offset,
                                       t_0=t_0,
                                       output_shape=output_shape,
                                       calculation_device=calculation_device,
                                       tile_shape=tile_shape, verbose=verbose)
    return focus.cpu()

    
