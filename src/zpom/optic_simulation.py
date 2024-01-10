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
import PIL
# May need to be updated if the ZP is too large
PIL.Image.MAX_IMAGE_PIXELS = 1000000000 
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


def simulate_focus(rasterized_zp,
                   input_offset,
                   focal_distance,
                   wavelength,
                   pix_size,
                   output_shape,
                   offset=None,
                   tile_shape=None,
                   calculation_device=None,
                   verbose=False):
    
    output_offset = - (np.array(output_shape)-1)/2
    if offset is not None:
        offset += np.array(offset) / pix_size
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

    
