
import torch as t
import numpy as np
from zpom.optic_design import inspect_sunflower_placement
import h5py
from matplotlib import pyplot as plt
import argparse

def main():

    parser = argparse.ArgumentParser(
        prog='inspect_sunflower_rzp_plan',
        description='Plot a visualization of a plan for a sunflower-style array.')

    parser.add_argument('filename', type=str, help='The filename to view the plan for.')

    args = parser.parse_args()
    with h5py.File(args.filename, 'r') as f:
        zp_idx = list(sorted(int(d[2:]) for d in list(f) if d[:2] == 'ZP'))
        zp_labels = ['ZP%03d' % zp_id for zp_id in zp_idx]
        zps = []
        for zp_label in zp_labels:
            new_zp = {}
            for key in list(f[zp_label]):
                new_zp[key] = np.array(f[zp_label][key])
            zps.append(new_zp)

    print('True inner zone:', int(zps[0]['start_zone']))
    print('True outer zone:', int(zps[-1]['end_zone']))
    print('True inner diameter:', 1e6 * 2 * zps[0]['inner_r'][0], 'um')
    print('True outer diameter:', 1e6 * 2 * zps[-1]['outer_r'][0], 'um')
    print('Inner frame center zone:', int(zps[0]['center_zone']))
    print('Outer frame center zone:', int(zps[-1]['center_zone']))
        
    inspect_sunflower_placement(zps, pix_size=1e-6)
    plt.set_cmap('gray_r')
    plt.xlabel('x (um)')
    plt.ylabel('y (um)')
    plt.show()
            
if __name__ == '__main__':
    main()

