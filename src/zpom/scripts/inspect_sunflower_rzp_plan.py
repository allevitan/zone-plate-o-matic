
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
    
    inspect_sunflower_placement(zps, pix_size=1e-6)
    plt.set_cmap('gray_r')
    plt.xlabel('x (um)')
    plt.ylabel('y (um)')
    plt.show()
            
if __name__ == '__main__':
    main()

