Zone-Plate-O-Matic
------------------

"Zone plates, zone plates! Get your zone plates here! Hot, fresh zone plates, best zone plates in all of Switzerland!"

Overview
--------

This python library, and it's associated command line entry points, produce design files for various kinds of zone plates. These include:

- Classic Fresnel zone plates
- Buttressed zone plates
- Apodized zone plates (In progress)
- Randomized zone plates
- Multi-frame time dependent zone plates


Installation
------------

The dependencies are listed in setup.py. If you manage your environment with conda, the following dependencies are available on the main anaconda repo:

- numpy
- matplotlib
- scikit-image
- h5py

Several more dependencies are available on conda-forge

- gdstk
- tqdm

And finally, pytorch can be installed by following the directions found [here](https://pytorch.org/). Make sure you install the correct packages for GPU compute on your system.

Once the environment is ready, `zpom` (zone-plate-o-matic) can be installed withby running:

```bash
$ pip install -e .
```

From the top level directory of this git repo. This installs the repo in developer mode, so that changes to the repo propagate immediately without a reinstall.

A new virtual environment can be created an
d set up by running the following lines in series:

```bash
$ conda create --name zpom
$ conda activate zpom
$ conda install numpy matplotlib scikit-image h5py
$ conda install -c conda-forge gdstk tqdm
$ <appropriate pytorch install command for your system>
$ pip install -e .
```

Usage
-----

Please contact Abe Levitan at abraham.levitan@psi.ch for more information on usage, as the code is not fully documented.

The python package provides several command line entry points, which can be used to design randomized zone plates and "sunflower-style" off-axis zone plate arrays for multi-frame randomized probe imaging.

The design process for all zone plates is conceptually broken down into a "design" phase, where light propagation is used to produce a set of raster images contining key wavefield information at the optic plane. In the "realization" step, these raster files are used to generate a specific, vector zone plate file in .gds format.

To design a sunflower-style off-axis zone plate array, the following sequence of commands is used:

```bash
$ plan-sunflower-rzp <zone plate parameters>
$ design-sunflower-rzp <plan file> -n <zone plate index> # once per off-axis zp
$ realize-sunflower-design <design file> <options> -n <zone plate index> # once per off-axis zp
$ collate-sunflower-design <.gds folder> # puts all zps into one file
```

Each of the above scripts has a lightweight help page accessible with

```bash
$ <script> --help
```

That explains the required parameters and options