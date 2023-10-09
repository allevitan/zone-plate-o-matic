Zone-Plate-O-Matic
------------------

"Zone plates, zone plates! Get your zone plates here! Hot, fresh zone plates, best zone plates in all of Switzerland!"

Overview
--------

This python library, and it's associated command line entry points, produce design files for various kinds of zone plates. These include:

- Classic Fresnel zone plates
- Buttressed zone plates
- Apodized zone plates
- Randomized zone plates
- Multi-frame time dependent zone plates

...or at least, that is the plan! 

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

