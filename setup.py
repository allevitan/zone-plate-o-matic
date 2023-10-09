import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="zpom",
    version="0.1.0",
    author="Abe Levitan",
    author_email="abraham.levitan@psi.ch",
    description="Zone plates! Zone plates! Get your zone plates here!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.psi.ch/cxi/zone-plate-o-matic",
    install_requires = [
        'numpy',
        'matplotlib',
        'torch',
        'scikit-image',
        'h5py',
        'gdstk',
        'tqdm'
    ],
    packages=setuptools.find_packages(
        where='src',
        include=['zpom'],
    ),
    package_dir={"": "src"},
   entry_points = {
        'console_scripts' : [
            'design-rzp = zpom.design_rzp:main',
            'design-sunflower-rzp = zpom.design_sunflower_rzp:main',
            'realize-design = zpom.realize_design:main',
            'realize-sunflower-design = zpom.realize_sunflower_design:main']
        },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
