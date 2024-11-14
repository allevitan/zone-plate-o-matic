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
            'plan-sunflower-rzp = zpom.scripts.plan_sunflower_rzp:main',
            'inspect-sunflower-rzp-plan = zpom.scripts.inspect_sunflower_rzp_plan:main',
            'design-rzp = zpom.scripts.design_rzp:main',
            'design-sunflower-rzp = zpom.scripts.design_sunflower_rzp:main',
            'realize-design = zpom.scripts.realize_design:main',
            'realize-sunflower-design = zpom.scripts.realize_sunflower_design:main',
            'collate-sunflower-design = zpom.scripts.collate_sunflower_design:main',
            'simulate-rzp-focus = zpom.scripts.simulate_rzp_focus:main',
            'simulate-sunflower-rzp-focus = zpom.scripts.simulate_sunflower_rzp_focus:main',
            'simulate-rzp-focus-t = zpom.scripts.simulate_rzp_focus_t:main',
            'simulate-sunflower-rzp-focus-t = zpom.scripts.simulate_sunflower_rzp_focus_t:main',
        ]
        },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
