import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="zpom",
    version="0.1.0",
    author="Abe Levitan",
    author_email="abraham.levitan@psi.ch",
    description="Zone plates! Zone plates! Get you zone plates here!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.psi.ch/cxi/zone-plate-o-matic",
    packages=setuptools.find_packages(
        where='src',
        include=['zpom'],
    ),
    package_dir={"": "src"},
#    entry_points = {
#        'console_scripts' : [
#            'make-base-rzp-design = zpom.make_base_rzp_design:main',
#            'realize-rzp-design = zpom.realize_rzp_design:main']
#        },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
