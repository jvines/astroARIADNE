#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import (setup, find_namespace_packages)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="astroARIADNE",
    version="1.2.1",
    author="Jose Vines",
    author_email="jose.vines@ug.uchile.cl",
    maintainer="Jose Vines",
    maintainer_email="jose.vines@ug.uchile.cl",
    description="Bayesian Model Averaging SED fitter",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/jvines/astroARIADNE",
    packages=find_namespace_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        'Intended Audience :: Science/Research',
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Unix",
        "Topic :: Scientific/Engineering :: Astronomy"
    ],
    install_requires=[
        "numpy>=1.26.0",
        "scipy>=1.15.0",
        "pandas>=2.2.0",
        "numba>=0.61.0",
        "astropy>=7.0.0",
        "astroquery>=0.4.9",
        "regions>=0.10",
        "PyAstronomy>=0.22.0",
        "tqdm>=4.67.0",
        "matplotlib>=3.10.0",
        "termcolor>=2.5.0",
        "extinction>=0.4.7",
        "pyphot>=1.6.0",
        "dustmaps>=1.0.13",
        "dynesty>=2.1.0",
        "corner>=2.2.3",
        "h5py>=3.12.0",
        "tabulate>=0.9.0",
        "isochrones>=2.1"
    ],
    package_data={'astroARIADNE': ['Datafiles']},
    include_package_data=True,
    python_requires='>=3.11',
)
