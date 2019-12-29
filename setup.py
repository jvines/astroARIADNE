#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="astro_ariadne",
    version="0.0.1",
    author="Jose Vines",
    author_email="jose.vines@ug.uchile.cl",
    maintainer="Jose Vines",
    maintainer_email="jose.vines@ug.uchile.cl",
    description="Bayesian Model Avergaing SED fitter",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Astronomy"
    ],
    requires=["numpy", "scipy", "matplotlib", "astropy", "astroquery",
              "tabulate", "tqdm", "regions", "pyphot", "PyAstronomy",
              "termcolor"],
    include_package_data=True,
    python_requires='>=3.6',
)
