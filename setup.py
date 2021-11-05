import os
from setuptools import setup

setup(
    name = "colabfit-tools",
    version = "0.0.1",
    author = "ColabFit",
    description = ("A suite of tools for working with traiing datasets for interatomic potentials"),
    license = "BSD",
    keywords = "machine learning interatomic potentials",
    url = "http://colabfit.org",
    packages=[
        'colabfit.tools',
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
    install_requires = [
        'ase',
        'kim_property',
        'numpy',
        'tqdm',
    ],
)