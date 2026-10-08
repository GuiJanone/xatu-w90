# Xatu

<div align=center>

![GitHub release (with filter)](https://img.shields.io/github/v/release/alejandrojuria/xatu)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/alejandrojuria/tightbinder/issues)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![arXiv](https://img.shields.io/badge/arXiv-2307.01572-red.svg)](https://arxiv.org/abs/2307.01572)
[![Documentation Status](https://readthedocs.org/projects/xatu-documentation/badge/?version=latest&style=flat)](https://xatu-documentation.readthedocs.io/en/latest/)

</div>  

Xatu (_eXcitons from ATomistic calcUlations_) is a program and library designed to solve the **Bethe-Salpeter equation** (BSE) of any material. Starting with electronic band structures from either tight-binding or DFT based on local orbitals, the BSE is constructed. Its diagonalization then yields the exciton spectrum, which can then be postprocessed to characterize the excitons and their optical properties.

<p align="center">
  <img src="hbn_xatu_example.png" width="90%" height="90%">
</p>

The theory behind the code plus details about the implementation and some usage examples can be found in our paper [Efficient computation of optical excitations in two-dimensional materials with the Xatu code](https://doi.org/10.1016/j.cpc.2023.109001). If you find 
our paper or the code useful, please consider citing us.

## Installation
Xatu is built upon the Armadillo C++ library for linear algebra, which is also based on the standard libraries por linear algebra, namely BLAS, LAPACK and ARPACK.
### Ubuntu 22.04 LTS native and WSL
Install the required libraries:
```
sudo apt-get install libopenblas-dev liblapack-dev libarpack2-dev libarmadillo-dev
```

The library (```libxatu.a```) can be automatically built running the following command:
```
make build
```

Then, the Xatu binary is built running:
```
make xatu
```

Alternatively, one can define scripts that make use of the functions defined in the library. To compile them, it suffices to put the script in the ```/main```folder and run
```
make [script]
```

### MacOS
For MacOS the dependencies can be installed via ```brew```. To build the library it is recommended to use brew's ```gcc``` compiler instead of ```clang```.
```
brew install gcc openblas lapack arpack armadillo
```

Then, specify in the Makefile the new compiler as well as the location of the libraries:
```
CC = g++-13
INCLUDE = -I$(PWD)/include -I/opt/homebrew/include -I/opt/homebrew/opt/openblas/include
LIBS = -DARMA_DONT_USE_WRAPPER -L$(PWD) -L/opt/homebrew/lib -L/opt/homebrew/opt/openblas/lib -lxatu -larmadillo -lopenblas -llapack -fopenmp -lgfortran -larpack
```

### General
In case that the libraries are not available through repositories, it is always possible to manually download and compile them. For specific instructions on how to install each library we refer to the documentation provided by each of these libraries. For instance, for Armadillo:

Clone the Armadillo library repository:
```git clone https://gitlab.com/conradsnicta/armadillo-code.git```

To install Armadillo run:
```
cd armadillo-code
cmake .
make install
```

Once they are compiled, to link the libraries we have to modify the Makefile to specify the directories where they are installed (e.g. Armadillo and OpenBLAS):
```
INCLUDE = -I/dir/armadillo/include -I/another_dir/OpenBLAS/include/
LIBS = -L/another_dir/OpenBLAS/lib
```

## HDF5 output
With HDF5 support (`libhdf5-dev`), build with `make build HDF5=1` and `make xatu HDF5=1`. The `-H` (`--hdf5`) flag then writes all
exciton outputs to one compressed archive, `<label>.h5`, instead of the separate text files. The energies are always stored;
`-c`, `-k`, `-r` and `-s` add the states, the k-space and real-space wavefunctions and the spin, as they do for the text output.
`--compression` sets the gzip level (0-9, default 4). The absorption spectrum (`-a`) and the self-energy (`-i`) are still written as text.
```
xatu system.model exciton.txt -n 100 -c -k -r 1 -s -H
```
Layout (energies in eV, lengths in the units of the system file, arrays in C order, band and k indices 0-based):
```
/                    attributes: calculation parameters (k mesh, bands, potential, dielectric, cutoff, TDA, command line, ...)
/system              bravais_lattice, reciprocal_lattice, motif, orbitals
/kpoints             (nk, 3) BZ mesh of the BSE
/bands               indices, energies (nk, nbands), eigenvectors (nk, nbands, norb) with -c
/basis               (dim, 3) electron-hole basis: valence band, conduction band, k index (with -c)
/realspace           positions, cell, atom of the sites of the real-space wavefunction (with -r)
/summary             energies, degeneracy_group, spin of all stored states
/excitons/0001 ...   one group per state, ordered by energy: eigval, state, state_antiresonant (no TDA),
                     kwf (nk), rswf (nsites), spin (total, hole, electron)
```
Here `kwf` is the probability per k point, summing to 1 (the text `.kwf` divides it by the spacing of the first two k points
and replicates it over the neighbouring zones), and `rswf[site]` is the electron probability at `/realspace/positions[site]`.
Reading it from Python:
```python
import h5py
with h5py.File("hBN.h5") as f:
    E = f["summary/energies"][()]
    X = f["excitons/0001/state"][()]      # complex128, indexed like /basis
    kwf = f["excitons/0001/kwf"][()]      # on f["kpoints"]
```
`test/check_hdf5_export.py` compares the archive with the text outputs of the same run.

## Documentation
The documentation for the library is generated using Doxygen. To build it, we have to install Doxygen and then run from the ```/docs```folder:
```
doxygen docs.cfg
```

The documentation can be accessed then in ```/docs/html```, by opening ```index.html```.
