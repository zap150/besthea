BESTHEA (Space-time boundary element methods for the heat equation)
===================================================================

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Installation

**Requirements**

The library can be compiled on Linux and OS X systems with GNU, Intel, or Clang compilers. For OS X the recommended toolchain is GNU or Clang installed with MacPorts. AppleClang si partially supported only with an external OpenMP installation.

Two branches are available in the project. The `master` branch is supposed to be stable (as stable as a research project can be), while `develop` inlcudes the newest enhancements before they are ready to be merged in `master`.

**Cloning the repository**

The repository and its submodules can be cloned by
```
git clone git@github.com:zap150/besthea.git
cd besthea
git submodule update --init --recursive
```

**Pulling updates**

When pulling updates one should also pull the possibly updated submodules as
```
git pull --recurse-submodules
git submodule update --init --recursive
```

**Dependencies**

Except for the Boost, Eigen, and Lyra submodules, BESTHEA requires the installation of MPI and Intel MKL. To configure the project make sure that the `MKLROOT` and `LIBRARY_PATH` by calling the scripts `mklvars.sh` and `compilervars.sh` provided by the MKL installation.

OpenMP is a dependency usually accompanying a compiler. For a note on AppleClang see the next section.

**Build**

The compilation of BESTHEA is based on CMake. Since in-source build is disabled, create a build directory and call cmake from within as
```
mkdir build
cd build
cmake ..
make
make install
```
To specify the compiler you can either use environment variables as
```
CC=icc CXX=icpc cmake ..
```
or CMake variables
```
cmake -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc ..
```

The project includes example executables testing the library. These can be disabled by setting
```
cmake -DBUILD_EXAMPLES=OFF ..
```

AppleClang does not come with a native support of OpenMP. This can be installed e.g. via MacPorts. As of now, an include hint for `omp.h` is necessary, e.g.
```
CXXFLAGS="-isystem /opt/local/include/libomp"
```

The BESTHEA library uses OpenMP SIMD. To fully leverage its potential we recommend the Intel compiler with optimisation flags set in accordance with the system. E.g. on a Skylake CPU we would use
```
CXXFLAGS="-xcore-avx512 -qopt-zmm-usage=high" cmake ..
```
Vectorisation with GNU and OpenMP does not work optimally from our experience, however, one can try to use `CXXFLAGS="-march=skylake-avx512"`, `CXXFLAGS="-mavx512*"`, and similar.
To specify the vector length processed in OpenMS SIMD loops, i.e. the `simdlen` clause, we provide the variable `DATA_WIDTH` implicitly set to 8. This can be modified by e.g.
```
cmake -DDATA_WIDTH=4 ..
```

**Usage**

The `install` target installs executable examples to `build/bin` together with example mesh files. After `cd ./bin`, one can run an example
```
./uniform_tensor_neumann --mesh cube_192.txt --grid grid_cube_xy.txt
```
or
```
./uniform_tensor_neumann --help
```
to see all command line options.

## Related Publications and Experiment Reproducibility

Please cite this research as:

* Jan Zapletal, Raphael Watschinger, Günther Of, Michal Merta, *Semi-analytic integration for a parallel space-time boundary element method modeling the heat equation*, 2021 [arXiv:2102.09811](https://arxiv.org/abs/2102.09811).

* Raphael Watschinger, Michal Merta, Günther Of, Jan Zapletal, *A parallel fast multipole method for a space-time boundary element method for the heat equation*, 2021.\
See this [guide](./examples/distributed_tensor_dirichlet/experiments_parallel_fmm.md) to **reproduce** experiments from the paper.


## Contact

See [the project website](https://sites.google.com/view/besthea/).

## Acknowledgements

Authors acknowledge the support provided by the Czech Science Foundation under the project 17-22615S, the Austrian Science Fund (FWF) under the project I 4033-N32, and by The Ministry of Education, Youth and Sports from the Large Infrastructures for Research, Experimental Development, and Innovations project 'e-INFRA CZ – LM2018140'.
