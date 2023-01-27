BESTHEA (Space-time boundary element methods for the heat equation)
===================================================================

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Acceleration of the space-time boundary element method using GPUs

For replication of the experiments published in the paper titled *Acceleration of the space-time boundary element method using GPUs*, clone this repository, checkout to the correct commit, and clone the submodules. Then see the the readme located in the `examples/gpu_onthefly` directory.

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

Beside the Boost, Eigen, and Lyra submodules, BESTHEA requires the installation of MPI and Intel MKL. To configure the project make sure that the `MKLROOT` and `LIBRARY_PATH` variables are set by calling the scripts `mklvars.sh` and `compilervars.sh` provided by the MKL installation (in the case of Intel OneAPI, use the script `setvars.sh`). 

OpenMP is a dependency usually accompanying a compiler. For a note on AppleClang see the next section.

If one wants to use the GPU-accelerated functionality implemented in the `besthea_cuda` library, CUDA should also be installed.

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
Vectorisation with GNU or Clang does not work optimally from our experience, however, one can try to use `CXXFLAGS="-march=skylake-avx512"`, `CXXFLAGS="-mavx512*"`, and similar.
To specify the vector length processed in OpenMP SIMD loops, i.e. the `simdlen` clause, we provide the variable `BESTHEA_SIMD_WIDTH` implicitly set to 8. This can be modified by e.g.
```
cmake -DBESTHEA_SIMD_WIDTH=4 ..
```

If CUDA is found on the system and CMake >= 1.18 is used, the library `besthea_cuda` containing the GPU-accelerated functionality is also built. To alter this behaviour, set the value of the `BESTHEA_CUDA` variable to `enable`, `disable` or `auto`, e.g.
```
cmake -DBESTHEA_CUDA=enable ..
```
If the value is `enable` and any of the prerequisities are not met, the configuration step fails with an error. `disable` entirely disables compilation of the GPU-accelerated code. `auto` is the default (equivalent to not setting the value of `BESTHEA_CUDA` at all), which tries to build the `besthea_cuda` library if it is possible, and if not, no errors are produced. By default we set the CUDA host compiler to be the same as the CXX compiler. To change it, set the `CMAKE_CUDA_HOST_COMPILER` variable. Using empty string reverts to nvcc's default behaviour. It is highly recommended to also specify the CUDA architecture to compile the CUDA code optimally for the the target system. E.g., use `-DCMAKE_CUDA_ARCHITECTURES=80-real` for the Nvidia A100 GPU. Minimum compute capability supported is 6.0, `60-virtual` is set as the default architecture.

**Usage**

The `install` target installs the static libraries to `${CMAKE_INSTALL_PREFIX}/lib/besthea`, include files to `${CMAKE_INSTALL_PREFIX}/include/besthea`, and executable examples to `${CMAKE_INSTALL_PREFIX}/bin/besthea` together with example mesh files. One can run an example
```
./uniform_tensor_neumann --mesh cube_192.txt --grid grid_cube_xy.txt
```
or
```
./uniform_tensor_neumann --help
```
to see all command line options.

## Related publications and experiment reproducibility

Please cite this research as:

* J. Zapletal, R. Watschinger, G. Of, M. Merta, *Semi-analytic integration for a parallel space-time boundary element method modeling the heat equation*, Comput. Math. Appl. 103 (2021) 156-170, [10.1016/j.camwa.2021.10.025](https://doi.org/10.1016/j.camwa.2021.10.025).

* R. Watschinger, M. Merta, G. Of, J. Zapletal, *A parallel fast multipole method for a space-time boundary element method for the heat equation*, 2021, [arXiv:2106.15911](https://arxiv.org/abs/2106.15911).\
See this [guide](./examples/distributed_tensor_dirichlet/experiments_parallel_fmm.md) to **reproduce** experiments from the paper.

Other related publications:

* R. Watschinger, G. Of, *An integration by parts formula for the bilinear form of the hypersingular boundary integral operator for the transient heat equation in three spatial dimensions*, 2021, J. Integral Equ. Appl. (accepted). [arXiv:2104.15024](https://arxiv.org/abs/2104.15024)


## Contact

See [the project website](https://sites.google.com/view/besthea/).

## Acknowledgements

Authors acknowledge the support provided by the Czech Science Foundation under the project 19-29698L, the Austrian Science Fund (FWF) under the project I 4033-N32, and by The Ministry of Education, Youth and Sports from the Large Infrastructures for Research, Experimental Development, and Innovations project 'e-INFRA CZ – LM2018140'.
