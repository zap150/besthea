# Requirements

The library can be compiled on Linux and OS X systems with GNU and Intel compilers. For OS X the recommended toolchain is GNU installed with macports.

Except for the Boost and Eigen submodules and the Catch header provided in the repository, besthea requires the installation of MPI and Intel MKL. To configure the project make sure that the `MKLROOT` and `LIBRARY_PATH` by calling the scripts `mklvars.sh` and `compilervars.sh` provided by the MKL installation. 

# Compilation

## CMake

The compilation of besthea is based on CMake. Since in-source build is disabled, create a build directory and call cmake from within as
```
mkdir build
cd build
cmake ..
make
```
To specify the compiler you can either use environment variables as
```
CC=icc CXX=icpc cmake ..
``` 
or CMake variables
```
cmake -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc ..
```

The project includes test executables testing the library. These can be enabled by setting 
```
cmake -DBUILD_TEST=ON ..
```

The besthea library uses OpenMP SIMD. To fully leverage its potential we recommend the Intel compiler with optimisation flags set in accordance with the system. E.g. on a Skylake CPU we would use 
```
CXXFLAGS="-xcore-avx512 -qopt-zmm-usage=high" cmake ..
```
Vectorisation with GNU and OpenMP does not work optimally from our experience, however, one can try to use `CXXFLAGS="-march=skylake-avx512"` and similar.
To specify the vector length processed in OpenMS SIMD loops, i.e. the `simdlen` clause, we provide the variable `DATA_WIDTH` implicitly set to 8. This can be modified by e.g.
```
cmake -DDATA_WIDTH=4 ..
```

## Legacy

The legacy makefiles can be found in `legacy/Makefile.default` and `legacy/test/Makefile.default`. After corresponding modifications and copying to the source directories by
```
cp legacy/Makefile.default Makefile
cp legacy/test/Makefile.default test/Makefile
```
these can be used for compilation as well. 
