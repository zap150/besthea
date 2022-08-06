















Compilation on Karolina:

mkdir build
cd build
ml CMake/3.18.4-GCCcore-10.2.0 intel/2020b CUDA/11.4.1
cmake -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DBESTHEA_CUDA=enable -DCMAKE_CUDA_ARCHITECTURES=80-real -DCMAKE_INSTALL_PREFIX=../installation ..


