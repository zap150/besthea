
# Experiments

In this directory, there are experiment sources and scripts for the onthefly (matrix-free) solution of the heat equation using the space-time boundary element method.



## How besthea was compiled and installed

Karolina cluster at IT4Innovations was used. The library was compiled using the following commands.

```
mkdir build
cd build
ml CMake/3.18.4-GCCcore-10.2.0 intel/2020b CUDA/11.7.0
cmake -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DBESTHEA_CUDA=enable -DCMAKE_CUDA_ARCHITECTURES=80-real -DCMAKE_INSTALL_PREFIX=../installation ..
make
make install
```

Experiments are compiled together with the library.



## Running the experiments

The experiments were run on the GPU accelerated nodes of Karolina (2x 64-core AMD EPYC 7763, 1TB RAM, 8x NVIDIA A100 40GB).

Each experiment has its own bash script, `run_<experiment>.sh`. The script should be run from the directory where besthea is installed (so it can easily find the executables). The results are written to the `gpu_onthefly_experiments_out/<experiment>/<datetime>` directory.

The jobs were launched using the command
```
qsub -q qgpu -A OPEN-00-00 -l select=1:ncpus=128:ngpus=8,walltime=48:00:00 job.sh
```
where `job.sh` only changes directory to the directory where besthea is installed and invokes the `run_*.sh` script, depending to the specific experiment.
