
# Acceleration of the space-time boundary element method using GPUs - numerical experiments

In this directory, there are experiment sources and scripts for the onthefly (matrix-free) solution of the heat equation using the space-time boundary element method. They are used in the paper

Jakub Homola, Michal Merta, Jan Zapletal, *Acceleration of the space-time boundary element method using GPUs*



## Compilation and installation the besthea library

Clone the repository using
```
git clone git@github.com:zap150/besthea.git
cd besthea
```
Checkout to the specific commit using
```
git checkout <COMMIT_HASH>
```
where `<COMMIT_HASH>` is the hexadecimal string at the end of the URL leading to this repository, located in the References section of the paper.

Clone the sumodules using
```
git submodule update --init --recursive
```

On the Karolina cluster, we compiled and installed the library along with the experiments using the following commands
```
mkdir build
cd build
ml CMake/3.18.4-GCCcore-10.2.0 intel/2020b CUDA/11.7.0
cmake -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DBESTHEA_CUDA=enable -DCMAKE_CUDA_ARCHITECTURES=80-real -DCMAKE_INSTALL_PREFIX=../installation ..
make
make install
```



## Running the experiments

The experiment programs are compiler and installed together with the library. There are two experiment programs located in the `installation/bin/besthea` directory - `onthefly_multiply` and `onthefly_solve`. These were used to perform the experiments in the paper, using command line arguments relevant to the specific experiment.

For each of the experiments in the paper, we prepared a bash script `run_<experiment>.sh`, which loads the necessary modules, sets the environment variables, runs the program multiple times with relevant arguments and parses the output to produce a `results.txt` file in the `installation/gpu_onthefly_experiments_out/<experiment>/<timestamp>` directory. The bash script should be run from the `installation` directory.

We used the following command to launch the batch jobs on the Karolina cluster
```
qsub -q qgpu -A OPEN-00-00 -l select=1:ncpus=128:ngpus=8,walltime=48:00:00 job.sh
```
where `job.sh` only changes to the besthea installation directory and invokes the `run_<experiment>.sh` script.
