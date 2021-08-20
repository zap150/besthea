
A parallel fast multipole method for a space-time boundary element method for the heat equation - A guide to the numerical experiments
======================================================================================

This is a short guide that helps to reproduce the numerical experiments from the paper:

Raphael Watschinger, Michal Merta, Günther Of, Jan Zapletal, *A parallel fast multipole method for a space-time boundary element method for the heat equation*, 2021.

To run the experiments, please build the BESTHEA library as explained in [`README.md`](../../README.md) with the option
```
-DBUILD_EXAMPLES=ON
```
In the following we will denote the build directory by `BUILD` and the root directory of the besthea library by `BESTHEA`. All experiments are based on the same executable `distributed_tensor_dirichlet`
which is, after calling `make install`, located in
```
BUILD/bin
```
The input mesh files are copied into the same directory during the installation.

In the following we list the parameters to use to reproduce all the numerical examples. For details about the individual parameters please execute `distributed_tensor_dirichlet --help`. For details regarding the used hardware and compiler options of each experiment we refer to the paper.

## Example 1: Shared memory performance

The results in Table 5.1 and Figure 5.1 were generated using the following parameters:

* `--mesh cube_24_half_scale.txt`
* `--space_init_refine 2`
* `--endtime 0.25` 
* `--timeslices 16`
* `--refine 1`
* `--dist_tree_levels 5`
* `--n_min_elems_refine 80`
* `--st_coupling_coeff 0.9`
* `--trunc_space 5`
* `--temp_order 6`
* `--spat_order 6`
* `--gmres_prec 1e-8`

The number of OpenMP threads was set using the `OMP_NUM_THREADS` environmental variable. 

The level of the vectorization (used in Figure 5.1) must be set when generating the Cmake configuration (see [`README.md`](../../README.md)) passing in appropriate compiler flags. In the case of the Intel Compiler, we used the `-no-vec -no-simd -qno-openmp-simd` flags for the non-vectorized version and `-xcore-avx512 -qopt-zmm-usage=high` for the vectorized version with the length of the vector registers set using the `DATA_WIDTH` variable.

## Example 2: Task measurement

Figures 5.2 and 5.3 were generated using the following parameters:

* `--mesh cube_24_half_scale.txt`
* `--space_init_refine 2`
* `--endtime 0.25` 
* `--timeslices 16`
* `--refine 2`
* `--dist_tree_levels 5`
* `--n_min_elems_refine 800`
* `--st_coupling_coeff 0.9`
* `--trunc_space 5`
* `--temp_order 6`
* `--spat_order 6`
* `--measure_tasks 1`
* `--gmres_prec 1e-8`

The measured execution times of process `p` are written to `./task_timer/process_p.m`. To plot the times of process `p` in a figure in the style of Figure 5.2 use Matlab to run `./task_timer/process_p.m` and to call the function `plot_tasks` provided in 
```
BESTHEA/examples/distributed_tensor_dirichlet/plot_tasks.m
```
as described in the file's documentation.

## Example 3: Scalability

The results in Table 5.2 were generated using the following parameters:

* `--mesh cube_12_half_scale.txt`
* `--space_init_refine 4`
* `--endtime 0.25` 
* `--timeslices 256`
* `--refine 1`
* `--dist_tree_levels 9`
* `--n_min_elems_refine 800`
* `--st_coupling_coeff 4.1`
* `--trunc_space 2`
* `--temp_order 4`
* `--spat_order 12`
* `--gmres_prec 1e-8`

## Example 4: Crankshaft

The results in Table 5.3 were generated using:

* `--mesh scaled_crankshaft_11k.txt`
* `--space_init_refine 0`
* `--endtime 0.25` 
* `--timeslices 256`
* `--refine 1`
* `--dist_tree_levels 8`
* `--n_min_elems_refine 800`
* `--st_coupling_coeff 4.5`
* `--trunc_space 2`
* `--temp_order 3`
* `--spat_order 12`
* `--gmres_prec 1e-6`

Figure 5.5 can be generated using EnSight, ParaView, or any program capable of processing the EnSight file format. For this purpose one has to provide an additional target directory to the executable `distributed_tensor_dirichlet` via the additional option

* `--ensight_dir TARGET_DIRECTORY`.

All files needed for the visualization of the approximated Neumann datum (in addition to the projections of the Neumann and Dirichlet data) are stored in this directory.
