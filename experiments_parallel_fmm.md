
A parallel fast multipole method for a space-time boundary element method for the heat equation - A guide to the numerical experiments
======================================================================================

This is a short guide that helps to reproduce the numerical experiments in the paper:

Raphael Watschinger, Michal Merta, Günther Of, Jan Zapletal, *A parallel fast multipole method for a space-time boundary element method for the heat equation*, 2021.

To run the experiments, please build the BESTHEA library as explained in [`README.md`](./README.md) with the option
```
-DBUILD_EXAMPLES=ON
```
In the following we will denote the build directory by `BUILD` and the root directory of the besthea library by `BESTHEA`. All experiments are based on the same executable `distributed_tensor_dirichlet`
which is located in
```
BUILD/examples/distributed_tensor_dirichlet
```
In the following we list the parameters to use to reproduce all the numerical examples. For details about the individual parameters please execute `distributed_tensor_dirichlet --help`. For details regarding the used hardware and compiler options of each experiment we refer to the paper.

## Example 1: Shared memory performance

The results in Table 5.1 and Figure 5.1 were generated using the following parameters:

* `--mesh BESTHEA/examples/mesh_files/cube_24_half_scale.txt`
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


## Example 2: Task measurement

Figures 5.2 and 5.3 were generated using the following parameters:

* `--mesh BESTHEA/examples/mesh_files/cube_24_half_scale.txt`
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

The measured execution times of process `p` are written to `./task_timer/process_p.m` and can be visualized using the matlab routine (TODO: should we add that?)

## Example 3: Scalability

The results in Table 5.2 were generated using:

* `--mesh BESTHEA/examples/mesh_files/cube_12_half_scale.txt`
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

## Example 4: Crankshaft

The results in Table 5.3 were generated using:

* `--mesh BESTHEA/examples/mesh_files/scaled_crankshaft_11k.txt`
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

TODO: the visualization is currently not featured in the example. Should we add that?


