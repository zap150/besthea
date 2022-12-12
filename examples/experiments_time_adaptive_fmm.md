
A Time-Adaptive Space-Time FMM for the Heat Equation - A guide to reproducing the numerical experiments
======================================================================================

This is a short guide that helps to reproduce the numerical experiments from the paper:

Raphael Watschinger, Günther Of, *A Time-Adaptive Space-Time FMM for the Heat Equation*, Comput. Methods Appl. Math., 2022. https://doi.org/10.1515/cmam-2022-0117

To run the experiments, please build the BESTHEA library as explained in [`README.md`](../README.md) with the option
```
-DBUILD_EXAMPLES=ON
```
In the following, we denote the build directory by `BUILD` and the root directory of the besthea library by `BESTHEA`. The results are obtained by running the executables `fast_dirichlet_ibvp` and `solve_ibvp_rapidly_changing_data`.
After calling `make install` these executables are located in
```
BUILD/bin/
```
The input mesh files are copied into the same directory during the installation.

In the following, we list the parameters to reproduce the numerical results in the paper. These parameters have to be provided when running the respective executables. For details about the individual parameters please execute them with the additional argument `--help`. For details regarding the used hardware, we refer to the paper.

## Example 1: Exponential Decay in Time

For the results in Table 2 in Section 5.1 the single layer operator matrix `V_h` is once approximated by the standard FMM and once by the time-adaptive FMM. The results for the standard FMM are obtained by calling `fast_dirichlet_ibvp` with the following parameters:

* `--cauchy_datum 1`
* `--time_mesh time_adaptive_mesh_exp_decay.txt`
* `--volume_mesh cube_12_vol_half_scale.txt`
* `--vol_init_refine 5`
* `--refine_large_leaves`
* `--temp_order 4`
* `--spat_order 12`
* `--diagonal_prec`

To use the time-adaptive FMM instead, one has to supply the parameter

* `--use_time_adaptive_operations`

in addition.

## Example 2: Rapid Change of Boundary Data

To reproduce the numerical results in Section 5.2 the right-hand side vectors of the linear systems have to be computed. For the results in Table 3 one has to call the routine `solve_ibvp_rapidly_changing_data` with the parameters

* `--time_mesh time_adaptive_mesh_rapidly_changing_boundary_data.txt`
* `--surface_mesh scaled_crankshaft_11k.txt`
* `--refine 1`
* `--temp_order 5`
* `--spat_order 20`
* `--refine_large_leaves`
* `--use_time_adaptive_operations`
* `--compute_right_hand_side`

to compute the corresponding right-hand side vector. It is stored in the binary file `./data_files/rhs_fine_mesh.bin`. To solve the corresponding linear system and obtain the actual results from Table 3 one has to call `solve_ibvp_rapidly_changing_data` again with different parameters. The parameters

* `--time_mesh time_adaptive_mesh_rapidly_changing_boundary_data.txt`
* `--surface_mesh scaled_crankshaft_11k.txt`
* `--temp_order 4`
* `--spat_order 12`
* `--refine_large_leaves`
* `--rhs_data ./data_files/rhs_fine_mesh.bin`
* `--diagonal_prec`

are used to solve the linear system with the standard FMM for the approximation of `V_h`. With the additional parameter

* `--use_time_adaptive_operations`

the time-adaptive FMM is used.

---

For the results in Table 4 the right-hand side vector is computed by calling `solve_ibvp_rapidly_changing_data` with the parameters

* `--surface_mesh scaled_crankshaft_11k.txt`
* `--endtime 0.25`
* `--timeslices 512`
* `--refine 1`
* `--temp_order 5`
* `--spat_order 20`
* `--compute_right_hand_side`

The resulting vector is again stored in the file `./data_files/rhs_fine_mesh.bin` replacing any file with the same name. In the paper we used 8 nodes of the VSC-4 cluster and all related CPU cores by setting `OMP_NUM_THREADS=48` and calling `mpirun -np 8 ...`

The linear system is solved by calling `solve_ibvp_rapidly_changing_data` with the parameters

* `--surface_mesh scaled_crankshaft_11k.txt`
* `--endtime 0.25`
* `--timeslices 512`
* `--temp_order 4`
* `--spat_order 12`
* `--refine_large_leaves`
* `--rhs_data ./data_files/rhs_fine_mesh.bin`
* `--diagonal_prec`

In the paper we used 16 nodes of the VSC-4 cluster and all related CPU cores by setting `OMP_NUM_THREADS=48` and calling `mpirun -np 16 ...`
