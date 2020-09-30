%% Test 1: Test BPX preconditioner for a uniform mesh
%
clc, clear

%accuracy of the GMRES
eps = 1e-6;

% number of temporal panels per temporal leaf cluster
N_steps = 16;

% end time
t_end = 1;
t_start = 0;

% set parameter eta for nearfield criterion for FMM
eta = 2;

% order of the Lagrange interpolant
order = 5;
% RHS function
rhs_fun = @(t) ((exp(t) / 4) ...
  .* (exp(2) * erfc((1 + t) ./ (sqrt(t))) ...
  + 2 * erf(sqrt(t)) - exp(-2) * erfc((1 - t) ...
  ./ (sqrt(t)))));
%rhs_fun =@(t) sin(4*pi*t)*exp(-t);
exact_solution = @(t) (exp(t));

%bounds for the maximal number of levels in the cluster tree for the loop
L_min = 4;
L_max = 10;

n_timesteps = N_steps * 2.^( L_min:L_max )';
iteration_numbers = zeros( L_max - L_min + 1, 4 );
iteration_numbers( :, 1 ) = n_timesteps;

for L = L_min : L_max
  % levels of the binary tree
  fprintf('\nSolving problem with %d time-steps \n', 2^L * N_steps);

  %construct uniform intervals (as in cFMM_solver)
  panels = zeros(2, 2^L * N_steps);
  ht = 1 / (2^L * N_steps);
  panels(1, :) = t_start : ht : t_end - ht;
  panels(2, :) = t_start + ht : ht : t_end;

  fmm_solver = cFMM_solver_adaptive(t_start, t_end, N_steps, L, order, ... 
        eta, rhs_fun, panels, true);

  fprintf('Solving with standard FMM without preconditioner. \n');
  tic
  [ x_no_prec, iter_no_prec ] ...
    = fmm_solver.solve_iterative_std_fmm( eps );
  toc
  fprintf('Solving with standard FMM and diagonal preconditioner. \n');
  tic
  [ x_diag_prec, iter_diag_prec ] ...
    = fmm_solver.solve_iterative_std_fmm_diag_prec( eps );
  toc
  fprintf('Solving with standard FMM and bpx preconditioner. \n');
  tic 
  [ x_bpx_prec, iter_bpx_prec ] ...
    = fmm_solver.solve_iterative_std_fmm_bpx_prec( eps );
  toc
  iteration_numbers( L - L_min + 1, 2 : 4 ) ...
    = [ iter_no_prec, iter_diag_prec, iter_bpx_prec ];
end

disp( iteration_numbers )