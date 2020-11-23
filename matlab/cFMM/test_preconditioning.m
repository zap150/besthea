%% Test 1: Test several preconditioners for a uniform mesh
%
clc, clear

%accuracy of the GMRES
eps = 1e-6;

% number of temporal panels per temporal leaf cluster
N_steps = 2;

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
% rhs_fun =@(t) sin(4*pi*t)*exp(-t);
% rhs_fun =@(t) 0;
exact_solution = @(t) (exp(t));
% exact_solution = @(t) (0);

%bounds for the maximal number of levels in the cluster tree for the loop
L_min = 4;
L_max = 11;

%method
method = 0; %use GMRES for solving the systems of equations
% method = 1; %use BiCGStab instead

n_timesteps = N_steps * 2.^( L_min:L_max )';
iteration_numbers = zeros( L_max - L_min + 1, 5 );
iteration_numbers( :, 1 ) = n_timesteps;

for L = L_min : L_max
  % levels of the binary tree
  fprintf('\nSolving problem with %d time-steps \n', 2^L * N_steps);
  if ( method == 0 )
    fprintf('Solver is GMRES\n');
  elseif ( method == 1 )
    fprintf('Solver is BiCGStab\n');
  end

  %construct uniform intervals (as in cFMM_solver)
  panels = zeros(2, 2^L * N_steps);
  ht = 1 / (2^L * N_steps);
  panels(1, :) = t_start : ht : t_end - ht;
  panels(2, :) = t_start + ht : ht : t_end;
%   rng( 0 );
%   rhs_fun = rand( 2^L * N_steps, 1 );
  
  fmm_solver = cFMM_solver_adaptive(t_start, t_end, N_steps, L, order, ... 
        eta, rhs_fun, panels, true);
  
%   init_guess = zeros( 2^L * N_steps, 1 );
  fprintf('Solving with standard FMM without preconditioner. \n');
  tic
  [ x_no_prec, iter_no_prec ] ...
    = fmm_solver.solve_iterative_std_fmm( eps, method );
  toc
  
  fprintf('Solving with standard FMM and diagonal preconditioner. \n');
  tic
  [ x_diag_prec, iter_diag_prec ] ...
    = fmm_solver.solve_iterative_std_fmm_diag_prec( eps, method );
  toc
  
  fprintf('Solving with standard FMM and bpx preconditioner. \n');
  tic 
  [ x_bpx_prec, iter_bpx_prec ] ...
    = fmm_solver.solve_iterative_std_fmm_bpx_prec( eps, method );
  toc
  
  fprintf( ['Solving with standard FMM and 2 level additive schwarz', ...
            ' preconditioner; coarse level = %d. \n' ] , min( L - 2, 6 ) );
  tic
    fmm_solver.initialize_2_lev_add_schw_preconditioner( min( L - 2, 6 ) );
    [ x_schw_prec, iter_schw_prec ] ...
      = fmm_solver.solve_iteartive_std_fmm_2_lev_add_schw_prec( ...
          eps, method );
  toc
  iteration_numbers( L - L_min + 1, 2 : 5 ) ...
    = [ iter_no_prec, iter_diag_prec, iter_bpx_prec, iter_schw_prec ];
end

disp( iteration_numbers )