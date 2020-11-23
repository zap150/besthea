%% Test 1: Check residuals for uniform mesh for different refinements
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

  %construct uniform intervals
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
  [ x_no_prec, iter_no_prec, res_vec_no_prec ] ...
    = fmm_solver.solve_iterative_std_fmm( eps, method );
  toc
  fprintf('Number of iterations: %d. \n', iter_no_prec);
  figure(L - L_min + 1)
  semilogy(res_vec_no_prec);
end

%% Test 2: Check residuals for non-uniform non-nested random meshes

clc, clear

%accuracy of the GMRES
eps = 1e-8;

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
L_max = 10;

%method
method = 0; %use GMRES for solving the systems of equations
% method = 1; %use BiCGStab instead

n_timesteps = N_steps * 2.^( L_min:L_max )';

for L = L_min : L_max
  % levels of the binary tree
  fprintf('\nSolving problem with %d time-steps \n', 2^L * N_steps);
  if ( method == 0 )
    fprintf('Solver is GMRES\n');
  elseif ( method == 1 )
    fprintf('Solver is BiCGStab\n');
  end

  %construct random non-uniform intervals
  panels = zeros(2, 2^L * N_steps);
  rng( 0 );
  panels(2, :) = cumsum( 0.01 + rand( 1,  2^L * N_steps ) );
  panels(2, :) ...
    = ( t_end - t_start ) * panels( 2, : ) ./ panels( 2, end ) + t_start;
  panels(1, 2 : end ) = panels(2, 1 : end - 1 );
  
  h_max = max( panels(2, :) - panels(1, :) );
  h_min = min( panels(2, :) - panels(1, :) );
  
  fmm_solver = cFMM_solver_adaptive(t_start, t_end, N_steps, L, order, ... 
        eta, rhs_fun, panels, true);
  
%   init_guess = zeros( 2^L * N_steps, 1 );
  fprintf('Timesteps = %d, h_min = %f, h_max = %f\n', 2^L * N_steps, h_min, h_max );
  tic
  [ x_no_prec, iter_no_prec, res_vec_no_prec ] ...
    = fmm_solver.solve_iterative_std_fmm( eps, method );
  toc
  fprintf('Number of iterations: %d. \n', iter_no_prec);
  approx_err = fmm_solver.l2_error(x_no_prec, exact_solution);
  fprintf( 'L2 error is: %f. \n', approx_err ); 
  
  
  figure(L - L_min + 1)
  semilogy( res_vec_no_prec );
 
end

%% Test 3: Check residuals for non-uniform meshes resulting from uniformly refined initial non-uniform mesh

clc, clear

%accuracy of the GMRES
eps = 1e-8;

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

%construct random non-uniform intervals
panels = zeros(2, 2^L_min * N_steps);
rng( 0 );
panels(2, :) = cumsum( 0.01 + rand( 1,  2^L_min * N_steps ) );
panels(2, :) ...
  = ( t_end - t_start ) * panels( 2, : ) ./ panels( 2, end ) + t_start;
panels(1, 2 : end ) = panels(2, 1 : end - 1 );

iteration_table = zeros( L_max - L_min + 1, 3 );
iteration_table( :, 1 ) = n_timesteps;

for L = L_min : L_max
  % levels of the binary tree
  fprintf('\nSolving problem with %d time-steps \n', 2^L * N_steps);
  if ( method == 0 )
    fprintf('Solver is GMRES\n');
  elseif ( method == 1 )
    fprintf('Solver is BiCGStab\n');
  end
  
  h_max = max( panels(2, :) - panels(1, :) );
  h_min = min( panels(2, :) - panels(1, :) );
  
  fmm_solver = cFMM_solver_adaptive(t_start, t_end, N_steps, L, order, ... 
        eta, rhs_fun, panels, true);
  
%   init_guess = zeros( 2^L * N_steps, 1 );
  fprintf('Timesteps = %d, h_min = %f, h_max = %f\n', 2^L * N_steps, h_min, h_max );
  fprintf('No preconditioner:\n');
  tic
  [ x_no_prec, iter_no_prec, res_vec_no_prec ] ...
    = fmm_solver.solve_iterative_std_fmm( eps, method );
  toc
  fprintf('Number of iterations: %d. \n', iter_no_prec);
  approx_err = fmm_solver.l2_error(x_no_prec, exact_solution);
  fprintf( 'L2 error is: %f. \n', approx_err ); 
  
  fprintf('BPX preconditioner:\n');
  tic 
  [ x_bpx_prec, iter_bpx_prec, res_vec_bpx_prec ] ...
    = fmm_solver.solve_iterative_std_fmm_bpx_prec( eps, method );
  toc
  fprintf('Number of iterations: %d. \n', iter_bpx_prec);
  approx_err_bpx = fmm_solver.l2_error(x_bpx_prec, exact_solution);
  fprintf( 'L2 error is: %f. \n', approx_err_bpx ); 
  
  fprintf('block diagonal preconditioner:\n');
  tic 
  [ x_diag_prec, iter_diag_prec, res_vec_diag_prec ] ...
    = fmm_solver.solve_iterative_std_fmm_diag_prec( eps, method );
  toc
  fprintf('Number of iterations: %d. \n', iter_diag_prec);
  approx_err_diag = fmm_solver.l2_error(x_diag_prec, exact_solution);
  fprintf( 'L2 error is: %f. \n', approx_err_diag ); 
  
  fprintf('two level additive Schwarz preconditioner:\n');
  tic 
  [ x_schw_prec, iter_schw_prec, res_vec_schw_prec ] ...
    = fmm_solver.solve_iteartive_std_fmm_2_lev_add_schw_prec( eps, method );
  toc
  fprintf('Number of iterations: %d. \n', iter_schw_prec);
  approx_err_schw = fmm_solver.l2_error(x_schw_prec, exact_solution);
  fprintf( 'L2 error is: %f. \n', approx_err_schw ); 
  
  % save iteration numbers in iteration table
  iteration_table(L - L_min + 1, 2 ) = iter_no_prec;
  iteration_table(L - L_min + 1, 3 ) = iter_bpx_prec;
  iteration_table(L - L_min + 1, 4 ) = iter_diag_prec;
  iteration_table(L - L_min + 1, 5 ) = iter_schw_prec;
  
  figure(L - L_min + 1)
  subplot(1,2,1)
  semilogy( res_vec_bpx_prec );
  subplot(1,2,2)
  plot( panels(2, :), x_bpx_prec);
 
  % refine panels for next iteration
  new_panels = zeros( 2, 2^(L+1) * N_steps );
  new_panels( 1, 1 : 2 : end ) = panels( 1, : );
  new_panels( 1, 2 : 2 : end ) = 0.5 * ( panels( 1, : ) + panels( 2, : )  ); 
  new_panels( 2, 2 : 2 : end ) = panels( 2, : );
  new_panels( 2, 1 : 2 : end ) = 0.5 * ( panels( 1, : ) + panels( 2, : )  );
  panels = new_panels;
end

%% Test 4: Analyze behavior of GMRES for matrix V
t_start = 0;
t_end = 1;
N_steps = 2;

L_min = 4;
L_max = 10;

rhs_fun = @(t) ((exp(t) / 4) ...
  .* (exp(2) * erfc((1 + t) ./ (sqrt(t))) ...
  + 2 * erf(sqrt(t)) - exp(-2) * erfc((1 - t) ...
  ./ (sqrt(t)))));
V_properties = zeros( L_max - L_min + 1, 7 );

%construct random non-uniform intervals
% panels = zeros(2, 2^L_min * N_steps);
% rng( 0 );
% panels(2, :) = cumsum( 0.01 + rand( 1,  2^L_min * N_steps ) );
% panels(2, :) ...
%   = ( t_end - t_start ) * panels( 2, : ) ./ panels( 2, end ) + t_start;
% panels(1, 2 : end ) = panels(2, 1 : end - 1 );
% h_panels = panels(2, :) - panels(1, :);
% uniform = 0;

%construct uniform intervals
panels = zeros(2, 2^L_min * N_steps);
ht = 1 / (2^L_min * N_steps);
panels(1, :) = t_start : ht : t_end - ht;
panels(2, :) = t_start + ht : ht : t_end;
h_panels = panels(2, :) - panels(1, :);
uniform = 1;

for L = L_min : L_max
  V_properties( L - L_min + 1, 1 ) = 2^L * N_steps;
  
  % assemble the matrix and compute its norm and the smallest eigenvalue of its
  % Hermitian part
  full_cluster = temporal_cluster( panels, 1, size(panels, 2), t_start, t_end, 0 );
  fprintf('Assembling the matrix for %d timesteps.\n', 2^L * N_steps );
  assembler = full_assembler_arb_timestep( );
  V_full = assembler.assemble_V( full_cluster, full_cluster );
  norm_V = norm( V_full );
  V_hermit_part = 1/2 * ( V_full + V_full' );
  if ( uniform == 1 && min( h_panels ) >= 1 / 256  )
    coerc_const_V = eigs( V_hermit_part , 1, 'sm');
  else
    % for these matrices Matlab fails to estimate the lowest eigenvalue of the
    % hermitian part of V_full, so we estimate it
    coerc_const_V = V_properties( L - L_min, 4 ) / 2^1.5;
  end
  V_properties( L - L_min + 1, 2 ) = norm_V;
  V_properties( L - L_min + 1, 4 ) = coerc_const_V;
  V_properties( L - L_min + 1, 6 ) = norm_V / coerc_const_V;
  V_properties( L - L_min + 1, 7 ) = 1 / norm( inv( V_full ) );
  
  % solve the linear system as in previous tests with the same rhs
  %
  % compute the projection of the rhs function to the space of piecewise
  % constant functions on the current mesh
  n_intervals = size(panels, 2);
  [ x, w, l ] = quadratures.line(10);    
  rhs_projection = zeros(n_intervals, 1);
  for j = 1 : l
    rhs_projection = rhs_projection + w(j) * rhs_fun(panels(1, :)' + ...
      x(j) * h_panels' );
  end
  [sol, ~, ~, iter, residua] = gmres( V_full, rhs_projection, size( V_full, 1 ), 1e-8, ...
    size( V_full, 1 ) );
  figure( L - L_min + 1 );
  semilogy( residua(2 : end ) ./ residua(1), 'k' );
  hold on
  % plot estimated convergence by Beckermann
  beta = acos( coerc_const_V / norm_V );
  gamma = 2 * sin ( beta / ( 4 - 2 * beta / pi ) );
  semilogy( (2 + 2/sqrt(3)) * (2 + gamma) * gamma.^(1:iter(2)) , 'r' );
  % plot the CG bound (NOTE: de facto not relevant)
  condition_nr = norm_V / coerc_const_V;
  q = (sqrt(condition_nr) + 1 )/(sqrt(condition_nr) - 1 );
  semilogy( 2*q.^(1:iter(2))./(1+q.^(2*(1:iter(2)))), 'g' );
  fprintf('GMRES iterations = %d \n', iter(2));
  hold off
  
  new_panels = zeros( 2, 2^(L+1) * N_steps );
  new_panels( 1, 1 : 2 : end ) = panels( 1, : );
  new_panels( 1, 2 : 2 : end ) = 0.5 * ( panels( 1, : ) + panels( 2, : )  ); 
  new_panels( 2, 2 : 2 : end ) = panels( 2, : );
  new_panels( 2, 1 : 2 : end ) = 0.5 * ( panels( 1, : ) + panels( 2, : )  );
  panels = new_panels;
  h_panels = panels(2, :) - panels(1, :);
end
V_properties(2 : end, 3 ) = ...
  V_properties(1 : end - 1, 2 ) ./ V_properties( 2 : end, 2 );
V_properties(2 : end, 5 ) = ...
  V_properties(1 : end - 1, 4 ) ./ V_properties( 2 : end, 4 );
disp( V_properties );

%% Test 5: Analyze behavior of GMRES for preconditioned matrix V.
t_start = 0;
t_end = 1;
N_steps = 2;
eta = 2; %nearfield parameter for FMM
order = 5; %interpolation degree for FMM

L_min = 4;
L_max = 8;

rhs_fun = @(t) ((exp(t) / 4) ...
  .* (exp(2) * erfc((1 + t) ./ (sqrt(t))) ...
  + 2 * erf(sqrt(t)) - exp(-2) * erfc((1 - t) ...
  ./ (sqrt(t)))));
V_properties = zeros( L_max - L_min + 1, 7 );

%construct random non-uniform intervals
panels = zeros(2, 2^L_min * N_steps);
rng( 0 );
panels(2, :) = cumsum( 0.01 + rand( 1,  2^L_min * N_steps ) );
panels(2, :) ...
  = ( t_end - t_start ) * panels( 2, : ) ./ panels( 2, end ) + t_start;
panels(1, 2 : end ) = panels(2, 1 : end - 1 );
h_panels = panels(2, :) - panels(1, :);
uniform = 0;

%construct uniform intervals
% panels = zeros(2, 2^L_min * N_steps);
% ht = 1 / (2^L_min * N_steps);
% panels(1, :) = t_start : ht : t_end - ht;
% panels(2, :) = t_start + ht : ht : t_end;
% h_panels = panels(2, :) - panels(1, :);
% uniform = 1;

for L = L_min : L_max
  V_properties( L - L_min + 1, 1 ) = 2^L * N_steps;
  
  % assemble the matrix and compute its norm and the smallest eigenvalue of its
  % Hermitian part
  fmm_solver = cFMM_solver_adaptive(t_start, t_end, N_steps, L, order, ... 
        eta, rhs_fun, panels, true);
  % To construct the preconditioned matrix as a full matrix apply it to the unit
  % vectors.
  V_prec_full = zeros( size(panels, 2 ) );
  fprintf('Assembling the full matrix for %d timesteps.\n', 2^L * N_steps );
  for i = 1 : size( panels, 2 )
    unit_vec = zeros( size( panels, 2 ), 1 );
    unit_vec(i) = 1;
    V_prec_full( :, i ) = fmm_solver.apply_fmm_matrix_std( unit_vec );
    V_prec_full( :, i ) = fmm_solver.apply_bpx_prec( V_prec_full( :, i ) );
  end
      
  norm_V_prec = norm( V_prec_full );
  V_prec_hermit_part = 1/2 * ( V_prec_full + V_prec_full' );
%   if ( uniform == 1 && min( h_panels ) >= 1 / 256  )
  coerc_const_V_prec = eigs( V_prec_hermit_part , 1, 'sm');
%   else
    % for these matrices Matlab fails to estimate the lowest eigenvalue of the
    % hermitian part of V_full, so we estimate it
%     coerc_const_V_prec = V_properties( L - L_min, 4 ) / 2^1.5;
%   end
  V_properties( L - L_min + 1, 2 ) = norm_V_prec;
  V_properties( L - L_min + 1, 4 ) = coerc_const_V_prec;
  V_properties( L - L_min + 1, 6 ) = norm_V_prec / coerc_const_V_prec;
  V_properties( L - L_min + 1, 7 ) = 1 / norm( inv( V_prec_full ) );
  
%   % solve the linear system as in previous tests with the same rhs, using the
%   % FMM solver.
%   %
  method = 0; %GMRES
  eps = 1e-8; %precision for GMRES
  [ x_bpx_prec, iter_bpx_prec, res_vec_bpx_prec  ] ...
    = fmm_solver.solve_iterative_std_fmm_bpx_prec( eps, method );
  
  % solve the linear system with the full matrix constructed above
%   rhs_proj = fmm_solver.get_rhs_proj( );
%   rhs_proj = fmm_solver.apply_bpx_prec( rhs_proj );
    
  
  figure( L - L_min + 1 );
  semilogy( res_vec_bpx_prec(2 : end ) ./ res_vec_bpx_prec(1), 'k' );
  hold on
  % plot estimated convergence by Beckermann
  beta = acos( coerc_const_V_prec / norm_V_prec );
  gamma = 2 * sin ( beta / ( 4 - 2 * beta / pi ) );
  semilogy( (2 + 2/sqrt(3)) * (2 + gamma) * gamma.^(1:iter_bpx_prec) , 'r' );
  % plot a CG like bound (NOTE: de facto not relevant)
%   condition_nr = norm_V_prec / coerc_const_V_prec;
  condition_nr = norm_V_prec / V_properties( L - L_min + 1, 7 );
  q = (sqrt(condition_nr) + 1 )/(sqrt(condition_nr) - 1 );
  semilogy( 2*q.^(1:iter_bpx_prec)./(1+q.^(2*(1:iter_bpx_prec))), 'g' );
  fprintf('GMRES iterations = %d \n', iter_bpx_prec);
  hold off
  
  new_panels = zeros( 2, 2^(L+1) * N_steps );
  new_panels( 1, 1 : 2 : end ) = panels( 1, : );
  new_panels( 1, 2 : 2 : end ) = 0.5 * ( panels( 1, : ) + panels( 2, : )  ); 
  new_panels( 2, 2 : 2 : end ) = panels( 2, : );
  new_panels( 2, 1 : 2 : end ) = 0.5 * ( panels( 1, : ) + panels( 2, : )  );
  panels = new_panels;
  h_panels = panels(2, :) - panels(1, :);
end
V_properties(2 : end, 3 ) = ...
  V_properties(1 : end - 1, 2 ) ./ V_properties( 2 : end, 2 );
V_properties(2 : end, 5 ) = ...
  V_properties(1 : end - 1, 4 ) ./ V_properties( 2 : end, 4 );
disp( V_properties );