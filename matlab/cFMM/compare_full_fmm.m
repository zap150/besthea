% Tests solution of a purely time-dependent problem on a sphere

% levels of the binary tree
L = 6;
% number of temporal panels per temporal cluster
N = 32;
% end time
T = 1;
% order of the Lagrange interpolant
order = 5;
% RHS function
rhs_fun = @( t ) ( ( exp( t ) / 4 ) ...
  .* ( exp( 2 ) * erfc( ( 1 + t ) ./ ( sqrt( t ) ) ) ...
  + 2 * erf( sqrt( t )) - exp( -2 ) * erfc( ( 1 - t ) ...
  ./ ( sqrt( t ) ) ) ) );
%rhs_fun =@( t ) sin( 4*pi*t )*exp( -t );
exact_solution = @(t) ( exp(t) );

fprintf('\nSolving problem with %d time-steps, end-time = %f. \n', N*2^L, T);

%solve using FMM
fprintf('\nSolving using time-stepping FMM. \n');
tic
fmm_solver = cFMM_solver(0, T, N, L, order, rhs_fun);
x_fmm_dir = fmm_solver.solve_direct( );
toc
err_fmm_dir = fmm_solver.l2_error(x_fmm_dir, exact_solution, T, N*2^L);

fprintf('\nSolving using GMRES with temporal FMM. \n');
fmm_solver = cFMM_solver(0, T, N, L, order, rhs_fun);
tic
x_fmm_iter = fmm_solver.solve_iterative( );
toc
err_fmm_iter = fmm_solver.l2_error(x_fmm_iter, exact_solution, T, N*2^L);

fprintf('\nSolving using GMRES with standard FMM. \n');
fmm_solver = cFMM_solver(0, T, N, L, order, rhs_fun);
tic
x_std_fmm_iter = fmm_solver.solve_iterative_std_fmm( );
toc
err_std_fmm_iter = fmm_solver.l2_error(x_std_fmm_iter, exact_solution, T, N*2^L);

% solve using full matrices
fprintf('\nSolving using dense matrices. \n');
tic
[ x_full, V ] = causal_full( T, N*2^L, rhs_fun );
toc
err_full = fmm_solver.l2_error(x_full, exact_solution, T, N*2^L);

ht = T / ( N * 2^L );
t = ht / 2 : ht : T - ht / 2;
analytical = exp( t );

figure;
title('Purely temporal problem solved using FMM and full matrices')
plot( t, x_fmm_dir );
hold on
plot( t, x_fmm_iter );
plot( t, x_std_fmm_iter );
plot( t, x_full );


%legend({'FMM','full'},'Location','southwest');

% analytical solution for the first testing RHS (comment out for the other)
plot( t, analytical );
legend({'FMM - direct', 'FMM - iterative', 'standard FMM - iterative', 'full', 'analytical'},'Location','northwest');
%legend({'FMM - direct', 'FMM - iterative', 'standard FMM - iterative', 'full'},'Location','southwest');

fprintf('err_fmm_dir:       %.8f \n', err_fmm_dir)
fprintf('err_fmm_iter:      %.8f \n', err_fmm_iter)
fprintf('err_std_fmm_iter:  %.8f \n', err_std_fmm_iter)
fprintf('err_full:          %.8f \n', err_full)
