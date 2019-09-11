% Tests solution of a purely time-dependent problem on a sphere

% levels of the binary tree
L = 7;
% number of temporal panels per temporal cluster
N = 20;
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

fprintf('\nSolving problem with %d time-steps, end-time = %f. \n', N*2^L, T ...
  );

% solve using FMM
fprintf('\nSolving using time-stepping FMM. \n');
fmm_solver = cFMM_solver(0, T, N, L, order, rhs_fun);
tic
x_fmm_dir = fmm_solver.solve_direct( );
toc

fprintf('\nSolving using GMRES with temporal FMM. \n');
fmm_solver = cFMM_solver(0, T, N, L, order, rhs_fun);
tic
x_fmm_iter = fmm_solver.solve_iterative( );
toc


fprintf('\nSolving using GMRES with standard FMM. \n');
fmm_solver = cFMM_solver(0, T, N, L, order, rhs_fun);
tic
x_std_fmm_iter = fmm_solver.solve_iterative_std_fmm( );
toc


% solve using full matrices
fprintf('\nSolving using dense matrices. \n');
tic
[ x_full, V ] = causal_full( T, N*2^L, rhs_fun );
toc


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

%error computation
panels = zeros(2, N * 2^L);
panels(1,:) = 0 : ht : T - ht;
panels(2,:) = ht : ht : T;
exact = @(t) (exp(t));
nr_int_points = 5;
err_fmm_dir = l2_error_pw_const(panels, x_fmm_dir, exact, nr_int_points);
err_fmm_iter = l2_error_pw_const(panels, x_fmm_iter, exact, nr_int_points);
err_std_fmm_iter = l2_error_pw_const(panels, x_std_fmm_iter, exact, nr_int_points);
err_full = l2_error_pw_const(panels, x_full, exact, nr_int_points);
fprintf('err_fmm_dir:       %.8f \n', err_fmm_dir)
fprintf('err_fmm_iter:      %.8f \n', err_fmm_iter)
fprintf('err_std_fmm_iter:  %.8f \n', err_std_fmm_iter)
fprintf('err_full:          %.8f \n', err_full)