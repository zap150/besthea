% Tests solution of a purely time-dependent problem on a sphere

% levels of the binary tree
L = 4;
% number of temporal panels per temporal cluster
N = 20;
% end time
T = 1;
% order of the Lagrange interpolant
order = 5;
% RHS function
rhs_fun = @( t ) ( ( exp( t ) / 4 ) ...
  .* ( exp( 2 ) * erfc( ( 1 + t ) ./ ( sqrt( t ) ) ) ...
  + 2 * erf( sqrt( t ) - exp( -2 ) * erfc( ( 1 - t ) ...
  ./ ( sqrt( t ) ) ) ) ) );
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
legend({'FMM - direct', 'FMM - iterative', 'standard FMM - iterative', 'full', 'analytical'},'Location','southwest');
%legend({'FMM - direct', 'FMM - iterative', 'standard FMM - iterative', 'full'},'Location','southwest');