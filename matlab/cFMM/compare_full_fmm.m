% Tests solution of a purely time-dependent problem on a sphere

% levels of the binary tree
L = 10;
% number of temporal panels per temporal cluster
N = 15;
% end time
T = 1;
% order of the Lagrange interpolant
order = 3;
% RHS function
rhs_fun = @( t ) ( ( exp( t ) / 4 ) ...
  .* ( exp( 2 ) * erfc( ( 1 + t ) ./ ( sqrt( t ) ) ) ...
  + 2 * erf( sqrt( t ) - exp( -2 ) * erfc( ( 1 - t ) ... 
  ./ ( sqrt( t ) ) ) ) ) );
%rhs_fun =@( t ) sin( 8*pi*t )*exp( -t );

fprintf('Solving problem with %d time-steps, end-time = %f. \n', N*2^L, T ...
  );

% solve using FMM
tic
fmm_solver = cFMM_solver(0, T, N, L, order, rhs_fun);
x_fmm = fmm_solver.solve( );
toc

% solve using full matrices
tic
x_full = causal_full( T, N*2^L, rhs_fun );
toc

ht = T / ( N * 2^L );
t = ht / 2 : ht : T - ht / 2;

analytical = exp( t );

figure;
title('Purely temporal problem solved using FMM and full matrices')
plot( t, x_fmm );
hold on
plot( t, x_full );
%legend({'FMM','full'},'Location','southwest');

% analytical solution for the first testing RHS (comment out for the other)
plot( t, analytical );
legend({'FMM','fulll', 'analytical'},'Location','southwest');