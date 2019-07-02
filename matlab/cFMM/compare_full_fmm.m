% Tests solution of a purely time-dependent problem on a sphere

% levels of the binary tree
L = 10;
% number of temporal panels per temporal cluster
N = 15;
% end time
T = 1;
% order of the Lagrange interpolant
order = 5;
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
x_fmm_dir = fmm_solver.solve_direct( );
toc

tic
fmm_solver = cFMM_solver(0, T, N, L, order, rhs_fun);
x_fmm_iter = fmm_solver.solve_iterative( );
toc

% solve using full matrices
tic
[ x_full, V ] = causal_full( T, N*2^L, rhs_fun );
toc

% compare multiplication
b = ones(size(V,2), 1);
tic
a1 = fmm_solver.apply_fmm_matrix(b);
toc
a2 = V * b;

% VVV = zeros(size(V));
% for i = 1 : size(V,2)
%   aa = zeros(size(V,2), 1);
%   aa(i) = 1;
%   col = fmm_solver.apply_fmm_matrix(b);
%   VVV(:, i) = col;
% end

ht = T / ( N * 2^L );
t = ht / 2 : ht : T - ht / 2;

analytical = exp( t );

figure;
title('Purely temporal problem solved using FMM and full matrices')
plot( t, x_fmm_dir );
hold on
plot( t, x_fmm_iter );
plot( t, x_full );

%legend({'FMM','full'},'Location','southwest');

% analytical solution for the first testing RHS (comment out for the other)
plot( t, analytical );
legend({'FMM - direct', 'FMM - iterative', 'full', 'analytical'},'Location','southwest');