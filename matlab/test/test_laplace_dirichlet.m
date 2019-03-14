function [ dir, neu, repr, err_bnd, err_vol ] = test_laplace_dirichlet( level )

if nargin < 1
  level = 0;
end

file='./input/cube_192.txt';
mesh = tri_mesh_3d( file );
mesh = mesh.refine( level );

% dir_fun = @( x, ~ ) ( 1 + x( :, 1 ) ) .* exp( 2 * pi * x( :, 2 ) ) .* ...
%   cos( 2 * pi * x( :, 3 ) );
% neu_fun = @( x, n ) exp( 2 * pi * x( :, 2 ) ) ...
%   .* ( n( 1 ) * cos( 2 * pi * x( :, 3 ) ) ...
%   + 2 * pi * ( 1 + x( :, 1 ) ) * n( 2 ) .* cos( 2 * pi * x( :, 3 ) ) ...
%   - 2 * pi * ( 1 + x( :, 1 ) ) * n( 3 ) .* sin( 2 * pi * x( :, 3 ) ) );

dir_fun = @( x, ~ ) x( :, 1 ) .* x( :, 2 ) .* x( :, 3 );
neu_fun = @( x, n ) n( 1 ) * x( :, 2 ) .* x( :, 3 ) ...
  + n( 2 ) * x( :, 1 ) .* x( :, 3 ) +  n( 3 ) * x( :, 1 ) .* x( :, 2 );

order_nf = 4;
order_ff = 4;

basis_p1 = p1( mesh );
basis_p0 = p0( mesh );

beas_v_laplace = be_assembler( mesh, kernel_laplace_sl, ...
  basis_p0, basis_p0, order_nf, order_ff );
fprintf( 1, 'Assembling V\n' );
tic;
V = beas_v_laplace.assemble( );
fprintf( 1, '  done in %f s.\n', toc );

beas_k_laplace = be_assembler( mesh, kernel_laplace_dl, ...
  basis_p0, basis_p1, order_nf, order_ff );
fprintf( 1, 'Assembling K\n' );
tic;
K = beas_k_laplace.assemble( );
fprintf( 1, '  done in %f s.\n', toc );

fprintf( 1, 'Assembling M\n' );
tic;
beid = be_identity( mesh, basis_p0, basis_p1, 1 );
M = beid.assemble( );
fprintf( 1, '  done in %f s.\n', toc );

fprintf( 1, 'Assembling rhs\n' );
tic;
L2_p1 = L2_tools( mesh, basis_p1, 5, 4 );
dir = L2_p1.projection( dir_fun );
rhs = 0.5 * M * dir;
rhs = rhs + K * dir;
fprintf( 1, '  done in %f s.\n', toc );

fprintf( 1, 'Solving the system\n' );
tic;
neu = V \ rhs;
fprintf( 1, '  done in %f s.\n', toc );

L2_p0 = L2_tools( mesh, basis_p0, 5, 4 );
err_bnd = L2_p0.relative_error( neu_fun, neu );
fprintf( 1, 'L2 relative error: %f.\n', err_bnd );

mesh.plot( dir, 'Dirichlet' );
mesh.plot( neu, 'Neumann' );

h = mesh.h / sqrt( 2 );
line( :, 1 ) = ( -1 + h ) : h : ( 1 - h );
l = size( line, 1 );
[ X, Y ] = meshgrid( line, line );
beev_v_laplace = be_evaluator( mesh, kernel_laplace_sl, p0( mesh ), neu, ...
  [ reshape( X, l^2, 1 ) reshape( Y, l^2, 1 ) 0.5 * ones( l^2, 1 ) ], ...
  order_ff );
fprintf( 1, 'Evaluating V\n' );
tic;
repr = beev_v_laplace.evaluate( );
fprintf( 1, '  done in %f s.\n', toc );

beev_k_laplace = be_evaluator( mesh, kernel_laplace_dl, p1( mesh ), dir, ...
  [ reshape( X, l^2, 1 ) reshape( Y, l^2, 1 ) 0.5 * ones( l^2, 1 ) ], ...
  order_ff );
fprintf( 1, 'Evaluating W\n' );
repr = repr - beev_k_laplace.evaluate( );
fprintf( 1, '  done in %f s.\n', toc );

figure;
handle = surf( X, Y, 0.5 * ones( l, l ), reshape( repr, l, l ) );
shading( 'interp' );
set( handle, 'EdgeColor', 'black' );
title( 'Solution' );
axis vis3d;

sol = ...
  dir_fun( [ reshape( X, l^2, 1 ) reshape( Y, l^2, 1 ) 0.5 * ones( l^2, 1 ) ] );
err_vol = sqrt( sum( ( repr - sol ).^2 ) / sum( sol.^2 ) );
fprintf( 1, 'l2 relative error: %f.\n', err_vol );

end
