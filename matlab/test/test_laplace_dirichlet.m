function [ dir, neu, err_bnd, err_vol ] = test_laplace_dirichlet( level )

if nargin < 1
  level = 0;
end

file='../../bem4i/input/cube_192.txt';
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

beas_v_laplace = be_assembler( mesh, kernel_laplace_sl, ...
  p0( mesh ), p0( mesh ), order_nf, order_ff );
fprintf( 1, 'Assembling V\n' );
tic;
V = beas_v_laplace.assemble( );
fprintf( 1, '  done in %f s.\n', toc );

beas_k_laplace = be_assembler( mesh, kernel_laplace_dl, ...
  p0( mesh ), p1( mesh ), order_nf, order_ff );
fprintf( 1, 'Assembling K\n' );
tic;
K = beas_k_laplace.assemble( );
fprintf( 1, '  done in %f s.\n', toc );

fprintf( 1, 'Assembling M\n' );
tic;
beid = be_identity( mesh, p0( mesh ), p1( mesh ), 1 );
M = beid.assemble( );
fprintf( 1, '  done in %f s.\n', toc );

beid_p1p1 = be_identity( mesh, p1( mesh ), p1( mesh ), 5 );

fprintf( 1, 'Assembling rhs\n' );
tic;
dir = beid_p1p1.L2_projection( dir_fun );
rhs = 0.5 * M * dir;
rhs = rhs + K * dir;
fprintf( 1, '  done in %f s.\n', toc );

fprintf( 1, 'Solving the system\n' );
tic;
neu = V \ rhs;
fprintf( 1, '  done in %f s.\n', toc );

[ x_ref, w, ~ ] = quadratures.tri( 5 );
l2_diff_err = 0;
l2_err = 0;
n_elems = mesh.n_elems;
for i_tau = 1 : n_elems
  %nodes = mesh.get_nodes( i_tau );
  nodes = mesh.nodes( mesh.elems( i_tau, : ), : );
  x = x_ref ...
    * [ nodes( 2, : ) - nodes( 1, : ); nodes( 3, : ) - nodes( 1, : ) ] ...
    + nodes( 1, : );
  f = neu_fun( x, mesh.normals( i_tau, : ) );
  area = mesh.areas( i_tau );
  l2_diff_err = l2_diff_err + ( w' * ( f - neu( i_tau ) ).^2 ) * area;
  l2_err = l2_err + ( w' * f.^2 ) * area;
end

err_bnd = sqrt( l2_diff_err / l2_err );
fprintf( 1, 'L2 relative error: %f.\n', err_bnd );

mesh.plot( dir, 'Dirichlet' );
mesh.plot( neu, 'Neumann' );

h = mesh.h / sqrt( 2 );
line( :, 1 ) = ( -1 + h ) : h : ( 1 - h );
l = size( line, 1 );
[ X, Y ] = meshgrid( line, line );
beev_v_laplace = be_evaluator( mesh, kernel_laplace_sl, p0( mesh ), neu, ...
  [ reshape( X, l^2, 1 ) reshape( Y, l^2, 1 ) zeros( l^2, 1 ) ], order_ff );
fprintf( 1, 'Evaluating V\n' );
tic;
repr = beev_v_laplace.evaluate( );
fprintf( 1, '  done in %f s.\n', toc );

beev_k_laplace = be_evaluator( mesh, kernel_laplace_dl, p1( mesh ), dir, ...
  [ reshape( X, l^2, 1 ) reshape( Y, l^2, 1 ) zeros( l^2, 1 ) ], order_ff );
fprintf( 1, 'Evaluating W\n' );
repr = repr - beev_k_laplace.evaluate( );
fprintf( 1, '  done in %f s.\n', toc );

figure;
handle = surf( X, Y, zeros( l, l ), reshape( repr, l, l ) );
shading( 'interp' );
set( handle, 'EdgeColor', 'black' );
title( 'Solution' );

sol = dir_fun( [ reshape( X, l^2, 1 ) reshape( Y, l^2, 1 ) zeros( l^2, 1 ) ] );
err_vol = abs( repr - sol );

end
