function [ dir, neu, repr, err_bnd, err_vol ] = helmholtz_neumann( level )

if nargin < 1
  level = 0;
end

file='./input/cube_192.txt';
mesh = tri_mesh_3d( file );
mesh = mesh.refine( level );

order_nf = 4;
order_ff = 4;

kappa = 2;

basis_p1 = p1( mesh );
basis_curl_p1 = curl_p1( mesh );
basis_p0 = p0( mesh );

beas_d1_helmholtz = be_assembler( mesh, kernel_helmholtz_hs1( kappa ), ...
  basis_curl_p1, basis_curl_p1, order_nf, order_ff );
fprintf( 1, 'Assembling D1\n' );
tic;
D = beas_d1_helmholtz.assemble( );
fprintf( 1, '  done in %f s.\n', toc );

beas_d2_helmholtz = be_assembler( mesh, kernel_helmholtz_hs2( kappa ), ...
  basis_p1, basis_p1, order_nf, order_ff );
fprintf( 1, 'Assembling D2\n' );
tic;
D2 = beas_d2_helmholtz.assemble( );
fprintf( 1, '  done in %f s.\n', toc );

D = D + D2;

beas_k_helmholtz = be_assembler( mesh, kernel_helmholtz_dl( kappa ), ...
  basis_p0, basis_p1, order_nf, order_ff );
fprintf( 1, 'Assembling K\n' );
tic;
K = beas_k_helmholtz.assemble( );
fprintf( 1, '  done in %f s.\n', toc );

fprintf( 1, 'Assembling M\n' );
tic;
beid = be_identity( mesh, basis_p0, basis_p1, order_ff );
M = beid.assemble( );
fprintf( 1, '  done in %f s.\n', toc );

fprintf( 1, 'Assembling rhs\n' );
tic;
L2_p0 = L2_tools( mesh, basis_p0, 5, 4 );
neu = L2_p0.projection( @neu_fun );
rhs = 0.5 * M' * neu;
rhs = rhs - K.' * neu;
fprintf( 1, '  done in %f s.\n', toc );

fprintf( 1, 'Solving the system\n' );
tic;
dir = D \ rhs;
fprintf( 1, '  done in %f s.\n', toc );

L2_p1 = L2_tools( mesh, basis_p1, 5, 4 );
err_bnd = L2_p1.relative_error( @dir_fun, dir );
fprintf( 1, 'L2 relative error: %f.\n', err_bnd );

mesh.plot( real( dir ), 'Dirichlet Re' );
mesh.plot( imag( dir ), 'Dirichlet Im' );
mesh.plot( real( neu ), 'Neumann Re' );
mesh.plot( imag( neu ), 'Neumann Im' );

h = mesh.h / sqrt( 2 );
line( :, 1 ) = ( -1 + h ) : h : ( 1 - h );
l = size( line, 1 );
[ X, Y ] = meshgrid( line, line );
z = 0.5;
points = [ reshape( X, l^2, 1 ) reshape( Y, l^2, 1 ) z * ones( l^2, 1 ) ];
beev_v_helmholtz = be_evaluator( mesh, kernel_helmholtz_sl( kappa ), ...
  basis_p0, neu, points, order_ff );
fprintf( 1, 'Evaluating V\n' );
tic;
repr = beev_v_helmholtz.evaluate( );
fprintf( 1, '  done in %f s.\n', toc );

beev_k_helmholtz = be_evaluator( mesh, kernel_helmholtz_dl( kappa ), ...
  basis_p1, dir, points, order_ff );
fprintf( 1, 'Evaluating W\n' );
repr = repr - beev_k_helmholtz.evaluate( );
fprintf( 1, '  done in %f s.\n', toc );

figure;
handle = surf( X, Y, z * ones( l, l ), reshape( real( repr ), l, l ) );
shading( 'interp' );
set( handle, 'EdgeColor', 'black' );
title( 'Solution Re' );
axis vis3d;

figure;
handle = surf( X, Y, z * ones( l, l ), reshape( imag( repr ), l, l ) );
shading( 'interp' );
set( handle, 'EdgeColor', 'black' );
title( 'Solution Im' );
axis vis3d;

sol = dir_fun( points );
err_vol = sqrt( sum( abs( repr - sol ).^2 ) / sum( sol.^2 ) );
fprintf( 1, 'l2 relative error: %f.\n', err_vol );

end

function value = dir_fun( x, ~ )

y = [ 0 0 1.5 ];
kappa = 2;
norm = sqrt( ( x - y ).^2 * [ 1; 1; 1 ] );
value = exp( 1i * kappa * norm ) ./ ( 4 * pi * norm );

end

function value = neu_fun( x, n )

y = [ 0 0 1.5 ];
kappa = 2;
xy = x - y;
norm = sqrt( xy.^2 * [ 1; 1; 1 ] );
value = -( xy * n' ) ./ ( 4 * pi * norm .* norm .* norm ) ...
  .* ( 1 - 1i * kappa * norm ) .* exp( 1i * kappa * norm );

end
