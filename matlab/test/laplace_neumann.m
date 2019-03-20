function [ dir, neu, repr, err_bnd, err_vol ] = laplace_neumann( level )

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
basis_curl_p1 = curl_p1( mesh );

beas_d_laplace = be_assembler( mesh, kernel_laplace_hs, ...
  basis_curl_p1, basis_curl_p1, order_nf, order_ff );
fprintf( 1, 'Assembling D\n' );
tic;
D = beas_d_laplace.assemble( );
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
L2_p0 = L2_tools( mesh, basis_p0, 5, 4 );
neu = L2_p0.projection( neu_fun );
rhs = 0.5 * M' * neu;
rhs = rhs - K' * neu;
fprintf( 1, '  done in %f s.\n', toc );

fprintf( 1, 'Stabilizing\n' );
tic;
a = M' * ones( mesh.n_elems, 1 );
D = D + a * a';
alpha = integral_continuous( mesh, dir_fun );
%L2_p1 = L2_tools( mesh, basis_p1, 5, 4 );
%dir_proj = L2_p1.projection( dir_fun );
%alpha = integral_discrete( mesh, basis_p1, dir_proj );
rhs = rhs + alpha * a;
fprintf( 1, '  done in %f s.\n', toc );

fprintf( 1, 'Solving the system\n' );
tic;
dir = D \ rhs;
fprintf( 1, '  done in %f s.\n', toc );

L2_p1 = L2_tools( mesh, basis_p1, 5, 4 );
err_bnd = L2_p1.relative_error( dir_fun, dir );
fprintf( 1, 'L2 relative error: %f.\n', err_bnd );

mesh.plot( dir, 'Dirichlet' );
mesh.plot( neu, 'Neumann' );

h = mesh.h / sqrt( 2 );
line( :, 1 ) = ( -1 + h ) : h : ( 1 - h );
l = size( line, 1 );
[ X, Y ] = meshgrid( line, line );
beev_v_laplace = be_evaluator( mesh, kernel_laplace_sl, basis_p0, neu, ...
  [ reshape( X, l^2, 1 ) reshape( Y, l^2, 1 ) 0.5 * ones( l^2, 1 ) ], ...
  order_ff );
fprintf( 1, 'Evaluating V\n' );
tic;
repr = beev_v_laplace.evaluate( );
fprintf( 1, '  done in %f s.\n', toc );

beev_k_laplace = be_evaluator( mesh, kernel_laplace_dl, basis_p1, dir, ...
  [ reshape( X, l^2, 1 ) reshape( Y, l^2, 1 ) 0.5 * ones( l^2, 1 ) ], ...
  order_ff );
fprintf( 1, 'Evaluating W\n' );
repr = repr - beev_k_laplace.evaluate( );
fprintf( 1, '  done in %f s.\n', toc );

figure;
handle = surf( X, Y, zeros( l, l ), reshape( repr, l, l ) );
shading( 'interp' );
set( handle, 'EdgeColor', 'black' );
title( 'Solution' );
axis vis3d;

sol = dir_fun( ...
  [ reshape( X, l^2, 1 ) reshape( Y, l^2, 1 ) 0.5 * ones( l^2, 1 ) ] );
err_vol = sqrt( sum( ( repr - sol ).^2 ) / sum( sol.^2 ) );
fprintf( 1, 'l2 relative error: %f.\n', err_vol );

end

function result = integral_continuous( mesh, fun )

[ x_ref, wx, ~ ] = quadratures.tri( 5 );
result = 0;
n_elems = mesh.n_elems;
for i_tau = 1 : n_elems
  nodes = mesh.nodes( mesh.elems( i_tau, : ), : );
  x = nodes( 1, : ) + x_ref ...
    * [ nodes( 2, : ) - nodes( 1, : ); nodes( 3, : ) - nodes( 1, : ) ];
  area = mesh.areas( i_tau );
  f = fun( x, mesh.normals( i_tau, : ) );
  result = result + ( wx' * f ) * area;
end

end

function result = integral_discrete( mesh, basis, fun )

[ x_ref, wx, ~ ] = quadratures.tri( 5 );
result = 0;
n_elems = mesh.n_elems;
basis_dim = basis.dim_local( );
for i_tau = 1 : n_elems
  basis_val = basis.eval( x_ref );
  basis_map = basis.l2g( i_tau );
  area = mesh.areas( i_tau );
  val = 0;
  for i_local_dim = 1 : basis_dim
    val = val + fun( basis_map( i_local_dim ) ) * basis_val( :, i_local_dim );
  end
  result = result + ( wx' * val ) * area;
end

end
