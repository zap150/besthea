function [ dir, neu, err_bnd ] = test_heat_dirichlet( level )
%function test_heat_dirichlet( level, neu )

if nargin < 1
  level = 0;
end

file='./input/cube_192.txt';
stmesh = spacetime_mesh( file, 1, 8 );
stmesh = stmesh.refine_xt( level, 2 );
% stmesh = spacetime_mesh( file, 1, 3 );
% stmesh = stmesh.refine_xt( level, 1 );

order_nf = 4;
order_ff = 4;

alpha = 1;

y = [ 0 0 1.5 ];
dir_fun = @( x, t, ~ ) ( 4 * pi * alpha * t )^( -3 / 2 ) ...
  .* exp( - ( ( x - y ).^2 * [ 1; 1; 1 ] ) / ( 4 * alpha * t ) );
neu_fun = @( x, t, n ) ( - 2 * t )^( -1 ) * dir_fun( x, t ) ...
  .* ( ( x - y ) * n' );

basis_p1 = p1( stmesh );
basis_p0 = p0( stmesh );

beas_v_heat = be_assembler( stmesh, kernel_heat_sl( alpha ), ...
  basis_p0, basis_p0, order_nf, order_ff );
fprintf( 1, 'Assembling V\n' );
tic;
V = beas_v_heat.assemble( );
fprintf( 1, '  done in %f s.\n', toc );

beas_k_heat = be_assembler( stmesh, kernel_heat_dl( alpha ), ...
  basis_p0, basis_p1, order_nf, order_ff );
fprintf( 1, 'Assembling K\n' );
tic;
K = beas_k_heat.assemble( );
fprintf( 1, '  done in %f s.\n', toc );

fprintf( 1, 'Assembling M\n' );
tic;
beid = be_identity( stmesh, basis_p0, basis_p1, 1 );
M = beid.assemble( );
fprintf( 1, '  done in %f s.\n', toc );

L2_p1 = L2_tools( stmesh, basis_p1, 5, 4 );
dir = L2_p1.projection( dir_fun );

solver = spacetime_solver( );

fprintf( 1, 'Solving the system\n' );
tic;
neu = solver.solve_dirichlet( V, K, M, dir );
fprintf( 1, '  done in %f s.\n', toc );

L2_p0 = L2_tools( stmesh, basis_p0, 5, 4 );
fprintf( 1, 'L2 relative error: %f.\n', L2_p0.relative_error( neu_fun, neu ) );

stmesh.plot( dir{ 1 }, sprintf( 'Dirichlet, t = %f', 0 ) );
stmesh.plot( dir{ stmesh.nt }, sprintf( 'Dirichlet, t = %f', stmesh.T ) );
stmesh.plot( neu{ 1 }, sprintf( 'Neumann, t = %f', 0 ) );
stmesh.plot( neu{ stmesh.nt }, sprintf( 'Neumann, t = %f', stmesh.T ) );

% h = mesh.h / sqrt( 2 );
% line( :, 1 ) = ( -1 + h ) : h : ( 1 - h );
% l = size( line, 1 );
% [ X, Y ] = meshgrid( line, line );
% beev_v_laplace = be_evaluator( mesh, kernel_laplace_sl, p0( mesh ), neu, ...
%   [ reshape( X, l^2, 1 ) reshape( Y, l^2, 1 ) zeros( l^2, 1 ) ], order_ff );
% fprintf( 1, 'Evaluating V\n' );
% tic;
% repr = beev_v_laplace.evaluate( );
% fprintf( 1, '  done in %f s.\n', toc );
% 
% beev_k_laplace = be_evaluator( mesh, kernel_laplace_dl, p1( mesh ), dir, ...
%   [ reshape( X, l^2, 1 ) reshape( Y, l^2, 1 ) zeros( l^2, 1 ) ], order_ff );
% fprintf( 1, 'Evaluating W\n' );
% repr = repr - beev_k_laplace.evaluate( );
% fprintf( 1, '  done in %f s.\n', toc );
% 
% figure;
% handle = surf( X, Y, zeros( l, l ), reshape( repr, l, l ) );
% shading( 'interp' );
% set( handle, 'EdgeColor', 'black' );
% title( 'Solution' );
% 
% sol = dir_fun( [ reshape( X, l^2, 1 ) reshape( Y, l^2, 1 ) zeros( l^2, 1 ) ] );
% err_vol = abs( repr - sol );

end
