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

beas_v_heat = be_assembler( stmesh, kernel_heat_sl( alpha ), ...
  p0( stmesh ), p0( stmesh ), order_nf, order_ff );
fprintf( 1, 'Assembling V\n' );
tic;
V = beas_v_heat.assemble( );
fprintf( 1, '  done in %f s.\n', toc );

beas_k_heat = be_assembler( stmesh, kernel_heat_dl( alpha ), ...
  p0( stmesh ), p1( stmesh ), order_nf, order_ff );
fprintf( 1, 'Assembling K\n' );
tic;
K = beas_k_heat.assemble( );
fprintf( 1, '  done in %f s.\n', toc );

fprintf( 1, 'Assembling M\n' );
tic;
beid = be_identity( stmesh, p0( stmesh ), p1( stmesh ), 1 );
M = beid.assemble( );
fprintf( 1, '  done in %f s.\n', toc );

beid_p1p1 = be_identity( stmesh, p1( stmesh ), p1( stmesh ), 5, 4 ); 
dir = beid_p1p1.L2_projection( dir_fun );

solver = spacetime_solver( );

fprintf( 1, 'Solving the system\n' );
tic;
neu = solver.solve_dirichlet( V, K, M, dir );
fprintf( 1, '  done in %f s.\n', toc );

[ x_ref, wx, ~ ] = quadratures.tri( 5 );
[ t_ref, wt, lt ] = quadratures.line( 4 );
l2_diff_err = 0;
l2_err = 0;
n_elems = stmesh.n_elems;
nt = stmesh.nt;
ht = stmesh.ht;
for d = 0 : nt - 1
  t = ht * ( t_ref + d );
  for i_tau = 1 : n_elems
    nodes = stmesh.nodes( stmesh.elems( i_tau, : ), : );
    x = x_ref ...
      * [ nodes( 2, : ) - nodes( 1, : ); nodes( 3, : ) - nodes( 1, : ) ] ...
      + nodes( 1, : );
    area = stmesh.areas( i_tau );
    for i_t = 1 : lt
      f = neu_fun( x, t( i_t ), stmesh.normals( i_tau, : ) );
      l2_diff_err = l2_diff_err + ( wx' * ( f - neu{ d + 1 }( i_tau ) ).^2 ) ...
        * area * ht * wt( i_t );
      l2_err = l2_err + ( wx' * f.^2 ) * area * ht * wt( i_t );
    end
  end
end
 
err_bnd = sqrt( l2_diff_err / l2_err );
fprintf( 1, 'L2 relative error: %f.\n', err_bnd );

stmesh.plot( dir{ 1 }, sprintf( 'Dirichlet, t = %f', 0 ) );
stmesh.plot( dir{ nt }, sprintf( 'Dirichlet, t = %f', stmesh.T ) );
stmesh.plot( neu{ 1 }, sprintf( 'Neumann, t = %f', 0 ) );
stmesh.plot( neu{ nt }, sprintf( 'Neumann, t = %f', stmesh.T ) );

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
