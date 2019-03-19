function ...
  [ dir, neu, neu_proj, repr, repr_interp, err_bnd, err_bnd_x, err_bnd_proj, ...
  err_bnd_proj_x, err_vol, err_vol_x ] = heat_dirichlet( level )

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
err_bnd =  L2_p0.relative_error( neu_fun, neu );
fprintf( 1, 'L2 relative error: %f.\n', err_bnd );
neu_proj = L2_p0.projection( neu_fun );
err_bnd_proj =  L2_p0.relative_error( neu_fun, neu_proj );
fprintf( 1, 'Projection L2 relative error: %f.\n', err_bnd_proj );

err_bnd_x = zeros( stmesh.nt, 1 );
err_bnd_proj_x = zeros( stmesh.nt, 1 );
for i= 1 : stmesh.nt
  t = ( i - 0.5 ) / stmesh.nt;
  neu_fun_t = @( x, n ) neu_fun( x, t, n );
  err_bnd_x( i ) = L2_p0.relative_error_s( neu_fun_t, neu{ i } );
  err_bnd_proj_x( i ) = L2_p0.relative_error_s( neu_fun_t, neu_proj{ i } );
end

stmesh.plot( dir{ 1 }, sprintf( 'Dirichlet, t = %f', stmesh.ht / 2 ) );
stmesh.plot( dir{ stmesh.nt }, sprintf( 'Dirichlet, t = %f', ...
  stmesh.T - stmesh.ht / 2 ) );
stmesh.plot( neu{ 1 }, sprintf( 'Neumann, t = %f', stmesh.ht / 2 ) );
stmesh.plot( neu{ stmesh.nt }, sprintf( 'Neumann, t = %f', ...
  stmesh.T - stmesh.ht / 2 ) );

h = stmesh.h / sqrt( 2 );
line( :, 1 ) = ( -1 + h ) : h : ( 1 - h );
l = size( line, 1 );
[ X, Y ] = meshgrid( line, line );
z = 0;
points = [ reshape( X, l^2, 1 ) reshape( Y, l^2, 1 ) z * ones( l^2, 1 ) ];
beev_v_heat = be_evaluator( stmesh, kernel_heat_sl( alpha ), basis_p0, ...
  neu, points, order_ff );
fprintf( 1, 'Evaluating V\n' );
tic;
repr = beev_v_heat.evaluate( );
fprintf( 1, '  done in %f s.\n', toc );

beev_w_heat = be_evaluator( stmesh, kernel_heat_dl( alpha ), basis_p1, ... 
  dir, points, order_ff );
fprintf( 1, 'Evaluating W\n' );
pot_k = beev_w_heat.evaluate( );
for d = 1 : stmesh.nt + 1
  repr{ d } = repr{ d } - pot_k{ d };
end
fprintf( 1, '  done in %f s.\n', toc );

% figure;
% handle = surf( X, Y, z * ones( l, l ), reshape( repr{ 1 }, l, l ) );
% shading( 'interp' );
% set( handle, 'EdgeColor', 'black' );
% title( sprintf( 'Solution, t = %f', 0 ) );
% axis vis3d;

figure;
handle = surf( X, Y, z * ones( l, l ), reshape( repr{ stmesh.nt + 1 }, l, l ) );
shading( 'interp' );
set( handle, 'EdgeColor', 'black' );
title( sprintf( 'Solution, t = %f', stmesh.T ) );
axis vis3d;

err_vol = 0;
aux = 0;
err_vol_x = zeros( stmesh.nt + 1, 1 );
repr_interp = cell( stmesh.nt + 1, 1 );
%%%%% initial condition
err_vol_x( 1 ) = 0;
repr_interp{ 1 } = zeros( l^2, 1 );
for d = 1 : stmesh.nt
  repr_interp{ d + 1 } = dir_fun( points, d * stmesh.ht );
  aux1 = sum( ( repr{ d + 1 } - repr_interp{ d + 1 } ).^2 );
  aux2 = sum( repr_interp{ d + 1 }.^2 );
  err_vol_x( d + 1 ) = sqrt( aux1 / aux2 );
  err_vol = err_vol + aux1;
  aux = aux + aux2;
end
err_vol = sqrt( err_vol / aux );
fprintf( 1, 'l2 relative error: %f.\n', err_vol );

end
