function ...
  [ dir, neu, dir_proj, repr, repr_interp, err_bnd, err_bnd_x, err_bnd_proj, ...
  err_bnd_proj_x, err_vol, err_vol_x ] = heat_neumann_prec( level )

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

alpha = 0.5;
y = [ 0 0 1.5 ];

dir_fun = @( x, t, ~ ) ( 4 * pi * alpha * t )^( -3 / 2 ) ...
  .* exp( - ( ( x - y ).^2 * [ 1; 1; 1 ] ) / ( 4 * alpha * t ) );
neu_fun = @( x, t, n ) ( - 2 * t )^( -1 ) * dir_fun( x, t ) ...
  .* ( ( x - y ) * n' );

% b = alpha;
% alpha = 1;
% stmesh = spacetime_mesh( file, b, 8 );
% stmesh = stmesh.refine_xt( level, 2 );
% dir_fun = @( x, t, ~ ) ( 4 * pi * t )^( -3 / 2 ) ...
%   .* exp( - ( ( x - y ).^2 * [ 1; 1; 1 ] ) / ( 4 * t ) );
% neu_fun = @( x, t, n ) ( - 2 * t )^( -1 ) * dir_fun( x, t ) ...
%   .* ( ( x - y ) * n' );

basis_p1 = p1( stmesh );
basis_p0 = p0( stmesh );
basis_curl_p1 = curl_p1( stmesh );

beas_d1_heat = be_assembler( stmesh, kernel_heat_hs1_2( alpha ), ...
  basis_curl_p1, basis_curl_p1, order_nf, order_ff );
fprintf( 1, 'Assembling D1\n' );
tic;
D = beas_d1_heat.assemble( );
fprintf( 1, '  done in %f s.\n', toc );

beas_d2_heat = be_assembler( stmesh, kernel_heat_hs2_2( alpha ), ...
  basis_p1, basis_p1, order_nf, order_ff );
fprintf( 1, 'Assembling D2\n' );
tic;
D2 = beas_d2_heat.assemble( );
fprintf( 1, '  done in %f s.\n', toc );

%D1 = D;
%return;

for i = 1 : stmesh.nt
  D{ i } = D{ i } + D2{ i };
end

beas_k_heat = be_assembler( stmesh, kernel_heat_dl_2( alpha ), ...
  basis_p0, basis_p1, order_nf, order_ff );
fprintf( 1, 'Assembling K\n' );
tic;
K = beas_k_heat.assemble( );
fprintf( 1, '  done in %f s.\n', toc );

fprintf( 1, 'Assembling M\n' );
tic;
beid = be_identity( stmesh, basis_p0, basis_p1, order_ff );
M = beid.assemble( );
fprintf( 1, '  done in %f s.\n', toc );

L2_p0 = L2_tools( stmesh, basis_p0, 5, 4 );
neu = L2_p0.projection( neu_fun );

fprintf( 1, 'Setting up rhs\n' );
tic;
rhs = apply_M( M, neu, true );
rhs2 = apply_toeplitz( K, neu, true );
for i= 1 : stmesh.nt
  rhs{ i } = 0.5 * rhs{ i } - rhs2{ i };
end
fprintf( 1, '  done in %f s.\n', toc );

beas_v11_heat = be_assembler( stmesh, kernel_heat_sl_2( alpha ), ...
  basis_p1, basis_p1, order_nf, order_ff );
fprintf( 1, 'Assembling V11\n' );
tic;
V11 = beas_v11_heat.assemble( );
fprintf( 1, '  done in %f s.\n', toc );

fprintf( 1, 'Assembling M11\n' );
tic;
beid11 = be_identity( stmesh, basis_p1, basis_p1, order_ff );
M11 = beid11.assemble( );
M11 = inv( M11 ); 
fprintf( 1, '  done in %f s.\n', toc );

fprintf( 1, 'Solving the system\n' );
tic;
[ ~, ~, res, iter ] = gmres( @( x ) to_contiguous( ...
  apply_toeplitz( D, ...
  from_contiguous( x, stmesh.nt, size( D{ 1 }, 1 ) ), false ) ), ...
  to_contiguous( rhs ), 500, 1e-8, 500 );
fprintf( 1, '    %d %e\n', iter( 2 ), res );
[ dir, ~, res, iter ] = gmres( @( x ) to_contiguous( ...
  apply_toeplitz( D, ...
  from_contiguous( x, stmesh.nt, size( D{ 1 }, 1 ) ), false ) ), ...
  to_contiguous( rhs ), 500, 1e-8, 500, ...
  @( x ) to_contiguous( ...
  apply_M( M11, apply_toeplitz( V11, ...
  apply_M( M11, from_contiguous( x, stmesh.nt, size( V11{ 1 }, 1 ) ), ...
  false ), false ), false ) ) );
fprintf( 1, '    %d %e\n', iter( 2 ), res );
fprintf( 1, '  done in %f s.\n', toc );
dir = from_contiguous( dir, stmesh.nt, size( D{ 1 }, 1 ) );

L2_p1 = L2_tools( stmesh, basis_p1, 5, 4 );
err_bnd = L2_p1.relative_error( dir_fun, dir );
fprintf( 1, 'L2 relative error: %f.\n', err_bnd );
dir_proj = L2_p1.projection( dir_fun );
err_bnd_proj = L2_p1.relative_error( dir_fun, dir_proj );
fprintf( 1, 'Projection L2 relative error: %f.\n', err_bnd_proj );

err_bnd_x = zeros( stmesh.nt, 1 );
err_bnd_proj_x = zeros( stmesh.nt, 1 );
for i= 1 : stmesh.nt
  t = ( i - 0.5 ) / stmesh.nt;
  dir_fun_t = @( x, n ) dir_fun( x, t, n );
  err_bnd_x( i ) = L2_p1.relative_error_s( dir_fun_t, dir{ i } );
  err_bnd_proj_x( i ) = L2_p1.relative_error_s( dir_fun_t, dir_proj{ i } );
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
beev_v_heat = be_evaluator( stmesh, kernel_heat_sl_2( alpha ), basis_p0, ...
  neu, points, order_ff );
fprintf( 1, 'Evaluating V\n' );
tic;
repr = beev_v_heat.evaluate( );
fprintf( 1, '  done in %f s.\n', toc );

beev_w_heat = be_evaluator( stmesh, kernel_heat_dl_2( alpha ), basis_p1, ...
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

function y = apply_toeplitz( A, x, trans )

block_dim = size( x, 1 );
y = cell( block_dim, 1 );
for d = 1 : block_dim
  if ~trans
    y{ d } = zeros( size( A{ d }, 1 ), 1 );
  else
    y{ d } = zeros( size( A{ d }, 2 ), 1 );
  end
end
for d = 1 : block_dim
  for block = 1 : block_dim - d + 1
    if ~trans
      y{ block + d - 1 } = y{ block + d - 1 } + A{ d } * x{ block };
    else
      y{ block + d - 1 } = y{ block + d - 1 } + A{ d }' * x{ block };
    end
  end
end

end

function y = apply_M( M, x, trans )

block_dim = size( x, 1 );
y = cell( block_dim, 1 );
for d = 1 : block_dim
  if ~trans
    y{ d } = M * x{ d };
  else
    y{ d } = M' * x{ d };
  end
end

end
function y = to_contiguous( x )

block_dim = size( x, 1 );
dim = size( x{ 1 }, 1 );
y = zeros( dim * block_dim, 1 );
counter = 1;

for i = 1 : block_dim
  for j = 1 : dim
    y( counter ) = x{ i }( j );
    counter = counter + 1;
  end
end

end

function y = from_contiguous( x, block_dim, dim )

y = cell( block_dim, 1 );
counter = 1;

for i = 1 : block_dim
  y{ i } = zeros( dim, 1 );
  for j = 1 : dim
    y{ i }( j ) = x( counter );
    counter = counter + 1;
  end
end

end


