function test_regular_quadrature( stmesh, i_trial, i_test, alpha, order_ff_x )

%% Settings

if nargin < 3
  file='./input/cube_192.txt';
  stmesh = spacetime_mesh( file, 1, 8 );
  i_trial = 1;
  i_test = 8;
end
if nargin < 4
  alpha = 5.0;
end
if nargin < 5
  order_ff_x = 5;
end

%% Initialize

area_trial = stmesh.areas( i_trial );
area_test = stmesh.areas( i_test );

fprintf( 1, '  trial h:           %f\n', sqrt( area_trial ) );
fprintf( 1, '  test h:            %f\n', sqrt( area_test ) );
fprintf( 1, '  centroid distance: %f\n\n', ...
  norm( stmesh.centroid( i_trial ) - stmesh.centroid( i_test ) ) );

plot_mesh( stmesh, i_test, i_trial );

areas = area_trial * area_test;

%% reference - analytic in time
[ x_ref, y_ref, w ] = ...
  reference_quad_analytic( order_ff_x );
[ x, y ] = global_quad( stmesh, i_test, i_trial, x_ref, y_ref );
k = kernel_analytic_in_time( alpha, 0, stmesh.ht, x, y );
value_analytic_in_time = w' * k * areas;

orders = quadratures.line_orders;
orders_size = length( orders );
line_num_points = zeros( orders_size, 1 );
value_numerical_single_square = zeros( orders_size, 1 );
value_numerical_two_squares = zeros( orders_size, 1 );
for i = 1 : orders_size
  [ x_ref, y_ref, w, t_ref, tau_ref ] = ...
    reference_quad_temporal_square( order_ff_x, orders( i ) );
  [ x, y ] = global_quad( stmesh, i_test, i_trial, x_ref, y_ref );
  line_num_points( i, 1 ) = quadratures.line_length( orders( i ) )^2;
  
  %% numerical over single square (one triangle returns 0)
  k = kernel( alpha, 0, stmesh.ht, x, y, t_ref, tau_ref );
  value_numerical_single_square( i, 1 ) = w' * k * areas;
  
  %% numerical over two squares
  k = kernel_two_squares( alpha, 0, stmesh.ht, x, y, t_ref, tau_ref );
  value_numerical_two_squares( i, 1 ) = w' * k * areas;
end

orders = quadratures.tri_orders;
orders_size = length( orders );
tri_num_points = zeros( orders_size, 1 );
value_numerical_triangle = zeros( orders_size, 1 );

for i = 1 : orders_size
  [ x_ref, y_ref, w, t_ref, tau_ref ] = ...
    reference_quad_temporal_triangle( order_ff_x, orders( i ) );
  [ x, y ] = global_quad( stmesh, i_test, i_trial, x_ref, y_ref );
  tri_num_points( i, 1 ) =  quadratures.tri_length( orders( i ) );
  
  %% numerical over triangle (one triangle returns 0)
  k = kernel( alpha, 0, stmesh.ht, x, y, t_ref, tau_ref );
  value_numerical_triangle( i, 1 ) = w' * k * areas * 0.5;
end

%% printing result
fprintf( 1, '  analytic in time:                  %.12e\n', ...
  value_analytic_in_time( end ) );
fprintf( 1, '  finest numerical over triangle:    %.12e\n', ...
  value_numerical_triangle( end ) );
fprintf( 1, '  finest numerical over square:      %.12e\n', ...
  value_numerical_single_square( end ) );
fprintf( 1, '  finest numerical over two squares: %.12e\n', ...
  value_numerical_two_squares( end ) );

figure;
hold on;
plot( [ min( line_num_points ) max( line_num_points ) ], ...
  value_analytic_in_time * [ 1 1 ], '-' );
plot( tri_num_points, value_numerical_triangle, '-o', 'LineWidth', 1 );
plot( line_num_points, value_numerical_single_square, '-o', 'LineWidth', 1 );
plot( line_num_points, value_numerical_two_squares, '-o', 'LineWidth', 1 );
legend( { 'analytic in time', ...
  'numerical over triangle', ...
  'numerical over square', ...
  'numericl over two squares' }, 'FontSize', 14, 'Location', 'southeast' );
set( gca, 'XScale', 'log' );
xlabel( 'Number of quadrature points' );
ylabel( 'Value' );
hold off;

end

function plot_mesh( stmesh, i_test, i_trial )

data = zeros( stmesh.n_elems, 1 );
data( i_test, 1 ) = 1;
data( i_trial, 1 ) = 1;

figure;
axis equal;
colormap( 'jet' );
trisurf( stmesh.elems, stmesh.nodes( :, 1 ), stmesh.nodes( :, 2 ), ...
  stmesh.nodes( :, 3 ), data, 'EdgeColor', 'black' );
axis vis3d;

end

%%%%% Transferred to (0,1)^2
function value = kernel( alpha, d, ht, x, y, t, tau )

ttau = d + t - tau;
mask = ( ttau <= 0 );
value( mask, 1 ) = 0;
mask = ~mask;
norm = sqrt( ( x - y ).^2 * [ 1; 1; 1 ] );
rr = norm / sqrt( alpha * ht );
value( mask, 1 ) = ( 4 * pi * ttau( mask, 1 ) ).^( -3/2 ) ...
  .* exp( -rr( mask, 1 ).^2 ./ ( 4 * ttau( mask, 1 ) ) );
value( mask, 1 ) = value( mask, 1 ) * sqrt( ht / alpha^3 );

end

%%%%% Transferred to (0,2)x(0,1)
function value = kernel_two_squares( alpha, d, ht, x, y, t, tau )

ttau = d + t - 2 * tau;
mask = ( ttau <= 0 );
value( mask, 1 ) = 0;
mask = ~mask;
norm = sqrt( ( x - y ).^2 * [ 1; 1; 1 ] );
rr = norm / sqrt( alpha * ht );
value( mask, 1 ) = ( 4 * pi * ttau( mask, 1 ) ).^( -3/2 ) ...
  .* exp( -rr( mask, 1 ).^2 ./ ( 4 * ttau( mask, 1 ) ) );
value( mask, 1 ) = value( mask, 1 ) * sqrt( ht / alpha^3 ) * 2;

end

function value = kernel_analytic_in_time( alpha, d, ht, x, y )

norm = sqrt( ( x - y ).^2 * [ 1; 1; 1 ] );
if d > 0
  value = - G_anti_tau_anti_t_regular( alpha, norm, ( d + 1 ) * ht ) ...
    + 2 * G_anti_tau_anti_t_regular( alpha, norm, d * ht ) ...
    - G_anti_tau_anti_t_regular( alpha, norm, ( d - 1 ) * ht );
else
  value = - G_anti_tau_anti_t_regular( alpha, norm, ht ) ...
    + G_anti_tau_anti_t_regular( alpha, norm, 0 ) ...
    + ht * G_anti_tau_limit( alpha, norm );
end

end

function res = G_anti_tau_anti_t_regular( alpha, norm, delta )

sqrt_d = sqrt( delta );
sqrt_pi_a = sqrt( pi * alpha );
mask = ( norm == 0 );
res( mask, 1 ) = sqrt_d / ( 2 * pi * alpha * sqrt_pi_a );
mask = ~mask;
res( mask, 1 ) = ( delta ./ ( 4 * pi * alpha * norm( mask ) ) ...
  + norm( mask ) / ( 8 * pi * alpha^2 ) ) ...
  .* erf( norm( mask ) / ( 2 * sqrt_d * sqrt( alpha ) ) ) ...
  + sqrt_d / ( 4 * pi * alpha * sqrt_pi_a ) ...
  .* exp( - norm( mask ).^2 / ( 4 * delta * alpha ) );

end

function res = G_anti_tau_limit( alpha, norm )

res = 1 ./ ( 4 * pi * alpha * norm );

end

function [ x_ref, y_ref, w, t_ref, tau_ref ] = ...
  reference_quad_temporal_triangle( order_ff_x, order_ff_t_tri )

size_ff = quadratures.tri_length( order_ff_x )^2 ...
  * quadratures.tri_length( order_ff_t_tri );

x_ref = zeros( size_ff, 2 );
y_ref = zeros( size_ff, 2 );
w = zeros( size_ff, 1 );
t_ref = zeros( size_ff, 1 );
tau_ref = zeros( size_ff, 1 );

[ x_tri, w_tri, l_tri ] = quadratures.tri( order_ff_x );
[ t_t, w_t, l_t ] = quadratures.tri( order_ff_t_tri );
% map to triagle t>tau
t_t = t_t * [ 1 0; 1 1 ];

counter = 1;
for i_t_tau = 1 : l_t
  for i_x = 1 : l_tri
    for i_y = 1 : l_tri
      x_ref( counter, : ) = x_tri( i_x, : );
      y_ref( counter, : ) = x_tri( i_y, : );
      w( counter ) = w_tri( i_x ) * w_tri( i_y ) * w_t( i_t_tau );
      t_ref( counter ) = t_t( i_t_tau, 1 );
      tau_ref( counter ) = t_t( i_t_tau, 2 );
      counter = counter + 1;
    end
  end
end

end

function [ x_ref, y_ref, w, t_ref, tau_ref ] = ...
  reference_quad_temporal_square( order_ff_x, order_ff_t )

size_ff = quadratures.tri_length( order_ff_x )^2 ...
  * quadratures.line_length( order_ff_t )^2;

x_ref = zeros( size_ff, 2 );
y_ref = zeros( size_ff, 2 );
w = zeros( size_ff, 1 );
t_ref = zeros( size_ff, 1 );
tau_ref = zeros( size_ff, 1 );

[ x_tri, w_tri, l_tri ] = quadratures.tri( order_ff_x );
[ t_t, w_t, l_t ] = quadratures.line( order_ff_t );

counter = 1;
for i_t = 1 : l_t
  for i_tau = 1 : l_t
    for i_x = 1 : l_tri
      for i_y = 1 : l_tri
        x_ref( counter, : ) = x_tri( i_x, : );
        y_ref( counter, : ) = x_tri( i_y, : );
        w( counter ) = w_tri( i_x ) * w_tri( i_y ) * w_t( i_t ) * w_t( i_tau );
        t_ref( counter ) = t_t( i_t );
        tau_ref( counter ) = t_t( i_tau );
        counter = counter + 1;
      end
    end
  end
end

end

function [ x_ref, y_ref, w ] = reference_quad_analytic( order_ff_x )

size_ff = quadratures.tri_length( order_ff_x )^2;

x_ref = zeros( size_ff, 2 );
y_ref = zeros( size_ff, 2 );
w = zeros( size_ff, 1 );

[ x_tri, w_tri, l_tri ] = quadratures.tri( order_ff_x );

counter = 1;
for i_x = 1 : l_tri
  for i_y = 1 : l_tri
    x_ref( counter, : ) = x_tri( i_x, : );
    y_ref( counter, : ) = x_tri( i_y, : );
    w( counter ) = w_tri( i_x ) * w_tri( i_y );
    counter = counter + 1;
  end
end

end

function [ x, y ] = global_quad( mesh, i_test, i_trial, x_ref, y_ref )

nodes = mesh.nodes( mesh.elems( i_test, : ), : );
x1 = nodes( 1, : );
x2 = nodes( 2, : );
x3 = nodes( 3, : );
x = x1 + x_ref * [ x2 - x1; x3 - x1 ];

nodes = mesh.nodes( mesh.elems( i_trial, : ), : );
y1 = nodes( 1, : );
y2 = nodes( 2, : );
y3 = nodes( 3, : );
y = y1 + y_ref * [ y2 - y1; y3 - y1 ];

end
