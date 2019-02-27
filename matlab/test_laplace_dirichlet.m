function [ dir, neu, err ] = test_laplace_dirichlet( mesh )

order_nf = 4;
order_ff = 4;

beas_v_laplace = be_assembler( mesh, kernel_laplace_sl, ...
  p0( mesh ), p0( mesh ), order_nf, order_ff );
fprintf( 1, 'Assembling V\n' );
tic;
V = beas_v_laplace.assemble( );
fprintf( 1, '  done in %f s.\n', toc );

beas_k_laplace=be_assembler( mesh, kernel_laplace_dl, ...
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

dir_fun = @( x, ~ ) ( 1 + x( :, 1 ) ) .* exp( 2 * pi * x( :, 2 ) ) .* ...
  cos( 2 * pi * x( :, 3 ) );
beid_p1p1 = be_identity( mesh, p1( mesh ), p1( mesh ), 2 );

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

neufun = @( x, n ) exp( 2 * pi * x( :, 2 ) ) ...
  .* ( n( 1 ) * cos( 2 * pi * x( :, 3 ) ) ...
  + 2 * pi * ( 1 + x( :, 1 ) ) * n( 2 ) .* cos( 2 * pi * x( :, 3 ) ) ...
  - 2 * pi * ( 1 + x( :, 1 ) ) * n( 3 ) .* sin( 2 * pi * x( :, 3 ) ) );

[ x_ref, w, ~ ] = quadratures.tri( 5 );
l2_diff_err = 0;
l2_err = 0;
n_elems = mesh.get_n_elems;
for i_tau = 1 : n_elems
  nodes = mesh.get_nodes( i_tau );
  R = [ nodes( 2, : ) - nodes( 1, : ); nodes( 3, : ) - nodes( 1, : ) ];
  x = x_ref * R;
  x = x + nodes( 1, : );
  f = neufun( x, mesh.get_normal( i_tau ) );
  area = mesh.get_area( i_tau );
  l2_diff_err = l2_diff_err + ( w' * ( f - neu( i_tau ) ).^2 ) * area;
  l2_err = l2_err + ( w' * f.^2 ) * area;
end

err = sqrt( l2_diff_err / l2_err );
fprintf( 1, 'L2 relative error: %f.\n', err );

end
