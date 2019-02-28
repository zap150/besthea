function test( file, test )

if nargin == 1
  test = false;
end

mesh = tri_mesh_3d( file );

order_nf = 4;
order_ff = 4;

p0_basis = p0( mesh );
p1_basis = p1( mesh );

if test
  load( 'data_ref.mat', 'V_lap', 'K_lap', 'V_helm', 'K_helm' );
end

bei = be_integrator( mesh, kernel_laplace_sl, p0_basis, p0_basis, ...
  order_nf, order_ff );
tic;
V_lap_test = bei.assemble( );
toc;
if test
 no = norm( V_lap_test - V_lap ) / norm( V_lap );
 fprintf( 1, 'Relative error is %e.\n\n', no );
end

bei = bei.set_kernel( kernel_laplace_dl );
bei = bei.set_trial( p1_basis );
tic;
K_lap_test = bei.assemble( );
toc;
if test
 no = norm( K_lap_test - K_lap ) / norm( K_lap );
 fprintf( 1, 'Relative error is %e.\n\n', no );
end

kappa = 2;
bei = bei.set_kernel( kernel_helmholtz_sl( kappa ) );
bei = bei.set_trial( p0_basis );
tic;
V_helm_test = bei.assemble( );
toc;
if test
 no = norm( V_helm_test - V_helm ) / norm( V_helm );
 fprintf( 1, 'Relative error is %e.\n\n', no );
end

bei = bei.set_kernel( kernel_helmholtz_dl( kappa ) );
bei = bei.set_trial( p1_basis );
tic;
K_helm_test = bei.assemble( );
toc;
if test
 no = norm( K_helm_test - K_helm ) / norm( K_helm );
 fprintf( 1, 'Relative error is %e.\n\n', no );
end

end

