function [ V, K ] = test( file )

mesh = tri_mesh_3d( file );

order_nf = 4;
order_ff = 4;

p0_basis = p0( mesh );
p1_basis = p1( mesh );

bei = be_integrator( mesh, kernel_laplace_sl, p0_basis, p0_basis, ...
  order_nf, order_ff );
tic;
V = bei.assemble( );
toc;

bei = bei.set_kernel( kernel_laplace_dl );
bei = bei.set_trial( p1_basis );
tic;
K = bei.assemble( );
toc;

end

