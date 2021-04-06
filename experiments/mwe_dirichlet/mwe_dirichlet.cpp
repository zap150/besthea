#include <cstring>
#include <besthea/besthea.h>

using namespace besthea::mesh;
using namespace besthea::bem;
using namespace besthea::linear_algebra;

sc bc_dir_func( sc x1, sc x2, sc x3, const coordinates< 3 > &, sc t ) {
    constexpr std::array< sc, 3 > _y{ 0.0, 0.0, 1.5 };
    sc alpha = 0.5;

    sc norm2 = ( x1 - _y[ 0 ] ) * ( x1 - _y[ 0 ] )
             + ( x2 - _y[ 1 ] ) * ( x2 - _y[ 1 ] )
             + ( x3 - _y[ 2 ] ) * ( x3 - _y[ 2 ] );

    sc value = std::pow( 4.0 * M_PI * alpha * t, -1.5 )
             * std::exp( -norm2 / ( 4.0 * alpha * t ) );

    return value;
}

int main()
{
    sc alpha = 0.5;
    
    // load and create mesh
    std::string mesh_file = "../examples/mesh_files/cube_192.txt";
    lo n_timesteps = 8;
    sc end_time = 10.0;
    triangular_surface_mesh space_mesh;
    space_mesh.load(mesh_file);
    uniform_spacetime_tensor_mesh spacetime_mesh(space_mesh, end_time, n_timesteps);
    spacetime_mesh.refine(1);

    // create BE spaces
    uniform_spacetime_be_space<basis_tri_p0> space_p0(spacetime_mesh);
    uniform_spacetime_be_space<basis_tri_p1> space_p1(spacetime_mesh);

    // project the boundary condition on the BE space
    block_vector bc_dir;
    space_p1.L2_projection( bc_dir_func, bc_dir );

    // create and assemble single layer matrix V
    block_lower_triangular_toeplitz_matrix V;
    spacetime_heat_sl_kernel_antiderivative kernel_v( alpha );
    uniform_spacetime_be_assembler assembler_v(kernel_v, space_p0, space_p0);
    assembler_v.assemble(V);

    // create and assemble double layer matrix K
    block_lower_triangular_toeplitz_matrix K;
    spacetime_heat_dl_kernel_antiderivative kernel_k( alpha );
    uniform_spacetime_be_assembler assembler_k(kernel_k, space_p0, space_p1);
    assembler_k.assemble(K);
    
    // create and assemble mass matrix M
    uniform_spacetime_be_identity M(space_p0, space_p1);
    M.assemble();

    // create and assemble right hand side vector
    block_vector rhs(K.get_block_dim(), K.get_n_rows());
    M.apply(bc_dir, rhs, false, 0.5, 0.0);
    K.apply(bc_dir, rhs, false, 1.0, 1.0);

    // solve the system
    block_vector sol;
    sc rel_error = 1e-6;
    lo n_iters = 1000;
    V.mkl_fgmres_solve(rhs, sol, rel_error, n_iters);

    // evaluation TODO

    // do something with the solution

    return 0;
}


/*
void nothing()
{
    uniform_spacetime_be_space<basis_tri_p0> space_p0(spacetime_mesh);
    uniform_spacetime_be_space<basis_tri_p1> space_p1(spacetime_mesh);

    spacetime_heat_sl_kernel_antiderivative kernel_v( alpha );
    spacetime_heat_dl_kernel_antiderivative kernel_k( alpha );
    spacetime_heat_hs_kernel_antiderivative kernel_d( alpha );
    
    uniform_spacetime_be_assembler assembler_v(kernel_v, space_p0, space_p0);
    uniform_spacetime_be_assembler assembler_k(kernel_k, space_p0, space_p1);
    uniform_spacetime_be_assembler assembler_d(kernel_d, space_p1, space_p1);
}
*/
