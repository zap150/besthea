classdef L2_tools
  
  properties (Access = public)
    mesh;
    basis;
    order_x;
    order_t;
  end
  
  methods
    function obj = L2_tools( mesh, basis, order_x, order_t )
      
      obj.mesh = mesh;
      obj.basis = basis;
      
      if( nargin < 3 )
        obj.order_x = 4;
      else
        obj.order_x = order_x;
      end
      
      if( nargin < 4 )
        obj.order_t = 4;
      else
        obj.order_t = order_t;
      end
    end
    
    function result = projection( obj, fun )
      if( isa( obj.mesh, 'spacetime_mesh' ) )
        result = projection_st( obj, fun );
      else
        result = projection_s( obj, fun );
      end
    end
    
    function result = relative_error( obj, fun, fun_disc )
      if( isa( obj.mesh, 'spacetime_mesh' ) )
        result = relative_error_st( obj, fun, fun_disc );
      else
        result = relative_error_s( obj, fun, fun_disc );
      end
    end
    
    function result = relative_error_s( obj, fun, fun_disc )
      [ x_ref, wx, ~ ] = quadratures.tri( obj.order_x );
      l2_diff_err = 0;
      l2_err = 0;
      n_elems = obj.mesh.n_elems;
      basis_dim = obj.basis.dim_local( );
      for i_tau = 1 : n_elems
        nodes = obj.mesh.nodes( obj.mesh.elems( i_tau, : ), : );
        x = nodes( 1, : ) + x_ref ...
          * [ nodes( 2, : ) - nodes( 1, : ); nodes( 3, : ) - nodes( 1, : ) ];
        basis_val = obj.basis.eval( x_ref );
        basis_map = obj.basis.l2g( i_tau );
        area = obj.mesh.areas( i_tau );
        val = 0;
        for i_local_dim = 1 : basis_dim
          val = val + fun_disc( basis_map( i_local_dim ) ) ...
            * basis_val( :, i_local_dim );
        end
        f = fun( x, obj.mesh.normals( i_tau, : ) );
        l2_diff_err = l2_diff_err + ( wx' * ( f - val ).^2 ) * area;
        l2_err = l2_err + ( wx' * f.^2 ) * area;
      end
      result = sqrt( l2_diff_err / l2_err );
    end
    
    function result = relative_error_st( obj, fun, fun_disc )
      [ x_ref, wx, ~ ] = quadratures.tri( obj.order_x );
      [ t_ref, wt, lt ] = quadratures.line( obj.order_t );
      l2_diff_err = 0;
      l2_err = 0;
      n_elems = obj.mesh.n_elems;
      basis_dim = obj.basis.dim_local( );
      nt = obj.mesh.nt;
      ht = obj.mesh.ht;
      for d = 0 : nt - 1
        t = ht * ( t_ref + d );
        for i_tau = 1 : n_elems
          nodes = obj.mesh.nodes( obj.mesh.elems( i_tau, : ), : );
          x = nodes( 1, : ) + x_ref ...
            * [ nodes( 2, : ) - nodes( 1, : ); nodes( 3, : ) - nodes( 1, : ) ];
          basis_val = obj.basis.eval( x_ref );
          basis_map = obj.basis.l2g( i_tau );
          area = obj.mesh.areas( i_tau );
          val = 0;
          for i_local_dim = 1 : basis_dim
            val = val + fun_disc{ d + 1 }( basis_map( i_local_dim ) ) ...
              * basis_val( :, i_local_dim );
          end
          for i_t = 1 : lt
            f = fun( x, t( i_t ), obj.mesh.normals( i_tau, : ) );
            l2_diff_err = l2_diff_err + ( wx' * ( f - val ).^2 ) ...
              * area * ht * wt( i_t );
            l2_err = l2_err + ( wx' * f.^2 ) * area * ht * wt( i_t );
          end
        end
      end
      result = sqrt( l2_diff_err / l2_err );
    end
    
%     function result = norm_continuous( obj, fun )
%       if( isa( obj.mesh, 'spacetime_mesh' ) )
%         result = norm_continuous_st( obj, fun );
%       else
%         result = norm_continuous_s( obj, fun );
%       end
%     end
  end
  
  methods (Access = private)
    function result = projection_s( obj, fun )
      beid = be_identity( obj.mesh, obj.basis, obj.basis, obj.order_x );
      M = beid.assemble(  );
      
      n_elems = obj.mesh.n_elems;
      basis_dim = obj.basis.dim_local( );
      rhs = zeros( obj.basis.dim_global( ), 1 );
      [ x_ref, w, ~ ] = quadratures.tri( obj.order_x );
      
      for i_tau = 1 : n_elems
        nodes = obj.mesh.nodes( obj.mesh.elems( i_tau, : ), : );
        x = nodes( 1, : ) + x_ref ...
          * [ nodes( 2, : ) - nodes( 1, : ); nodes( 3, : ) - nodes( 1, : ) ];
        f = fun( x, obj.mesh.normals( i_tau, : ) );
        basis_fun = obj.basis.eval( x_ref );
        basis_map = obj.basis.l2g( i_tau );
        
        for i_loc_test = 1 : basis_dim
          rhs( basis_map( i_loc_test ) ) = rhs( basis_map( i_loc_test ) ) ...
            + ( ( basis_fun( :, i_loc_test ) .* f )' * w ) ...
            * obj.mesh.areas( i_tau );
        end
      end
      result = M \ rhs;
    end
    
    function result = projection_st( obj, fun )
      nt = obj.mesh.nt;
      result = cell( nt, 1 );
      
      beid = ...
        be_identity( obj.mesh, obj.basis, obj.basis, obj.order_x, obj.order_t );
      M = beid.assemble(  );
      
      n_elems = obj.mesh.n_elems;
      basis_dim = obj.basis.dim_local( );
      rhs = zeros( obj.basis.dim_global( ), 1 );
      [ x_ref, wx, ~ ] = quadratures.tri( obj.order_x );
      [ t_ref, wt, lt ] = quadratures.line( obj.order_t );
      
      for d = 0 : nt - 1
        t = obj.mesh.ht * ( t_ref + d );
        for i_tau = 1 : n_elems
          nodes = obj.mesh.nodes( obj.mesh.elems( i_tau, : ), : );
          x = nodes( 1, : ) + x_ref ...
            * [ nodes( 2, : ) - nodes( 1, : ); nodes( 3, : ) - nodes( 1, : ) ];
          basis_fun = obj.basis.eval( x_ref );
          basis_map = obj.basis.l2g( i_tau );
          area = obj.mesh.areas( i_tau );
          for i_t = 1 : lt
            f = fun( x, t( i_t ), obj.mesh.normals( i_tau, : ) );
            for i_loc_test = 1 : basis_dim
              rhs( basis_map( i_loc_test ) ) = ...
                rhs( basis_map( i_loc_test ) ) ...
                + ( ( basis_fun( :, i_loc_test ) .* f )' * wx ) ...
                * area * wt( i_t );
            end
          end
        end
        rhs = rhs * obj.mesh.ht;
        result{ d + 1 } = M \ rhs;
        rhs( :, : ) = 0;
      end
    end
        
%     function result = norm_continuous_s( obj, fun )
%       [ x_ref, wx, ~ ] = quadratures.tri( obj.order_x );
%       l2_norm = 0;
%       n_elems = obj.mesh.n_elems;
%       for i_tau = 1 : n_elems
%         nodes = obj.mesh.nodes( obj.mesh.elems( i_tau, : ), : );
%         x = nodes( 1, : ) + x_ref ...
%           * [ nodes( 2, : ) - nodes( 1, : ); nodes( 3, : ) - nodes( 1, : ) ];
%         area = obj.mesh.areas( i_tau );
%         f = fun( x, obj.mesh.normals( i_tau, : ) );
%         l2_norm = l2_norm + ( wx' * f.^2 ) * area;
%       end
%       result = sqrt( l2_norm );
%     end
%     
%     function result = norm_continuous_st( obj, fun )
%       [ x_ref, wx, ~ ] = quadratures.tri( obj.order_x );
%       [ t_ref, wt, lt ] = quadratures.line( obj.order_t );
%       l2_norm = 0;
%       n_elems = obj.mesh.n_elems;
%       nt = obj.mesh.nt;
%       ht = obj.mesh.ht;
%       for d = 0 : nt - 1
%         t = ht * ( t_ref + d );
%         for i_tau = 1 : n_elems
%           nodes = obj.mesh.nodes( obj.mesh.elems( i_tau, : ), : );
%           x = nodes( 1, : ) + x_ref ...
%             * [ nodes( 2, : ) - nodes( 1, : ); nodes( 3, : ) - nodes( 1, : ) ];
%           area = obj.mesh.areas( i_tau );
%           for i_t = 1 : lt
%             f = fun( x, t( i_t ), obj.mesh.normals( i_tau, : ) );
%             l2_norm = l2_norm + ( wx' * f.^2 ) * area * ht * wt( i_t );
%           end
%         end
%       end
%       result = sqrt( l2_norm );
%     end
    
  end
  
end

