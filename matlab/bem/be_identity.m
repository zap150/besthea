classdef be_identity < handle
  
  properties (Access = public)
    mesh;
    test;
    trial;
    order_x;
    order_t;
  end
  
  methods
    function obj = be_identity( mesh, test, trial, order_x, order_t )
      
      obj.mesh = mesh;
      obj.test = test;
      obj.trial = trial;
      
      if( nargin < 4 )
        obj.order_x = 4;
      else
        obj.order_x = order_x;
      end
      
      if( nargin < 5 )
        obj.order_t = 4;
      else
        obj.order_t = order_t;
      end
    end
    
    function M = assemble( obj )
      n_elems = obj.mesh.n_elems;
      dim_test = obj.test.dim_local( );
      dim_trial = obj.trial.dim_local( );
      
      i = zeros( n_elems * obj.test.dim_local( ) * obj.trial.dim_local( ), 1 );
      j = zeros( n_elems * obj.test.dim_local( ) * obj.trial.dim_local( ), 1 );
      v = zeros( n_elems * obj.test.dim_local( ) * obj.trial.dim_local( ), 1 );
      
      [ x_ref, w, ~ ] = quadratures.tri( obj.order_x );
      
      counter = 1;
      for i_tau = 1 : n_elems    
        test_fun = obj.test.eval( x_ref );        
        trial_fun = obj.trial.eval( x_ref );
        map_test = obj.test.l2g( i_tau );
        map_trial = obj.trial.l2g( i_tau );
        
        for i_loc_test = 1 : dim_test
          for i_loc_trial = 1 : dim_trial
            i( counter ) = map_test( i_loc_test );
            j( counter ) = map_trial( i_loc_trial );
            v( counter ) = ( ( test_fun( :, i_loc_test ) ...
              .* trial_fun( :, i_loc_trial ) )' * w ) * obj.mesh.areas( i_tau );
            counter = counter + 1;
          end
        end
      end
      
      if( isa( obj.mesh, 'spacetime_mesh' ) )
        v = v * obj.mesh.ht;
      end
      
      M = sparse(  i, j, v, obj.test.dim_global( ), obj.trial.dim_global( ) );
    end
    
    function result = L2_projection( obj, fun )
      if( isa( obj.mesh, 'spacetime_mesh' ) )
        result = L2_projection_st( obj, fun );
      else
        result = L2_projection_s( obj, fun );
      end
    end
  end
  
  methods (Access = private)
    function result = L2_projection_s( obj, fun )
      M = obj.assemble(  );
      
      n_elems = obj.mesh.n_elems;
      dim_test = obj.test.dim_local( );
      rhs = zeros( obj.test.dim_global( ), 1 );
      [ x_ref, w, ~ ] = quadratures.tri( obj.order_x );
      
      for i_tau = 1 : n_elems
        x = obj.global_quad( x_ref, i_tau );
        f = fun( x, obj.mesh.normals( i_tau, : ) );
        test_fun = obj.test.eval( x_ref );
        map_test = obj.test.l2g( i_tau );
        
        for i_loc_test = 1 : dim_test
          rhs( map_test( i_loc_test ) ) = rhs( map_test( i_loc_test ) ) ...
            + ( ( test_fun( :, i_loc_test ) .* f )' * w ) ...
            * obj.mesh.areas( i_tau );
        end
      end
      result = M \ rhs;
    end
    
    function result = L2_projection_st( obj, fun )
      nt = obj.mesh.nt;
      result = cell( nt, 1 );
      
      M = obj.assemble(  );
      
      n_elems = obj.mesh.n_elems;
      dim_test = obj.test.dim_local( );
      rhs = zeros( obj.test.dim_global( ), 1 );
      [ x_ref, wx, ~ ] = quadratures.tri( obj.order_x );
      [ t_ref, wt, lt ] = quadratures.line( obj.order_t );
      
      for d = 0 : nt - 1
        t = global_quad_t( obj, t_ref, d + 1 );
        for i_tau = 1 : n_elems
          x = global_quad( obj, x_ref, i_tau );
          test_fun = obj.test.eval( x_ref );
          map_test = obj.test.l2g( i_tau );
          area = obj.mesh.areas( i_tau );
          for i_t = 1 : lt            
            f = fun( x, t( i_t ), obj.mesh.normals( i_tau, : ) );                        
            for i_loc_test = 1 : dim_test
              rhs( map_test( i_loc_test ) ) = rhs( map_test( i_loc_test ) ) ...
                + ( ( test_fun( :, i_loc_test ) .* f )' * wx ) ...
                * area * wt( i_t );
            end
          end
        end
        rhs = rhs * obj.mesh.ht;
        result{ d + 1 } = M \ rhs;
        rhs( :, : ) = 0;
      end
    end
    
    function x = global_quad( obj, x_ref, i )
      nodes = obj.mesh.nodes( obj.mesh.elems( i, : ), : );
      x = nodes( 1, : ) + x_ref ...
        * [ nodes( 2, : ) - nodes( 1, : ); nodes( 3, : ) - nodes( 1, : ) ];
    end
    
    function t = global_quad_t( obj, t_ref, i )
      t = obj.mesh.get_time_node( i ) + obj.mesh.ht * t_ref;
    end
  end
  
end

