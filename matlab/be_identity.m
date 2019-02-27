classdef be_identity
  
  properties (Access = private)
    mesh;
    test;
    trial;
    order;
  end
  
  methods
    function obj = be_identity( mesh, test, trial, order )
      
      obj.mesh = mesh;
      obj.test = test;
      obj.trial = trial;
      
      if( nargin < 4 )
        obj.order = 4;
      else
        obj.order = order;
      end
    end
    
    function obj = set_test( obj, test )
      obj.test = test;
    end
    
    function obj = set_trial( obj, trial )
      obj.trial = trial;
    end
    
    function M = assemble( obj )
      n_elems = obj.mesh.get_n_elems( );
      dim_test = obj.test.dim_local( );
      dim_trial = obj.trial.dim_local( );
      
      i = zeros( n_elems * obj.test.dim_local( ) * obj.trial.dim_local( ), 1 );
      j = zeros( n_elems * obj.test.dim_local( ) * obj.trial.dim_local( ), 1 );
      v = zeros( n_elems * obj.test.dim_local( ) * obj.trial.dim_local( ), 1 );
      
      [ x_ref, w, ~ ] = quadratures.tri( obj.order );
      
      counter = 1;
      for i_tau = 1 : n_elems    
        test_fun = obj.test.eval( x_ref );        
        trial_fun = obj.trial.eval( x_ref );
        map_test = obj.test.l2g( i_tau );
        map_trial = obj.trial.l2g( i_tau );
        area = obj.mesh.get_area( i_tau );
        
        for i_loc_test = 1 : dim_test
          for i_loc_trial = 1 : dim_trial
            i( counter ) = map_test( i_loc_test );
            j( counter ) = map_trial( i_loc_trial );
            v( counter ) = ( ( test_fun( :, i_loc_test ) ...
              .* trial_fun( :, i_loc_trial ) )' * w ) * area;
            counter = counter + 1;
          end
        end
      end
      
      M = sparse(  i, j, v, obj.test.dim_global( ), obj.trial.dim_global( ) );
    end
    
    function result = L2_projection( obj, fun )
      M = obj.assemble(  );
      
      n_elems = obj.mesh.get_n_elems( );
      dim_test = obj.test.dim_local( );
      rhs = zeros( obj.test.dim_global( ), 1 );
      [ x_ref, w, ~ ] = quadratures.tri( obj.order );
      
      for i_tau = 1 : n_elems
        x = obj.global_quad( x_ref, i_tau );
        f = fun( x, obj.mesh.get_normal( i_tau ) );
        test_fun = obj.test.eval( x_ref );
        map_test = obj.test.l2g( i_tau );
        area = obj.mesh.get_area( i_tau );
        
        for i_loc_test = 1 : dim_test
          rhs( map_test( i_loc_test ) ) = rhs( map_test( i_loc_test ) ) ...
            + ( ( test_fun( :, i_loc_test ) .* f )' * w ) * area;
        end
      end
      
      result = M \ rhs;
    end
  end
  
  methods (Access = private)
    function x = global_quad( obj, x_ref, i_trial )
      nodes = obj.mesh.get_nodes( i_trial );
      R = [ nodes( 2, : ) - nodes( 1, : ); nodes( 3, : ) - nodes( 1, : ) ];
      x = x_ref * R;
      x = x + nodes( 1, : );
    end
  end
  
end

