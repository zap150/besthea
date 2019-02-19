classdef be_integrator
  
  properties (Access = private)
    mesh;
    kernel;
    test;
    trial;
    order_nf;
    size_nf;
    order_ff;
    size_ff;
      
    %%%% type = 1 ... disjoint
    %%%% type = 2 ... vertex
    %%%% type = 3 ... edge
    %%%% type = 4 ... identical
    
    %%%% Tausch
    % n_simplex = [ 1 2 4 6 ];
    %%%% Sauter, Schwab
    n_simplex = [ 1 2 5 6 ];
    
    x_ref = cell( 4, 1 );
    y_ref = cell( 4, 1 );
    w = cell( 4, 1 );
    
    map = [ 1 2 3 1 2 ];

  end
  
  methods
    function obj = ...
        be_integrator( mesh, kernel, test, trial, order_nf, order_ff )
      
      obj.mesh = mesh;
      obj.kernel = kernel;
      obj.test = test;
      obj.trial = trial;
      
      if( nargin < 4 )
        obj.order_nf = 4;
        obj.order_ff = 4;
      else
        obj.order_nf = order_nf;
        obj.order_ff = order_ff;
      end
      length_nf = quadratures.line_length( order_nf );
      obj.size_nf = length_nf * length_nf * length_nf * length_nf;
      length_ff = quadratures.tri_length( order_ff );
      obj.size_ff = length_ff * length_ff;
      
      for i = 1 : 4
        obj.x_ref{ i } = cell( obj.n_simplex( i ), 1 );
        obj.y_ref{ i } = cell( obj.n_simplex( i ), 1 );
        obj.w{ i } = cell( obj.n_simplex( i ), 1 );
      end
      
      obj = init_quadrature_data( obj );   
    end
    
    function obj = set_kernel( obj, kernel )
      obj.kernel = kernel;
    end
    
    function obj = set_test( obj, test )
      obj.test = test;
    end
    
    function obj = set_trial( obj, trial )
      obj.trial = trial;
    end
        
    function A = assemble( obj )
      A = zeros( obj.test.dim_global( ), obj.trial.dim_global( ) );
      
      n_elems = obj.mesh.get_n_elems( );
      dim_test = obj.test.dim_local( );
      dim_trial = obj.trial.dim_local( );
            
      for i_trial = 1 : n_elems
        for i_test = 1 : n_elems
          
          [ type, rot_test, rot_trial ] = obj.get_type( i_test, i_trial );
                    
          A_local = zeros( dim_test, dim_trial );
          for i_simplex = 1 : obj.n_simplex( type )
            [ x, y ] = global_quad( ... 
              obj, i_test, i_trial, type, rot_test, rot_trial, i_simplex );
            
            k = obj.kernel.eval( x, y, obj.mesh.get_normal( i_trial ) );
            
            test_fun = obj.test.eval( obj.x_ref{ type }{ i_simplex } );
            
            trial_fun = obj.trial.eval( obj.y_ref{ type }{ i_simplex } );
            
            for i_loc_test = 1 : dim_test
              for i_loc_trial = 1 : dim_trial
                A_local( i_loc_test, i_loc_trial ) = ...
                  A_local( i_loc_test, i_loc_trial ) ...
                  + ( obj.w{ type }{ i_simplex } ...
                  .* test_fun( :, i_loc_test ) ...
                  .* trial_fun( :, i_loc_trial ) )' * k;
              end
            end
          end
          
          map_test = obj.test.l2g( i_test, type, rot_test, false );
          map_trial = obj.trial.l2g( i_trial, type, rot_trial, true );
          A( map_test, map_trial ) = A( map_test, map_trial ) ...
            + A_local * obj.mesh.get_area( i_trial ) ...
            * obj.mesh.get_area( i_test );
        end
      end          
    end
    
  end
  
  methods (Access = private)
    function [ x, y ] = global_quad( obj, i_test, i_trial, type, ... 
        rot_test, rot_trial, i_simplex )
      
      nodes = obj.mesh.get_nodes( i_test );          
      z1 = nodes( obj.map( rot_test + 1 ), : );
      z2 = nodes( obj.map( rot_test + 2 ), : );
      z3 = nodes( obj.map( rot_test + 3 ), : );
      R = [ z2 - z1; z3 - z1 ];
      x = obj.x_ref{ type }{ i_simplex } * R; 
      x = x + z1;
      
      nodes = obj.mesh.get_nodes( i_trial );
      %%%% inverting trial element
      if type == 3
        z1 = nodes( obj.map( rot_trial + 2 ), : );
        z2 = nodes( obj.map( rot_trial + 1 ), : );
      else
        z1 = nodes( obj.map( rot_trial + 1 ), : );
        z2 = nodes( obj.map( rot_trial + 2 ), : );
      end
      z3 = nodes( obj.map( rot_trial + 3 ), : );
      R = [ z2 - z1; z3 - z1 ];
      y = obj.y_ref{ type }{ i_simplex } * R;
      y = y + z1;
      
    end
    
    function [ type, rot_test, rot_trial ] = get_type( obj, i_test, i_trial )
      
      rot_test = 0;
      rot_trial = 0;
      
      %%%% identical
      if i_test == i_trial
        type = 4;
        return;
      end
      
      elem_test = obj.mesh.get_element( i_test );
      elem_trial = obj.mesh.get_element( i_trial );
            
      %%%% common edge
      for i_trial = 1 : 3
        for i_test = 1 : 3
          if ( ...
            ( elem_trial( i_trial ) == elem_test( obj.map( i_test + 1 ) ) ) ...
            && ...
            ( elem_trial( obj.map( i_trial + 1 ) ) == elem_test( i_test ) ) ) 
            
            type = 3;
            rot_test = i_test - 1;
            rot_trial = i_trial - 1;
            return;
          end
        end
      end
      
      %%%% common vertex
      for i_trial = 1 : 3
        for i_test = 1 : 3
          if elem_test( i_test ) == elem_trial( i_trial )
            type = 2;
            rot_test = i_test - 1;
            rot_trial = i_trial - 1;
          return;
          end
        end
      end
      
      %%%% disjoint
      type = 1;
      
    end
    
    function obj = init_quadrature_data( obj )
         
      %%%% disjoint
      obj.x_ref{ 1 }{ 1 } = zeros( obj.size_ff, 2 );
      obj.y_ref{ 1 }{ 1 } = zeros( obj.size_ff, 2 );
      obj.w{ 1 }{ 1 } = zeros( obj.size_ff, 1 );
      
      [ x_tri, w_tri, l_tri ] = quadratures.tri( obj.order_ff );
      
      counter = 1;
      for i_x = 1 : l_tri
        for i_y = 1 : l_tri
          obj.x_ref{ 1 }{ 1 }( counter, : ) = x_tri( i_x, : );
          obj.y_ref{ 1 }{ 1 }( counter, : ) = x_tri( i_y, : );
          obj.w{ 1 }{ 1 }( counter ) = w_tri( i_x ) * w_tri( i_y );
          counter = counter + 1;
        end
      end
      
      %%%% singular
      for type = 2 : 4
        ns = obj.n_simplex( type );
        for i_simplex = 1 : ns
          obj.x_ref{ type }{ i_simplex } = zeros( obj.size_nf, 2 );
          obj.y_ref{ type }{ i_simplex } = zeros( obj.size_nf, 2 );
          obj.w{ type }{ i_simplex } = zeros( obj.size_nf, 1 );
        end
      end
            
      [ x_line, w_line, l_line ] = quadratures.line( obj.order_nf );
      
      counter = 1;
      for i_ksi = 1 : l_line
        for i_eta1 = 1 : l_line
          for i_eta2 = 1 : l_line
            for i_eta3 = 1 : l_line
              
              weight = 4 * w_line( i_ksi ) * w_line( i_eta1 ) ...
                * w_line( i_eta2 ) * w_line( i_eta3 );
              
              for type = 2 : 4
                ns = obj.n_simplex( type );
                for i_simplex = 1 : ns
                  [ x_single, y_single, jac ] = obj.cube_to_tri( ...
                    x_line( i_ksi ), x_line( i_eta1 ), ...
                    x_line( i_eta2 ), x_line( i_eta3 ), type, i_simplex );
                  obj.x_ref{ type }{ i_simplex }( counter, : ) = x_single;
                  obj.y_ref{ type }{ i_simplex }( counter, : ) = y_single;
                  obj.w{ type }{ i_simplex }( counter ) = weight * jac;
                end
              end
                           
              counter = counter + 1;
            end
          end
        end
      end    
    end
    
    function [ x, y, jac ] = ...
        cube_to_tri( obj, ksi, eta1, eta2, eta3, type, simplex )
      
      switch type
        case 1
          [ x, y, jac ] = ...
            obj.cube_to_tri_disjoint( ksi, eta1, eta2, eta3 );
        case 2
          [ x, y, jac ] = ...
            obj.cube_to_tri_vertex( ksi, eta1, eta2, eta3, simplex );
        case 3
          [ x, y, jac ] = ...
            obj.cube_to_tri_edge( ksi, eta1, eta2, eta3, simplex );
        case 4
          [ x, y, jac ] = ...
            obj.cube_to_tri_identical( ksi, eta1, eta2, eta3, simplex );
      end
    end
  
    % Tausch
%     function [ x, y, jac ] = ...
%         cube_to_tri_identical( ~, ksi, eta1, eta2, eta3, simplex )
%       
%       switch simplex
%         case 1
%           x( 1 ) = ( 1 - ksi ) * eta2 * ( 1 - eta3 );
%           x( 2 ) = ksi + ( 1 - ksi ) * eta2 * eta3;
%           y( 1 ) = x( 1 ) + ksi * ( 1 - eta1 );
%           y( 2 ) = x( 2 ) - ksi;
%         case 2
%           x( 1 ) = ksi * ( 1 - eta1 ) + ( 1 - ksi ) * eta2 * ( 1 - eta3 );
%           x( 2 ) = ksi * eta1 + ( 1 - ksi ) * eta2 * eta3;
%           y( 1 ) = x( 1 ) + ksi * ( 1 - eta1 );
%           y( 2 ) = x( 2 ) - ksi;
%         case 3
%           x( 1 ) = ksi + ( 1 - ksi ) * eta2 * ( 1 - eta3 );
%           x( 2 ) = ( 1 - ksi ) * eta2 * eta3;
%           y( 1 ) = x( 1 ) - ksi;
%           y( 2 ) = x( 2 ) + ksi * ( 1 - eta1 );
%         case 4
%           x( 1 ) = ( 1 - ksi ) * eta2 * ( 1 - eta3 );
%           x( 2 ) = ksi * eta1 + ( 1 - ksi ) * eta2 * eta3;
%           y( 1 ) = x( 1 ) + ksi;
%           y( 2 ) = x( 2 ) - ksi * eta1;
%         case 5
%           x( 1 ) = ( 1 - ksi ) * eta2 * ( 1 - eta3 );
%           x( 2 ) = ( 1 - ksi ) * eta2 * eta3;
%           y( 1 ) = x( 1 ) + ksi * ( 1 - eta1 );
%           y( 2 ) = x( 2 ) + ksi * eta1;
%         case 6
%           x( 1 ) = ksi * ( 1 - eta1 ) + ( 1 - ksi ) * eta2 * ( 1 - eta3 );
%           x( 2 ) = ( 1 - ksi ) * eta2 * eta3;
%           y( 1 ) = x( 1 ) - ksi * ( 1 - eta1 );
%           y( 2 ) = x( 2 ) + ksi;
%       end
%       
%       jac = ksi * ( 1 - ksi ) * ( 1 - ksi ) * eta2;       
%     end
    
    % Sauter, Schwab
    function [ x, y, jac ] = ...
        cube_to_tri_identical( ~, ksi, eta1, eta2, eta3, simplex )
      
      jac = ksi * ksi * ksi * eta1 * eta1 * eta2;
      
      switch simplex
        case 1
          x( 1 ) = ksi * eta1 * ( 1 - eta2 );
          x( 2 ) = ksi * ( 1 - eta1 * ( 1 - eta2 ) );
          y( 1 ) = ksi * eta1 * ( 1 - eta2 * eta3 );
          y( 2 ) = ksi * ( 1 - eta1 );
        case 2
          x( 1 ) = ksi * eta1 * ( 1 - eta2 * eta3 );
          x( 2 ) = ksi * ( 1 - eta1 );
          y( 1 ) = ksi * eta1 * ( 1 - eta2 );
          y( 2 ) = ksi * ( 1 - eta1 * ( 1 - eta2 ) );
        case 3
          x( 1 ) = ksi * ( 1 - eta1 * ( 1 - eta2 * ( 1 - eta3 ) ) );
          x( 2 ) = ksi * eta1 * ( 1 - eta2 * ( 1 - eta3 ) );
          y( 1 ) = ksi * ( 1 - eta1 );
          y( 2 ) = ksi * eta1 * ( 1 - eta2 );
        case 4
          x( 1 ) = ksi * ( 1 - eta1 );
          x( 2 ) = ksi * eta1 * ( 1 - eta2 );
          y( 1 ) = ksi * ( 1 - eta1 * ( 1 - eta2 * ( 1 - eta3 ) ) );
          y( 2 ) = ksi * eta1 * ( 1 - eta2 * ( 1 - eta3 ) );
        case 5
          x( 1 ) = ksi * ( 1 - eta1 );
          x( 2 ) = ksi * eta1 * ( 1 - eta2 * eta3 );
          y( 1 ) = ksi * ( 1 - eta1 * ( 1 - eta2 ) );
          y( 2 ) = ksi * eta1 * ( 1 - eta2 );
        case 6
          x( 1 ) = ksi * ( 1 - eta1 * ( 1 - eta2 ) );
          x( 2 ) = ksi * eta1 * ( 1 - eta2 );
          y( 1 ) = ksi * ( 1 - eta1 );
          y( 2 ) = ksi * eta1 * ( 1 - eta2 * eta3 );
      end           
    end
    
    % Tausch
%     function [ x, y, jac ] = ...
%         cube_to_tri_edge( ~, ksi, eta1, eta2, eta3, simplex )
%       
%       switch simplex
%         case 1
%           x( 1 ) = ksi * ( 1 - eta2 ) + ( 1 - ksi ) * eta3;
%           x( 2 ) = ksi * eta2;
%           y( 1 ) = ( 1 - ksi ) * eta3;
%           y( 2 ) = ksi * ( 1 - eta1 );
%           jac = ( 1 - ksi ) * ksi * ksi;
%         case 2
%           x( 1 ) = ( 1 - ksi ) * eta3;
%           x( 2 ) = ksi;
%           y( 1 ) = ksi * ( 1 - eta2 ) + ( 1 - ksi ) * eta3;
%           y( 2 ) = ksi * ( 1 - eta1 ) * eta2;
%           jac = ( 1 - ksi ) * ksi * ksi * eta2;
%         case 3
%           x( 1 ) = ( 1 - 2 * eta1 ) * ksi + ( 1 - ksi ) * eta3;
%           x( 2 ) = ksi * eta1;
%           y( 1 ) = ksi * ( 2 - 2 * eta1 - eta2 ) + ( 1 - ksi ) * eta3;
%           y( 2 ) = ksi * eta2;
%           jac = ( 1 - ksi ) * ksi * ksi;
%         case 4
%           x( 1 ) = ksi * ( 1 - eta2 ) + ( 1 - ksi ) * eta3;
%           x( 2 ) = ksi * ( 1 - eta1 ) * eta2;
%           y( 1 ) = ( 1 - ksi ) * eta3;
%           y( 2 ) = ksi;
%           jac = ( 1 - ksi ) * ksi * ksi * eta2;
%       end
%     end

    % Sauter, Schwab
        function [ x, y, jac ] = ...
        cube_to_tri_edge( ~, ksi, eta1, eta2, eta3, simplex )
      
        jac = ksi * ksi * ksi * eta1 * eta1;
      
      switch simplex
        case 1
          x( 1 ) = ksi * ( 1 - eta1 * eta3 );
          x( 2 ) = ksi * ( eta1 * eta3 );
          y( 1 ) = ksi * ( 1 - eta1 );
          y( 2 ) = ksi * eta1 * ( 1 - eta2 );
        case 2
          x( 1 ) = ksi * ( 1 - eta1 );
          x( 2 ) = ksi * eta1;
          y( 1 ) = ksi * ( 1 - eta1 * eta2 );
          y( 2 ) = ksi * eta1 * eta2 * ( 1 - eta3 );
          jac = jac * eta2;
        case 3
          x( 1 ) = ksi * ( 1 - eta1 );
          x( 2 ) = ksi * eta1 * ( 1 - eta2 );
          y( 1 ) = ksi * ( 1 - eta1 * eta2 * eta3 );
          y( 2 ) = ksi * eta1 * eta2 * eta3;
          jac = jac * eta2;
        case 4
          x( 1 ) = ksi * ( 1 - eta1 * eta2 );
          x( 2 ) = ksi * eta1 * eta2 * ( 1 - eta3 );
          y( 1 ) = ksi * ( 1 - eta1 );
          y( 2 ) = ksi * eta1;
          jac = jac * eta2;
        case 5
          x( 1 ) = ksi * ( 1 - eta1 );
          x( 2 ) = ksi * eta1 * ( 1 - eta2 * eta3 );
          y( 1 ) = ksi * ( 1 - eta1 * eta2 );
          y( 2 ) = ksi * eta1 * eta2;
          jac = jac * eta2;
      end
    end
    
    function [ x, y, jac ] = ...
        cube_to_tri_vertex( ~, ksi, eta1, eta2, eta3, simplex )
      
      switch simplex
        case 1
          x( 1 ) = ksi * ( 1 - eta1 );
          x( 2 ) = ksi * eta1;
          y( 1 ) = ksi * eta2 * ( 1 - eta3 );
          y( 2 ) = ksi * eta2 * eta3;
        case 2
          x( 1 ) = ksi * eta2 * ( 1 - eta3 );
          x( 2 ) = ksi * eta2 * eta3;
          y( 1 ) = ksi * ( 1 - eta1 );
          y( 2 ) = ksi * eta1;
      end
      
      jac = ksi * ksi * ksi * eta2;      
    end
    
    function [ x, y, jac ] = ...
        cube_to_tri_disjoint( ~, ksi, eta1, eta2, eta3 )
      
      x( 1 ) = ksi * ( 1 - eta1 );
      x( 2 ) = ksi * eta1;
      y( 1 ) = eta2 * ( 1 - eta3 );
      y( 2 ) = eta2 * eta3;
      
      jac = ksi * eta2;    
    end
  end
end

