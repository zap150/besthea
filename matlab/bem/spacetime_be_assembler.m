classdef spacetime_be_assembler < handle
  
  properties (Access = public)
    mesh;
    kernel;
    test;
    trial;
    
    %%%% spatial and temporal
    order_nf;
    size_nf;
    size_ff;
    w = cell( 3, 1 );
    
    %%%% spatial
    order_ff_x;
    x_ref = cell( 3, 1 );
    y_ref = cell( 3, 1 );
    
    %%%% temporal
    order_ff_t;
    t_ref = cell( 3, 1 );
    tau_ref = cell( 3, 1 );
    w_t_tau = cell( 3, 1 );
  end
  
  properties (Constant)
    %%%% type_x = 1 ... disjoint
    %%%% type_x = 2 ... vertex
    %%%% type_x = 3 ... edge
    %%%% type_x = 4 ... identical
    
    %%%% type_t = 1 ... disjoint
    %%%% type_t = 2 ... vertex
    %%%% type_t = 3 ... identical
    
    %%%% Sauter, Schwab
    n_simplex = [ 1 4 10 12 ];
    
    map = [ 1 2 3 1 2 ];
  end
  
  methods
    function obj = ...
        spacetime_be_assembler( mesh, kernel, test, trial, order_nf, ...
        order_ff_x, order_ff_t )
      
      obj.mesh = mesh;
      obj.kernel = kernel;
      obj.test = test;
      obj.trial = trial;
      
      if( nargin < 7 )
        obj.order_nf = 4;
        obj.order_ff_x = 4;
        obj.order_ff_t = 4;
      else
        obj.order_nf = order_nf;
        obj.order_ff_x = order_ff_x;
        obj.order_ff_t = order_ff_t;
      end
      
      obj.size_nf = quadratures.line_length( order_nf )^6;
      obj.size_ff = quadratures.tri_length( order_ff_x )^2 ...
        * quadratures.line_length( order_ff_t )^2;
      
      %%%% regular in time, no Sauter-Schwab in space
      obj.x_ref{ 1 } = cell( 1, 1 );
      obj.y_ref{ 1 } = cell( 1, 1 );
      obj.w{ 1 } = cell( 1, 1 );
      obj.x_ref{ 1 }{ 1 } = ...
        cell( obj.n_simplex( 1 ), 1 );
      obj.y_ref{ 1 }{ 1 } = ...
        cell( obj.n_simplex( 1 ), 1 );
      obj.w{ 1 }{ 1 } = ...
        cell( obj.n_simplex( 1 ), 1 );
      obj.t_ref{ 1 }{ 1 } = ...
        cell( obj.n_simplex( 1 ), 1 );
      obj.tau_ref{ 1 }{ 1 } = ...
        cell( obj.n_simplex( 1 ), 1 );

      %%%% singular in time
      for i_type_t = 2 : 3
        obj.x_ref{ i_type_t } = cell( 4, 1 );
        obj.y_ref{ i_type_t } = cell( 4, 1 );
        obj.w{ i_type_t } = cell( 4, 1 );
        for i_type_x = 2 : 4
          obj.x_ref{ i_type_t }{ i_type_x } = ...
            cell( obj.n_simplex( i_type_x ), 1 );
          obj.y_ref{ i_type_t }{ i_type_x } = ...
            cell( obj.n_simplex( i_type_x ), 1 );
          obj.w{ i_type_t }{ i_type_x } = ...
            cell( obj.n_simplex( i_type_x ), 1 );
          obj.t_ref{ i_type_t }{ i_type_x } = ...
            cell( obj.n_simplex( i_type_x ), 1 );
          obj.tau_ref{ i_type_t }{ i_type_x } = ...
            cell( obj.n_simplex( i_type_x ), 1 );
        end
      end
           
      obj = init_quadrature_data( obj );
    end
    
    function A = assemble( obj )
      
      nt = obj.mesh.nt;
      A = cell( nt, 1 );
      for d = 0 : nt - 1
        A{ d + 1 } = zeros( obj.test.dim_global( ), obj.trial.dim_global( ) );
      end
      
      n_elems = obj.mesh.n_elems;
      dim_test = obj.test.dim_local( );
      dim_trial = obj.trial.dim_local( );
      obj.kernel.ht = obj.mesh.ht;
      
      msg = sprintf( 'assembling %s, singular part', class( obj.kernel ) );
      f = waitbar( 0, msg );
      f.Children.Title.Interpreter = 'none';
      
%      my_kernel = obj.kernel;
      A_local = zeros( dim_test, dim_trial );
%       for d = 0 : 1
%         my_kernel.d = d;
%         msgd = [ msg sprintf( ', d = %d/%d', d + 1, 2 ) ];
%         waitbar( d / 2, f, msgd );
%         
%         for i_trial = 1 : n_elems
%           waitbar( ( d + ( i_trial - 1 ) / n_elems ) / 2, f );
%           for i_test = 1 : n_elems
%             
%             [ type_x, rot_test, rot_trial ] = get_type( obj, i_test, i_trial );
%           if d == 0
%             type_t = 3;
%           elseif d == 1
%             type_t = 2;
%           end
%           if type_x == 1
%             type_t = 1; 
%           end
%             A_local( :, : ) = 0;
%             for i_simplex = 1 : obj.n_simplex( type_x )
%               %[ x, y ] = global_quad( obj, ...
%               %  i_test, i_trial, type_x, rot_test, rot_trial, i_simplex, type_t );
%               
%               k = zeros( size( obj.x_ref{ type_t }{ type_x }{ i_simplex }, 1 ), 1 );
%               %k = my_kernel.eval( x, y, obj.mesh.normals( i_test, : ), ...
%               %  obj.mesh.normals( i_trial, : ) );
%               
%               test_fun = obj.test.eval( obj.x_ref{ type_t }{ type_x }{ i_simplex }, ...
%                 obj.mesh.normals( i_test, : ), i_test, type_x, rot_test, ...
%                 false );
%               
%               trial_fun = obj.trial.eval( obj.y_ref{ type_t }{ type_x }{ i_simplex }, ...
%                 obj.mesh.normals( i_trial, : ), i_trial, type_x, rot_trial, ...
%                 true );
%               
%               if( isa( obj.test, 'curl_p1' ) )
%                 for i_loc_test = 1 : dim_test
%                   for i_loc_trial = 1 : dim_trial
%                     A_local( i_loc_test, i_loc_trial ) = ...
%                       A_local( i_loc_test, i_loc_trial ) ...
%                       + ( test_fun( ( i_loc_test - 1 ) * 3 + 1 : ...
%                       i_loc_test * 3 ) ...
%                       * trial_fun( ( i_loc_trial - 1 ) * 3 + 1 : ...
%                       i_loc_trial * 3 )' ) ...
%                       * ( obj.w_x_y{ type_t }{ type_x }{ i_simplex } )' * k;
%                   end
%                 end
%               else
%                 for i_loc_test = 1 : dim_test
%                   for i_loc_trial = 1 : dim_trial
%                     A_local( i_loc_test, i_loc_trial ) = ...
%                       A_local( i_loc_test, i_loc_trial ) ...
%                       + ( obj.w_x_y{ type_t }{ type_x }{ i_simplex } ...
%                       .* test_fun( :, i_loc_test ) ...
%                       .* trial_fun( :, i_loc_trial ) )' * k;
%                   end
%                 end
%               end
%             end
%             
%             map_test = obj.test.l2g( i_test, type_x, rot_test, false );
%             map_trial = obj.trial.l2g( i_trial, type_x, rot_trial, true );
%             A{ d + 1 }( map_test, map_trial ) = ...
%               A{ d + 1 }( map_test, map_trial ) ...
%               + A_local * obj.mesh.areas( i_trial ) ...
%               * obj.mesh.areas( i_test );
%           end
%         end
%       end
      waitbar( 1, f );
      close( f );
      
      msg = sprintf( 'assembling %s, regular part', class( obj.kernel ) );
      f = waitbar( 0, msg );
      f.Children.Title.Interpreter = 'none';
            
      my_kernel = obj.kernel;
      for d = 2 : nt - 1
        %       parfor d = 0 : nt - 1
        %         my_kernel = copy( obj.kernel );
        %         A_local = zeros( dim_test, dim_trial );
        
        my_kernel.d = d;
        msgd = [ msg sprintf( ', d = %d/%d', d - 1, nt - 2 ) ];
        waitbar( ( d - 2 ) / ( nt - 2 ), f, msgd );
        
        for i_trial = 1 : n_elems
          waitbar( ( d - 2 + ( i_trial - 1 ) / n_elems ) / ( nt - 2 ), f );
          for i_test = 1 : n_elems
            
            type_x = 1;
            type_t = 1;
            rot_test = 0;
            rot_trial = 0;
            i_simplex = 1;
            [ x, y ] = global_quad( obj, ...
              i_test, i_trial, type_x, rot_test, rot_trial, i_simplex, type_t );
            map_test = obj.test.l2g( i_test, type_x, rot_test, false );
            map_trial = obj.trial.l2g( i_trial, type_x, rot_trial, true );
            
            test_fun = obj.test.eval( ...
              obj.x_ref{ type_t }{ type_x }{ i_simplex }, ...
              obj.mesh.normals( i_test, : ), i_test, type_x, rot_test, ...
              false );
            
            trial_fun = obj.trial.eval( ...
              obj.y_ref{ type_t }{ type_x }{ i_simplex }, ...
              obj.mesh.normals( i_trial, : ), i_trial, type_x, rot_trial, ...
              true );
            
            A_local( :, : ) = 0;
            
            k = my_kernel.eval( x, y, obj.mesh.normals( i_test, : ), ...
              obj.mesh.normals( i_trial, : ), ...
              obj.t_ref{ type_t }{ type_x }{ i_simplex }, ...
              obj.tau_ref{ type_t }{ type_x }{ i_simplex } );
            
            if( isa( obj.test, 'curl_p1' ) )
              for i_loc_test = 1 : dim_test
                for i_loc_trial = 1 : dim_trial
                  A_local( i_loc_test, i_loc_trial ) = ...
                    A_local( i_loc_test, i_loc_trial ) ...
                    + ( test_fun( ( i_loc_test - 1 ) * 3 + 1 : ...
                    i_loc_test * 3 ) ...
                    * trial_fun( ( i_loc_trial - 1 ) * 3 + 1 : ...
                    i_loc_trial * 3 )' ) ...
                    * ( obj.w{ type_t }{ type_x }{ i_simplex } )' * k;
                end
              end
            else
              for i_loc_test = 1 : dim_test
                for i_loc_trial = 1 : dim_trial
                  A_local( i_loc_test, i_loc_trial ) = ...
                    A_local( i_loc_test, i_loc_trial ) ...
                    + ( obj.w{ type_t }{ type_x }{ i_simplex } ...
                    .* test_fun( :, i_loc_test ) ...
                    .* trial_fun( :, i_loc_trial ) )' * k;
                end
              end
            end
            
            A{ d + 1 }( map_test, map_trial ) = ...
              A{ d + 1 }( map_test, map_trial ) ...
              + A_local * obj.mesh.areas( i_trial ) ...
              * obj.mesh.areas( i_test );
          end
        end
      end
      waitbar( 1, f );
      close( f );
    end
  end
  
  methods (Access = private)
    
    function [ x, y ] = global_quad( obj, i_test, i_trial, type_x, ...
        rot_test, rot_trial, i_simplex, type_t )
      
      nodes = obj.mesh.nodes( obj.mesh.elems( i_test, : ), : );
      z1 = nodes( obj.map( rot_test + 1 ), : );
      z2 = nodes( obj.map( rot_test + 2 ), : );
      z3 = nodes( obj.map( rot_test + 3 ), : );
      x = z1 ...
        + obj.x_ref{ type_t }{ type_x }{ i_simplex } * [ z2 - z1; z3 - z1 ];
      
      nodes = obj.mesh.nodes( obj.mesh.elems( i_trial, : ), : );
      %%%% inverting trial element
      if type_x == 3
        z1 = nodes( obj.map( rot_trial + 2 ), : );
        z2 = nodes( obj.map( rot_trial + 1 ), : );
      else
        z1 = nodes( obj.map( rot_trial + 1 ), : );
        z2 = nodes( obj.map( rot_trial + 2 ), : );
      end
      z3 = nodes( obj.map( rot_trial + 3 ), : );
      y = z1 ...
        + obj.y_ref{ type_t }{ type_x }{ i_simplex } * [ z2 - z1; z3 - z1 ];
      
    end
    
    function [ type_x, rot_test, rot_trial ] = get_type( obj, i_test, i_trial )
      
      rot_test = 0;
      rot_trial = 0;
      
      elem_test = obj.mesh.elems( i_test, : );
      elem_trial = obj.mesh.elems( i_trial, : );
      
      %%%% disjoint
      if ~any( elem_test( : ) == elem_trial, 'all' )
        type_x = 1;
        return;
      end
      
      %%%% identical
      if i_test == i_trial
        type_x = 4;
        return;
      end
      
      %%%% common edge
      for i_trial = 1 : 3
        for i_test = 1 : 3
          if ( ...
              ( elem_trial( i_trial ) == elem_test( obj.map( i_test + 1 ) ) ) ...
              && ...
              ( elem_trial( obj.map( i_trial + 1 ) ) == elem_test( i_test ) ) )
            
            type_x = 3;
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
            type_x = 2;
            rot_test = i_test - 1;
            rot_trial = i_trial - 1;
            return;
          end
        end
      end
      
    end
    
    function obj = init_quadrature_data( obj )
      
      %%%% regular in space
      %%%% no need to store three times for type_t=1,2,3
      obj.x_ref{ 1 }{ 1 }{ 1 } = ...
        zeros( obj.size_ff, 2 );
      obj.y_ref{ 1 }{ 1 }{ 1 } = ...
        zeros( obj.size_ff, 2 );
      obj.w{ 1 }{ 1 }{ 1 } = ...
        zeros( obj.size_ff, 1 );
      obj.t_ref{ 1 }{ 1 }{ 1 } = ...
        zeros( obj.size_ff, 1 );
      obj.tau_ref{ 1 }{ 1 }{ 1 } = ...
        zeros( obj.size_ff, 1 );
        
      [ x_tri, w_tri, l_tri ] = quadratures.tri( obj.order_ff_x );    
      [ t_t, w_t, l_t ] = quadratures.line( obj.order_ff_t );
      
      counter = 1;
      for i_t = 1 : l_t
        for i_tau = 1 : l_t
          for i_x = 1 : l_tri
            for i_y = 1 : l_tri
              obj.x_ref{ 1 }{ 1 }{ 1 }( counter, : ) = x_tri( i_x, : );
              obj.y_ref{ 1 }{ 1 }{ 1 }( counter, : ) = x_tri( i_y, : );
              obj.w{ 1 }{ 1 }{ 1 }( counter ) = ...
                w_tri( i_x ) * w_tri( i_y ) * w_t( i_t ) * w_t( i_tau );
              obj.t_ref{ 1 }{ 1 }{ 1 }( counter, : ) = t_t( i_t );
              obj.tau_ref{ 1 }{ 1 }{ 1 }( counter, : ) = t_t( i_tau );
              counter = counter + 1;
            end
          end
        end
      end

      %%%% singular in time, singular in space
      for type_t = 2 : 3
        for type_x = 2 : 4
          ns = obj.n_simplex( type_x );
          for i_simplex = 1 : ns
            obj.x_ref{ type_t }{ type_x }{ i_simplex } = ...
              zeros( obj.size_nf, 2 );
            obj.y_ref{ type_t }{ type_x }{ i_simplex } = ...
              zeros( obj.size_nf, 2 );
            obj.w{ type_t }{ type_x }{ i_simplex } = ...
              zeros( obj.size_nf, 1 );
            obj.t_ref{ type_t }{ type_x }{ i_simplex } = ...
              zeros( obj.size_nf, 1 );
            obj.tau_ref{ type_t }{ type_x }{ i_simplex } = ...
              zeros( obj.size_nf, 1 );
          end
        end
      end
      
      [ x_line, w_line, l_line ] = quadratures.line( obj.order_nf );
      
      counter = 1;
      for i_eta1 = 1 : l_line
        for i_eta2 = 1 : l_line
          for i_eta3 = 1 : l_line
            for i_eta4 = 1 : l_line
              for i_lambda = 1 : l_line
                for i_mu = 1 : l_line
                  
                  weight = 4 * w_line( i_lambda ) * w_line( i_eta1 ) ...
                    * w_line( i_eta2 ) * w_line( i_eta3 ) ...
                    * w_line( i_lambda ) * w_line( i_mu );
                  
                  for type_t = 2 : 3
                    for type_x = 2 : 4
                      ns = obj.n_simplex( type_x );
                      for i_simplex = 1 : ns
                        [ x_single, y_single, t_single, tau_single, jac ] = ...
                          obj.cube_to_tritime( ...
                          x_line( i_eta1 ), x_line( i_eta2 ), ...
                          x_line( i_eta3 ), x_line( i_eta4 ), ...
                          x_line( i_lambda ), x_line( i_mu ), ...
                          type_x, i_simplex, type_t );
                        obj.x_ref{ type_t }{ type_x }{ i_simplex }...
                          ( counter, : ) = x_single;
                        obj.y_ref{ type_t }{ type_x }{ i_simplex }...
                          ( counter, : ) = y_single;
                        obj.t_ref{ type_t }{ type_x }{ i_simplex }...
                          ( counter, : ) = t_single;
                        obj.tau_ref{ type_t }{ type_x }{ i_simplex }...
                          ( counter, : ) = tau_single;
                        obj.w{ type_t }{ type_x }{ i_simplex }...
                          ( counter ) = weight * jac;
                      end
                    end
                  end
                  
                  counter = counter + 1;
                end
              end
            end
          end
        end
      end
    end
    
    function [ x, y, t, tau, jac ] = cube_to_tritime( ...
        obj, eta1, eta2, eta3, eta4, lambda, mu, type_x, simplex, type_t )
      
      switch type_x
        case 2
          [ x, y, t, tau, jac ] = ...
            cube_to_tritime_vertex_x_singular_t( ...
            obj, eta1, eta2, eta3, eta4, lambda, mu, simplex, type_t );
        case 3
          [ x, y, t, tau, jac ] = ...
            cube_to_tritime_edge_x_singular_t( ...
            obj, eta1, eta2, eta3, eta4, lambda, mu, simplex, type_t );
        case 4
          [ x, y, t, tau, jac ] = ...
            cube_to_tritime_identical_x_singular_t( ...
            obj, eta1, eta2, eta3, eta4, lambda, mu, simplex, type_t );
      end
    end
    
    function [ x, y, t, tau, jac ] = ...
        cube_to_tritime_identical_x_singular_t( ...
        ~, eta1, eta2, eta3, eta4, lambda, mu, simplex, type_t )
           
      switch simplex
        case { 1, 2, 3, 4, 5, 6 }
          ksi = lambda;
          zeta = lambda * mu;
        case { 7, 8, 9, 10, 11, 12 }
          ksi = lambda * mu;
          zeta = lambda;
      end 
      zeta2 = zeta * zeta;
      if ( type_t == 3 )
        t = zeta2 + ( 1 - zeta2 ) * eta4;
        tau = ( 1 - zeta2 ) * eta4;
      elseif ( type_t == 2 )
        t = zeta2 * ( 1 - eta4 );
        tau = 1 - zeta2 * eta4;
      end
      jac = ksi * ksi * ksi * eta1 * eta1 * eta2 ...
        * lambda * zeta * ( 1 - zeta2 ) * 2;     
      
      switch simplex
        case { 1, 7 }
          x( 1 ) = ksi * eta1 * ( 1 - eta2 );
          x( 2 ) = ksi * ( 1 - eta1 * ( 1 - eta2 ) );
          y( 1 ) = ksi * eta1 * ( 1 - eta2 * eta3 );
          y( 2 ) = ksi * ( 1 - eta1 );
        case { 2, 8 }
          x( 1 ) = ksi * eta1 * ( 1 - eta2 * eta3 );
          x( 2 ) = ksi * ( 1 - eta1 );
          y( 1 ) = ksi * eta1 * ( 1 - eta2 );
          y( 2 ) = ksi * ( 1 - eta1 * ( 1 - eta2 ) );
        case { 3, 9 }
          x( 1 ) = ksi * ( 1 - eta1 * ( 1 - eta2 * ( 1 - eta3 ) ) );
          x( 2 ) = ksi * eta1 * ( 1 - eta2 * ( 1 - eta3 ) );
          y( 1 ) = ksi * ( 1 - eta1 );
          y( 2 ) = ksi * eta1 * ( 1 - eta2 );
        case { 4, 10 }
          x( 1 ) = ksi * ( 1 - eta1 );
          x( 2 ) = ksi * eta1 * ( 1 - eta2 );
          y( 1 ) = ksi * ( 1 - eta1 * ( 1 - eta2 * ( 1 - eta3 ) ) );
          y( 2 ) = ksi * eta1 * ( 1 - eta2 * ( 1 - eta3 ) );
        case { 5, 11 } 
          x( 1 ) = ksi * ( 1 - eta1 );
          x( 2 ) = ksi * eta1 * ( 1 - eta2 * eta3 );
          y( 1 ) = ksi * ( 1 - eta1 * ( 1 - eta2 ) );
          y( 2 ) = ksi * eta1 * ( 1 - eta2 );
        case { 6, 12 }
          x( 1 ) = ksi * ( 1 - eta1 * ( 1 - eta2 ) );
          x( 2 ) = ksi * eta1 * ( 1 - eta2 );
          y( 1 ) = ksi * ( 1 - eta1 );
          y( 2 ) = ksi * eta1 * ( 1 - eta2 * eta3 );
      end
    end
            
    function [ x, y, t, tau, jac ] = ...
        cube_to_tritime_edge_x_singular_t( ...
        ~, eta1, eta2, eta3, eta4, lambda, mu, simplex, type_t )     
           
      switch simplex
        case { 1, 2, 3, 4, 5 }
          ksi = lambda;
          zeta = lambda * mu;
        case { 6, 7, 8, 9, 10 }
          ksi = lambda * mu;
          zeta = lambda;
      end 
      zeta2 = zeta * zeta;
      if ( type_t == 3 )
        t = zeta2 + ( 1 - zeta2 ) * eta4;
        tau = ( 1 - zeta2 ) * eta4;
      elseif ( type_t == 2 )
        t = zeta2 * ( 1 - eta4 );
        tau = 1 - zeta2 * eta4;
      end
      jac = ksi * ksi * ksi * eta1 * eta1 ...
        * lambda * zeta * ( 1 - zeta2 ) * 2;
      
      switch simplex
        case { 1, 6 }
          x( 1 ) = ksi * ( 1 - eta1 * eta3 );
          x( 2 ) = ksi * ( eta1 * eta3 );
          y( 1 ) = ksi * ( 1 - eta1 );
          y( 2 ) = ksi * eta1 * ( 1 - eta2 );
        case { 2, 7 }
          x( 1 ) = ksi * ( 1 - eta1 );
          x( 2 ) = ksi * eta1;
          y( 1 ) = ksi * ( 1 - eta1 * eta2 );
          y( 2 ) = ksi * eta1 * eta2 * ( 1 - eta3 );
          jac = jac * eta2;
        case { 3, 8 }
          x( 1 ) = ksi * ( 1 - eta1 );
          x( 2 ) = ksi * eta1 * ( 1 - eta2 );
          y( 1 ) = ksi * ( 1 - eta1 * eta2 * eta3 );
          y( 2 ) = ksi * eta1 * eta2 * eta3;
          jac = jac * eta2;
        case { 4, 9 }
          x( 1 ) = ksi * ( 1 - eta1 * eta2 );
          x( 2 ) = ksi * eta1 * eta2 * ( 1 - eta3 );
          y( 1 ) = ksi * ( 1 - eta1 );
          y( 2 ) = ksi * eta1;
          jac = jac * eta2;
        case { 5, 10 }
          x( 1 ) = ksi * ( 1 - eta1 );
          x( 2 ) = ksi * eta1 * ( 1 - eta2 * eta3 );
          y( 1 ) = ksi * ( 1 - eta1 * eta2 );
          y( 2 ) = ksi * eta1 * eta2;
          jac = jac * eta2;
      end
    end
       
    function [ x, y, t, tau, jac ] = ...
        cube_to_tritime_vertex_x_singular_t( ...
        ~, eta1, eta2, eta3, eta4, lambda, mu, simplex, type_t )
   
      switch simplex
        case { 1, 2 }
          ksi = lambda;
          zeta = lambda * mu;
        case { 3, 4 }
          ksi = lambda * mu;
          zeta = lambda;
      end 
      zeta2 = zeta * zeta;
      if ( type_t == 3 )
        t = zeta2 + ( 1 - zeta2 ) * eta4;
        tau = ( 1 - zeta2 ) * eta4;
      elseif ( type_t == 2 )
        t = zeta2 * ( 1 - eta4 );
        tau = 1 - zeta2 * eta4;
      end    
      
      switch simplex
        case { 1, 3 }
          x( 1 ) = ksi * ( 1 - eta1 );
          x( 2 ) = ksi * eta1;
          y( 1 ) = ksi * eta2 * ( 1 - eta3 );
          y( 2 ) = ksi * eta2 * eta3;
        case { 2, 4 }
          x( 1 ) = ksi * eta2 * ( 1 - eta3 );
          x( 2 ) = ksi * eta2 * eta3;
          y( 1 ) = ksi * ( 1 - eta1 );
          y( 2 ) = ksi * eta1;
      end
      
      jac = ksi * ksi * ksi * eta2 ...
        * lambda * zeta * ( 1 - zeta2 ) * 2;
    end
    
  end
end

