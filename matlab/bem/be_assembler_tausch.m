classdef be_assembler_tausch < handle
  
  properties (Access = public)
    mesh;
    kernel;
    test;
    trial;
    order_nf_1d;
    order_nf_2d;
    order_nf_3d;
    order_ff;
    size_xy;
    size_zp;
    
    x_ref = cell( 4, 1 );
    y_ref = cell( 4, 1 );
    weight = cell( 4, 1 );
  end
  
  properties (Constant)
    %%%% type = 1 ... disjoint
    %%%% type = 2 ... vertex
    %%%% type = 3 ... edge
    %%%% type = 4 ... identical
    
    n_simplex = [ 1 6 6 6 ];
    
    map = [ 1 2 3 1 2 ];
  end
  
  methods
    function obj = be_assembler_tausch( mesh, kernel, test, trial, ...
        order_nf_1d, order_nf_2d, order_nf_3d, order_ff )
      
      obj.mesh = mesh;
      obj.kernel = kernel;
      obj.test = test;
      obj.trial = trial;
      
      obj.order_nf_1d = order_nf_1d;
      obj.order_nf_2d = order_nf_2d;
      obj.order_nf_3d = order_nf_3d;
      obj.order_ff = order_ff;
      
      length_ff = quadratures.tri_length( order_ff );
      length_nf_1d = quadratures.line_length( order_nf_1d );
      length_nf_2d = quadratures.tri_length( order_nf_2d );
      length_nf_3d = quadratures.tetra_length( order_nf_3d );
      
      obj.size_xy( 1 ) = length_ff * length_ff;
      obj.size_xy( 2 ) = length_nf_1d * length_nf_3d;
      obj.size_xy( 3 ) = length_nf_1d * length_nf_2d * length_nf_1d;
      obj.size_xy( 4 ) = length_nf_1d * length_nf_1d * length_nf_2d;
            
      for type = 1 : 4
        obj.x_ref{ type } = cell( obj.n_simplex( type ), 1 );
        obj.y_ref{ type } = cell( obj.n_simplex( type ), 1 );
        obj.weight{ type } = cell( obj.n_simplex( type ), 1 );
      end
      
      init_quadrature_data( obj );
    end
    
    function A = assemble( obj )
      if( isa( obj.mesh, 'spacetime_mesh' ) )
        A = assemble_st( obj );
      else
        A = assemble_s( obj );
      end
    end
    
  end
  
  methods (Access = private)
    
    function A = assemble_s( obj )
      A = zeros( obj.test.dim_global( ), obj.trial.dim_global( ) );
      
      n_elems = obj.mesh.n_elems;
      dim_test = obj.test.dim_local( );
      dim_trial = obj.trial.dim_local( );
      A_local = zeros( dim_test, dim_trial );
      
      msg = sprintf( 'assembling %s', class( obj.kernel ) );
      f = waitbar( 0, msg );
      f.Children.Title.Interpreter = 'none';
      
      for i_trial = 1 : n_elems
        waitbar( ( i_trial - 1 ) / n_elems, f );
        for i_test = 1 : n_elems
          
          [ type, rot_test, rot_trial ] = get_type( obj, i_test, i_trial );
          
          A_local( :, : ) = 0;
          for i_simplex = 1 : obj.n_simplex( type )
            [ x, y ] = global_quad( obj, ...
              i_test, i_trial, type, rot_test, rot_trial, i_simplex );
            
            k = obj.kernel.eval( x, y, obj.mesh.normals( i_test, : ), ...
              obj.mesh.normals( i_trial, : ) );
            
            test_fun = obj.test.eval( obj.x_ref{ type }{ i_simplex }, ...
              obj.mesh.normals( i_test, : ), i_test, type, rot_test, false );
            
            trial_fun = obj.trial.eval( obj.y_ref{ type }{ i_simplex }, ...
              obj.mesh.normals( i_trial, : ), i_trial, type, rot_trial, true );
            
            if( isa( obj.test, 'curl_p1' ) )
              for i_loc_test = 1 : dim_test
                for i_loc_trial = 1 : dim_trial
                  A_local( i_loc_test, i_loc_trial ) = ...
                    A_local( i_loc_test, i_loc_trial ) ...
                    + ( test_fun( ( i_loc_test - 1 ) * 3 + 1 : ...
                    i_loc_test * 3 ) ...
                    * trial_fun( ( i_loc_trial - 1 ) * 3 + 1 : ...
                    i_loc_trial * 3 )' ) ...
                    * ( obj.weight{ type }{ i_simplex } )' * k;
                end
              end
            else
              for i_loc_test = 1 : dim_test
                for i_loc_trial = 1 : dim_trial
                  A_local( i_loc_test, i_loc_trial ) = ...
                    A_local( i_loc_test, i_loc_trial ) ...
                    + ( obj.weight{ type }{ i_simplex } ...
                    .* test_fun( :, i_loc_test ) ...
                    .* trial_fun( :, i_loc_trial ) )' * k;
                end
              end
            end
          end
          
          map_test = obj.test.l2g( i_test, type, rot_test, false );
          map_trial = obj.trial.l2g( i_trial, type, rot_trial, true );
          A( map_test, map_trial ) = A( map_test, map_trial ) ...
            + A_local * obj.mesh.areas( i_trial ) ...
            * obj.mesh.areas( i_test );
        end
      end
      waitbar( 1, f );
      close( f );
    end
    
    function A = assemble_st( obj )
      
      nt = obj.mesh.nt;
      A = cell( nt, 1 );
      for d = 0 : nt - 1
        A{ d + 1 } = zeros( obj.test.dim_global( ), obj.trial.dim_global( ) );
      end
      
      n_elems = obj.mesh.n_elems;
      dim_test = obj.test.dim_local( );
      dim_trial = obj.trial.dim_local( );
      obj.kernel.ht = obj.mesh.ht;
      
      msg = sprintf( 'assembling %s', class( obj.kernel ) );
      f = waitbar( 0, msg );
      f.Children.Title.Interpreter = 'none';
      
      my_kernel = obj.kernel;
      A_local = zeros( dim_test, dim_trial );
      for d = 0 : nt - 1
        my_kernel.d = d;
        msgd = [ msg sprintf( ', d = %d/%d', d + 1, nt ) ];
        waitbar( d / nt, f, msgd );
        
        for i_trial = 1 : n_elems
          waitbar( ( d + ( i_trial - 1 ) / n_elems ) / nt, f );
          for i_test = 1 : n_elems
            
            if d <= 1
              [ type, rot_test, rot_trial ] = get_type( obj, i_test, i_trial );
            else
              type = 1;
              rot_test = 0;
              rot_trial = 0;
            end
            
            A_local( :, : ) = 0;
            for i_simplex = 1 : obj.n_simplex( type )
              [ x, y ] = global_quad( obj, ...
                i_test, i_trial, type, rot_test, rot_trial, i_simplex );
              
              k = my_kernel.eval( x, y, obj.mesh.normals( i_test, : ), ...
                obj.mesh.normals( i_trial, : ) );
              
              test_fun = obj.test.eval( obj.x_ref{ type }{ i_simplex }, ...
                obj.mesh.normals( i_test, : ), i_test, type, rot_test, ...
                false );
              
              trial_fun = obj.trial.eval( obj.y_ref{ type }{ i_simplex }, ...
                obj.mesh.normals( i_trial, : ), i_trial, type, rot_trial, ...
                true );
              
              if( isa( obj.test, 'curl_p1' ) )
                for i_loc_test = 1 : dim_test
                  for i_loc_trial = 1 : dim_trial
                    A_local( i_loc_test, i_loc_trial ) = ...
                      A_local( i_loc_test, i_loc_trial ) ...
                      + ( test_fun( ( i_loc_test - 1 ) * 3 + 1 : ...
                      i_loc_test * 3 ) ...
                      * trial_fun( ( i_loc_trial - 1 ) * 3 + 1 : ...
                      i_loc_trial * 3 )' ) ...
                      * ( obj.weight{ type }{ i_simplex } )' * k;
                  end
                end
              else
                for i_loc_test = 1 : dim_test
                  for i_loc_trial = 1 : dim_trial
                    A_local( i_loc_test, i_loc_trial ) = ...
                      A_local( i_loc_test, i_loc_trial ) ...
                      + ( obj.weight{ type }{ i_simplex } ...
                      .* test_fun( :, i_loc_test ) ...
                      .* trial_fun( :, i_loc_trial ) )' * k;
                  end
                end
              end
            end
            
            map_test = obj.test.l2g( i_test, type, rot_test, false );
            map_trial = obj.trial.l2g( i_trial, type, rot_trial, true );
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
    
    function [ x, y ] = global_quad( obj, i_test, i_trial, type, ...
        rot_test, rot_trial, i_simplex )
      
      nodes = obj.mesh.nodes( obj.mesh.elems( i_test, : ), : );
      z1 = nodes( obj.map( rot_test + 1 ), : );
      z2 = nodes( obj.map( rot_test + 2 ), : );
      z3 = nodes( obj.map( rot_test + 3 ), : );
      x = z1 + obj.x_ref{ type }{ i_simplex } * [ z2 - z1; z3 - z1 ];
      
      nodes = obj.mesh.nodes( obj.mesh.elems( i_trial, : ), : );
      %%%% inverting trial element
      if type == 3
        z1 = nodes( obj.map( rot_trial + 2 ), : );
        z2 = nodes( obj.map( rot_trial + 1 ), : );
      else
        z1 = nodes( obj.map( rot_trial + 1 ), : );
        z2 = nodes( obj.map( rot_trial + 2 ), : );
      end
      z3 = nodes( obj.map( rot_trial + 3 ), : );
      y = z1 + obj.y_ref{ type }{ i_simplex } * [ z2 - z1; z3 - z1 ];
      
    end
    
    function [ type, rot_test, rot_trial ] = get_type( obj, i_test, i_trial )
      
      rot_test = 0;
      rot_trial = 0;
      
      elem_test = obj.mesh.elems( i_test, : );
      elem_trial = obj.mesh.elems( i_trial, : );
      
      %%%% disjoint
      if ~any( elem_test( : ) == elem_trial, 'all' )
        type = 1;
        return;
      end
      
      %%%% identical
      if i_test == i_trial
        type = 4;
        return;
      end
      
      %%%% common edge
      for ii_trial = 1 : 3
        for ii_test = 1 : 3
          if ( ( elem_trial( ii_trial ) == ...
              elem_test( obj.map( ii_test + 1 ) ) ) ...
              && ...
              ( elem_trial( obj.map( ii_trial + 1 ) ) ...
              == elem_test( ii_test ) ) )
            
            type = 3;
            rot_test = ii_test - 1;
            rot_trial = ii_trial - 1;
            return;
          end
        end
      end
      
      %%%% common vertex
      for ii_trial = 1 : 3
        for ii_test = 1 : 3
          if elem_test( ii_test ) == elem_trial( ii_trial )
            type = 2;
            rot_test = ii_test - 1;
            rot_trial = ii_trial - 1;
            return;
          end
        end
      end
      
    end
    
    function obj = init_quadrature_data( obj )
      
      %%%% disjoint
      obj.x_ref{ 1 }{ 1 } = zeros( obj.size_xy( 1 ), 2 );
      obj.y_ref{ 1 }{ 1 } = zeros( obj.size_xy( 1 ), 2 );
      obj.weight{ 1 }{ 1 } = zeros( obj.size_xy( 1 ), 1 );
      
      [ x_tri, w_tri, l_tri ] = quadratures.tri( obj.order_ff );
      
      counter = 1;
      for i_x = 1 : l_tri
        for i_y = 1 : l_tri
          obj.x_ref{ 1 }{ 1 }( counter, : ) = x_tri( i_x, : );
          obj.y_ref{ 1 }{ 1 }( counter, : ) = x_tri( i_y, : );
          obj.weight{ 1 }{ 1 }( counter ) = w_tri( i_x ) * w_tri( i_y );
          counter = counter + 1;
        end
      end
      
      %%%% singular
      for type = 2 : 4
        for i_simplex = 1 : obj.n_simplex( type )
          obj.x_ref{ type }{ i_simplex } = zeros( obj.size_xy( type ), 2 );
          obj.y_ref{ type }{ i_simplex } = zeros( obj.size_xy( type ), 2 );
          obj.weight{ type }{ i_simplex } = zeros( obj.size_xy( type ), 1 );
        end
      end
      
      [ xref1d, weight1d, l1d ] = quadratures.line( obj.order_nf_1d );
      [ xref2d, weight2d, l2d ] = quadratures.tri( obj.order_nf_2d );
      [ xref3d, weight3d, l3d ] = quadratures.tetra( obj.order_nf_3d );
      
      % identical
      type = 4;
      for i_simplex = 1 : obj.n_simplex( type )
        counter = 1;
        for i_ksi = 1 : l1d
          ksi = xref1d( i_ksi );
          for i_w = 1 : l1d
            w = xref1d( i_w );
            for i_wb = 1 : l2d
              wb = xref2d( i_wb, : );
              
              [ x_single, y_single, jac ] = ...
                ksi_w_wb_to_x_y( obj, ksi, w, wb, type, i_simplex );
              obj.x_ref{ type }{ i_simplex }( counter, : ) = x_single;
              obj.y_ref{ type }{ i_simplex }( counter, : ) = y_single;
              obj.weight{ type }{ i_simplex }( counter ) = 4 ...
                * weight1d( i_ksi ) * weight1d( i_w ) * jac * weight2d( i_wb );
              
              counter = counter + 1;
            end
          end
        end
      end
      
      % common edge
      type = 3;
      for i_simplex = 1 : obj.n_simplex( type )
        counter = 1;
        for i_ksi = 1 : l1d
          ksi = xref1d( i_ksi );
          for i_w = 1 : l2d
            w = xref2d( i_w, : );
            for i_wb = 1 : l1d
              wb = xref1d( i_wb );
              
              [ x_single, y_single, jac ] = ...
                ksi_w_wb_to_x_y( obj, ksi, w, wb, type, i_simplex );
              obj.x_ref{ type }{ i_simplex }( counter, : ) = x_single;
              obj.y_ref{ type }{ i_simplex }( counter, : ) = y_single;
              obj.weight{ type }{ i_simplex }( counter ) = 4 ...
                * weight1d( i_ksi ) * weight2d( i_w ) * jac * weight1d( i_wb );
              
              counter = counter + 1;
            end
          end
        end
      end
      
      % common vertex
      type = 2;
      for i_simplex = 1 : obj.n_simplex( type )
        counter = 1;
        for i_ksi = 1 : l1d
          ksi = xref1d( i_ksi );
          for i_w = 1 : l3d
            w = xref3d( i_w, : );
            
            [ x_single, y_single, jac ] = ...
              ksi_w_wb_to_x_y( obj, ksi, w, [], type, i_simplex );
            obj.x_ref{ type }{ i_simplex }( counter, : ) = x_single;
            obj.y_ref{ type }{ i_simplex }( counter, : ) = y_single;
            obj.weight{ type }{ i_simplex }( counter ) = 4 ...
              * weight1d( i_ksi ) * weight3d( i_w ) * jac / 3;
            
            counter = counter + 1;
          end
        end
      end
      
    end
    
    function [ x, y, jac ] = ksi_w_wb_to_x_y( obj, ksi, w, wb, type, simplex )
      switch type
        case 2
          [ x, y, jac ] = ...
            ksi_w_wb_to_x_y_vertex( obj, ksi, w( 1 ), w( 2 ), w( 3 ), simplex );
        case 3
          [ x, y, jac ] = ...
            ksi_w_wb_to_x_y_edge( obj, ksi, w( 1 ), w( 2 ), wb, simplex );
        case 4
          [ x, y, jac ] = ...
            ksi_w_wb_to_x_y_identical( obj, ksi, w, wb( 1 ), wb( 2 ), simplex );
      end
    end
    
    function [ x, y, jac ] = ...
        ksi_w_wb_to_x_y_identical( ~, ksi, w, wb1, wb2, simplex )
      
      jac = ksi^3*(ksi^2 - 1)^2;
      
      switch simplex
        case 1
          x( 1 ) = -wb1*(ksi^2 - 1);
          x( 2 ) = -wb2*(ksi^2 - 1);
          y( 1 ) = wb1 - ksi^2*w - ksi^2*wb1 + ksi^2;
          y( 2 ) = ksi^2*w - wb2*(ksi^2 - 1);
        case 2
          x( 1 ) = wb1 + ksi^2*w - ksi^2*wb1;
          x( 2 ) = -wb2*(ksi^2 - 1);
          y( 1 ) = -wb1*(ksi^2 - 1);
          y( 2 ) = ksi^2 - wb2*(ksi^2 - 1);
        case 3
          x( 1 ) = -wb1*(ksi^2 - 1);
          x( 2 ) = ksi^2*w - wb2*(ksi^2 - 1);
          y( 1 ) = wb1 - ksi^2*wb1 + ksi^2;
          y( 2 ) = -wb2*(ksi^2 - 1);
        case 4
          x( 1 ) = wb1 - ksi^2*wb1 + ksi^2;
          x( 2 ) = -wb2*(ksi^2 - 1);
          y( 1 ) = -wb1*(ksi^2 - 1);
          y( 2 ) = ksi^2*w - wb2*(ksi^2 - 1);
        case 5
          x( 1 ) = -wb1*(ksi^2 - 1);
          x( 2 ) = ksi^2 - wb2*(ksi^2 - 1);
          y( 1 ) = wb1 + ksi^2*w - ksi^2*wb1;
          y( 2 ) = -wb2*(ksi^2 - 1);
        case 6
          x( 1 ) = wb1 - ksi^2*w - ksi^2*wb1 + ksi^2;
          x( 2 ) = ksi^2*w - wb2*(ksi^2 - 1);
          y( 1 ) = -wb1*(ksi^2 - 1);
          y( 2 ) = -wb2*(ksi^2 - 1);
      end
    end
    
    function [ x, y, jac ] = ...
        ksi_w_wb_to_x_y_edge( ~, ksi, w1, w2, wb, simplex )
      
      jac = ksi^5 * ( 1 - ksi^2 );
      
      switch simplex
        case 1
          x( 1 ) = -wb*(ksi^2 - 1);
          x( 2 ) = -ksi^2*(w1 + w2 - 1);
          y( 1 ) = ksi^2*w1 - wb*(ksi^2 - 1);
          y( 2 ) = -ksi^2*(w1 - 1);
        case 2
          x( 1 ) = ksi^2*w2 - wb*(ksi^2 - 1);
          x( 2 ) = -ksi^2*(w1 + w2 - 1);
          y( 1 ) = -wb*(ksi^2 - 1);
          y( 2 ) = ksi^2;
        case 3
          x( 1 ) = -wb*(ksi^2 - 1);
          x( 2 ) = -ksi^2*(w1 - 1);
          y( 1 ) = ksi^2*w1 - wb*(ksi^2 - 1) + ksi^2*w2;
          y( 2 ) = -ksi^2*(w1 + w2 - 1);
        case 4
          x( 1 ) = ksi^2*w1 - wb*(ksi^2 - 1) + ksi^2*w2;
          x( 2 ) = -ksi^2*(w1 + w2 - 1);
          y( 1 ) = -wb*(ksi^2 - 1);
          y( 2 ) = -ksi^2*(w1 - 1);
        case 5
          x( 1 ) = -wb*(ksi^2 - 1);
          x( 2 ) = ksi^2;
          y( 1 ) = ksi^2*w2 - wb*(ksi^2 - 1);
          y( 2 ) = -ksi^2*(w1 + w2 - 1);
        case 6
          x( 1 ) = ksi^2*w1 - wb*(ksi^2 - 1);
          x( 2 ) = -ksi^2*(w1 - 1);
          y( 1 ) = -wb*(ksi^2 - 1);
          y( 2 ) = -ksi^2*(w1 + w2 - 1);
      end
    end
    
    function [ x, y, jac ] = ...
        ksi_w_wb_to_x_y_vertex( ~, ksi, w1, w2, w3, simplex )
      
      jac = ksi^7;
      
      switch simplex
        case 1
          x( 1 ) = -ksi^2*(w1 + w2 + w3 - 1);
          x( 2 ) = ksi^2*w1;
          y( 1 ) = -ksi^2*(w1 + w3 - 1);
          y( 2 ) = ksi^2*(w1 + w3);
        case 2
          x( 1 ) = -ksi^2*(w1 + w2 - 1);
          x( 2 ) = ksi^2*w1;
          y( 1 ) = -ksi^2*(w1 + w2 + w3 - 1);
          y( 2 ) = ksi^2*(w1 + w2 + w3);
        case 3
          x( 1 ) = -ksi^2*(w1 + w2 + w3 - 1);
          x( 2 ) = ksi^2*(w1 + w3);
          y( 1 ) = -ksi^2*(w1 - 1);
          y( 2 ) = ksi^2*w1;
        case 4
          x( 1 ) = -ksi^2*(w1 - 1);
          x( 2 ) = ksi^2*w1;
          y( 1 ) = -ksi^2*(w1 + w2 + w3 - 1);
          y( 2 ) = ksi^2*(w1 + w3);
        case 5
          x( 1 ) = -ksi^2*(w1 + w2 + w3 - 1);
          x( 2 ) = ksi^2*(w1 + w2 + w3);
          y( 1 ) = -ksi^2*(w1 + w2 - 1);
          y( 2 ) = ksi^2*w1;
        case 6
          x( 1 ) = -ksi^2*(w1 + w3 - 1);
          x( 2 ) = ksi^2*(w1 + w3);
          y( 1 ) = -ksi^2*(w1 + w2 + w3 - 1);
          y( 2 ) = ksi^2*w1;
      end
    end
    
  end
end

