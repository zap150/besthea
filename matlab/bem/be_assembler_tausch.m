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
    zp = cell( 4, 1 );
    wxy = cell( 4, 1 );
    wzp = cell( 4, 1 );
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
      
      length_nf_1d = quadratures.line_length( order_nf_1d );
      length_nf_2d = quadratures.tri_length( order_nf_2d );
      length_nf_3d = quadratures.tetra_length( order_nf_3d );
      
      obj.size_zp( 2 ) = length_nf_1d * length_nf_3d;
      obj.size_zp( 3 ) = length_nf_1d * length_nf_2d;
      obj.size_zp( 4 ) = length_nf_1d * length_nf_1d;
      obj.size_xy( 2 ) = 1;
      obj.size_xy( 3 ) = length_nf_1d;
      obj.size_xy( 4 ) = length_nf_2d;
      
      length_ff = quadratures.tri_length( order_ff );
      obj.size_xy( 1 ) = length_ff * length_ff;
      obj.size_zp( 1 ) = 1;
      
      type = 1;
      obj.x_ref{ type } = cell( 1, 1 );
      obj.y_ref{ type } = cell( 1, 1 );
      obj.zp{ type } = [];
      obj.wxy{ type } = cell( 1, 1 );
      obj.wzp{ type } = [];
      
      for type = 2 : 4
        obj.x_ref{ type } = cell( obj.n_simplex( type ), 1 );
        obj.y_ref{ type } = cell( obj.n_simplex( type ), 1 );
        obj.zp{ type } = cell( obj.n_simplex( type ), 1 );
        obj.wxy{ type } = cell( obj.n_simplex( type ), 1 );
        obj.wzp{ type } = cell( obj.n_simplex( type ), 1 );
      end
      
      init_quadrature_data( obj );
    end
    
    function A = assemble( obj )
      if( isa( obj.mesh, 'spacetime_mesh' ) )
        %A = assemble_st( obj );
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
          
          [ type, rot_test, rot_trial ] = obj.get_type( i_test, i_trial );
          
          A_local( :, : ) = 0;
          if type == 1
            i_simplex = 1;
            xy = obj.global_quad( i_test, i_trial, type, rot_test, ...
              rot_trial, i_simplex );
            
            k = obj.kernel.eval( xy, zeros( size ( xy ) ), ...
              obj.mesh.normals( i_test, : ), obj.mesh.normals( i_trial, : ) );
            
            test_fun = obj.test.eval( obj.x_ref{ type }{ 1 }{ i_simplex }, ...
              obj.mesh.normals( i_test, : ), i_test, type, rot_test, false );
            trial_fun = obj.trial.eval( obj.y_ref{ type }{ 1 }{ i_simplex }, ...
              obj.mesh.normals( i_trial, : ), i_trial, type, rot_trial, true );
            for i_loc_test = 1 : dim_test
              for i_loc_trial = 1 : dim_trial
                A_local( i_loc_test, i_loc_trial ) = ...
                  A_local( i_loc_test, i_loc_trial ) ...
                  + ( test_fun( :, i_loc_test ) ...
                  .* trial_fun( :, i_loc_trial ) ...
                  .* obj.wxy{ type }{ i_simplex }{ 1 } )' ...
                  * k;
              end
            end
            
          else
            for i_simplex = 1 : obj.n_simplex( type )
              xy = obj.global_quad( i_test, i_trial, type, rot_test, ...
                rot_trial, i_simplex );
              
              k = obj.kernel.eval( xy, zeros( size ( xy ) ), ...
                obj.mesh.normals( i_test, : ), obj.mesh.normals( i_trial, : ) );
              k = k .* obj.wzp{ type }{ i_simplex };
              
              for i_xy = 1 : obj.size_zp( type )
                test_fun = obj.test.eval( ...
                  obj.x_ref{ type }{ i_simplex }{ i_xy }, ...
                  obj.mesh.normals( i_test, : ), ...
                  i_test, type, rot_test, false );
                trial_fun = obj.trial.eval( ...
                  obj.y_ref{ type }{ i_simplex }{ i_xy }, ...
                  obj.mesh.normals( i_trial, : ), ...
                  i_trial, type, rot_trial, true );
                for i_loc_test = 1 : dim_test
                  for i_loc_trial = 1 : dim_trial
                    A_local( i_loc_test, i_loc_trial ) = ...
                      A_local( i_loc_test, i_loc_trial ) ...
                      + ( test_fun( :, i_loc_test ) ...
                      .* trial_fun( :, i_loc_trial ) )' ...
                      * obj.wxy{ type }{ i_simplex }{ i_xy } ...
                      * k( i_xy );
                  end
                end
              end
              
              %             if( isa( obj.test, 'curl_p1' ) )
              %               for i_loc_test = 1 : dim_test
              %                 for i_loc_trial = 1 : dim_trial
              %                   A_local( i_loc_test, i_loc_trial ) = ...
              %                     A_local( i_loc_test, i_loc_trial ) ...
              %                     + ( test_fun( ( i_loc_test - 1 ) * 3 + 1 : ...
              %                     i_loc_test * 3 ) ...
              %                     * trial_fun( ( i_loc_trial - 1 ) * 3 + 1 : ...
              %                     i_loc_trial * 3 )' ) * ( obj.w{ type }{ i_simplex } )' * k;
              %                 end
              %               end
              %             else
              %             for i_loc_test = 1 : dim_test
              %               for i_loc_trial = 1 : dim_trial
              %                 A_local( i_loc_test, i_loc_trial ) = ...
              %                   A_local( i_loc_test, i_loc_trial ) ...
              %                   + ( obj.wzp{ type }{ i_simplex } .* tt{ type } )' * k;
              %               end
              %             end
              %             end
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
    
%     function A = assemble_st( obj )
%       
%       nt = obj.mesh.nt;
%       A = cell( nt, 1 );
%       for d = 0 : nt - 1
%         A{ d + 1 } = zeros( obj.test.dim_global( ), obj.trial.dim_global( ) );
%       end
%       
%       n_elems = obj.mesh.n_elems;
%       dim_test = obj.test.dim_local( );
%       dim_trial = obj.trial.dim_local( );
%       obj.kernel.ht = obj.mesh.ht;
%       
%       msg = sprintf( 'assembling %s', class( obj.kernel ) );
%       f = waitbar( 0, msg );
%       f.Children.Title.Interpreter = 'none';
%       
%       my_kernel = obj.kernel;
%       A_local = zeros( dim_test, dim_trial );
%       for d = 0 : nt - 1
%         my_kernel.d = d;
%         msgd = [ msg sprintf( ', d = %d/%d', d + 1, nt ) ];
%         waitbar( d / nt, f, msgd );
%         
%         for i_trial = 1 : n_elems
%           waitbar( ( d + ( i_trial - 1 ) / n_elems ) / nt, f );
%           for i_test = 1 : n_elems
%             
%             if d <= 1
%               [ type, rot_test, rot_trial ] = get_type( obj, i_test, i_trial );
%             else
%               type = 1;
%               rot_test = 0;
%               rot_trial = 0;
%             end
%             
%             A_local( :, : ) = 0;
%             for i_simplex = 1 : obj.n_simplex( type )
%               [ x, y ] = global_quad( obj, ...
%                 i_test, i_trial, type, rot_test, rot_trial, i_simplex );
%               
%               k = my_kernel.eval( x, y, obj.mesh.normals( i_test, : ), ...
%                 obj.mesh.normals( i_trial, : ) );
%               
%               test_fun = obj.test.eval( obj.x_ref{ type }{ i_simplex }, ...
%                 obj.mesh.normals( i_test, : ), i_test, type, rot_test, ...
%                 false );
%               
%               trial_fun = obj.trial.eval( obj.y_ref{ type }{ i_simplex }, ...
%                 obj.mesh.normals( i_trial, : ), i_trial, type, rot_trial, ...
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
%                       i_loc_trial * 3 )' ) * ( obj.w{ type }{ i_simplex } )' ...
%                       * k;
%                   end
%                 end
%               else
%                 for i_loc_test = 1 : dim_test
%                   for i_loc_trial = 1 : dim_trial
%                     A_local( i_loc_test, i_loc_trial ) = ...
%                       A_local( i_loc_test, i_loc_trial ) ...
%                       + ( obj.w{ type }{ i_simplex } ...
%                       .* test_fun( :, i_loc_test ) ...
%                       .* trial_fun( :, i_loc_trial ) )' * k;
%                   end
%                 end
%               end
%             end
%             
%             map_test = obj.test.l2g( i_test, type, rot_test, false );
%             map_trial = obj.trial.l2g( i_trial, type, rot_trial, true );
%             A{ d + 1 }( map_test, map_trial ) = ...
%               A{ d + 1 }( map_test, map_trial ) ...
%               + A_local * obj.mesh.areas( i_trial ) ...
%               * obj.mesh.areas( i_test );
%           end
%         end
%       end
%       waitbar( 1, f );
%       close( f );
%     end
    
    function xy = global_quad( obj, i_test, i_trial, type, ...
        rot_test, rot_trial, i_simplex )
      
      nodes = obj.mesh.nodes( obj.mesh.elems( i_test, : ), : );
      z1 = nodes( obj.map( rot_test + 1 ), : );
      z2 = nodes( obj.map( rot_test + 2 ), : );
      z3 = nodes( obj.map( rot_test + 3 ), : );
      Ax = [ z2 - z1; z3 - z2 ];
      x = z1 + cat( 1, obj.x_ref{ type }{ i_simplex }{ : } ) ...
        * [ z2 - z1; z3 - z1 ];
      
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
      Ay = [ z2 - z1; z3 - z2 ];
      y = z1 + cat( 1, obj.y_ref{ type }{ i_simplex }{ : } ) ...
        * [ z2 - z1; z3 - z1 ];
      
      switch type
        case 4
          xy = -obj.zp{ type }{ i_simplex } * Ax;
        case 3
          xy = obj.zp{ type }{ i_simplex } ...
            * [ -Ax( 1, : ); Ax( 2, : ); -Ay( 2, : ) ];
        case 2
          xy = obj.zp{ type }{ i_simplex } * [ Ax; -Ay ];
        case 1
          xy = x - y;
      end
      
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
      obj.x_ref{ 1 }{ 1 } = cell( 1, 1 );
      obj.y_ref{ 1 }{ 1 } = cell( 1, 1 );
      obj.wxy{ 1 }{ 1 } = cell( 1, 1 );
      obj.x_ref{ 1 }{ 1 }{ 1 } = zeros( obj.size_xy( 1 ), 2 );
      obj.y_ref{ 1 }{ 1 }{ 1 } = zeros( obj.size_xy( 1 ), 2 );
      obj.wxy{ 1 }{ 1 }{ 1 } = zeros( obj.size_xy( 1 ), 1 );
      obj.zp{ 1 }{ 1 } = [];
      obj.wzp{ 1 }{ 1 } = [];
      
      [ x_tri, w_tri, l_tri ] = quadratures.tri( obj.order_ff );
      
      counter = 1;
      for i_x = 1 : l_tri
        for i_y = 1 : l_tri
          obj.x_ref{ 1 }{ 1 }{ 1 }( counter, : ) = x_tri( i_x, : );
          obj.y_ref{ 1 }{ 1 }{ 1 }( counter, : ) = x_tri( i_y, : );
          obj.wxy{ 1 }{ 1 }{ 1 }( counter ) = w_tri( i_x ) * w_tri( i_y );
          counter = counter + 1;
        end
      end
      
      %%%% singular
      for type = 2 : 4
        for i_simplex = 1 : obj.n_simplex( type )
          obj.x_ref{ type }{ i_simplex } = cell( obj.size_zp( type ), 1 );
          obj.y_ref{ type }{ i_simplex } = cell( obj.size_zp( type ), 1 );
          obj.wxy{ type }{ i_simplex } = cell( obj.size_zp( type ), 1 );
          obj.zp{ type }{ i_simplex } = zeros( obj.size_zp( type ), 6 - type );
          obj.wzp{ type }{ i_simplex } = zeros( obj.size_zp( type ), 1 );
          for i_zp = 1 : obj.size_zp( type )
            obj.x_ref{ type }{ i_simplex }{ i_zp } = ...
              zeros( obj.size_xy( type ), 2 );
            obj.y_ref{ type }{ i_simplex }{ i_zp } = ...
              zeros( obj.size_xy( type ), 2 );
            obj.wxy{ type }{ i_simplex }{ i_zp } = ...
              zeros( obj.size_xy( type ), 1 );
          end
        end
      end
      
      [ xref1d, weight1d, l1d ] = quadratures.line( obj.order_nf_1d );
      [ xref2d, weight2d, l2d ] = quadratures.tri( obj.order_nf_2d );
      [ xref3d, weight3d, l3d ] = quadratures.tetra( obj.order_nf_3d );
      
      % identical
      type = 4;
      for i_simplex = 1 : obj.n_simplex( type )
        counter_zp = 1;
        for i_ksi = 1 : l1d
          for i_w = 1 : l1d
            w = xref1d( i_w );
            ksi = xref1d( i_ksi );
            [ zp_single, jac ] = ksi_w_to_zp( obj, ksi, w, type, i_simplex );
            obj.zp{ type }{ i_simplex }( counter_zp, : ) = zp_single;
            obj.wzp{ type }{ i_simplex }( counter_zp, : ) = ...
              4 * weight1d( i_ksi ) * weight1d( i_w ) * jac;
            for i_wb = 1 : l2d
              wb = xref2d( i_wb, : );
              [ x_single, y_single ] = ...
                ksi_w_wb_to_x_y( obj, ksi, w, wb, type, i_simplex );
              obj.x_ref{ type }{ i_simplex }{ counter_zp }( i_wb, : ) = ...
                x_single;
              obj.y_ref{ type }{ i_simplex }{ counter_zp }( i_wb, : ) = ...
                y_single;
              obj.wxy{ type }{ i_simplex }{ counter_zp }( i_wb, : ) = ...
                weight2d( i_wb );
            end
            counter_zp = counter_zp + 1;
          end
        end
      end
      
      % common edge
      type = 3;
      for i_simplex = 1 : obj.n_simplex( type )
        counter_zp = 1;
        for i_ksi = 1 : l1d
          for i_w = 1 : l2d
            w = xref2d( i_w, : );
            ksi = xref1d( i_ksi );
            [ zp_single, jac ] = ksi_w_to_zp( obj, ksi, w, type, i_simplex );
            obj.zp{ type }{ i_simplex }( counter_zp, : ) = zp_single;
            obj.wzp{ type }{ i_simplex }( counter_zp, : ) = ...
              4 * weight1d( i_ksi ) * weight2d( i_w ) * jac;
            for i_wb = 1 : l1d
              wb = xref1d( i_wb );
              [ x_single, y_single ] = ...
                ksi_w_wb_to_x_y( obj, ksi, w, wb, type, i_simplex );
              obj.x_ref{ type }{ i_simplex }{ counter_zp }( i_wb, : ) = ...
                x_single;
              obj.y_ref{ type }{ i_simplex }{ counter_zp }( i_wb, : ) = ...
                y_single;
              obj.wxy{ type }{ i_simplex }{ counter_zp }( i_wb, : ) = ...
                weight1d( i_wb );
            end
            counter_zp = counter_zp + 1;
          end
        end
      end
      
      % common vertex
      type = 2;
      for i_simplex = 1 : obj.n_simplex( type )
        counter_zp = 1;
        for i_ksi = 1 : l1d
          for i_w = 1 : l3d
            w = xref3d( i_w, : );
            ksi = xref1d( i_ksi );
            [ zp_single, jac ] = ksi_w_to_zp( obj, ksi, w, type, i_simplex );
            obj.zp{ type }{ i_simplex }( counter_zp, : ) = zp_single;
            obj.wzp{ type }{ i_simplex }( counter_zp, : ) = ...
              4 * weight1d( i_ksi ) * weight3d( i_w ) * jac / 3;
            [ x_single, y_single ] = ...
              ksi_w_wb_to_x_y( obj, ksi, w, [], type, i_simplex );
            obj.x_ref{ type }{ i_simplex }{ counter_zp }( 1, : ) = x_single;
            obj.y_ref{ type }{ i_simplex }{ counter_zp }( 1, : ) = y_single;
            obj.wxy{ type }{ i_simplex }{ counter_zp }( 1 ) = 1;
            counter_zp = counter_zp + 1;
          end
        end
      end
      
    end
    
    function [ zp, jac ] = ksi_w_to_zp( obj, ksi, w, type, simplex )
      switch type
        case 2
          [ zp, jac ] = ...
            ksi_w_to_zp_vertex( obj, ksi, w( 1 ), w( 2 ), w( 3 ), simplex );
        case 3
          [ zp, jac ] = ksi_w_to_zp_edge( obj, ksi, w( 1 ), w( 2 ), simplex );
        case 4
          [ zp, jac ] = ksi_w_to_zp_identical( obj, ksi, w, simplex );
      end
    end
    
    function [ zp, jac ] = ksi_w_to_zp_identical( ~, ksi, w, simplex )
      
      jac = ksi^3*(ksi^2 - 1)^2;
      
      switch simplex
        case 1
          zp( 1 ) = ksi^2;
          zp( 2 ) = ksi^2*w;
        case 2
          zp( 1 ) = -ksi^2*(w - 1);
          zp( 2 ) = ksi^2;
        case 3
          zp( 1 ) = -ksi^2*(w - 1);
          zp( 2 ) = -ksi^2*w;
        case 4
          zp( 1 ) = ksi^2*(w - 1);
          zp( 2 ) = ksi^2*w;
        case 5
          zp( 1 ) = ksi^2*(w - 1);
          zp( 2 ) = -ksi^2;
        case 6
          zp( 1 ) = -ksi^2;
          zp( 2 ) = -ksi^2*w;
      end
    end
    
    function [ zp, jac ] = ksi_w_to_zp_edge( ~, ksi, w1, w2, simplex )
      
      jac = ksi^5 * ( 1 - ksi^2 );
      
      switch simplex
        case 1
          zp( 1 ) = ksi^2*(w1 + w2);
          zp( 2 ) = -ksi^2*(w1 + w2 - 1);
          zp( 3 ) = -ksi^2*(w1 - 1);
        case 2
          zp( 1 ) = ksi^2*w1;
          zp( 2 ) = -ksi^2*(w1 + w2 - 1);
          zp( 3 ) = ksi^2;
        case 3
          zp( 1 ) = ksi^2*w1;
          zp( 2 ) = -ksi^2*(w1 - 1);
          zp( 3 ) = -ksi^2*(w1 + w2 - 1);
        case 4
          zp( 1 ) = -ksi^2*w1;
          zp( 2 ) = -ksi^2*(w1 + w2 - 1);
          zp( 3 ) = -ksi^2*(w1 - 1);
        case 5
          zp( 1 ) = -ksi^2*w1;
          zp( 2 ) = ksi^2;
          zp( 3 ) = -ksi^2*(w1 + w2 - 1);
        case 6
          zp( 1 ) = -ksi^2*(w1 + w2);
          zp( 2 ) = -ksi^2*(w1 - 1);
          zp( 3 ) = -ksi^2*(w1 + w2 - 1);
      end
    end
    
    function [ zp, jac ] = ksi_w_to_zp_vertex( ~, ksi, w1, w2, w3, simplex )
      
      jac = ksi^7;
      
      switch simplex
        case 1
          zp( 1 ) = -ksi^2*(w2 + w3 - 1);
          zp( 2 ) = ksi^2*w1;
          zp( 3 ) = ksi^2;
          zp( 4 ) = ksi^2*(w1 + w3);
        case 2
          zp( 1 ) = -ksi^2*(w2 - 1);
          zp( 2 ) = ksi^2*w1;
          zp( 3 ) = ksi^2;
          zp( 4 ) = ksi^2*(w1 + w2 + w3);
        case 3
          zp( 1 ) = -ksi^2*(w2 - 1);
          zp( 2 ) = ksi^2*(w1 + w3);
          zp( 3 ) = ksi^2;
          zp( 4 ) = ksi^2*w1;
        case 4
          zp( 1 ) = ksi^2;
          zp( 2 ) = ksi^2*w1;
          zp( 3 ) = -ksi^2*(w2 - 1);
          zp( 4 ) = ksi^2*(w1 + w3);
        case 5
          zp( 1 ) = ksi^2;
          zp( 2 ) = ksi^2*(w1 + w2 + w3);
          zp( 3 ) = -ksi^2*(w2 - 1);
          zp( 4 ) = ksi^2*w1;
        case 6
          zp( 1 ) = ksi^2;
          zp( 2 ) = ksi^2*(w1 + w3);
          zp( 3 ) = -ksi^2*(w2 + w3 - 1);
          zp( 4 ) = ksi^2*w1;
      end
    end
    
    function [ x, y ] = ksi_w_wb_to_x_y( obj, ksi, w, wb, type, simplex )
      switch type
        case 2
          [ x, y ] = ...
            ksi_w_wb_to_x_y_vertex( obj, ksi, w( 1 ), w( 2 ), w( 3 ), simplex );
        case 3
          [ x, y ] = ...
            ksi_w_wb_to_x_y_edge( obj, ksi, w( 1 ), w( 2 ), wb, simplex );
        case 4
          [ x, y ] = ...
            ksi_w_wb_to_x_y_identical( obj, ksi, w, wb( 1 ), wb( 2 ), simplex );
      end
    end
    
    function [ x, y ] = ...
        ksi_w_wb_to_x_y_identical( ~, ksi, w, wb1, wb2, simplex )
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
    
    function [ x, y ] = ...
        ksi_w_wb_to_x_y_edge( ~, ksi, w1, w2, wb, simplex )
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
    
    function [ x, y ] = ...
        ksi_w_wb_to_x_y_vertex( ~, ksi, w1, w2, w3, simplex )
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

