classdef be_integrator
  
  properties (Access = private)
    mesh;
    kernel;
    order_nf;
    order_ff;
    
    % type = 1 ... disjoint
    % type = 2 ... vertex
    % type = 3 ... edge
    % type = 4 ... identical
    
    % Tausch
    % n_simplex = [ 1 2 4 6 ];
    % Sauter, Schwab
    n_simplex = [ 1 2 5 6 ];
    
    x1_ref = cell( 4, 1 );
    x2_ref = cell( 4, 1 );
    y1_ref = cell( 4, 1 );
    y2_ref = cell( 4, 1 );
    w = cell( 4, 1 );

  end
  
  methods
    function obj = be_integrator( mesh, kernel, order_nf, order_ff )
      
      obj.mesh = mesh;
      obj.kernel = kernel;
      
      if( nargin < 4 )
        obj.order_nf = 4;
        obj.order_ff = 4;
      else
        obj.order_nf = order_nf;
        obj.order_ff = order_ff;
      end
      
      obj = init_quadrature_data( obj );
      
    end
    
    function V = assemble_v_p0_p0( obj )
      n_elems = obj.mesh.n_elems;
      
      V = zeros( n_elems, n_elems );
      
      for i_trial = 1 : n_elems
        for i_test = 1 : n_elems
          
          [ type, rot_test, rot_trial  ] = obj.get_type( i_test, i_trial );
                    
          value = 0;      
          for i_simplex = 1 : obj.n_simplex( type )
            [ x, y ] = global_quad( obj, i_test, i_trial, type, ...
              rot_test, rot_trial, i_simplex );
            k = obj.kernel.eval( x, y )';
            value = value + k * obj.w{ type }( :, i_simplex );
          end
          
          V( i_test, i_trial ) = value * 4 ...
            * obj.mesh.get_area( i_trial ) * obj.mesh.get_area( i_test );
        end
      end
           
    end
    
  end
  
  methods (Access = private)
    function [ x, y ] = global_quad( obj, i_test, i_trial, type, ... 
        rot_test, rot_trial, i_simplex )
      
      map = [ 1 2 3 1 2 ];

      nodes = obj.mesh.get_nodes( i_test );          
      z1 = nodes( map( rot_test + 1 ), : );
      z2 = nodes( map( rot_test + 2 ), : );
      z3 = nodes( map( rot_test + 3 ), : );
      R = [ z2 - z1; z3 - z1 ];
      x = [ obj.x1_ref{ type }( :, i_simplex ) ...
        obj.x2_ref{ type }( :, i_simplex ) ] * R;     
      x = x + z1;
      
      nodes = obj.mesh.get_nodes( i_trial );
      % inverting trial element
      if type == 3
        z1 = nodes( map( rot_trial + 2 ), : );
        z2 = nodes( map( rot_trial + 1 ), : );
      else
        z1 = nodes( map( rot_trial + 1 ), : );
        z2 = nodes( map( rot_trial + 2 ), : );
      end
      z3 = nodes( map( rot_trial + 3 ), : );
      R = [ z2 - z1; z3 - z1 ];
      y = [ obj.y1_ref{ type }( :, i_simplex ) ...
        obj.y2_ref{ type }( :, i_simplex ) ] * R;
      y = y + z1;
      
    end
    
    function [ type, rot_test, rot_trial ] = ...
        get_type( obj, i_test, i_trial )
      
      rot_test = 0;
      rot_trial = 0;
      
      % identical
      if i_test == i_trial
        type = 4;
        return;
      end
      
      elem_test = obj.mesh.get_element( i_test );
      elem_trial = obj.mesh.get_element( i_trial );
      
      [ c, c_test, c_trial ] = intersect( elem_test, elem_trial );
      
      nc = length( c );
      
      % disjoint
      if nc == 0
        type = 1;
        return
      end
      
      % edge
      if nc == 2
        type = 3;
        map = [ 2 1 0 2 1 0 ];
        rot_test = map( c_test( 1 ) + c_test( 2 ) );
        rot_trial = map( c_trial( 1 ) + c_trial( 2 ) );
        return;
      end
      
      % vertex
      if nc == 1
        type = 2;
        rot_test = c_test - 1;
        rot_trial = c_trial - 1;
        return;
      end
      
    end
    
    function obj = init_quadrature_data( obj )
      
      size = obj.order_nf * obj.order_nf * obj.order_nf * obj.order_nf;      
      for type = 1 : 4
        ns = obj.n_simplex( type );
        obj.x1_ref{ type } = zeros( size, ns );
        obj.x2_ref{ type } = zeros( size, ns );
        obj.y1_ref{ type } = zeros( size, ns );
        obj.y2_ref{ type } = zeros( size, ns );
        obj.w{ type } = zeros( size, ns );
      end
      
      [ x_line, w_line, l_line ] = quadratures.line( obj.order_nf );
      
      counter = 1;
      for i_ksi = 1 : l_line
        for i_eta1 = 1 : l_line
          for i_eta2 = 1 : l_line
            for i_eta3 = 1 : l_line
              
              weight = w_line( i_ksi ) * w_line( i_eta1 ) ...
                * w_line( i_eta2 ) * w_line( i_eta3 );
              
              for type = 1 : 4
                ns = obj.n_simplex( type );
                for i_simplex = 1 : ns
                  [ x_ref, y_ref, jac ] = obj.cube_to_tri( ...
                    x_line( i_ksi ), x_line( i_eta1 ), ...
                    x_line( i_eta2 ), x_line( i_eta3 ), ...
                    type, i_simplex );
                  obj.x1_ref{ type }( counter, i_simplex ) = ...
                    x_ref( 1 );
                  obj.x2_ref{ type }( counter, i_simplex ) = ...
                    x_ref( 2 );
                  obj.y1_ref{ type }( counter, i_simplex ) = ...
                    y_ref( 1 );
                  obj.y2_ref{ type }( counter, i_simplex ) = ...
                    y_ref( 2 );
                  obj.w{ type }( counter, i_simplex ) = weight * jac;
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
%       
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

