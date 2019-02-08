classdef be_integrator
  
  properties (Access = private)
    mesh;
    kernel;
    order_nf;
    order_ff;
    
    % Tausch
    % n_simplex = [ 6 2 4 1 ];
    % Sauter, Schwab
    n_simplex = [ 6 2 5 1 ];
    
    x1_ref_identical;
    x2_ref_identical;
    y1_ref_identical;
    y2_ref_identical;
    w_identical;
    
    x1_ref_edge;
    x2_ref_edge;
    y1_ref_edge;
    y2_ref_edge;
    w_edge;
    
    x1_ref_vertex;
    x2_ref_vertex;
    y1_ref_vertex;
    y2_ref_vertex;
    w_vertex;
    
    x1_ref_disjoint;
    x2_ref_disjoint;
    y1_ref_disjoint;
    y2_ref_disjoint;
    w_disjoint;
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
          
          [ type, rot_test, rot_trial  ] = obj.type( i_test, i_trial );
                    
          value = 0;      
          for i_simplex = 1 : obj.n_simplex( type + 1 )
            [ x, y ] = global_quad( obj, i_test, i_trial, type, ...
              rot_test, rot_trial, i_simplex );
            k = obj.kernel.eval( x, y )';
            switch type
              case 0
                value = value + k * obj.w_identical( :, i_simplex );
              case 1
                value = value + k * obj.w_vertex( :, i_simplex );
              case 2
                value = value + k * obj.w_edge( :, i_simplex );
              case 3
                value = value + k * obj.w_disjoint( :, i_simplex );
            end
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

      nodes = obj.mesh.get_nodes( i_trial );          
      z1 = nodes( map( rot_test + 1 ), : );
      z2 = nodes( map( rot_test + 2 ), : );
      z3 = nodes( map( rot_test + 3 ), : );
      R = [ z2 - z1; z3 - z1 ];
      switch type
        case 0
          x = [ obj.x1_ref_identical( :, i_simplex ) ...
            obj.x2_ref_identical( :, i_simplex ) ] * R;
        case 1
          x = [ obj.x1_ref_vertex( :, i_simplex ) ...
            obj.x2_ref_vertex( :, i_simplex ) ] * R;
        case 2
          x = [ obj.x1_ref_edge( :, i_simplex ) ...
            obj.x2_ref_edge( :, i_simplex ) ] * R;
        case 3
          x = [ obj.x1_ref_disjoint obj.x2_ref_disjoint ] * R;
      end
      x = x + z1;
      
      nodes = obj.mesh.get_nodes( i_test );
      z1 = nodes( map( rot_trial + 1 ), : );
      z2 = nodes( map( rot_trial + 2 ), : );
      % inverting trial element
      if type == 2
        z1 = nodes( map( rot_trial + 2 ), : );
        z2 = nodes( map( rot_trial + 1 ), : );
      end
      z3 = nodes( map( rot_trial + 3 ), : );
      R = [ z2 - z1; z3 - z1 ];
      switch type
        case 0
          y = [ obj.y1_ref_identical( :, i_simplex ) ...
            obj.y2_ref_identical( :, i_simplex ) ] * R;
        case 1
          y = [ obj.y1_ref_vertex( :, i_simplex ) ...
            obj.y2_ref_vertex( :, i_simplex ) ] * R;
        case 2
          y = [ obj.y1_ref_edge( :, i_simplex ) ...
            obj.y2_ref_edge( :, i_simplex ) ] * R;
        case 3
          y = [ obj.y1_ref_disjoint obj.y2_ref_disjoint ] * R;
      end
      y = y + z1;
      
    end
    
    function [ type, rot_test, rot_trial ] = ...
        type( obj, i_test, i_trial )
      
      rot_test = 0;
      rot_trial = 0;
      
      % identical
      if i_test == i_trial
        type = 0;
        return;
      end
      
      elem_test = obj.mesh.get_element( i_test );
      elem_trial = obj.mesh.get_element( i_trial );
      
      [ c, c_test, c_trial ] = intersect( elem_test, elem_trial );
      
      nc = length( c );
      
      % disjoint
      if nc == 0
        type = 3;
        return
      end
      
      % edge
      if nc == 2
        type = 2;
        map = [ 2 1 0 2 1 0 ];
        rot_test = map( c_test( 1 ) + c_test( 2 ) );
        rot_trial = map( c_trial( 1 ) + c_test( 2 ) );
        return;
      end
      
      % vertex
      if nc == 1
        type = 1;
        rot_test = c_test - 1;
        rot_trial = c_trial - 1;
        return;
      end
      
    end
    
    function obj = init_quadrature_data( obj )
      size = obj.order_nf * obj.order_nf * obj.order_nf * obj.order_nf;
      
      type = 0;
      ns = obj.n_simplex( type + 1 );
      obj.x1_ref_identical = zeros( size, ns );
      obj.x2_ref_identical = zeros( size, ns );
      obj.y1_ref_identical = zeros( size, ns );
      obj.y2_ref_identical = zeros( size, ns );
      obj.w_identical = zeros( size, ns );
      
      type = 2;
      ns = obj.n_simplex( type + 1 );
      obj.x1_ref_edge = zeros( size, ns );
      obj.x2_ref_edge = zeros( size, ns );
      obj.y1_ref_edge = zeros( size, ns );
      obj.y2_ref_edge = zeros( size, ns );
      obj.w_edge = zeros( size, ns );
      
      type = 1;
      ns = obj.n_simplex( type + 1 );
      obj.x1_ref_vertex = zeros( size, ns );
      obj.x2_ref_vertex = zeros( size, ns );
      obj.y1_ref_vertex = zeros( size, ns );
      obj.y2_ref_vertex = zeros( size, ns );
      obj.w_vertex = zeros( size, ns );
      
      type = 3;
      ns = obj.n_simplex( type + 1 );
      obj.x1_ref_disjoint = zeros( size, ns );
      obj.x2_ref_disjoint = zeros( size, ns );
      obj.y1_ref_disjoint = zeros( size, ns );
      obj.y2_ref_disjoint = zeros( size, ns );
      obj.w_disjoint = zeros( size, ns );
      
      [ x, w, l ] = quadratures.line( obj.order_nf );
      
      counter = 1;
      for i_ksi = 1 : l
        for i_eta1 = 1 : l
          for i_eta2 = 1 : l
            for i_eta3 = 1 : l
              
              weight = ...
                w( i_ksi ) * w( i_eta1 ) * w( i_eta2 ) * w( i_eta3 );
              
              type = 0;
              ns = obj.n_simplex( type + 1 );
              for i_simplex = 1 : ns
                [ x_ref, y_ref, jac ] = obj.cube_to_tri_identical( ...
                  x( i_ksi ), x( i_eta1 ), x( i_eta2 ), x( i_eta3 ), ...
                  i_simplex );
                obj.x1_ref_identical( counter, i_simplex ) = x_ref( 1 );
                obj.x2_ref_identical( counter, i_simplex ) = x_ref( 2 );
                obj.y1_ref_identical( counter, i_simplex ) = y_ref( 1 );
                obj.y2_ref_identical( counter, i_simplex ) = y_ref( 2 );
                obj.w_identical( counter, i_simplex ) = weight * jac;
              end
              
              type = 2;
              ns = obj.n_simplex( type + 1 );
              for i_simplex = 1 : ns
                [ x_ref, y_ref, jac ] = obj.cube_to_tri_edge( ...
                  x( i_ksi ), x( i_eta1 ), x( i_eta2 ), x( i_eta3 ), ...
                  i_simplex );
                obj.x1_ref_edge( counter, i_simplex ) = x_ref( 1 );
                obj.x2_ref_edge( counter, i_simplex ) = x_ref( 2 );
                obj.y1_ref_edge( counter, i_simplex ) = y_ref( 1 );
                obj.y2_ref_edge( counter, i_simplex ) = y_ref( 2 );
                obj.w_edge( counter, i_simplex ) = weight * jac;
              end
              
              type = 1;
              ns = obj.n_simplex( type + 1 );
              for i_simplex = 1 : ns
                [ x_ref, y_ref, jac ] = obj.cube_to_tri_vertex( ...
                  x( i_ksi ), x( i_eta1 ), x( i_eta2 ), x( i_eta3 ), ...
                  i_simplex );
                obj.x1_ref_vertex( counter, i_simplex ) = x_ref( 1 );
                obj.x2_ref_vertex( counter, i_simplex ) = x_ref( 2 );
                obj.y1_ref_vertex( counter, i_simplex ) = y_ref( 1 );
                obj.y2_ref_vertex( counter, i_simplex ) = y_ref( 2 );
                obj.w_vertex( counter, i_simplex ) = weight * jac;
              end
              
              type = 3;
              ns = obj.n_simplex( type + 1 );
              for i_simplex = 1 : ns
                [ x_ref, y_ref, jac ] = obj.cube_to_tri_disjoint( ...
                  x( i_ksi ), x( i_eta1 ), x( i_eta2 ), x( i_eta3 ) );
                obj.x1_ref_disjoint( counter, i_simplex ) = x_ref( 1 );
                obj.x2_ref_disjoint( counter, i_simplex ) = x_ref( 2 );
                obj.y1_ref_disjoint( counter, i_simplex ) = y_ref( 1 );
                obj.y2_ref_disjoint( counter, i_simplex ) = y_ref( 2 );
                obj.w_disjoint( counter, i_simplex ) = weight * jac;
              end
              
              counter = counter + 1;
            end
          end
        end
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

