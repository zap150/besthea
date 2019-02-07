classdef be_integrator
  
  properties (Access = private)
    mesh;
    kernel;
    order_nf;
    order_ff;
    
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
      V = zeros( obj.mesh.n_elems, obj.mesh.n_elems );
    end
    
  end
  
  methods (Access = private)
    function [ type, rot_test, rot_trial ] = type( obj, i_test, i_trial )
      rot_test = 0;
      rot_trial = 0;
      
      if i_test == i_trial
        type = 0;
        return;
      end
      
      elem_test = obj.mesh.get_element( i_test );
      elem_trial = obj.mesh.get_element( i_trial );
      
      [ c, c_test, c_trial ] = intersect( elem_test, elem_trial );
      
      nc = length( c );
      
      if nc == 0
        type = 3;
        return
      end
      
      if nc == 2
        
        return;
      end
      
      if nc == 1
        
        return;
      end
      
    end
    
    function obj = init_quadrature_data( obj )
      size = obj.order_nf * obj.order_nf * obj.order_nf * obj.order_nf;
      
      obj.x1_ref_identical = zeros( size, 6 );
      obj.x2_ref_identical = zeros( size, 6 );
      obj.y1_ref_identical = zeros( size, 6 );
      obj.y2_ref_identical = zeros( size, 6 );
      obj.w_identical = zeros( size, 6 );
      
      obj.x1_ref_edge = zeros( size, 4 );
      obj.x2_ref_edge = zeros( size, 4 );
      obj.y1_ref_edge = zeros( size, 4 );
      obj.y2_ref_edge = zeros( size, 4 );
      obj.w_edge = zeros( size, 4 );
      
      obj.x1_ref_vertex = zeros( size, 2 );
      obj.x2_ref_vertex = zeros( size, 2 );
      obj.y1_ref_vertex = zeros( size, 2 );
      obj.y2_ref_vertex = zeros( size, 2 );
      obj.w_vertex = zeros( size, 2 );
      
      obj.x1_ref_disjoint = zeros( size, 1 );
      obj.x2_ref_disjoint = zeros( size, 1 );
      obj.y1_ref_disjoint = zeros( size, 1 );
      obj.y2_ref_disjoint = zeros( size, 1 );
      obj.w_disjoint = zeros( size, 1 );
      
      [ x, w, l ] = quadratures.line( obj.order_nf );
      
      counter = 1;
      for i_ksi = 1 : l
        for i_eta1 = 1 : l
          for i_eta2 = 1 : l
            for i_eta3 = 1 : l
              
              weight = ...
                w( i_ksi ) * w( i_eta1 ) * w( i_eta2 ) * w( i_eta3 );
              
              for i_simplex = 1 : 6
                [ x_ref, y_ref, jac ] = obj.cube_to_tri_identical( ...
                  x( i_ksi ), x( i_eta1 ), x( i_eta2 ), x( i_eta3 ), ...
                  i_simplex );
                obj.x1_ref_identical( counter, i_simplex ) = x_ref( 1 );
                obj.x2_ref_identical( counter, i_simplex ) = x_ref( 2 );
                obj.y1_ref_identical( counter, i_simplex ) = y_ref( 1 );
                obj.y2_ref_identical( counter, i_simplex ) = y_ref( 2 );
                obj.w_identical( counter, i_simplex ) = weight * jac;
              end
              
              for i_simplex = 1 : 4
                [ x_ref, y_ref, jac ] = obj.cube_to_tri_edge( ...
                  x( i_ksi ), x( i_eta1 ), x( i_eta2 ), x( i_eta3 ), ...
                  i_simplex );
                obj.x1_ref_edge( counter, i_simplex ) = x_ref( 1 );
                obj.x2_ref_edge( counter, i_simplex ) = x_ref( 2 );
                obj.y1_ref_edge( counter, i_simplex ) = y_ref( 1 );
                obj.y2_ref_edge( counter, i_simplex ) = y_ref( 2 );
                obj.w_edge( counter, i_simplex ) = weight * jac;
              end
              
              for i_simplex = 1 : 2
                [ x_ref, y_ref, jac ] = obj.cube_to_tri_vertex( ...
                  x( i_ksi ), x( i_eta1 ), x( i_eta2 ), x( i_eta3 ), ...
                  i_simplex );
                obj.x1_ref_vertex( counter, i_simplex ) = x_ref( 1 );
                obj.x2_ref_vertex( counter, i_simplex ) = x_ref( 2 );
                obj.y1_ref_vertex( counter, i_simplex ) = y_ref( 1 );
                obj.y2_ref_vertex( counter, i_simplex ) = y_ref( 2 );
                obj.w_vertex( counter, i_simplex ) = weight * jac;
              end
              
              [ x_ref, y_ref, jac ] = obj.cube_to_tri_disjoint( ...
                x( i_ksi ), x( i_eta1 ), x( i_eta2 ), x( i_eta3 ) );
              obj.x1_ref_disjoint( counter ) = x_ref( 1 );
              obj.x2_ref_disjoint( counter ) = x_ref( 2 );
              obj.y1_ref_disjoint( counter ) = y_ref( 1 );
              obj.y2_ref_disjoint( counter ) = y_ref( 2 );
              obj.w_disjoint( counter ) = weight * jac;
              
              counter = counter + 1;
            end
          end
        end
      end
      
      scatter( obj.x1_ref_disjoint( : ), obj.x2_ref_disjoint( : ) );
      
    end
    
    function [ x, y, jac ] = ...
        cube_to_tri_identical( ~, ksi, eta1, eta2, eta3, simplex )
      
      switch simplex
        case 1
          x( 1 ) = ( 1 - ksi ) * eta2 * ( 1 - eta3 );
          x( 2 ) = ksi + ( 1 - ksi ) * eta2 * eta3;
          y( 1 ) = x( 1 ) + ksi * ( 1 - eta1 );
          y( 2 ) = x( 2 ) - ksi;
        case 2
          x( 1 ) = ksi * ( 1 - eta1 ) + ( 1 - ksi ) * eta2 * ( 1 - eta3 );
          x( 2 ) = ksi * eta1 + ( 1 - ksi ) * eta2 * eta3;
          y( 1 ) = x( 1 ) + ksi * ( 1 - eta1 );
          y( 2 ) = x( 2 ) - ksi;
        case 3
          x( 1 ) = ksi + ( 1 - ksi ) * eta2 * ( 1 - eta3 );
          x( 2 ) = ( 1 - ksi ) * eta2 * eta3;
          y( 1 ) = x( 1 ) - ksi;
          y( 2 ) = x( 2 ) + ksi * ( 1 - eta1 );
        case 4
          x( 1 ) = ( 1 - ksi ) * eta2 * ( 1 - eta3 );
          x( 2 ) = ksi * eta1 + ( 1 - ksi ) * eta2 * eta3;
          y( 1 ) = x( 1 ) + ksi;
          y( 2 ) = x( 2 ) - ksi * eta1;
        case 5
          x( 1 ) = ( 1 - ksi ) * eta2 * ( 1 - eta3 );
          x( 2 ) = ( 1 - ksi ) * eta2 * eta3;
          y( 1 ) = x( 1 ) + ksi * ( 1 - eta1 );
          y( 2 ) = x( 2 ) + ksi * eta1;
        case 6
          x( 1 ) = ksi * ( 1 - eta1 ) + ( 1 - ksi ) * eta2 * ( 1 - eta3 );
          x( 2 ) = ( 1 - ksi ) * eta2 * eta3;
          y( 1 ) = x( 1 ) - ksi * ( 1 - eta1 );
          y( 2 ) = x( 2 ) + ksi;
      end
      
      jac = ksi * ( 1 - ksi ) * ( 1 - ksi ) * eta2;
      
    end
    
    function [ x, y, jac ] = ...
        cube_to_tri_edge( ~, ksi, eta1, eta2, eta3, simplex )
      
      switch simplex
        case 1
          x( 1 ) = ksi * ( 1 - eta2 ) * ( 1 - ksi ) * eta3;
          x( 2 ) = ksi * eta2;
          y( 1 ) = ( 1 - ksi ) * eta3;
          y( 2 ) = ksi * ( 1 - eta1 );
          jac = ( 1 - ksi ) * ksi * ksi;
        case 2
          x( 1 ) = ( 1 - ksi ) * eta3;
          x( 2 ) = ksi;
          y( 1 ) = ksi * ( 1 - eta2 ) + ( 1 - ksi ) * eta3;
          y( 2 ) = ksi * ( 1 - eta1 ) * eta2;
          jac = ( 1 - ksi ) * ksi * ksi * eta2;
        case 3
          x( 1 ) = ( 1 - 2 * eta1 ) * ksi + ( 1 - ksi ) * eta3;
          x( 2 ) = ksi * eta1;
          y( 1 ) = ksi * ( 2 - 2 * eta1 - eta2 ) + ( 1 - ksi ) * eta3;
          y( 2 ) = ksi * eta2;
          jac = ( 1 - ksi ) * ksi * ksi;
        case 4
          x( 1 ) = ksi * ( 1 - eta2 ) + ( 1 - ksi ) * eta3;
          x( 2 ) = ksi * ( 1 - eta1 ) * eta2;
          y( 1 ) = ( 1 - ksi ) * eta3;
          y( 2 ) = ksi;
          jac = ( 1 - ksi ) * ksi * ksi * eta2;
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

