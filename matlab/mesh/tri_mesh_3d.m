classdef tri_mesh_3d
  
  properties (Access = public)
    nodes;
    n_nodes;
    elems;
    n_elems;
    
    edges;
    n_edges;
    elem_to_edges;
    
    areas;
    normals;
    
    r_tinv;
  end
  
  properties (Dependent)
    h;
  end
    
  methods
    function obj = tri_mesh_3d( file )
      fid = fopen( file );
      
      % skip first lines
      fgetl( fid );
      fgetl( fid );
      fgetl( fid );
      
      obj.n_nodes = str2double( fgetl( fid ) );
      obj.nodes = zeros( obj.n_nodes, 3 );
      for i = 1 : obj.n_nodes
        line = fgets( fid );
        row = textscan( line, '%f' );
        obj.nodes( i, : ) = row{ 1 };
      end
      
      % skip empty line
      fgetl( fid );
      
      obj.n_elems = str2double( fgetl( fid ) );
      obj.elems = zeros( obj.n_elems, 3 );
      for i = 1 : obj.n_elems
        line = fgets( fid );
        row = textscan( line, '%d' );
        obj.elems( i, : ) = row{ 1 } + 1;
      end
      
      fclose( fid );
      
      obj = obj.init_areas( );
      obj = obj.init_normals( );  
      obj = obj.init_edges( );
      obj = obj.init_r_tinv( );
    end
    
%     function e = get_nodes( obj, i )
%       e = obj.nodes( obj.elems( i, : ), : );
%     end
    
    function h = get.h( obj )
      h = max( sqrt( obj.areas ) );
    end
    
    function obj = refine( obj, level )
      
      if nargin == 1
        level = 1;
      end
      
      for l = 1 : level
        new_n_elems = 4 * obj.n_elems;
        new_elems = zeros( new_n_elems, 3 );
        new_n_nodes = obj.n_nodes + obj.n_edges;
        new_nodes = zeros( new_n_nodes, 3 );
        new_n_edges = 2 * obj.n_edges + 3 * obj.n_elems;
        new_edges = zeros( new_n_edges, 2 );
        new_elem_to_edges = zeros( new_n_elems, 3 );
        
        new_nodes( 1 : obj.n_nodes, : ) = obj.nodes;
        for i = 1 : obj.n_edges
          edge = obj.edges( i, : );
          x1 = obj.nodes( edge( 1 ), : );
          x2 = obj.nodes( edge( 2 ), : );
          new_nodes( obj.n_nodes + i, : ) = ( x1 + x2 ) / 2;
          new_edges( 2 * i - 1, : ) = [ edge( 1 ) obj.n_nodes + i ];
          new_edges( 2 * i, : ) = [ edge( 2 ) obj.n_nodes + i ];
        end
        
        for i = 1 : obj.n_elems
          i_element = obj.elems( i, : );
          i_edges = obj.elem_to_edges( i, : );
          
          node1 = i_element( 1 );
          node2 = i_element( 2 );
          node3 = i_element( 3 );
          node4 = obj.n_nodes + i_edges( 1 );
          node5 = obj.n_nodes + i_edges( 2 );
          node6 = obj.n_nodes + i_edges( 3 );
          
          new_elems( 4 * i - 3, : ) = [ node1 node4 node6 ];
          new_elems( 4 * i - 2, : ) = [ node4 node2 node5 ];
          new_elems( 4 * i - 1, : ) = [ node5 node3 node6 ];
          new_elems( 4 * i - 0, : ) = [ node4 node5 node6 ];
          
          new_edges( 2 * obj.n_edges + 3 * i - 2, : ) = sort( [ node4 node5 ] );
          new_edges( 2 * obj.n_edges + 3 * i - 1, : ) = sort( [ node5 node6 ] );
          new_edges( 2 * obj.n_edges + 3 * i - 0, : ) = sort( [ node4 node6 ] );
          
          new_elem_to_edges( 4 * i - 3, : ) = [ 2 * i_edges( 1 ) - 1 ...
            2 * obj.n_edges + 3 * i - 0 ...
            2 * i_edges( 3 ) - 1 ];
          
          new_elem_to_edges( 4 * i - 2, : ) = [ 2 * i_edges( 1 ) - 1 ...
            2 * i_edges( 2 ) - 1 ...
            2 * obj.n_edges + 3 * i - 2 ];
          
          new_elem_to_edges( 4 * i - 1, : ) = [ 2 * i_edges( 2 ) - 1 ...
            2 * i_edges( 3 ) - 1 ...
            2 * obj.n_edges + 3 * i - 1 ];
          
          new_elem_to_edges( 4 * i - 0, : ) = [ 2 * obj.n_edges + 3 * i - 2 ...
            2 * obj.n_edges + 3 * i - 1 ...
            2 * obj.n_edges + 3 * i - 0 ];
          
          if node1 > node2
            new_elem_to_edges( 4 * i - 3, 1 ) = ...
              new_elem_to_edges( 4 * i - 3, 1 ) + 1;
          else
            new_elem_to_edges( 4 * i - 2, 1 ) = ...
              new_elem_to_edges( 4 * i - 2, 1 ) + 1;            
          end
          
          if node1 > node3
            new_elem_to_edges( 4 * i - 3, 3 ) = ...
              new_elem_to_edges( 4 * i - 3, 3 ) + 1;
          else
            new_elem_to_edges( 4 * i - 1, 2 ) = ...
              new_elem_to_edges( 4 * i - 1, 2 ) + 1;
          end
          
          if node2 > node3
            new_elem_to_edges( 4 * i - 2, 2 ) = ...
              new_elem_to_edges( 4 * i - 2, 2 ) + 1;
          else
            new_elem_to_edges( 4 * i - 1, 1 ) = ...
              new_elem_to_edges( 4 * i - 1, 1 ) + 1;
          end
        end
        
        obj.n_elems = new_n_elems;
        obj.elems = new_elems;
        obj.n_nodes = new_n_nodes;
        obj.nodes = new_nodes;
        obj.n_edges = new_n_edges;
        obj.edges = new_edges;
        obj.elem_to_edges = new_elem_to_edges;
      end
    
      obj = obj.init_areas( );
      obj = obj.init_normals( );
      obj = obj.init_r_tinv( );
    end
    
    function plot( obj, data, name )
      if nargin < 2
        data = zeros( obj.n_elems, 1 );
      end
      if nargin < 3
        name = '';
      end
      figure;
      axis equal;
      colormap( 'jet' );
      handle = trisurf( obj.elems, obj.nodes( :, 1 ), obj.nodes( :, 2 ), ...
        obj.nodes( :, 3 ), data, 'EdgeColor', 'black' );
      if( size( data, 1 ) == obj.n_nodes )
        shading( 'interp' );
        set( handle, 'EdgeColor', 'black' );
      end
      title( name );
    end
  end
  
  methods (Access = private)
    function obj = init_areas( obj )
      obj.areas = zeros( obj.n_elems, 1 );
      for i = 1 : obj.n_elems
        e = obj.nodes( obj.elems( i, : ), : );
        u = e( 2, : ) - e( 1, : );
        v = e( 3, : ) - e( 1, : );
        s( 1 ) = u( 2 ) * v( 3 ) - u( 3 ) * v( 2 );
        s( 2 ) = u( 3 ) * v( 1 ) - u( 1 ) * v( 3 );
        s( 3 ) = u( 1 ) * v( 2 ) - u( 2 ) * v( 1 );     
        obj.areas( i ) = 0.5 * sqrt( s * s' );
      end
    end
    
    function obj = init_normals( obj )
      obj.normals = zeros( obj.n_elems, 1 );
      for i = 1 : obj.n_elems
        e = obj.nodes( obj.elems( i, : ), : );
        u = e( 2, : ) - e( 1, : );
        v = e( 3, : ) - e( 1, : );
        obj.normals( i, 1 ) = u( 2 ) * v( 3 ) - u( 3 ) * v( 2 );
        obj.normals( i, 2 ) = u( 3 ) * v( 1 ) - u( 1 ) * v( 3 );
        obj.normals( i, 3 ) = u( 1 ) * v( 2 ) - u( 2 ) * v( 1 );
        norm = sqrt( obj.normals( i, : ) * obj.normals( i, : )' );
        obj.normals( i, : ) = obj.normals( i, : ) / norm;
      end
    end
    
    function obj = init_edges( obj )
      [ obj.edges, ~, ic ] = unique( sort( [ ...
        obj.elems( :, [ 1 2 ] )
        obj.elems( :, [ 2 3 ] )
        obj.elems( :, [ 3 1 ] ) ], 2 ), 'rows' );
      obj.n_edges = size( obj.edges, 1 );
      obj.elem_to_edges = reshape( ic, obj.n_elems, 3 );
    end
    
    function obj = init_r_tinv( obj )
      obj.r_tinv = cell( obj.n_elems, 1 );
      for i = 1 : obj.n_elems
        nod = obj.nodes( obj.elems( i, : ), : );
        obj.r_tinv{ i } = inv( [ ...
          nod( 2, : ) - nod( 1, : )
          nod( 3, : ) - nod( 1, : )
          obj.normals( i, : ) ...
          ] );
      end
    end
  end
end

