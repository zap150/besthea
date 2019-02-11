classdef tri_mesh_3d
  
  properties (Access = private)
    nodes;
    elems;
    areas;
  end
  
  properties (Dependent)
    n_nodes;
    n_elems;
  end
  
  methods
    function obj = tri_mesh_3d( file )
      fid = fopen( file );
      
      % skip first lines
      fgetl( fid );
      fgetl( fid );
      fgetl( fid );
      
      n_nodes = str2num( fgetl( fid ) );
      obj.nodes = zeros( n_nodes, 3 );
      for i = 1 : n_nodes
        line = fgets( fid );
        row = textscan( line, '%f' );
        obj.nodes( i, : ) = row{ 1 };
      end
      
      % skip empty line
      fgetl( fid );
      
      n_elems = str2num( fgetl( fid ) );
      obj.elems = zeros( n_elems, 3 );
      for i = 1 : n_elems
        line = fgets( fid );
        row = textscan( line, '%d' );
        obj.elems( i, : ) = row{ 1 };
      end
      
      fclose( fid );
      
      obj = obj.init_areas( );
      
    end
    
    function value = get.n_nodes( obj )
      value = size( obj.nodes, 1 );
    end
    
    function value = get.n_elems( obj )
      value = size( obj.elems, 1 );
    end
    
    function e = get_element( obj, i )
      e = obj.elems( i, : ) + 1;
    end
    
    function e = get_node( obj, i )
      e = obj.nodes( i, : );
    end
    
    function e = get_nodes( obj, i )
      e = obj.nodes( obj.elems( i, : ) + 1, : );
    end
    
    function value = get_area( obj, i )
      value = obj.areas( i );
    end
  end
  
  methods (Access = private)
    function obj = init_areas( obj )
      obj.areas = zeros( obj.n_elems, 1 );
      for i = 1 : obj.n_elems
        e = obj.get_nodes( i );
        u = e( 1, : ) - e( 2, : );
        v = e( 3, : ) - e( 2, : );
        s( 1 ) = u( 2 ) * v( 3 ) - u( 3 ) * v( 2 );
        s( 2 ) = u( 3 ) * v( 1 ) - u( 1 ) * v( 3 );
        s( 3 ) = u( 1 ) * v( 2 ) - u( 2 ) * v( 1 );     
        obj.areas( i ) = 0.5 * sqrt( s * s' );
      end
    end
  end
end

