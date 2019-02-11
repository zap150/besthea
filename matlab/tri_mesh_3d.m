classdef tri_mesh_3d
  
  properties (Access = private)
    nodes;
    n_nodes;
    elems;
    n_elems;
    areas;
    normals;
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
      
    end
    
    function value = get_n_nodes( obj )
      value = obj.n_nodes;
    end
    
    function value = get_n_elems( obj )
      value = obj.n_elems;
    end
    
    function e = get_element( obj, i )
      e = obj.elems( i, : );
    end
    
    function e = get_node( obj, i )
      e = obj.nodes( i, : );
    end
    
    function e = get_nodes( obj, i )
      e = obj.nodes( obj.elems( i, : ), : );
    end
    
    function value = get_area( obj, i )
      value = obj.areas( i );
    end
    
    function value = get_normal( obj, i )
      value = obj.normals( i, : );
    end
  end
  
  methods (Access = private)
    function obj = init_areas( obj )
      obj.areas = zeros( obj.n_elems, 1 );
      for i = 1 : obj.n_elems
        e = obj.get_nodes( i );
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
        e = obj.get_nodes( i );
        u = e( 2, : ) - e( 1, : );
        v = e( 3, : ) - e( 1, : );
        obj.normals( i, 1 ) = u( 2 ) * v( 3 ) - u( 3 ) * v( 2 );
        obj.normals( i, 2 ) = u( 3 ) * v( 1 ) - u( 1 ) * v( 3 );
        obj.normals( i, 3 ) = u( 1 ) * v( 2 ) - u( 2 ) * v( 1 );
        norm = sqrt( obj.normals( i, : ) * obj.normals( i, : )' );
        obj.normals( i, : ) = obj.normals( i, : ) / norm;
      end
    end
  end
end

