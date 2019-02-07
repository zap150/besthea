classdef tri_mesh_3d
  
  properties (Access = private)
    nodes;
    elems;
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
      
    end
    
    function value = get.n_nodes( obj )
      value = size( obj.nodes, 1 );
    end
    
    function value = get.n_elems( obj )
      value = size( obj.elems, 1 );
    end
    
    function e = get_element( obj, i )
      e = obj.elems( i, : );
    end
  end
end

