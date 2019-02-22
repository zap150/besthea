classdef spacetime_mesh < tri_mesh_3d
  
  properties (Access = private)
    T;
    nt;
    ht;
  end
  
  methods
    function obj = spacetime_mesh( file, T, nt )
      obj = obj@tri_mesh_3d( file );
      
      obj.T = T;
      obj.nt = nt;
      obj.ht = T / nt;
    end
    
    function value = get_nt( obj )
      value = obj.nt;
    end
    
    function value = get_ht( obj )
      value = obj.ht;
    end
    
    function node = get_time_node( obj, i )
      node = ( i - 1 ) * obj.ht;
    end
    
    function nodes = get_time_nodes( obj, i )
      nodes = [ ( i - 1 ) * obj.ht i * obj.ht ];
    end
  end

end
