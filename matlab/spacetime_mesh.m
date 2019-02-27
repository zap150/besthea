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
    
    function value = get_hx( obj )
      value = obj.get_h( );
    end
    
    function node = get_time_node( obj, i )
      node = ( i - 1 ) * obj.ht;
    end
    
    function nodes = get_time_nodes( obj, i )
      nodes = [ ( i - 1 ) * obj.ht i * obj.ht ];
    end
    
    function obj = refine_xt( obj, level, order )
      if nargin < 3
        order = 2;
      end
      if nargin < 2
        level = 1;
      end
      obj = obj.refine_x( level );
      obj = obj.refine_t( level * order );
    end
    
    function obj = refine_x( obj, level )
      if nargin < 2
        level = 1;
      end
      obj = obj.refine( level );
    end
    
    function obj = refine_t( obj, level )
      if nargin < 2
        level = 1;
      end
      obj.nt = 2^level * obj.nt;
      obj.ht = obj.T / obj.nt;
    end
  end

end
