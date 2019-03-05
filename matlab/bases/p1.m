classdef p1 < basis_function
  
  properties (Constant)      
    map = [ 1 2 3 1 2 ];
  end
  
  methods
    function obj = p1( mesh )
      obj.mesh = mesh;
    end
    
    function value = dim( ~ )
      value = 1;
    end
    
    function value = dim_global( obj )
      value = obj.mesh.n_nodes;
    end
    
    function value = dim_local( ~ )
      value = 3;
    end
    
    function value = eval( ~, x, ~, ~, ~, ~, ~ )
      value( :, 1 ) = 1 - x( :, 1 ) - x( :, 2 );
      value( :, 2 ) = x( :, 1 );
      value( :, 3 ) = x( :, 2 );
    end
    
    function value = l2g( obj, i, type, rot, swap )  
      if nargin < 3
        type = 1;
        rot = 0;
        swap = false;
      end
      
      nodes = obj.mesh.elems( i, : );
      if type == 3 && swap
        value( 1 ) = nodes( obj.map( rot + 2 ) );
        value( 2 ) = nodes( obj.map( rot + 1 ) );
      else
        value( 1 ) = nodes( obj.map( rot + 1 ) );
        value( 2 ) = nodes( obj.map( rot + 2 ) );
      end
      value( 3 ) = nodes( obj.map( rot + 3 ) );
    end
  end
  
end

