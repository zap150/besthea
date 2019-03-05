classdef curl_p1 < basis_function
  
  properties (Constant)      
    map = [ 1 2 3 1 2 ];
  end
  
  methods
    function obj = curl_p1( mesh )
      obj.mesh = mesh;
    end
    
    function value = dim( ~ )
      value = 3;
    end
    
    function value = dim_global( obj )
      value = obj.mesh.n_nodes;
    end
    
    function value = dim_local( ~ )
      value = 3;
    end
    
    function value = eval( obj, ~, n, i, type, rot, swap )
      if( type == 1 )
        r_tinv = obj.mesh.r_tinv{ i };
      else
        nodes = obj.mesh.nodes( obj.mesh.elems( i, : ), : );
        if type == 3 && swap
          z1 = nodes( obj.map( rot + 2 ), : );
          z2 = nodes( obj.map( rot + 1 ), : );
        else
          z1 = nodes( obj.map( rot + 1 ), : );
          z2 = nodes( obj.map( rot + 2 ), : );
        end
        z3 = nodes( obj.map( rot + 3 ), : );
        r_tinv = inv( [ z2 - z1; z3 - z1; n ] );
      end
      g = ( r_tinv * [ -1; -1; 0 ] )';
      value( 1, 1 ) = ( n( 2 ) * g( 3 ) - n( 3 ) * g( 2 ) );
      value( 1, 2 ) = ( n( 3 ) * g( 1 ) - n( 1 ) * g( 3 ) );
      value( 1, 3 ) = ( n( 1 ) * g( 2 ) - n( 2 ) * g( 1 ) );
      g = ( r_tinv * [ 1; 0; 0 ] )';
      value( 1, 4 ) = ( n( 2 ) * g( 3 ) - n( 3 ) * g( 2 ) );
      value( 1, 5 ) = ( n( 3 ) * g( 1 ) - n( 1 ) * g( 3 ) );
      value( 1, 6 ) = ( n( 1 ) * g( 2 ) - n( 2 ) * g( 1 ) );
      g = ( r_tinv * [ 0; 1; 0 ] )';
      value( 1, 7 ) = ( n( 2 ) * g( 3 ) - n( 3 ) * g( 2 ) );
      value( 1, 8 ) = ( n( 3 ) * g( 1 ) - n( 1 ) * g( 3 ) );
      value( 1, 9 ) = ( n( 1 ) * g( 2 ) - n( 2 ) * g( 1 ) );
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

