classdef p0 < basis_function

  properties (Access = private)
    mesh;
  end  
  
  methods
    function obj = p0( mesh )
      obj.mesh = mesh;
    end
    
    function value = dim_global( obj )
      value = obj.mesh.n_elems;
    end
    
    function value = dim_local( ~ )
      value = 1;
    end
    
    function value = eval( ~, x, ~ )
      value = ones( length( x ), 1 );
    end
    
    function value = map( ~, i, ~, ~, ~ )
      value = i;
    end
  end
end

