classdef p0 < basis_function

  methods
    function obj = p0( mesh )
      obj.mesh = mesh;
    end
    
    function value = dim( ~ )
      value = 1;
    end
    
    function value = dim_global( obj )
      value = obj.mesh.n_elems;
    end
    
    function value = dim_local( ~ )
      value = 1;
    end
    
    function value = eval( ~, x, ~, ~, ~, ~, ~ )
      value = ones( size( x, 1 ), 1 );
    end
    
    function value = l2g( ~, i, ~, ~, ~ )
      value = i;
    end
  end
  
end

