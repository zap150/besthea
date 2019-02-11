classdef p1 < basis_function
  
  properties (Access = private)
    mesh;
  end
  
  methods
    function obj = p0( mesh )
      obj.mesh = mesh;
    end
    
    function value = eval( ~, x, ~ )
      value( :, 1 ) = 1 - x( 1 ) - x( 2 );
      value( :, 2 ) = x( 1 );
      value( :, 3 ) = x( 2 );
    end
  end
end

