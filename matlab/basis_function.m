classdef (Abstract) basis_function
  
  properties (Access = private)
    mesh;
  end
  
  methods (Abstract)
    value = dim_global( obj )
    value = dim_local( obj )
    value = eval( obj, x, n )
    value = l2g( obj, i, type, rot, swap )
  end
end
