classdef (Abstract) basis_function
  
  properties (Access = public)
    mesh;
  end
  
  methods (Abstract)
    value = dim( obj )
    value = dim_global( obj )
    value = dim_local( obj )
    value = eval( obj, x, n, i, type, rot, swap )
    value = l2g( obj, i, type, rot, swap )
  end
end

