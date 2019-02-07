classdef (Abstract) kernel
  
  methods (Abstract)
    value = eval( obj, x, y )
    value = eval_derivative( obj, x, y, n )
  end
end

