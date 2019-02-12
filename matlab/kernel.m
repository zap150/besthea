classdef (Abstract) kernel
  
  methods (Abstract)
    value = eval( obj, x, y, n, data )
  end
end

