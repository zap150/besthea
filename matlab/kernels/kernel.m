classdef (Abstract) kernel
  
  methods (Abstract)
    value = eval( obj, x, y, n )
  end
end

