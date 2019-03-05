classdef (Abstract) kernel
  
  methods (Abstract)
    value = eval( obj, x, y, nx, ny )
  end
end

