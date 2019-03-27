classdef (Abstract) kernel < handle
  
  methods (Abstract)
    value = eval( obj, x, y, nx, ny )
  end
end

