classdef (Abstract) spacetime_kernel < handle
  
  methods (Abstract)
    value = eval( obj, x, y, nx, ny, t, tau )
  end
end

