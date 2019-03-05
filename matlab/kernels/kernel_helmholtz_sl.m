classdef kernel_helmholtz_sl < kernel
  
  properties (Access = public)
    kappa;
  end
  
  methods
    function obj = kernel_helmholtz_sl( kappa )
      obj.kappa = kappa;
    end
    
    function value = eval( obj, x, y, ~, ~ )
      norm = sqrt( ( x - y ).^2 * [ 1; 1; 1 ] );
      value = exp( 1i * obj.kappa * norm ) ./ ( 4 * pi * norm );
    end
  end
  
end

