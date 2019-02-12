classdef kernel_helmholtz_dl < kernel
  
  properties (Access = private)
    kappa;
  end
  
  methods
    function obj = kernel_helmholtz_dl( kappa )
      obj.kappa = kappa;
    end
    
    function value = eval( obj, x, y, n )
      xy = x - y;
      norm = sqrt( xy.^2 * [ 1; 1; 1 ] );
      value = ( xy * n' ) ./ ( 4 * pi * norm .* norm .* norm ) ...
        .* ( 1 - 1i * obj.kappa * norm ) .* exp( 1i * obj.kappa * norm );
    end
  end
  
end
