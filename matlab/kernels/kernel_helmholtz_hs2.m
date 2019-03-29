classdef kernel_helmholtz_hs2 < kernel
  
  properties (Access = public)
    kappa;
  end
  
  methods
    function obj = kernel_helmholtz_hs2( kappa )
      obj.kappa = kappa;
    end
    
    function value = eval( obj, x, y, nx, ny )
      dot = nx * ny';
%       if dot == 0
      if abs( dot ) < 1e-8
        value = zeros( size( x, 1 ), 1 );
        return;
      end
      norm = sqrt( ( x - y ).^2 * [ 1; 1; 1 ] );
      value = exp( 1i * obj.kappa * norm ) ./ ( 4 * pi * norm );
      value = -value * dot * obj.kappa * obj.kappa;
    end
  end
  
end

