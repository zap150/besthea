classdef spacetime_kernel_heat_sl < spacetime_kernel & matlab.mixin.Copyable
  
  properties (Access = public)
    alpha;
    ht;
    d;
  end
  
  methods    
    function obj = spacetime_kernel_heat_sl( alpha )
      obj.alpha = alpha;
      obj.ht = 0;
      obj.d = 0;
    end
        
    %%%%% Assuming t > tau
    %%%%% Transferred to (0,1)^2
    function value = eval( obj, x, y, ~, ~, t, tau )
      norm = sqrt( ( x - y ).^2 * [ 1; 1; 1 ] );     
      rr = norm / sqrt( obj.alpha * obj.ht );
      value = ( 4 * pi * ( obj.d + t - tau ) ).^( -3/2 ) ...
        .* exp( -rr.^2 ./ ( 4 * ( obj.d + t - tau ) ) );
      value = value * sqrt( obj.ht / obj.alpha^3 );
    end
  end
  
  methods (Access = protected)
    function cp = copyElement( obj )
      cp = kernel_heat_sl( obj.alpha );
      cp.ht = obj.ht;
      cp.d = obj.d;
    end  
  end
  
end

