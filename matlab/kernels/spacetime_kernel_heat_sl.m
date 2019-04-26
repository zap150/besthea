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
        
    %%%%% Transferred to (0,1)^2
    function value = eval( obj, x, y, ~, ~, t, tau )
      ttau = obj.d + t - tau;
      mask = ( ttau <= 0 );
      value( mask, 1 ) = 0;
      mask = ~mask;
      norm = sqrt( ( x - y ).^2 * [ 1; 1; 1 ] );     
      rr = norm / sqrt( obj.alpha * obj.ht );
      value( mask, 1 ) = ( 4 * pi * ttau( mask, 1 ) ).^( -3/2 ) ...
        .* exp( -rr( mask, 1 ).^2 ./ ( 4 * ttau( mask, 1 ) ) );
      value( mask, 1 ) = value( mask, 1 ) * sqrt( obj.ht / obj.alpha^3 );
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

