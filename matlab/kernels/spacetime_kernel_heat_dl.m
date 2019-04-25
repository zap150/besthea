classdef spacetime_kernel_heat_dl < spacetime_kernel & matlab.mixin.Copyable
  
  properties (Access = public)
    alpha;
    ht;
    d;
  end
  
  methods    
    function obj = spacetime_kernel_heat_dl( alpha )
      obj.alpha = alpha;
      obj.ht = 0;
      obj.d = 0;
    end
        
    %%%%% Assuming t > tau
    %%%%% Transferred to (0,1)^2
    function value = eval( obj, x, y, ~, ny, t, tau )
      xy = x - y;
      norm = sqrt( xy.^2 * [ 1; 1; 1 ] );   
      rr = norm / sqrt( obj.alpha * obj.ht );   
      dot = ( xy * ny' ) / sqrt( obj.alpha * obj.ht );
      value = dot ./ ( 16 .* pi^( 3/2 ) ) .* ( obj.d + t - tau ).^( -5/2 ) ...
        .* exp( -rr.^2 ./ ( 4 * ( obj.d + t - tau ) ) );
      value = value / obj.alpha;
    end
  end
  
  methods (Access = protected)
    function cp = copyElement( obj )
      cp = kernel_heat_dl( obj.alpha );
      cp.ht = obj.ht;
      cp.d = obj.d;
    end  
  end
  
end

