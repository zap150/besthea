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
        
    %%%%% Transferred to (0,1)^2
    function value = eval( obj, x, y, ~, ny, t, tau )
      ttau = obj.d + t - tau;
      mask = ( ttau <= 0 );
      value( mask, 1 ) = 0;
      mask = ~mask;
      xy = x - y;
      norm = sqrt( xy.^2 * [ 1; 1; 1 ] );   
      rr = norm / sqrt( obj.alpha * obj.ht );   
      dot = ( xy * ny' ) / sqrt( obj.alpha * obj.ht );
      value( mask, 1 ) = dot( mask, 1 ) ./ ( 16 .* pi^( 3/2 ) ) ...
        .* ttau( mask, 1 ).^( -5/2 ) ...
        .* exp( -rr( mask, 1 ).^2 ./ ( 4 * ttau( mask, 1 ) ) );
      value( mask, 1 ) = value( mask, 1 ) / obj.alpha;
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

