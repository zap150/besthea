classdef kernel_heat_hs2 < kernel & matlab.mixin.Copyable
  
  properties (Access = public)
    alpha;
    ht;
    d;
  end
  
  methods    
    function obj = kernel_heat_hs2( alpha )
      obj.alpha = alpha;
      obj.ht = 0;
      obj.d = 0;
    end
        
    function value = eval( obj, x, y, nx, ny )
      dot = nx * ny';
%       if dot == 0
      if abs( dot ) < 1e-8
        value = zeros( size( x, 1 ), 1 );
        return;
      end
      norm = sqrt( ( x - y ).^2 * [ 1; 1; 1 ] );
      rr = norm / sqrt( obj.alpha * obj.ht );      
      if obj.d > 0
        value = - G_anti_t( obj, rr, obj.d + 1 ) ...
          + 2 * G_anti_t( obj, rr, obj.d ) - G_anti_t( obj, rr, obj.d - 1 );
      else
        value = - G_anti_t( obj, rr, 1 ) + G_anti_t( obj, rr, 0 );
      end
      value = - value * dot * 1 / sqrt( obj.alpha * obj.ht );
    end
  end
  
  methods (Access = protected)
    function cp = copyElement( obj )
      cp = kernel_heat_hs2( obj.alpha );
      cp.ht = obj.ht;
      cp.d = obj.d;
    end
    
    function res = G_anti_t( obj, rr, delta )
      if( delta > 0 )
        res = G_anti_t_regular( obj, rr, delta );
      else
        res = G_anti_t_limit( obj, rr );
      end     
    end
 
    %%%%% int G dt
    %%%%% delta > 0, rr > 0 or limit for rr -> 0
    function res = G_anti_t_regular( ~, rr, delta )
      sqrt_d = sqrt( delta );
      mask = ( rr == 0 );
      res( mask, 1 ) = -1 / ( 4 * pi * sqrt( pi * delta ) );
      mask = ~mask;
      res( mask, 1 ) = -erf( rr( mask ) / ( 2 * sqrt_d ) ) ...
        ./ ( 4 * pi * rr( mask ) );
    end 
 
    %%%%% int G dt    
    %%%%% Limit for delta -> 0, assuming rr > 0
    function res = G_anti_t_limit( ~, rr )
        res = -1 ./ ( 4 * pi * rr );   
    end
 
  end
  
end

