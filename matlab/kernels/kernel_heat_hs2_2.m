classdef kernel_heat_hs2_2 < kernel & matlab.mixin.Copyable
  
  properties (Access = public)
    alpha;
    ht;
    d;
  end
  
  methods    
    function obj = kernel_heat_hs2_2( alpha )
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
      if obj.d > 0
        value = - G_anti_t( obj, norm, ( obj.d + 1 ) * obj.ht ) ...
          + 2 * G_anti_t( obj, norm, obj.d * obj.ht ) ...
          - G_anti_t( obj, norm, ( obj.d - 1 ) * obj.ht );
      else
        value = - G_anti_t( obj, norm, obj.ht ) + G_anti_t( obj, norm, 0 );
      end
      value = - value * dot * obj.alpha;
    end
  end
  
  methods (Access = protected)
    function cp = copyElement( obj )
      cp = kernel_heat_hs2_2( obj.alpha );
      cp.ht = obj.ht;
      cp.d = obj.d;
    end
    
    function res = G_anti_t( obj, norm, delta )
      if( delta > 0 )
        res = G_anti_t_regular( obj, norm, delta );
      else
        res = G_anti_t_limit( obj, norm );
      end     
    end
 
    %%%%% int G dt
    %%%%% delta > 0, norm > 0 or limit for norm -> 0
    function res = G_anti_t_regular( obj, norm, delta )
      mask = ( norm == 0 );
      res( mask, 1 ) = ...
        -1 / ( 4 * pi * obj.alpha * sqrt( pi * delta * obj.alpha ) );
      mask = ~mask;
      res( mask, 1 ) = -erf( norm( mask ) / sqrt( 4 * obj.alpha * delta ) ) ...
        ./ ( 4 * pi * obj.alpha * norm( mask ) );
    end 
 
    %%%%% int G dt    
    %%%%% Limit for delta -> 0, assuming norm > 0
    function res = G_anti_t_limit( obj, norm )
        res = -1 ./ ( 4 * pi * obj.alpha * norm );   
    end
 
  end
  
end

