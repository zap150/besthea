classdef kernel_heat_adl_2 < kernel & matlab.mixin.Copyable
  
  properties (Access = public)
    alpha;
    ht;
    d;
  end
  
  methods    
    function obj = kernel_heat_adl_2( alpha )
      obj.alpha = alpha;
      obj.ht = 0;
      obj.d = 0;
    end
        
    function value = eval( obj, x, y, nx, ~ )
      xy = x - y;
      norm = sqrt( xy.^2 * [ 1; 1; 1 ] );
      dot = -xy * nx';
      if obj.d > 0
        value = ...
          - dnG_anti_tau_anti_t( obj, norm, dot, ( obj.d + 1 ) * obj.ht ) ...
          + 2 * dnG_anti_tau_anti_t( obj, norm, dot, obj.d * obj.ht ) ...
          - dnG_anti_tau_anti_t( obj, norm, dot, ( obj.d - 1 ) * obj.ht );
      else
        value = - dnG_anti_tau_anti_t( obj, norm, dot, obj.ht ) ...
          + dnG_anti_tau_anti_t( obj, norm, dot, 0 ) ...
          + obj.ht * dnG_anti_tau_limit( obj, norm, dot );
      end
    end
  end
  
  methods (Access = protected)
    function cp = copyElement( obj )
      cp = kernel_heat_dl_2( obj.alpha );
      cp.ht = obj.ht;
      cp.d = obj.d;
    end
    
    function res = dnG_anti_tau_anti_t( obj, norm, dot, delta )
      if( delta > 0 )
        res = dnG_anti_tau_anti_t_regular( obj, norm, dot, delta );
      else
        res = dnG_anti_tau_anti_t_limit( obj, norm, dot );
      end     
    end
 
    %%%%% int int dG/dn dtau dt
    %%%%% delta > 0, norm > 0 or integration over the same element (or plane)
    function res = dnG_anti_tau_anti_t_regular( obj, norm, dot, delta )
      sqrt_d = sqrt( delta );
      sqrt_a = sqrt( obj.alpha );
      %%%%% Integration over the same element (or plane)
      mask = ( dot == 0 );
      res( mask, 1 ) = 0;
      mask = ~mask;
      res( mask, 1 ) = -dot( mask ) ./ ( 4 * pi * norm( mask ) ) ...
        .* ( ( 1 / ( 2 * obj.alpha ) - delta ./ norm( mask ).^2 ) ...
        .* erf( norm( mask ) / ( 2 * sqrt_d * sqrt_a ) ) ...
        + sqrt_d ./ ( sqrt( pi ) * sqrt_a * norm( mask ) ) ...
        .* exp( - norm( mask ).^2 / ( 4 * obj.alpha * delta ) ) );
    end
 
    %%%%% int int dG/dn dtau dt    
    %%%%% Limit for delta -> 0, assuming norm > 0
    function res = dnG_anti_tau_anti_t_limit( obj, norm, dot )
        res = - dot ./ ( 8 * pi * norm * obj.alpha );  
    end   
    
    %%%%% int dG/dn dtau      
    %%%%% Limit for delta -> 0, assuming norm > 0
    function res = dnG_anti_tau_limit( ~, norm, dot )
      res = dot ./ ( 4 * pi * norm.^3 );
    end
  end
  
end

