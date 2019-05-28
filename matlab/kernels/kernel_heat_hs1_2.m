classdef kernel_heat_hs1_2 < kernel & matlab.mixin.Copyable
  
  properties (Access = public)
    alpha;
    ht;
    d;
  end
  
  methods    
    function obj = kernel_heat_hs1_2( alpha )
      obj.alpha = alpha;
      obj.ht = 0;
      obj.d = 0;
    end
        
    function value = eval( obj, x, y, ~, ~ )
      norm = sqrt( ( x - y ).^2 * [ 1; 1; 1 ] );     
      if obj.d > 0
        value = - G_anti_tau_anti_t( obj, norm, ( obj.d + 1 ) * obj.ht ) ...
          + 2 * G_anti_tau_anti_t( obj, norm, obj.d * obj.ht ) ...
          - G_anti_tau_anti_t( obj, norm, ( obj.d - 1 ) * obj.ht );
      else
        value = - G_anti_tau_anti_t( obj, norm, obj.ht ) ...
          + G_anti_tau_anti_t( obj, norm, 0 ) ...
          + obj.ht * G_anti_tau_limit( obj, norm );
      end
      value = value * obj.alpha^2;
    end
    
  end
  
  methods (Access = protected)
    function cp = copyElement( obj )
      cp = kernel_heat_hs1_2( obj.alpha );
      cp.ht = obj.ht;
      cp.d = obj.d;
    end

    function res = G_anti_tau_anti_t( obj, norm, delta )
      if( delta > 0 )
        res = G_anti_tau_anti_t_regular( obj, norm, delta );
      else
        res = G_anti_tau_anti_t_limit( obj, norm );
      end     
    end
     
    %%%%% int int G dtau dt
    %%%%% delta > 0, norm > 0 or limit for norm -> 0
    function res = G_anti_tau_anti_t_regular( obj, norm, delta )
      sqrt_d = sqrt( delta );
      sqrt_pi_a = sqrt( pi * obj.alpha );
      mask = ( norm == 0 );
      res( mask, 1 ) = sqrt_d / ( 2 * pi * obj.alpha * sqrt_pi_a );
      mask = ~mask;
      res( mask, 1 ) = ( delta ./ ( 4 * pi * obj.alpha * norm( mask ) ) ...
        + norm( mask ) / ( 8 * pi * obj.alpha^2 ) ) ...
        .* erf( norm( mask ) / ( 2 * sqrt_d * sqrt( obj.alpha ) ) ) ...
        + sqrt_d / ( 4 * pi * obj.alpha * sqrt_pi_a ) ...
        .* exp( - norm( mask ).^2 / ( 4 * delta * obj.alpha ) );
    end 
 
    %%%%% int int G dtau dt    
    %%%%% Limit for delta -> 0, assuming norm > 0
    function res = G_anti_tau_anti_t_limit( obj, norm )
        res = norm / ( 8 * pi * obj.alpha^2 );   
    end
 
    %%%%% int G dtau      
    %%%%% Limit for delta -> 0, assuming norm > 0
    function res = G_anti_tau_limit( obj, norm )
      res = 1 ./ ( 4 * pi * obj.alpha * norm );
    end
  end
  
end

