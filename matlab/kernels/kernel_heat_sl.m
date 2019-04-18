classdef kernel_heat_sl < kernel & matlab.mixin.Copyable
  
  properties (Access = public)
    alpha;
    ht;
    d;
  end
  
  methods    
    function obj = kernel_heat_sl( alpha )
      obj.alpha = alpha;
      obj.ht = 0;
      obj.d = 0;
    end
        
    function value = eval( obj, x, y, ~, ~ )
      norm = sqrt( ( x - y ).^2 * [ 1; 1; 1 ] );
      rr = norm / sqrt( obj.alpha * obj.ht );      
      if obj.d > 0
        value = - G_anti_tau_anti_t( obj, rr, obj.d + 1 ) ...
          + 2 * G_anti_tau_anti_t( obj, rr, obj.d ) ...
          - G_anti_tau_anti_t( obj, rr, obj.d - 1 );
      else
        value = - G_anti_tau_anti_t( obj, rr, 1 ) ...
          + G_anti_tau_anti_t( obj, rr, 0 ) + G_anti_tau_limit( obj, rr );
      end
      value = value * sqrt( obj.ht / obj.alpha^3 );
    end
    
    function value = eval_repr( obj, x, y, ~ )
      norm = sqrt( ( x - y ).^2 * [ 1; 1; 1 ] );
      rr = norm / sqrt( obj.alpha * obj.ht );      
      value = G_anti_tau( obj, rr, obj.d - 1 ) ...
          - G_anti_tau( obj, rr, obj.d );
      value = value / sqrt( obj.ht * obj.alpha^3 );
    end
  end
  
  methods (Access = protected)
    function cp = copyElement( obj )
      cp = kernel_heat_sl( obj.alpha );
      cp.ht = obj.ht;
      cp.d = obj.d;
    end

    function res = G_anti_tau_anti_t( obj, rr, delta )
      if( delta > 0 )
        res = G_anti_tau_anti_t_regular( obj, rr, delta );
      else
        res = G_anti_tau_anti_t_limit( obj, rr );
      end     
    end
    
    function res = G_anti_tau( obj, rr, delta )
      if( delta > 0 )
        res = G_anti_tau_regular( obj, rr, delta );
      else
        res = G_anti_tau_limit( obj, rr );
      end   
    end    
 
    %%%%% int int G dtau dt
    %%%%% delta > 0, rr > 0 or limit for rr -> 0
    function res = G_anti_tau_anti_t_regular( ~, rr, delta )
      sqrt_d = sqrt( delta );
      mask = ( rr == 0 );
      res( mask, 1 ) = sqrt_d / ( 2 * pi * sqrt( pi ) );
      mask = ~mask;
      res( mask, 1 ) = ( sqrt_d / ( 4 * pi ) ) ...
        * ( erf( rr( mask ) / ( 2 * sqrt_d ) ) ...
        .* ( rr( mask ) / ( 2 * sqrt_d ) + sqrt_d ./ rr( mask ) ) ...
        + exp( - rr( mask ).^2 / ( 4 * delta ) ) / sqrt( pi ) );
    end 
 
    %%%%% int int G dtau dt    
    %%%%% Limit for delta -> 0, assuming rr > 0
    function res = G_anti_tau_anti_t_limit( ~, rr )
        res = rr / ( 8 * pi );   
    end
    
    %%%%% int G dtau      
    %%%%% delta > 0 && rr > 0
    function res = G_anti_tau_regular( ~, rr, delta )
      res = erf( rr / sqrt( 4 * delta ) ) ./ ( 4 * pi * rr );
    end
 
    %%%%% int G dtau      
    %%%%% Limit for delta -> 0, assuming rr > 0
    function res = G_anti_tau_limit( ~, rr )
      res = 1 ./ ( 4 * pi * rr );
    end
  end
  
end

