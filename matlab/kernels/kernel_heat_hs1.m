classdef kernel_heat_hs1 < kernel
  
  properties (Access = public)
    alpha;
    ht;
    nt;
    d;
  end
  
  methods    
    function obj = kernel_heat_hs1( alpha )
      obj.alpha = alpha;
      obj.ht = 0;
      obj.nt = 0;
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
      value = value * sqrt( obj.ht * obj.alpha );
    end
  end
  
  methods (Access = private)
    function res = G_anti_tau_anti_t( obj, rr, delta )
      if( delta > 0 )
        res = G_anti_tau_anti_t_regular( obj, rr, delta );
      else
        res = G_anti_tau_anti_t_limit( obj, rr );
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
    %%%%% Limit for delta -> 0, assuming rr > 0
    function res = G_anti_tau_limit( ~, rr )
      res = 1 ./ ( 4 * pi * rr );
    end
  end
  
end

