classdef kernel_heat_sl < kernel
  
  properties (Access = private)
    alpha;
    ht;
    nt;
    d;
  end
  
  methods    
    function obj = kernel_heat_sl( alpha )
      obj.alpha = alpha;
      obj.ht = 0;
      obj.nt = 0;
      obj.d = 0;
    end
    
    function obj = set_d( obj, d )
      obj.d = d;
    end
    
    function obj = set_ht( obj, ht )
      obj.ht = ht;
    end
    
    function obj = set_nt( obj, nt )
      obj.nt = nt;
    end
    
    function value = eval( obj, x, y, ~ )
      norm = sqrt( ( x - y ).^2 * [ 1; 1; 1 ] );
      rr = norm / sqrt( obj.alpha * obj.ht );      
      if obj.d > 0
        value = - obj.G_anti_tau_anti_t( rr, obj.d + 1 ) ...
          + 2 * obj.G_anti_tau_anti_t( rr, obj.d ) ...
          - obj.G_anti_tau_anti_t( rr, obj.d - 1 );
      else
        value = - obj.G_anti_tau_anti_t( rr, 1 ) ...
          + obj.G_anti_tau_anti_t( rr, 0 ) + obj.G_anti_tau( rr );
      end
      value = value * sqrt( obj.ht / obj.alpha^3 );
    end
  end
  
  methods (Access = private)
    function res = G_anti_tau_anti_t( obj, rr, delta )
      if( delta > 0 )
        res = obj.G_anti_tau_anti_t_regular( rr, delta );
      else
        res = obj.G_anti_tau_anti_t_limit( rr );
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
    %%%%% Assuming delta == 0 && rr > 0
    function res = G_anti_tau( ~, rr )
      res = 1 ./ ( 4 * pi * rr );
    end
  end
  
end

