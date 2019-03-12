classdef kernel_heat_dl < kernel
  
  properties (Access = public)
    alpha;
    ht;
    nt;
    d;
  end
  
  methods    
    function obj = kernel_heat_dl( alpha )
      obj.alpha = alpha;
      obj.ht = 0;
      obj.nt = 0;
      obj.d = 0;
    end
        
    function value = eval( obj, x, y, ~, ny )
      xy = x - y;
      norm = sqrt( xy.^2 * [ 1; 1; 1 ] );
      rr = norm / sqrt( obj.alpha * obj.ht );   
      dot = ( xy * ny' ) / sqrt( obj.alpha * obj.ht );
      if obj.d > 0
        value = - obj.dnG_anti_tau_anti_t( rr, dot, obj.d + 1 ) ...
          + 2 * obj.dnG_anti_tau_anti_t( rr, dot, obj.d ) ...
          - obj.dnG_anti_tau_anti_t( rr, dot, obj.d - 1 );
      else
        value = - obj.dnG_anti_tau_anti_t( rr, dot, 1 ) ...
          + obj.dnG_anti_tau_anti_t( rr, dot, 0 ) + obj.dnG_anti_tau( rr, dot );
      end
      value = value / obj.alpha;
    end
  end
  
  methods (Access = private)
    function res = dnG_anti_tau_anti_t( obj, rr, dot, delta )
      if( delta > 0 )
        res = obj.dnG_anti_tau_anti_t_regular( rr, dot, delta );
      else
        res = obj.dnG_anti_tau_anti_t_limit( rr, dot );
      end     
    end
 
    %%%%% int int dG/dn dtau dt
    %%%%% delta > 0, rr > 0 or integration over the same element (or plane)
    function res = dnG_anti_tau_anti_t_regular( ~, rr, dot, delta )
      sqrt_d = sqrt( delta );
      %%%%% Integration over the same element (or plane)
      mask = ( dot == 0 );
      res( mask, 1 ) = 0;
      mask = ~mask;
      res( mask, 1 ) = -( sqrt_d * dot( mask ) ) ...
        ./ ( 4 * pi * rr( mask ).^2 ) ...
        .* ( erf( rr( mask ) / ( 2 * sqrt_d ) ) ...
        .* ( rr( mask ) / ( 2 * sqrt_d ) - sqrt_d ./ rr( mask ) ) ...
        + exp( - rr( mask ).^2 / ( 4 * delta ) ) / sqrt( pi ) );
    end
 
    %%%%% int int dG/dn dtau dt    
    %%%%% Limit for delta -> 0, assuming rr > 0
    function res = dnG_anti_tau_anti_t_limit( ~, rr, dot )
        res = - dot ./ ( 8 * pi * rr );  
    end
 
    %%%%% int dG/dn dtau      
    %%%%% Assuming delta == 0 && rr > 0
    function res = dnG_anti_tau( ~, rr, dot )
      res = dot ./ ( 4 * pi * rr.^3 );
    end
  end
  
end
