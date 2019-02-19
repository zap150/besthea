classdef kernel_heat_sl < kernel
  
  properties (Access = private)
    alpha;
    ht;
    d;
  end
  
  methods    
    function obj = kernel_heat_sl( alpha, ht )
      obj.alpha = alpha;
      obj.ht = ht;
      obj.d = 0;
    end
    
    function obj = set_d( obj, d )
      obj.d = d;
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
    end
  end
  
  methods (Access = private)
    function res = G_anti_tau_anti_t( ~, rr, delta )
      if( rr == 0 && delta > 0 )
        res = sqrt( delta ) / ( 2 * pi * sqrt( pi ) );
      elseif( delta == 0 && rr > 0 )
        res = rr / ( 8 * pi );
      else
        res = ( sqrt( delta ) / ( 4 * pi ) ) ...
          * ( erf( rr / sqrt( 4 * delta ) ) ...
          * ( rr / sqrt( 4 * delta ) + sqrt( delta ) ./ rr ) ...
          + exp( - rr^2 / ( 4 * delta ) ) / sqrt( pi ) );
      end     
    end
    
    %%%%% Assuming delta == 0 && rr > 0
    function res = G_anti_tau( ~, rr )
      res = 1 ./ ( 4 * pi * rr );
    end
  end
  
end

