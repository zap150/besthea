classdef lagrange_interpolant < handle
  %LAGRANGE_INTERPOLANT Serves for interpolation of function by L. p.
  
  properties ( Access = private ) 
    order
    nodes
  end
  
  methods
    function obj = lagrange_interpolant( order )
      obj.order = order;
      obj.nodes = zeros( 1, order + 1 );
      for i = 0 : order
        obj.nodes( i + 1 ) = cos( ( pi * ( 2 * i + 1 ) ) / ...
          ( 2 * ( obj.order + 1 ) ) );
      end
    end

    function value = lagrange( obj, i, t )
      % evaluates i-th Lagrange function on the interval [ -1, 1 ]
      value = ones(size(t));
      for k = 0 : obj.order
        if ( i ~= k )
          value = value .* ( t - obj.nodes( k + 1 ) ) ./ ...
            ( obj.nodes( i + 1 ) - obj.nodes( k + 1) );
        end 
      end
    end
    
    function nodes = get_nodes( obj )
      nodes = obj.nodes;
    end
  end
end

