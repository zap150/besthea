classdef spacetime_kernel_heat_hs2 < spacetime_kernel & matlab.mixin.Copyable

  properties (Access = public)
    alpha;
    ht;
    d;
  end

  methods
    function obj = spacetime_kernel_heat_hs2( alpha )
      obj.alpha = alpha;
      obj.ht = 0;
      obj.d = 0;
    end

    %%%%% Transferred to (0,1)^2
    function value = eval( obj, x, y, nx, ny, t, tau )
      dot = nx * ny';
%       if dot == 0
      if abs( dot ) < 1e-8
        value = zeros( size( x, 1 ), 1 );
        return;
      end
      ttau = obj.d + t - tau;
      mask = ( ttau <= 0 );
      value( mask, 1 ) = 0;
      mask = ~mask;
      norm = sqrt( ( x - y ).^2 * [ 1; 1; 1 ] );
      rr = norm / sqrt( obj.alpha * obj.ht );
      rr2 = rr.^2;
      value( mask, 1 ) = exp( -rr2( mask, 1 ) ./ ( 4 * ttau( mask, 1 ) ) ) ...
        .* ( 3 ./ ttau( mask, 1 ).^( 5/2 ) ...
        - rr2( mask, 1 ) ./ ( 2 * ttau( mask, 1 ).^( 7/2 ) ) );
      value( mask, 1 ) = value( mask, 1 ) * dot * pi^( -3/2 ) / 16;
      %value( mask, 1 ) = - value( mask, 1 ) * sqrt( obj.ht / obj.alpha );
      value( mask, 1 ) = - value( mask, 1 ) * 1 / sqrt( obj.ht * obj.alpha );
    end
  end

  methods (Access = protected)
    function cp = copyElement( obj )
      cp = spacetime_kernel_heat_hs2( obj.alpha );
      cp.ht = obj.ht;
      cp.d = obj.d;
    end
  end

end
