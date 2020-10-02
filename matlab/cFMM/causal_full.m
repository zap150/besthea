function [result, V] = causal_full( T, nT, rhs_fun )
%CAUSAL_FULL Summary of this function goes here
%   Detailed explanation goes here

    function V = assemble_matrix( end_time, n_steps )
        v = zeros( nT, 1 );
        V = zeros( nT, nT );
        for i = 0 : nT - 1
            v( i + 1 ) = Vd( i, end_time / n_steps );
        end
        for i = 1 : nT
            V( i : end, i ) = v( 1 : ( nT - i + 1 ) );
        end
    end

    function vd = Vd( d, ht )
        vd = sqrt( ht^3 ) * ( VV( d + 1, ht ) - 2 * VV( d, ht ) ...
          + VV( d - 1, ht ) );
    end

    function vv = VV( d, ht )
        if d <= 0
            vv = 0;
        else
            vv = sqrt( ( 4 * d ) / ( 9 * pi ) ) * ( d + ( 1 / sqrt( ht ) ) ...
              * ( sqrt( pi / d ) * ( 1 / ht ) + 1.5 * ...
                sqrt( pi * d ) ) * ( erfc( ( 1 / sqrt( ht ) ) / ...
                sqrt( d ) ) ) - ( d + ( 1 / ht ) ) * exp( - ( 1 / ht ) / d ) );
        end
    end
  
    function projection = project( rhs, T, nT )
      diag = ones( nT,1 ) * ( T / nT );
      M = spdiags( diag, 0, nT, nT );
      [ x, w, l ] = quadratures.line( 4 );
      
      ht = T / nT;
      proj_rhs = zeros( nT, 1 );
      for i = 1 : nT
        for j = 1 : l
          proj_rhs( i ) = proj_rhs( i ) + rhs( ( i - 1 ) * ht + ...
            x ( j ) * ht ) * w( j );
        end
        proj_rhs( i ) = proj_rhs( i ) * ht;
      end
      projection = M \ proj_rhs;
    end

    V = assemble_matrix( T, nT );
    ht = T / nT;

    rhs_proj = project( rhs_fun, T, nT );
    rhs_proj = rhs_proj * ht;

    result = V \ rhs_proj;
    
end



%     rhs_fun = @( t ) ( ( exp( t ) / 4 ) ...
%       .* ( exp( 2 ) * erfc( ( 1 + t ) ./ ( sqrt( t ) ) ) ...
%       + 2 * erf( sqrt( t ) - exp( -2 ) * erfc( ( 1 - t ) ... 
%       ./ ( sqrt( t ) ) ) ) ) );

%     rhs_fun = @( t ) sin(8*pi*t)*exp(-t);

%     result = zeros( size(rhs_proj ));
%     step = 2;    
%     for i = 1 : size(rhs_proj, 1) / step
%       f = zeros( step, 1 );
%       for j = 1 : i - 1
%         f = f + V( (i - 1 ) * step + 1 : i * step, (j - 1) * step + 1 : j * step) * ...
%           result((j - 1) * step + 1 : j * step);
%       end
%       rhs_proj((i - 1 ) * step + 1 : i * step) = rhs_proj((i - 1 ) * step + 1 : i * step) - f;
%       result((i - 1 ) * step + 1 : i * step) = ...
%         V( (i-1)*step + 1 : i * step, (i-1)*step + 1 : i * step )\ ...
%         rhs_proj((i - 1 ) * step + 1 : i * step);
%     end
%    result = V\rhs_proj;
%     figure
%     hold on
%     t=ht/2:ht:T-ht/2;
%    plot(t, result);
    %plot(t, exp(t));
%    hold off;