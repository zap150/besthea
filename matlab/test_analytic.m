function [ res1_num, res_anal_1, res_anal_2 ] = test_analytic( r, d )

if( nargin < 2 )
  r = 0;
  d = 1;
end
alpha = 2.9;
ht = 0.17;
order = 32;
n = [ 2 4 -1 ]';
n = n / norm( n );
x = [ 1 2 3 ]';
xy = ( r / norm( x ) ) * x;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rrpn = ( n' * xy ) / sqrt( alpha * ht );
rr = r / sqrt( alpha * ht );

% if d > 0
%   res_anal_1 = - Gm2_1( rr, d + 1 ) + 2 * Gm2_1( rr, d ) - Gm2_1( rr, d - 1 );
% else
%   res_anal_1 = - Gm2_1( rr, 1 ) + Gm2_1( rr, 0 ) + Gm1_1( rr, 0 );
% end
% 
% if d > 0
%   res_anal_2 = - Gm2_2( rr, d + 1 ) + 2 * Gm2_2( rr, d ) - Gm2_2( rr, d - 1 );
% else
%   res_anal_2 = - Gm2_2( rr, 1 );
% end
% 
% res1_num = V_num( rr, d, order );

if d > 0
  res_anal_1 = 1 / alpha ...
    * ( - dnGm2_1( rr, rrpn, d + 1 ) + 2 * dnGm2_1( rr, rrpn, d ) ...
    - dnGm2_1( rr, rrpn, d - 1 ) );
else
  res_anal_1 = 1 / alpha ...
    * ( - dnGm2_1( rr, rrpn, 1 ) + dnGm2_1( rr, rrpn, 0 ) ...
    + dnGm1_1( rr, rrpn, 0 ) );
end

res1_num = K_num( rr, rrpn, d, order ) / alpha;


end

function res = V_num( rr, d, order )

[ tt, w, l ] = quadratures.line( order );
res = 0;
for i_t = 1 : l
  for i_tau = 1 : l
    res = res + w( i_t ) * w( i_tau ) ...
      * G1( rr, d + tt( i_t ) - tt( i_tau ) );
  end
end

end 

function res = K_num( rr, rrpn, d, order )

[ tt, w, l ] = quadratures.line( order );
res = 0;
for i_t = 1 : l
  for i_tau = 1 : l
    delta = d + tt( i_t ) - tt( i_tau );
    res = res + w( i_t ) * w( i_tau ) ...
      * dnG1( rr, rrpn, delta );
  end
end

end 

function res = Gm2_1( rr, delta )

if( rr == 0 && delta > 0 )
  res = sqrt( delta ) / ( 2 * pi^( 3/2 ) );
elseif( delta == 0 && rr > 0 )
  res = rr / ( 8 * pi );
else
  res = ( sqrt( delta ) / ( 4 * pi ) ) * ( erf( rr / sqrt( 4 * delta ) ) ...
   * ( rr / sqrt( 4 * delta ) + sqrt( delta ) / rr ) ...
   + exp( - rr^2 / ( 4 * delta ) ) / sqrt( pi ) );
end

end

function res = dnGm2_1( rr, rrpn, delta )

%%%%% Assuming integration over the same spatial element
if( rr == 0 && delta > 0 )
  res = 0;
elseif( delta == 0 && rr > 0 )
  res = - rrpn / ( 8 * pi * rr );
else
  res = -( sqrt( delta ) * rrpn ) / ( 4 * pi * rr^2 ) ...
    * ( erf( rr / sqrt( 4 * delta ) ) ...
    * ( rr / sqrt( 4 * delta ) - sqrt( delta ) / rr ) ...
    + exp( - rr^2 / ( 4 * delta ) ) / sqrt( pi ) );
end

end

function res = Gm1_1( rr, delta )

if( delta == 0 && rr > 0 )
  res = 1 / ( 4 * pi * rr );
end

end

function res = dnGm1_1( rr, rrpn, delta )

if( delta == 0 && rr > 0 )
  res = rrpn / ( 4 * pi * rr^3 );
end

end

function res = Gm2_2( rr, delta )

if( delta == 0 && rr > 0 )
  res = 0;
else
  res = ( sqrt( delta ) / ( 4 * pi ) ) * ( -erfc( rr / sqrt( 4 * delta ) ) ...
    * ( rr / sqrt( 4 * delta ) + sqrt( delta ) / rr ) ...
    + exp( - rr^2 / ( 4 * delta ) ) / sqrt( pi ) );
end

end

function res = G1( rr, delta )

if(delta > 0)
  res = ( 4 * pi * ( delta ) )^( -3/2 ) * exp( - rr^2 / ( 4 * ( delta ) ) );
else
  res = 0;
end

end

function res = dnG1( rr, rrpn, delta )

if(delta > 0)
  res = ( 4 * pi * ( delta ) )^( -3/2 ) * exp( - rr^2 / ( 4 * ( delta ) ) ) ... 
    * rrpn / ( 2 * delta );
else
  res = 0;
end

end
  