function [ res1_num, res_anal_1, res_anal_2 ] = test_analytic( r, d )

[ tt, w, l ] = quadratures.line( 32 );

if( nargin < 2 )
  r = 0;
  d = 1;
end
alpha = 2;
ht = 0.1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rr = r / sqrt( alpha * ht );

if d > 0
  res_anal_1 = - Gm2_1( rr, d + 1 ) + 2 * Gm2_1( rr, d ) - Gm2_1( rr, d - 1 );
else
  res_anal_1 = - Gm2_1( rr, 1 ) + Gm2_1( rr, 0 ) + Gm1_1( rr, 0 );
end

if d > 0
  res_anal_2 = - Gm2_2( rr, d + 1 ) + 2 * Gm2_2( rr, d ) - Gm2_2( rr, d - 1 );
else
  res_anal_2 = - Gm2_2( rr, 1 );
end

res1_num = 0;
for i_t = 1 : l
  for i_tau = 1 : l
    res1_num = res1_num + w( i_t ) * w( i_tau ) ...
      * G0( rr, d + tt( i_t ) - tt( i_tau ) );
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

function res = Gm1_1( rr, delta )

if( delta == 0 && rr > 0 )
  res = 1 / ( 4 * pi * rr );
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

function res = G0( rr, delta )

if(delta > 0)
  res = ( 4 * pi * ( delta ) )^( -3/2 ) * exp( - rr^2 / ( 4 * ( delta ) ) );
else
  res = 0;
end

end
  