classdef kernel_laplace_dl < kernel
  
  methods
    function value = eval( ~, x, y, n, ~ )
      xy = x - y;
      norm = sqrt( xy.^2 * [ 1; 1; 1 ] );
      value = ( xy * n' ) ./ ( 4 * pi * norm .* norm .* norm );
    end
  end
  
end

