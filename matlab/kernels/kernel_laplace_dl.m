classdef kernel_laplace_dl < kernel
  
  methods
    function value = eval( ~, x, y, ~, ny )
      xy = x - y;
      norm = sqrt( xy.^2 * [ 1; 1; 1 ] );
      value = ( xy * ny' ) ./ ( 4 * pi * norm .* norm .* norm );
    end
  end
  
end

