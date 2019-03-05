classdef kernel_laplace_hs < kernel
  
  methods
    function value = eval( ~, x, y, ~, ~ )
      value = 1 ./ ( 4 * pi * sqrt( ( x - y ).^2 * [ 1; 1; 1 ] ) );
    end
  end
  
end

