classdef kernel_laplace_sl < kernel
  
  methods
    function value = eval( ~, x, y, ~ )
      value = 1 ./ ( 4 * pi * vecnorm( ( x - y )' ) )';
    end
  end
  
end

