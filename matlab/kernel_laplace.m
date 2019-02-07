classdef kernel_laplace < kernel
  
  methods
    function value = eval( ~, x, y )
      value = 1 ./ ( 4 * pi * vecnorm( ( x - y )' ) );
    end
    
    function value = eval_derivative( ~, x, y, n )
      xy = x - y;
      norm = vecnorm( xy' )';
      dot = xy * n';
      value = dot ./ ( 4 * pi * norm .* norm .* norm );
    end
  end
  
end

