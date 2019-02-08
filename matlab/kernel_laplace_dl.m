classdef kernel_laplace_dl < kernel
  
  methods
    function value = eval( ~, x, y, n )
      xy = x - y;
      norm = vecnorm( xy' )';
      dot = xy * n';
      value = dot ./ ( 4 * pi * norm .* norm .* norm );
    end
  end
  
end

