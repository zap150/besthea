%help file to determine the eoc of the approximation error for piecewise
%constant projection
clc, clear
t_start = 0;
t_end = 1;
lambda = 0.25;
tar_function = @(t) (t.^(lambda));
% compute l2 projection and l2 error for various uniform partitions
L_max = 15;
errors = zeros(1, L_max);
for l=1: L_max
  n_panels = 2^l;
  h_init = (t_end - t_start) / n_panels;
  panels = zeros(2, n_panels);
  panels(1, :) = t_start : h_init : t_end - h_init;
  panels(2, :) = t_start + h_init : h_init : t_end;
  projection = project_function(panels, tar_function);
  panelwise_error = l2_error_panelwise(panels, projection, tar_function);
  errors(l) = sqrt(sum(panelwise_error.^2));
end
eoc = log2(errors(1 : end-1) ./ errors(2 : end))
%errors




% L2 projection of a function
function func_projection = project_function(panels, func)
  n_intervals = size(panels, 2);
  h_panels = panels(2, :) - panels(1, :);
  [ x, w, l ] = quadratures.line(10);

  func_projection = zeros(n_intervals, 1);
  for j = 1 : l
    func_projection = func_projection + w(j) * func(panels(1, :)' + ...
      x (j) * h_panels');
    % for numerical integration a factor h comes into play, which cancels
    % with the factor 1/h from the projection
  end
end

% panelwise error of a function
function panelwise_error = l2_error_panelwise(panels, sol, analytic)
  [ x, w, ~ ] = quadratures.line(10);
  h = panels(2, :) - panels(1, :);
  panelwise_error = zeros(1, size(panels, 2));
  for i = 1 : size(panels, 2)
    panelwise_error(i) = panelwise_error(i) + h(i) * ...
      (sol(i) - analytic(panels(1, i) + x * h(i))).^2' * w;
  end
  panelwise_error = sqrt(panelwise_error);
end