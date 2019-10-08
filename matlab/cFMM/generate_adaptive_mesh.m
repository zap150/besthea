% generate adaptive mesh for a 1d function by local refinement based on l2
% projection to piecewise constant functions and reduction of projection error

turn_on = false; %change to generate desired mesh files
plot_figures = true;

% avoid changes below this line
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t_start = 0;
t_end = 1; 
% construct function handle for function t^lambda, with lambda = 0.25
lambda = 0.25;
f_lambda = @(t) (t.^lambda);
% settings for refinement using Doerfler marking
theta_marking = 0.5;
eps = 1e-7;
if turn_on
  % construct initial mesh
  n_panels = 2^1;
  panels = zeros(2, n_panels);
  h_init = (t_end - t_start) / n_panels;
  panels(1, :) = t_start : h_init : t_end - h_init;
  panels(2, :) = t_start + h_init : h_init : t_end;
  % set target function for which the mesh is generated
  bisection_point = calc_bisection_point(11, t_start, t_end);
  tar_function = @(t) (turn_on_function(f_lambda, t, bisection_point, 0.2));
  % construct adaptive mesh
  max_step = 22;
  [out_panels, error, steps] = refine_locally(t_start, t_end, panels, ...
    tar_function, theta_marking, eps, max_step);
  nr_refinements = 3;
  panels = refine_uniformly(out_panels, nr_refinements);
  % store mesh and relevant parameters
  save panels_turn_on_refined.mat panels error max_step t_start t_end
else
  n_panels = 2^5;
  h_init = (t_end - t_start) / n_panels;
  % set target function for which the mesh is generated
  tar_function = f_lambda;
  step_list = [8, 8 : 3 : 26];
  for j = 1 : length(step_list)
    %reset initial mesh
    panels = zeros(2, n_panels);
    panels(1, :) = t_start : h_init : t_end - h_init;
    panels(2, :) = t_start + h_init : h_init : t_end;
    save_file = ['panels_lambda_0p25_0-1_step', num2str(step_list(j)), '.mat'];
    % construct adaptive mesh for current number of steps
    [panels, error, steps] = refine_locally(t_start, t_end, panels, ...
      tar_function, theta_marking, eps, step_list(j));
    save (save_file, 'panels', 'error', 'max_step', 't_start', 't_end')
  end
end

if (plot_figures)
  % print bar plot of last generated panels
  h_intervals = panels(2,:)-panels(1, :);
  figure
  bar(-log2(h_intervals /(t_end - t_start)))
end

% % Generate basic text file which allows to generate intervals with tikz
% fileID=fopen('intervals.txt','w')
% fprintf(fileID, '\\draw (%.4f, -0.02) -- (%.4f, 0.02);\n', [panels(2,:);panels(2,:)]);
% fclose(fileID);

function [out_panels, error, step_countr] = refine_locally(t_start, t_end, ...
    in_panels, tar_function, theta_marking, eps, max_step)
  old_panels = in_panels;
  projection = project_function(old_panels, tar_function);
  panelwise_error = l2_error_panelwise(old_panels, projection, tar_function);
  error = sqrt(sum(panelwise_error.^2));
  step_countr = 0;
  while (error > eps && step_countr < max_step) % refine until certain accuracy is achieved
    % construct a doerfler marking of the panels for refinement
    [sorted_errors_squared, indices] = sort(panelwise_error.^2, 'descend');
    nr_marked = 1;
    marked_error_squared = sorted_errors_squared(1);
    error_squared = error^2;
    while (marked_error_squared < theta_marking * error_squared)
      nr_marked = nr_marked + 1;
      marked_error_squared = marked_error_squared + ...
        sorted_errors_squared(nr_marked);
    end
    marked_panels = sort(indices(1 : nr_marked));
    new_panels = zeros(2, size(old_panels, 2) + nr_marked);
    new_panels(:, 1:marked_panels(1)-1) = old_panels(:, 1:marked_panels(1)-1);
    for i = 1 : nr_marked-1
      % split marked panel into two panels via bisection
      marked_panel = old_panels(:, marked_panels(i));
      panel_mid = 0.5 * (marked_panel(1) + marked_panel(2));
      new_panels(:, marked_panels(i)+i-1) = ...
        [old_panels(1, marked_panels(i)); panel_mid];
      new_panels(:, marked_panels(i)+i) = ...
        [panel_mid; old_panels(2, marked_panels(i))];
      % leave other panels as they are
      new_panels(:, marked_panels(i)+i+1 : marked_panels(i+1)-1+i) = ...
        old_panels(:, marked_panels(i)+1 : marked_panels(i+1)-1);
    end
    % do the same for the last panel
    marked_panel = old_panels(:, marked_panels(end));
    panel_mid = 0.5 * (marked_panel(1) + marked_panel(2));
    new_panels(:, marked_panels(nr_marked)+(nr_marked-1)) = ...
      [old_panels(1, marked_panels(nr_marked)); panel_mid];
    new_panels(:, marked_panels(nr_marked)+nr_marked) = ...
      [panel_mid; old_panels(2, marked_panels(nr_marked))];
    new_panels(:, marked_panels(nr_marked)+nr_marked+1 : end) = ...
      old_panels(:, marked_panels(nr_marked)+1 : end);
    % overwrite old panels
    old_panels = new_panels;
    projection = project_function(old_panels, tar_function);
    panelwise_error = l2_error_panelwise(old_panels, projection, tar_function);
    error = sqrt(sum(panelwise_error.^2));
    step_countr = step_countr + 1;
  end
  out_panels = old_panels;
end
 
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

function x = calc_bisection_point(L, t_start, t_end)
  t_start_temp = t_start;
  t_end_temp = t_end;
  for i=1:L
      t_start_temp = (t_start_temp+t_end_temp)/2;
      t_end_temp = (t_start_temp+t_end_temp)/2;
  end
  x = (t_start_temp+t_end_temp)/2;
end

function y = turn_on_function(func, t, x, c)
  y = zeros(size(t));
  y(t <= x) = c;
  y(t > x) = c + func(t(t>x));
end

function new_panels = refine_uniformly(old_panels, nr_refinements)
  nr_old_panels = size(old_panels, 2);
  mul_factor = 2^nr_refinements;
  new_panels = zeros(2, mul_factor * nr_old_panels);
  for j = 1 : nr_old_panels
    h_curr = old_panels(2, j) - old_panels(1, j);
    h_curr_split = h_curr / mul_factor;
    new_panels(1, (j-1)*mul_factor+1 : j*mul_factor) = ...
      old_panels(1, j) : h_curr_split : old_panels(2, j) - h_curr_split;
    new_panels(2, (j-1)*mul_factor+1 : j*mul_factor) = ...
      old_panels(1, j) + h_curr_split : h_curr_split : old_panels(2, j);
  end
end