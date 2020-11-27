% Tests solution of a purely time-dependent problem on a sphere

%% Test 1: Compare standard code and adaptive code for uniform timesteps
%
clc, clear

%accuracy of the GMRES
eps = 1e-6;
% levels of the binary tree
L = 4;
% number of temporal panels per temporal cluster
N_steps = 16;
% end time
t_end = 1;
t_start = 0;

% order of the Lagrange interpolant
order = 5;
% RHS function
rhs_fun = @(t) ((exp(t) / 4) ...
  .* (exp(2) * erfc((1 + t) ./ (sqrt(t))) ...
  + 2 * erf(sqrt(t)) - exp(-2) * erfc((1 - t) ...
  ./ (sqrt(t)))));
%rhs_fun =@(t) sin(4*pi*t)*exp(-t);
exact_solution = @(t) (exp(t));

fprintf('\nSolving problem with %d time-steps, end-time = %f. \n', ...
  2^L * N_steps, t_end);
fprintf('GMRES prec = 1e%d \n', log10( eps ) ); 

fprintf('\nSolving using GMRES with standard FMM. \n');
fmm_solver = cFMM_solver(t_start, t_end, N_steps, L, order, rhs_fun);


%x_stndrd = fmm_solver.apply_fmm_matrix_std(vec);
tic
x_stndrd = fmm_solver.solve_iterative_std_fmm_prec( eps );
toc
err_stndrd = fmm_solver.l2_error(x_stndrd, exact_solution, t_end, N_steps*2^L);

% ##############################################################################
% solve with new code 
% ##############################################################################

% set parameter eta for nearfield criterion 
eta = 2;

%construct uniform intervals (as in cFMM_solver)
panels = zeros(2, 2^L * N_steps);
ht = 1 / (2^L * N_steps);
panels(1, :) = t_start : ht : t_end - ht;
panels(2, :) = t_start + ht : ht : t_end;

fprintf('\nSolving using GMRES with standard FMM (adaptive code basis). \n');

fmm_solver = cFMM_solver_adaptive(t_start, t_end, N_steps, L, order, ... 
      eta, rhs_fun, panels, true);
    
% vec = ones(size(panels,2), 1);
% x_adptv = fmm_solver.apply_fmm_matrix_std(vec);
    
tic
x_adptv = fmm_solver.solve_iterative_std_fmm_diag_prec( eps );
toc
err_adptv = fmm_solver.l2_error(x_adptv, exact_solution);

t = 0.5 * (panels(1, :) + panels(2, :));
analytical = exp(t);

figure;
title('Purely temporal problem solved using FMM and full matrices')
plot(t, x_stndrd);
hold on
plot(t, x_adptv);

% analytical solution for the first testing RHS (comment out for the other)
plot(t, analytical);
legend({'standard FMM algorithm', 'adaptive FMM algorithm', 'analytic'},'Location','northwest');
%legend({'FMM - direct', 'FMM - iterative', 'standard FMM - iterative', 'full'},'Location','southwest');
hold off

fprintf('err_stndrd:       %.8f \n', err_stndrd)
fprintf('err_adptv:        %.8f \n', err_adptv)
%
%% Test adaptive code for non-uniform time steps (WITHOUT ADAPTIVE ROUTINES!)
%
t_start = 0;
t_end = 1;
L = 5;
N_steps = 16;
mul_factor = 4; %for case 1
nr_refinement_levels = 3; %for case 1

panel_case = 1; %switch between different partitions of the time interval

exact_solution = @(t) (exp(t));

switch panel_case
  case 1 % adaptivily refined intervals close to 1
    h_list = ones(1, nr_refinement_levels + 2);
    h_list(1) = 0;
    for j = 2 : nr_refinement_levels + 1
      h_list(j) = h_list(j-1) + 2^(-j+1);
    end
    panels = zeros(2, (nr_refinement_levels+1) * N_steps * mul_factor);
    for j = 1 : nr_refinement_levels + 1
      curr_h = (h_list(j+1) - h_list(j))/(mul_factor * N_steps);
      panels(1, (mul_factor * N_steps * (j-1) + 1) : (mul_factor * N_steps * j)) = ...
        h_list(j) : curr_h : h_list(j+1) - curr_h;
      panels(2, (mul_factor * N_steps * (j-1) + 1) : (mul_factor * N_steps * j)) = ...
        h_list(j) + curr_h : curr_h : h_list(j+1);
    end
    panels = 1-panels;
    panels = panels(end:-1:1, :);
    panels = panels(:, end:-1:1);
  case 2 % intervals in [0, 0.25] + interval [0.25, 1]
    N = 24 * 2^4;
    h = 0.25/N;
    panels = zeros(2, N+1);
    panels(1, 1:end-1) = 0 : h : 0.25-h;
    panels(2, 1:end-1) = h : h : 0.25;
    panels(2, end) = 1;
    panels(1, end) = 0.25;
  case 3 % only intervals in [0, 0.25]
    N = 24 * 2^4;
    h = 0.25/N;
    panels = zeros(2, N);
    panels(1, 1:end) = 0 : h : 0.25-h;
    panels(2, 1:end) = h : h : 0.25;
    t_end = 0.25;
  case 4 % interval [0, 0.75] + intervals in [0.75, 1]
    N = 24 * 2^4;
    h = 0.25/N;
    panels = zeros(2, N+1);
    panels(1, 2:end) = 0.75 : h : 1-h;
    panels(2, 2:end) = 0.75 + h : h : 1;
    panels(2, 1) = 0.75;
  otherwise % random intervals in [0, 1]
    seed = 1; %set seed for rng to get always the same result
    rng(seed);
    % construct intervals with a cumulative sum of random numbers
    panels = zeros(2, (nr_refinement_levels+1) * N_steps * mul_factor);
    panels(2, :) = cumsum(0.01 + rand(1, size(panels, 2)));
    panels(2, :) = panels(2, :) / panels(2, end);
    panels(1, 2:end) = panels(2, 1:end-1);
end

% order of the Lagrange interpolant
order = 5;
% RHS function
rhs_fun = @(t) ((exp(t) / 4) ...
  .* (exp(2) * erfc((1 + t) ./ (sqrt(t))) ...
  + 2 * erf(sqrt(t)) - exp(-2) * erfc((1 - t) ...
  ./ (sqrt(t)))));
%rhs_fun =@(t) sin(4*pi*t)*exp(-t);

fprintf('\nSolving problem with %d non-uniform time-steps, end-time =  %f. \n', ... 
  size(panels,2), t_end);

% set parameter eta for nearfield criterion 
eta = 2;

fprintf('\nSolving using GMRES with standard FMM (adaptive code basis). \n');
fmm_solver = cFMM_solver_adaptive(t_start, t_end, N_steps, L, order, ... 
      eta, rhs_fun, panels, true);
%vec = ones(size(panels,2), 1);
%x_adptv = fmm_solver.apply_fmm_matrix_std(vec);
tic
x_adptv = fmm_solver.solve_iterative_std_fmm_diag_prec(1e-5);
toc
err_adptv = fmm_solver.l2_error(x_adptv, exact_solution);

t = 0.5 * (panels(1, :) + panels(2, :));

figure;
title('Purely temporal problem solved using FMM (non-uniform time intervals)')
plot(t, x_adptv);
hold on
analytical = exp(t);
plot(t, analytical);

legend({'adaptive FMM algorithm', 'analytical solution'},'Location','northwest');
hold off
fprintf('err_adptv:        %.8f \n', err_adptv)