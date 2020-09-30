computation_case = 3;
plot_figures = true;
print_to_file = false;
turn_on = true;

switch computation_case
  case 1 
    % compute an approximation of the rhs function for a given function f_lambda
    % on a suitably fine grid.
    
    % choose f_lambda
    lambda = 0.25;
    f_lambda = @(t) (t^lambda);
    
    if (turn_on)
      bisection_point = calc_bisection_point(11, 0, 1);
      tar_function = @(t) (turn_on_function(f_lambda, t, bisection_point, 0.2));
    else
      tar_function = f_lambda;
    end
    % number of levels in standard cluster tree 
    L = 18;
    % number of intervals per leaf cluster
    N = 2^7;
    % The total number of intervals will be 2^L * N
    
    % end time
    T = 1;
    % order of the Lagrange interpolant
    order = 5;
    % dummy function for rhs and 
    dummy_fun = @(t) (ones(size(t)));
    % compute rhs on suitably fine grid
    fprintf('\n Initializing FMM structure. \n');
    tic
    fmm_solver = cFMM_solver(0, T, N, L, order, dummy_fun);
    toc
    fprintf('\n Projecting function f_lambda to test space. \n');
    tic
    projection = fmm_solver.apply_const_l2_project( tar_function );
    toc
    fprintf('\n Computing rhs. \n');
    tic
    rhs = fmm_solver.apply_fmm_matrix( projection );
    toc
    % plot rhs function
    ht = T/(N*2^L);
    rhs_projection = (1/ht) * rhs; % compute projection coefficients from rhs 
    t = ht/2 : ht : T - ht/2;
    plot(t, rhs_projection)
    if turn on
      save adaptive_turn_on_2p25.mat L N rhs rhs_projection t
    else
      save adaptive_test_2p25.mat L N rhs rhs_projection t
    end
  case 2
    if (exist('adaptive_turn_on_2p25.mat', 'file') == 2)
      if (exist('panels_turn_on_refined.mat', 'file') == 2)
        lambda = 0.25;
        f_lambda = @(t) (t.^lambda);
        bisection_point = calc_bisection_point(11, 0, 1);
        tar_function = @(t) (turn_on_function(f_lambda, t, bisection_point, 0.2));
        load adaptive_turn_on_2p25.mat
        h = 1/(2^L*N);
        fine_interval_ends = h : h : 1;
        L = 22;
        % number of temporal panels per temporal cluster
        N_steps = 8;
        order = 5;
        % constant for admissibility criterion
        eta = 2;
        % load t_start, t_end, panels and others (rest not needed here)
        load panels_turn_on_refined.mat 
        % project rhs to coarse grid
        coarse_rhs_projection = project_to_coarse_grid(rhs_projection, ...
          fine_interval_ends, h, panels);
        t_coarse_adptv = 0.5 * (panels(1, :) + panels(2, :));
        if (plot_figures)
          figure
          plot(t_coarse_adptv, coarse_rhs_projection)
          hold on
          plot(t, rhs_projection)
          legend({'coarse rhs','fine rhs'},'Location','southeast');
          hold off
          if (print_to_file)
            fileID = fopen('rhs_turn_on.txt','w');
            fprintf(fileID,'(%6.10f, %12.10f)\n',[t(1:1024*128:end); rhs_projection(1:1024*128:end)']);
            fclose(fileID);

            fileID = fopen('solution_turn_on.txt','w');
            t_print = 0:1/1000:1;
            func_print = tar_function(t_print);
            fprintf(fileID,'(%6.10f, %12.10f)\n',[t_print; func_print]);
            fclose(fileID);
          end
        end
        % rhs_fun =@(t) sin(4*pi*t)*exp(-t);
        exact_solution = tar_function;
        % do computations without adaptive routines
        fprintf('\nSolving without adaptive routines. \n');
        tic
        fmm_solver = cFMM_solver_adaptive(t_start, t_end, N_steps, ...
          L, order, eta, coarse_rhs_projection, panels, false);
        x_non_adptv = fmm_solver.solve_iterative_std_fmm_diag_prec(1e-7);
        toc
        tic
        fmm_solver = cFMM_solver_adaptive(t_start, t_end, N_steps, ...
          L, order, eta, coarse_rhs_projection, panels, false);
        x_non_adptv = fmm_solver.solve_iterative_std_fmm_diag_prec(1e-7);
        toc
        err_non_adptv = fmm_solver.l2_error(x_non_adptv, exact_solution);
        fmm_solver.print_info();
        % do computations with adaptive routines
        fprintf('\nSolving with adaptive routines. \n');
        tic
        fmm_solver_adaptive = cFMM_solver_adaptive(t_start, t_end, N_steps, ...
          L, order, eta, coarse_rhs_projection, panels, true);
        x_adptv = fmm_solver_adaptive.solve_iterative_std_fmm_diag_prec(1e-7);
        toc
        err_adptv = fmm_solver_adaptive.l2_error(x_adptv, exact_solution);
        fmm_solver_adaptive.print_info();
        fprintf('err_adptv:       %.8f \n', err_adptv)
        fprintf('err_non_adptv:   %.8f \n', err_non_adptv)
      else
        disp('panels_turn_on_refined.mat not found!')
        disp('Run generate_adaptive_mesh, settings:  turn_on = true;')
        disp('### ABORT ###')
      end
    else
      disp('adaptive_turn_on_2p25.mat not found!')
      disp('Rerun with settings: computation_case = 1; turn_on = true;')
      disp('ATTENTION: This may take some time')
      disp('### ABORT ###')
    end
  case 3
    if (exist('adaptive_test_2p25.mat', 'file') == 2)
      if (exist('panels_lambda_0p25_0-1_step8.mat', 'file') == 2)
        % load t (midpoint of intervals), rhs_projection, projection (of t^0.25),
        % N (number of intervals per leaf) and L (number of refinements)
        step_list = [8, 8 : 3 : 26];

        load adaptive_test_2p25.mat
        h = 1/(2^L*N);
        fine_interval_ends = h : h : 1;
        L = 22;
        % number of temporal panels per temporal cluster
        N_steps = 32;
        % order of the Lagrange interpolant
        order_list = [2, 2, 4, 5 * ones(1, 5)];
        % constant for admissibility criterion
        eta = 2;
        err_adptv = zeros(length(step_list), 1);
        time_adptv = zeros(length(step_list), 1);
        N_adptv = zeros(length(step_list), 1);
        for j = 1 : length(step_list)
          % load data of coarse grid: panels, lambda, tar_function, error 
          % (of projection to the grid)
          load_file = ['panels_lambda_0p25_0-1_step', num2str(step_list(j)), '.mat'];
          load(load_file)
          N_adptv(j) = size(panels, 2);
          % project rhs to coarse grid
          coarse_rhs_projection = project_to_coarse_grid(rhs_projection, ...
            fine_interval_ends, h, panels);
          t_coarse_adptv = 0.5 * (panels(1, :) + panels(2, :));
          if (plot_figures && j==1)
            figure
            plot(t_coarse_adptv, coarse_rhs_projection)
            hold on
            plot(t, rhs_projection)
            legend({'coarse rhs','fine rhs'},'Location','southeast');
            hold off
            if (print_to_file)
              fileID = fopen('rhs.txt','w');
              fprintf(fileID,'(%6.10f, %12.10f)\n',[t_coarse_adptv; coarse_rhs_projection']);
              fclose(fileID);

              fileID = fopen('solution.txt','w');
              t_print = 0:1/500:1;
              func_print = tar_function(t_print);
              fprintf(fileID,'(%6.10f, %12.10f)\n',[t_print; func_print]);
              fclose(fileID);
            end
          end
          % do computations on an adaptive grid
          order = order_list(j);
          % rhs_fun =@(t) sin(4*pi*t)*exp(-t);
          exact_solution = tar_function;
          fprintf('\nSolving on adaptive grid, order = %d, nr_intervals = %d\n', ...
            order, size(panels, 2));
          tic
          fmm_solver = cFMM_solver_adaptive(t_start, t_end, N_steps, ...
            L, order, eta, coarse_rhs_projection, panels, false);
          x_adptv = fmm_solver.solve_iterative_std_fmm_diag_prec(1e-7);
          time_adptv(j) = toc;
          err_adptv(j)=fmm_solver.l2_error(x_adptv, exact_solution);
      %     %  plot info about used routines
      %     count_matrix_not_adaptive =fmm_solver_not_adaptive.print_info();
        end
        output_mat = zeros(6, length(step_list)-1);
        output_mat(1, :) = N_adptv(2:end)';
        output_mat(2, :) = log2(N_adptv(2:end)');
        output_mat(3, :) = order_list(2:end);
        output_mat(4, :) = err_adptv(2:end)' / sqrt(2/3);
        eoc = log2(err_adptv(2 : end-1)' ./ err_adptv(3 : end)') ./ ...
          log2(N_adptv(3:end)'./ N_adptv(2:end-1)');
        output_mat(5, :) = [0, eoc];
        output_mat(6, :) = time_adptv(2:end)';
        if (print_to_file)
          fileID = fopen('results_adaptive.txt','w');
          fprintf(fileID,'%d, %.2f, %d, %f, %.2f, %f\n', output_mat);
          fclose(fileID);
        end


        % do computations on a regular grid
        nr_refinements = 13;
        ref_start = 5;
        % manually selected order list (s.t. approximation quality is not affected)
        order_list = [2, 2, 3, 3, 3, 4, 5 * ones(1, 7)];
        err_non_adptv = zeros(nr_refinements, 1);
        time_non_adptv = zeros(nr_refinements, 1);
        for L_coarse = ref_start : ref_start + nr_refinements-1;
          fprintf('\n L_coarse = %d.\n', L_coarse);
          order = order_list(L_coarse - ref_start + 1);
          fprintf('\nSolving on regular grid, order = %d, nr_intervals = %d\n', ...
           order, 2^L_coarse);
          h_coarse = (t_end - t_start) * 2^(-L_coarse);
          panels = zeros(2, 2^L_coarse);
          panels(1, :) = t_start : h_coarse : t_end - h_coarse;
          panels(2, :) = t_start + h_coarse : h_coarse : t_end;
          coarse_rhs_projection = project_to_coarse_grid(rhs_projection, ...
            fine_interval_ends, h, panels);
          tic
          fmm_solver = cFMM_solver_adaptive(t_start, t_end, N_steps, ...
            L, order, eta, coarse_rhs_projection, panels, false);
          x_non_adptv = fmm_solver.solve_iterative_std_fmm_diag_prec(1e-7);
          time_non_adptv(L_coarse -ref_start + 1) = toc;
          err_non_adptv(L_coarse -ref_start + 1) = ...
            fmm_solver.l2_error(x_non_adptv, exact_solution);
        end
        output_mat = zeros(5, nr_refinements);
        output_mat(1, :) = ref_start : ref_start + nr_refinements-1;
        output_mat(2, :) = order_list;
        output_mat(3, :) = err_non_adptv' / sqrt(2/3);
        output_mat(4, :) = [0, log2(err_non_adptv(1 : end-1)' ./ err_non_adptv(2 : end)')];
        output_mat(5, :) = time_non_adptv';
        if (print_to_file)
          fileID = fopen('results_non_adaptive.txt','w');
          fprintf(fileID,'%d, %d, %f, %.2f, %f\n', output_mat);
          fclose(fileID);
        end

        % do computations with adaptive routines
    %     fprintf('\nSolving with adaptive routines. \n');
    %     tic
    %     fmm_solver_adaptive = cFMM_solver_adaptive(t_start, t_end, N_steps, ...
    %       L, order, eta, coarse_rhs_projection, panels, true);
    %     x_adptv = fmm_solver_adaptive.solve_iterative_std_fmm_diag_prec();
    %     toc
    %     err_adptv = fmm_solver_adaptive.l2_error(x_adptv, exact_solution);
    %     % plot info about used routines
    %     count_matrix_adaptive = fmm_solver_adaptive.print_info();

        %
    %     fprintf('err_adptv:       %.8f \n', err_adptv)
    %     fprintf('err_non_adptv:   %.8f \n', err_non_adptv)
        if (plot_figures)
          t_coarse_non_adptv = 0.5 * (panels(1, :) + panels(2, :)); 
          figure
          hold on
          plot(t_coarse_adptv, x_adptv);
          plot(t_coarse_non_adptv, x_non_adptv);
          plot(t_coarse_adptv, exact_solution(t_coarse_adptv))
          legend({'FMM adaptive grid', 'FMM non-adaptive grid','exact solution'}, ...
            'Location','southeast');
          hold off
        end
      else
        disp('panels_lambda_0p25_0-1_step8.mat not found!')
        disp('Run generate_adaptive_mesh, settings:  turn_on = false;')
        disp('### ABORT ###')
      end
    else
      disp('adaptive_test_2p25.mat not found!')
      disp('Rerun with settings: computation_case = 1; turn_on = false;')
      disp('ATTENTION: This may take some time')
      disp('### ABORT ###')
    end
  otherwise
    disp('nothing to do')
end

function coarse_func = project_to_coarse_grid(fine_func, fine_interval_ends, ...
  fine_h, coarse_panels)
  nr_coarse_panels = size(coarse_panels,2);
  coarse_func = zeros(nr_coarse_panels, 1);
  fine_pos = 1;
  for j = 1 : nr_coarse_panels
    curr_start = coarse_panels(1, j);
    curr_end = coarse_panels(2, j);
    curr_h = curr_end - curr_start;
    while ( (fine_pos <= length(fine_interval_ends)) && ...
        (fine_interval_ends(fine_pos) <= curr_end) )
      coarse_func(j) = coarse_func(j) + fine_h * fine_func(fine_pos);
      fine_pos = fine_pos + 1;
    end
    coarse_func(j) = coarse_func(j) / curr_h;
  end
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
  y(t > x) = c + func(t(t>x)-x);
end