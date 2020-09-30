classdef cFMM_solver_adaptive < handle
%CFMM_SOLVER Summary of this class goes here
  %   Detailed explanation goes here
  
  properties (Access = private)
    T_start
    T_end
    % function handle to the rhs, or a vector containing the projection ...
    % coefficients of this function
    rhs_fun
    rhs_proj
    cluster_tree
    % maximal number of intervals in each leaf cluster
    n_max
    % highest level allowed in the cluster tree
    L
    % effective maximal level in the cluster tree
    max_level
    % temporal panels
    panels
    % length of all the panels
    h_panels
    % maximum length of all the panels
    h_max_panels
    
    % order of the Lagrange interolation
    L_order
    % parameter for nearfield criterion
    eta
    % nearfield clusters
    cluster_1
    cluster_2
    
    % leaf nodes of the tree stored levelwise 
    % ATTENTION: level j is stored at position j + 1 
    levelwise_leaves
    % number of leaves stored levelwise 
    % ATTENTION: level j is stored at position j + 1 
    nr_levelwise_leaves
    % total number of leaves
    nr_leaves
    % list containing leaf clusters sorted in time
    ordered_leaves
    
    % list of levelwise padding
    % ATTENTION: level j is stored at position j + 1 
    levelwise_pad_list
    
    % list of levelwise cluster sizes (including padding)
    % ATTENTION: level j is stored at position j + 1 
    levelwise_cluster_size
    
    %lowest level on which interactions (M2L operations) happen
    lowest_interaction_level
    
    % list of levelwise m2m matrices, left and right
    % ATTENTION: level j is stored at position j + 1 
    levelwise_m2m_left
    levelwise_m2m_right
    
    % list of levelwise m2l matrices, distances and their numbers
    % ATTENTION: level j is stored at position j + 1 
    levelwise_m2l_mat
    levelwise_m2l_diff
    nr_levelwise_m2l
    
    % bool indicating whether or not M2T and S2L operations should be used
    use_m2t_and_s2l
    
    % nearfield
    V1
    V2
    
    % for preconditioning
    V_diag
  end
  
  methods
    
    function obj = cFMM_solver_adaptive(t_start, t_end, n_max, L, L_order, ... 
      eta, rhs_fun, panels, use_m2t_and_s2l)
      obj.T_start = t_start;
      obj.T_end = t_end;
      obj.rhs_fun = rhs_fun;
      obj.n_max = n_max;
      obj.L = L;
      obj.L_order = L_order;
      obj.eta = eta;
      obj.panels = panels;
      obj.h_panels = panels(2,:) - panels(1,:);
      obj.h_max_panels = max( obj.h_panels );
      obj.use_m2t_and_s2l = use_m2t_and_s2l;
      obj.lowest_interaction_level = obj.L+1;
      
      % construct temporal cluster tree
      obj.levelwise_leaves = cell(1, L+1);
      obj.nr_levelwise_leaves = zeros(1, L+1); 
      obj.nr_leaves = 0;
      for l = 0 : L
        obj.levelwise_leaves{l+1} = cell(1, 2^l);
      end
      obj.ordered_leaves = cell(1, 2^obj.L);
      
      cluster = temporal_cluster(obj.panels, 1, size(obj.panels,2), ...
        t_start, t_end, 0);
      obj.cluster_tree = binary_tree(cluster, 0, 0);
      obj.construct_cluster_tree(obj.cluster_tree);
      obj.max_level = obj.L;
      while (obj.nr_levelwise_leaves(obj.max_level+1) == 0)
        obj.max_level = obj.max_level - 1;
      end
      obj.set_pad_list();
      obj.pad_cluster_tree(obj.cluster_tree);
      % compute levelwise cluster size.
      obj.levelwise_cluster_size = zeros( obj.max_level + 2, 1 );
      obj.levelwise_cluster_size( 1 : obj.max_level + 1 ) = ...
        ( t_end - t_start ) * 2.^( - ( 0 : obj.max_level )' ) ...
        + obj.levelwise_pad_list( 1 : obj.max_level + 1, 2 ) ...
        + obj.levelwise_pad_list( 1 : obj.max_level + 1, 1 );
      % add h_max_panels as fictive size of clusters at level max_level + 1 
      % (useful for bpx preconditioning)
      obj.levelwise_cluster_size( obj.max_level + 2 ) = obj.h_max_panels;
          
      obj.assmbl_intrctn_nghbr_lsts(obj.cluster_tree);
      
      obj.find_lowest_interaction_level(obj.cluster_tree);
      
      obj.compute_nearfield_matrices();
      obj.compute_q2m();
      obj.compute_m2m(obj.cluster_tree);
      
      obj.nr_levelwise_m2l = zeros(1, obj.L+1);
      obj.levelwise_m2l_diff = cell(1, obj.L+1);
      obj.find_m2l(obj.cluster_tree);
      obj.compute_m2l();
      obj.compute_m2t_matrices();
      obj.compute_s2l_matrices();
      %obj.cluster_tree.print();
      
      % RHS assembly
      if (~isnumeric(obj.rhs_fun))
        obj.rhs_proj = obj.project_rhs(obj.rhs_fun);
        obj.rhs_proj = obj.rhs_proj .* obj.h_panels';
      else
        obj.rhs_proj = obj.rhs_fun .* obj.h_panels';
      end
    end
    
    function x = apply_diag_prec(obj, y)
      %loop through all levels and apply inverse nearfield matrices
      x = zeros(size(y));
      for i = 1 : obj.nr_leaves
        curr_leaf = obj.ordered_leaves{i}.get_value();
        % We want to access the nearfield matrix which describes the action
        % of the cluster with itself. Since the cluster itself is always
        % its last neighbor (by construction), the index nf_mat_ind to 
        % access the matrix is the size of the lists of neighbors.
        nf_mat_ind = size(curr_leaf.get_neighbors(), 2);
        start_ind = curr_leaf.get_start_index();
        end_ind = curr_leaf.get_end_index();
        x(start_ind : end_ind) = curr_leaf.apply_inv_nearfield_mat( ...
          y(start_ind : end_ind), nf_mat_ind);
      end
    end
    
    function z = apply_bpx_prec( obj, r )
      %Applies the bpx preconditioner based on the cluster hierarchy to the 
      %vector r.
      obj.bpx_precondition_upward_path( obj.cluster_tree, r );
      obj.bpx_precondition_downward_path( obj.cluster_tree );
      z = obj.eval_bpx_for_leaves( r );
    end
    
    function bpx_precondition_upward_path( obj, cluster_tree, r )
      % Realizes the upward path of the bpx preconditioner: 
      % for all clusters starting at leaf level the projection coefficients of
      % the right hand side are computed by restrictions (simple summations due 
      % to the use of p0 basis functions).
      % The routine is realized by a recursive tree traversal.
      cluster = cluster_tree.get_value( );
      cluster.set_bpx_rhs_projection_coeffs( 0 );
      left_subtree = cluster_tree.get_left_child( );
      right_subtree = cluster_tree.get_right_child( );
      % add the projection coefficients of the children to those of the current 
      % cluster, if it is not a leaf
      if ( left_subtree ~= 0 ) 
        bpx_precondition_upward_path( obj, left_subtree, r );
        cluster.set_bpx_rhs_projection_coeffs( ...
          cluster.get_bpx_rhs_projection_coeffs( ) ...
          + left_subtree.get_value( ).get_bpx_rhs_projection_coeffs( ) );
      end
      if ( right_subtree ~= 0 )
        bpx_precondition_upward_path( obj, right_subtree, r );
        cluster.set_bpx_rhs_projection_coeffs( ...
          cluster.get_bpx_rhs_projection_coeffs( ) ...
          + right_subtree.get_value( ).get_bpx_rhs_projection_coeffs( ) );
      end
      % compute the projection coefficients directly for leaf clusters.
      if ( left_subtree == 0 && right_subtree == 0 ) 
        cluster.set_bpx_rhs_projection_coeffs( ...
          sum( r( cluster.get_start_index( ) : cluster.get_end_index( ) ) ) );
      end
    end
    
    function bpx_precondition_downward_path( obj, cluster_tree )
      % Realizes the downward path of the bpx preconditioner.
      % for all clusters starting at the root the appropriate contributions are
      % computed and passed downwards to the children. The evaluation for leaf
      % clusters is done in a separate step.
      % The routine is realized by a recursive tree traversal.
      
      left_subtree = cluster_tree.get_left_child( );
      right_subtree = cluster_tree.get_right_child( );
      parent_subtree = cluster_tree.get_parent( );
      level = cluster_tree.get_level( );
      cluster = cluster_tree.get_value( );
      % compute the size of the cluster
      start_panel = cluster.get_panel( cluster.get_start_index( ) );
      end_panel = cluster.get_panel( cluster.get_end_index( ) );
      cluster_size = end_panel( 2 ) - start_panel( 1 );
      % compute the bpx contribution for the current cluster
      cluster.set_bpx_contribution( ...
        cluster.get_bpx_rhs_projection_coeffs( ) / cluster_size ...
        * ( 1 / sqrt( obj.levelwise_cluster_size( level + 1 ) ) ...
          - 1 / sqrt( obj.levelwise_cluster_size( level + 2 ) ) ) );
      % add the parent bpx contribution if available
      if ( parent_subtree ~= 0 ) 
        cluster.set_bpx_contribution( ...
          cluster.get_bpx_contribution( ) ...
          + parent_subtree.get_value( ).get_bpx_contribution( ) );
      end
      % call the routine recursively for children, if they exist
      if ( left_subtree ~= 0 )
        bpx_precondition_downward_path( obj, left_subtree );
      end
      if ( right_subtree ~= 0 )
        bpx_precondition_downward_path( obj, right_subtree );
      end
    end
    
    function z = eval_bpx_for_leaves( obj, r )
      % Realizes the final evaluation step of the bpx preconditioner for leaf
      % clusters in the cluster tree.
      z = zeros( size( r ) );
      for i = 1 : obj.nr_leaves
        curr_level = obj.ordered_leaves{ i }.get_level( );
        curr_leaf = obj.ordered_leaves{ i }.get_value( );
        start_idx = curr_leaf.get_start_index( );
        end_idx = curr_leaf.get_end_index( );
        z( start_idx : end_idx ) = ...
          curr_leaf.get_bpx_contribution( ) ...
          + 1 / sqrt( obj.levelwise_cluster_size( curr_level + 2 ) ) ...
          * r( start_idx : end_idx ) ./ obj.h_panels( start_idx : end_idx )';
      end
    end
      
%      function y = apply_fmm_matrix(obj, x)
%        
%        % downward pass
%        parent = obj.leaves{1};
%        for l = obj.L : -1 : 1
%          parent = parent.get_parent();
%        end
%        obj.reset(parent);
%        
%        y = zeros(size(obj.rhs_proj));
%        
%        n = obj.N_steps;
%        y(1 : n) = obj.V1 * x(1 : n);
%        y(n + 1 : 2 * n) = obj.V1 * x(n + 1 : 2 * n) + obj.V2 * x(1 : n) ;
%        
%        for m = 2 : 2^obj.L - 1 %  
%          % find the coarsest level where parents change
%          lt = 2;
%          
%          for i = 2 : obj.L - 1
%            if obj.get_id_on_level(m - 1, i) == obj.get_id_on_level(m, i)
%              lt = lt + 1;
%            end
%          end
%          
%          % compute moments
%          cluster = obj.leaves{m - 2 + 1}.get_value();
%          ind_start = cluster.get_start_index();
%          ind_end = cluster.get_end_index();
%          cluster.set_moments(cluster.apply_q2m(x(ind_start : ind_end)));
%          parent_node = obj.leaves{m - 2 + 1}.get_parent();
%          cluster.inc_c_moments_count();
%          cluster.inc_c_moments_count();
%          
%          % upward path
%          % transfer moments to the highest possible level
%          for l = obj.L - 1 : -1 :  2
%            parent_cluster = parent_node.get_value();
%            left_child = parent_cluster.get_left_child();
%            right_child = parent_cluster.get_right_child();
%            
%            % transfer up only if there are contributions from both descendants
%            % included
%            if (left_child.get_c_moments_count() == 2)
%              parent_cluster.apply_m2m_left(left_child.get_moments());
%            end
%            
%            if (right_child.get_c_moments_count() == 2)
%              parent_cluster.apply_m2m_right(right_child.get_moments());
%            end
%            parent_node = parent_node.get_parent();
%          end
%          
%          % interaction phase
%          % go through interaction lists of the clusters and apply M2L
%          cluster = obj.leaves{m + 1}.get_value();
%          parent_node = obj.leaves{m + 1};
%          for l = obj.L : -1 : lt
%            cluster.apply_m2l();
%            parent_node = parent_node.get_parent();
%            cluster = parent_node.get_value();
%          end
%          
%          % downward pass
%          parent = obj.leaves{m + 1};
%          for l = obj.L : -1 : lt
%            parent = parent.get_parent();
%          end
%          
%          obj.downward_pass(parent, m);
%          
%          cluster = obj.leaves{m + 1}.get_value();
%          
%          f = cluster.apply_l2p();
%          ind_start = obj.leaves{m - 1 + 1}.get_value().get_start_index();
%          ind_end = obj.leaves{m - 1 + 1}.get_value().get_end_index();
%          tst = obj.V2 * x(ind_start : ind_end);
%          f = f + tst; 
%          
%          ind_start = obj.leaves{m  + 1}.get_value().get_start_index();
%          ind_end = obj.leaves{m + 1}.get_value().get_end_index();
%          tst = obj.V1 * x(ind_start : ind_end);
%          f = f + tst;
%          
%          y(ind_start : ind_end) = f; 
%        end
%        
%      end
    
    
    function y = apply_fmm_matrix_std(obj, x)
      % reset data and set y to 0
      obj.reset(obj.cluster_tree());
      y = zeros(size(obj.rhs_proj));
    
      % compute moments in all leaves first (S2M)
      for l = obj.lowest_interaction_level : obj.L
        for j = 1 : obj.nr_levelwise_leaves(l+1)
          cluster = obj.levelwise_leaves{l+1}{j}.get_value();
          ind_start = cluster.get_start_index();
          ind_end = cluster.get_end_index();
          cluster.set_moments(cluster.apply_q2m(x(ind_start : ind_end)));
        end 
      end
      
      % upward pass (M2M)
      obj.upward_pass(obj.cluster_tree());
      
      % interaction phase (M2L)
      obj.farfield_interaction(obj.cluster_tree());
      
      % S2L
      if (obj.use_m2t_and_s2l)
        obj.apply_s2l_operations(x);
      end
      
      % downward pass (L2L)
      obj.downward_pass_std(obj.cluster_tree());
      
      % evaluate local expansions in all leaves (L2T)
      for l = obj.lowest_interaction_level : obj.L
        for j = 1 : obj.nr_levelwise_leaves(l+1)
          cluster = obj.levelwise_leaves{l+1}{j}.get_value();
          ind_start = cluster.get_start_index();
          ind_end = cluster.get_end_index();
          y(ind_start : ind_end) = cluster.apply_l2p();
        end
      end
      
      % nearfield evaluation
      y = y + obj.nearfield_evaluation(x);
      
      if (obj.use_m2t_and_s2l)
      % compute part of y from M2T operations
        y = y + obj.apply_m2t_operations();
      end
    end
    
%     function x = solve_iterative(obj)
%       %x = gmres(@obj.apply_fmm_matrix, obj.rhs_proj, 100, 1e-5, 600);
% 
%       x = gmres(@obj.apply_fmm_matrix, obj.rhs_proj, 100, 1e-5, 600, ...
%        @obj.apply_diag_prec);
%     end
    
    function [ x, inner_iter ] = solve_iterative_std_fmm(obj, eps)
      [ x, ~, ~, iter ] = gmres(@obj.apply_fmm_matrix_std, ...
        obj.rhs_proj, 100, eps, 600);
      inner_iter = ( iter( 1 ) - 1 ) * 100 + iter( 2 );
    end
    
    function [ x, inner_iter ] = solve_iterative_std_fmm_diag_prec(obj, eps)
      [ x, ~, ~, iter ] = gmres(@obj.apply_fmm_matrix_std, ...
        obj.rhs_proj, 100, eps, 600, @obj.apply_diag_prec);
      inner_iter = ( iter( 1 ) - 1 ) * 100 + iter( 2 );
    end
    
    function [x, inner_iter] = solve_iterative_std_fmm_bpx_prec(obj, eps)
      [ x, ~, ~, iter ] = gmres(@obj.apply_fmm_matrix_std, ...
        obj.rhs_proj, 100, eps, 600, @obj.apply_bpx_prec);
      inner_iter = ( iter( 1 ) - 1 ) * 100 + iter( 2 );
    end
    
%     function x = solve_direct(obj)  
%       x = zeros(size(obj.rhs_proj)); 
%       n = obj.N_steps;
%       x(1 : n) = obj.V1 \ obj.rhs_proj(1 : n);
%       x(n + 1 : 2 * n) = obj.V1 \ ...
%         (obj.rhs_proj(n + 1 : 2 * n) - obj.V2 * x(1 : n));
%       
%       for m = 2 : 2^obj.L - 1 %  -1 ?
%         % find the coarsest level where parents change
%         lt = 2;
%         
%         for i = 2 : obj.L - 1
%           if obj.get_id_on_level(m - 1, i) == obj.get_id_on_level(m, i)
%             lt = lt + 1;
%           end
%         end
%         
%         % compute moments
%         cluster = obj.leaves{m - 2 + 1}.get_value();
%         ind_start = cluster.get_start_index();
%         ind_end = cluster.get_end_index();
%         cluster.set_moments(cluster.apply_q2m(x(ind_start : ind_end)));
%         parent_node = obj.leaves{m - 2 + 1}.get_parent();
%         cluster.inc_c_moments_count();
%         cluster.inc_c_moments_count();
%         
%         % upward path
%         % transfer moments to the highest possible level
%         for l = obj.L - 1 : -1 :  2
%           parent_cluster = parent_node.get_value();
%           left_child = parent_cluster.get_left_child();
%           right_child = parent_cluster.get_right_child();
%           
%           % transfer up only if there are contributions from both descendants
%           % included
%           if (left_child.get_c_moments_count() == 2)
%             parent_cluster.apply_m2m_left(left_child.get_moments());
%           end
%           
%           if (right_child.get_c_moments_count() == 2)
%             parent_cluster.apply_m2m_right(right_child.get_moments());
%           end
%           parent_node = parent_node.get_parent();
%         end
%         
%         % interaction phase
%         % go through interaction lists of the clusters and apply M2L
%         cluster = obj.leaves{m + 1}.get_value();
%         parent_node = obj.leaves{m + 1};
%         for l = obj.L : -1 : lt
%           cluster.apply_m2l();
%           parent_node = parent_node.get_parent();
%           cluster = parent_node.get_value();
%         end
%         
%         % downward pass
%         parent = obj.leaves{m + 1};
%         for l = obj.L : -1 : lt
%           parent = parent.get_parent();
%         end
%         
%         obj.downward_pass(parent, m);
%         
%         cluster = obj.leaves{m + 1}.get_value();
%         f = cluster.apply_l2p();
%         ind_start = obj.leaves{m - 1 + 1}.get_value().get_start_index();
%         ind_end = obj.leaves{m - 1 + 1}.get_value().get_end_index();
%         tst = obj.V2 * x(ind_start : ind_end);
%         f = f + tst; %V2 * x(ind_start : ind_end);
%         ind_start = obj.leaves{m  + 1}.get_value().get_start_index();
%         ind_end = obj.leaves{m + 1}.get_value().get_end_index();
%         obj.rhs_proj(ind_start : ind_end) = ...
%           obj.rhs_proj(ind_start : ind_end) - f;
%         x(ind_start : ind_end) = obj.V1 \ obj.rhs_proj(ind_start : ind_end);
%       end
%     end
    
    function error = l2_error(obj, sol, analytic)
      [ x, w, ~ ] = quadratures.line(10);
      h = obj.panels(2, :) - obj.panels(1, :);
      error = 0;
      for i = 1 : size(obj.panels, 2)
        error = error + h(i) * (sol(i) - analytic(obj.panels(1, i) ...
             + x * h(i))).^2' * w;
      end
      error = sqrt(error);
    end
    
    function panelwise_error = l2_error_panelwise(obj, sol, analytic)
      [ x, w, ~ ] = quadratures.line(10);
      h = obj.panels(2, :) - obj.panels(1, :);
      panelwise_error = zeros(1, size(obj.panels,2));
      for i = 1 : size(obj.panels, 2)
        panelwise_error(i) = panelwise_error(i) + h(i) * ...
          (sol(i) - analytic(obj.panels(1, i) + x * h(i))).^2' * w;
      end
      panelwise_error = sqrt(panelwise_error);
    end
    
    % project a function to the space of piecewise constant functions on the 
    % panels of the current object
    % ATTENTION: Works only for t_start = 0 and uniform timesteps
    function projection = apply_const_l2_project( obj, fnctn)
      projection = obj.project_rhs( fnctn );
    end
    
    % plot information about the FMM structure and return count matrix of
    % operations
    function count_matrix = print_info( obj )
      fprintf('Information about cFFM_solver_adaptive:')
      fprintf('Max level in cluster tree: %d\n', obj.max_level);
      num_string = sprintf('%d ', obj.nr_levelwise_leaves);
      fprintf('Leaves per level: %s\n', num_string);
      % print minimal level where M2L operations are done
      fprintf('Minimal M2L level: %d\n', obj.lowest_interaction_level);
      % count operations
      count_matrix = obj.count_operations(obj.cluster_tree, []);
      num_string = sprintf('%d ', count_matrix(1, :));
      fprintf('M2M operations: %s\n', num_string);
      num_string = sprintf('%d ', count_matrix(2, :));
      fprintf('L2L operations: %s\n', num_string);
      num_string = sprintf('%d ', count_matrix(3, :));
      fprintf('M2L operations: %s\n', num_string);
      num_string = sprintf('%d ', count_matrix(4, :));
      fprintf('M2T entries: %s\n', num_string);
      num_string = sprintf('%d ', count_matrix(5, :));
      fprintf('S2L entries: %s\n', num_string);
      num_string = sprintf('%d ', count_matrix(6, :));
      fprintf('S2M entries: %s\n', num_string);
      num_string = sprintf('%d ', count_matrix(7, :));
      fprintf('L2T entries: %s\n', num_string);
      num_string = sprintf('%d ', count_matrix(8, :));
      fprintf('nearfield entries: %s\n', num_string);
      fprintf('\nTotal M2T entries: %d\n', sum(count_matrix(4, :)));
      fprintf('Total S2L entries: %d\n', sum(count_matrix(5, :)));
      fprintf('Total nearfield entries: %d\n', sum(count_matrix(8, :)));
    end
    
  end
  
  methods (Access = private)
    % structure of the count_matrix:
    % row 1: m2m operations levelwise
    % row 2: l2l operations levelwise
    % row 3: m2l operations levelwise
    % row 4: m2t entries levelwise
    % row 5: s2l entries levelwise
    % row 6: s2m entries levelwise
    % row 7: l2t entries levelwise
    % row 8: nearfield entries levelwise
    % Call this function from outside with count_matrix_in = []
    function count_matrix_out = count_operations(obj, root, count_matrix_in)
      if (isempty(count_matrix_in))
        % initialize count cell
        count_matrix_out = zeros(8, obj.L+1);
        % count m2t, s2l, s2m, l2t and nearfield entries
        for l = obj.lowest_interaction_level : obj.L
          for j = 1 : obj.nr_levelwise_leaves(l+1)
            curr_leaf_cluster = obj.levelwise_leaves{l+1}{j}.get_value();
            % add number of m2t entries
            count_matrix_out(4, l+1) = count_matrix_out(4, l+1) + ...
              curr_leaf_cluster.get_nr_m2t_matrix_entries();
            % add number of s2l entries
            count_matrix_out(5, l+1) = count_matrix_out(5, l+1) + ...
              curr_leaf_cluster.get_nr_s2l_matrix_entries();
            % add number of s2m entries
            count_matrix_out(6, l+1) = count_matrix_out(6, l+1) + ...
              curr_leaf_cluster.get_n_steps() * (obj.L_order + 1);
            % add number of l2t entries
            count_matrix_out(7, l+1) = count_matrix_out(7, l+1) + ...
              curr_leaf_cluster.get_n_steps() * (obj.L_order + 1);
          end
        end
        for l = 0 : obj.L
          for j = 1 : obj.nr_levelwise_leaves(l+1)
            curr_leaf_cluster = obj.levelwise_leaves{l+1}{j}.get_value();
            % add number of nearfield entries
            count_matrix_out(8, l+1) = count_matrix_out(8, l+1) + ...
              curr_leaf_cluster.get_nr_nearfield_matrix_entries();
          end
        end
      else
        count_matrix_out = count_matrix_in;
      end
      curr_level = root.get_level();
      curr_cluster = root.get_value();

      interaction_list = curr_cluster.get_interaction_list();
      % count m2l operations of current cluster
      count_matrix_out(3, curr_level+1) = count_matrix_out(3, curr_level+1)...
        + size(interaction_list, 2);
      if (root.get_left_child() ~= 0)
        % add 1 to number of m2m and l2l operations for left child
        count_matrix_out(1, curr_level+1) = count_matrix_out(1, curr_level+1) + 1;
        count_matrix_out(2, curr_level+1) = count_matrix_out(2, curr_level+1) + 1;
        % call function recursively to count m2m, l2l and m2l ops for child
        count_matrix_out = obj.count_operations(root.get_left_child(), ...
          count_matrix_out);
      end
      if (root.get_right_child() ~= 0)
        % add 1 to number of m2m and l2l operations for right child
        count_matrix_out(1, curr_level+1) = count_matrix_out(1, curr_level+1) + 1;
        count_matrix_out(2, curr_level+1) = count_matrix_out(2, curr_level+1) + 1;
        % call function recursively to count m2m, l2l and m2l ops for child
        count_matrix_out = obj.count_operations(root.get_right_child(), ...
          count_matrix_out);
      end
      
      if (curr_level == 0)
        % correct wrong numbers for m2m and l2l operations
        count_matrix_out(1:2, 1:obj.lowest_interaction_level) = ...
          zeros(2, obj.lowest_interaction_level);
      end
    end
    
    
    function construct_cluster_tree(obj, root)
    % construct a cluster tree for arbitrary, possibly non-uniform intervals
      curr_lev = root.get_level();
      if ((curr_lev < obj.L) && ...
           (root.get_value().get_n_steps() > obj.n_max))
        % split current cluster (root) into (up to two) subclusters
        current_cluster = root.get_value();
        t_start = current_cluster.get_start();
        t_end = current_cluster.get_end();
        t_mid = 0.5 * (t_start + t_end);
        start_index = current_cluster.get_start_index();
        end_index = current_cluster.get_end_index();
        %check if cluster has only one child
        start_panel = current_cluster.get_panel(start_index);
        end_panel = current_cluster.get_panel(end_index);
        if (0.5 * (start_panel(1) + start_panel(2)) > t_mid) 
          %all midpoints of intervals are in the right half of the cluster
          right_cluster = temporal_cluster(obj.panels, start_index, ...
            end_index, t_mid, t_end, 2 * current_cluster.get_idx_nl() + 1);
          root.set_right_child(right_cluster);
          if (curr_lev == obj.L - 1) %insert new leaf
            obj.nr_leaves = obj.nr_leaves + 1;
            obj.nr_levelwise_leaves(obj.L+1) = ...
              obj.nr_levelwise_leaves(obj.L+1) + 1;
            obj.levelwise_leaves{obj.L+1}{obj.nr_levelwise_leaves(obj.L+1)}= ...
              root.get_right_child();
            obj.ordered_leaves{obj.nr_leaves} = root.get_right_child();
          end
          current_cluster.set_children(0, right_cluster);
          %recursively construct subtree
          obj.construct_cluster_tree(root.get_right_child());
        elseif (0.5 * (end_panel(1) + end_panel(2)) <= t_mid)
          %all midpoints of intervals are in the left half of the cluster
          left_cluster = temporal_cluster(obj.panels, start_index, ...
            end_index, t_start, t_mid, 2 * current_cluster.get_idx_nl());
          root.set_left_child(left_cluster);
          if (curr_lev == obj.L - 1) %insert new leaf
            obj.nr_leaves = obj.nr_leaves + 1;
            obj.nr_levelwise_leaves(obj.L+1) = ...
              obj.nr_levelwise_leaves(obj.L+1) + 1;
            obj.levelwise_leaves{obj.L+1}{obj.nr_levelwise_leaves(obj.L+1)}= ...
              root.get_left_child();
            obj.ordered_leaves{obj.nr_leaves} = root.get_left_child();
          end
          current_cluster.set_children(left_cluster, 0);
          %recursively construct subtree
          obj.construct_cluster_tree(root.get_left_child());
        else 
          %two children; find index where the cluster is split (via bisection)
          middle_index = floor((start_index + end_index) / 2);
          middle_panel = current_cluster.get_panel(middle_index);
          mid_middle_panel = 0.5 * (middle_panel(1) + middle_panel(2));
          next_panel = current_cluster.get_panel(middle_index + 1);
          mid_next_panel = 0.5 * (next_panel(1) + next_panel(2));
          while ((mid_middle_panel > t_mid) || (mid_next_panel <= t_mid))
            if (mid_middle_panel > t_mid)
              end_index = middle_index;
            else
              start_index = middle_index;
            end
            middle_index = floor((start_index + end_index) / 2);
            middle_panel = current_cluster.get_panel(middle_index);
            mid_middle_panel = 0.5 * (middle_panel(1) + middle_panel(2));
            next_panel = current_cluster.get_panel(middle_index + 1);
            mid_next_panel = 0.5 * (next_panel(1) + next_panel(2));
          end
          %reset start and end index.
          start_index = current_cluster.get_start_index();
          end_index = current_cluster.get_end_index();
          % recursively create binary subtrees for left and right descendants
          left_cluster = temporal_cluster(obj.panels, start_index, ...
            middle_index, t_start, t_mid, 2 * current_cluster.get_idx_nl());
          root.set_left_child(left_cluster);
          child_lev = curr_lev + 1;
          if (child_lev == obj.L || middle_index-start_index+1 <= obj.n_max)
            obj.nr_leaves = obj.nr_leaves + 1;
            obj.nr_levelwise_leaves(child_lev+1) = ...
              obj.nr_levelwise_leaves(child_lev+1) + 1;
            obj.levelwise_leaves{child_lev+1}{...
              obj.nr_levelwise_leaves(child_lev+1)} = root.get_left_child();
            obj.ordered_leaves{obj.nr_leaves} = root.get_left_child();
          end
          obj.construct_cluster_tree(root.get_left_child());
          
          right_cluster = temporal_cluster(obj.panels, middle_index + 1, ...
            end_index, t_mid, t_end, 2 * current_cluster.get_idx_nl() + 1);
          root.set_right_child(right_cluster);
          current_cluster.set_children(left_cluster, right_cluster);
          if (child_lev == obj.L || end_index-middle_index <= obj.n_max)
            obj.nr_leaves = obj.nr_leaves + 1;
            obj.nr_levelwise_leaves(child_lev+1) = ...
              obj.nr_levelwise_leaves(child_lev+1) + 1;
            obj.levelwise_leaves{child_lev+1}{...
              obj.nr_levelwise_leaves(child_lev+1)} = root.get_right_child();
            obj.ordered_leaves{obj.nr_leaves} = root.get_right_child();
          end
          obj.construct_cluster_tree(root.get_right_child());
        end
      elseif (curr_lev == 0)
        obj.nr_leaves = obj.nr_leaves + 1;
        obj.nr_levelwise_leaves(1) = ...
          obj.nr_levelwise_leaves(1) + 1;
        obj.levelwise_leaves{1}{obj.nr_levelwise_leaves(1)} = root;
        obj.ordered_leaves{obj.nr_leaves} = root;
      end
    end
    
    function set_pad_list (obj)
      %find needed padding length levelwise, bottom up
      obj.levelwise_pad_list = zeros(obj.L + 1, 2);
      for l = obj.L : -1 : 0
        for j = 1 : obj.nr_levelwise_leaves(l+1)
          curr_leaf = obj.levelwise_leaves{l+1}{j}.get_value();
          t_start = curr_leaf.get_start();
          start_panel = curr_leaf.get_panel(curr_leaf.get_start_index());
          t_int_start = start_panel(1);
          obj.levelwise_pad_list(l+1, 1) = ...
            max(obj.levelwise_pad_list(l+1, 1), t_start - t_int_start);
          t_end = curr_leaf.get_end();
          end_panel = curr_leaf.get_panel(curr_leaf.get_end_index());
          t_int_end = end_panel(2);
          obj.levelwise_pad_list(l+1, 2) = ...
            max(obj.levelwise_pad_list(l+1, 2), t_int_end - t_end);
        end
        if l > 0
          obj.levelwise_pad_list(l,:) = obj.levelwise_pad_list(l+1,:);
        end
      end
    end
    
    function pad_cluster_tree (obj, root)
      level = root.get_level();
      cluster = root.get_value();
      cluster.update_bounds(...
        cluster.get_start() - obj.levelwise_pad_list(level+1,1), ...
        cluster.get_end() + obj.levelwise_pad_list(level+1,2));
      if (root.get_left_child() ~= 0)
        obj.pad_cluster_tree(root.get_left_child());
      end
      if (root.get_right_child() ~= 0)
        obj.pad_cluster_tree(root.get_right_child());
      end
    end
%      function ctree = construct_cluster_tree_bottom_up(obj)
%        t_start = obj.t_start;
%        counter = 0;
%        ind_start = 0;
%        nr_intrvls = size(obj.panels,2);
%        h_t = (obj.t_end - obj.t_start)/(2^(obj.L));
%        leaf_cluster_bound = t_start + h_t : h_t : obj.t_end;
%        curr_ind = 1;
%        ind_pos_stop = zeros(1, obj.L+1);
%        j=1;
%        while (j <= nr_intrvls)
%          nr_pos_stop = 1;
%          ind_pos_stop(1) = curr_ind;
%          ref_ind = curr_ind - 1;
%          step = 1;
%          while ((mod(ref_ind,2)==0) && (nr_pos_stop < obj.L + 1))
%            nr_pos_stop = nr_pos_stop + 1;
%            step = step * 2;
%            ind_pos_stop(nr_pos_stop) = j-1 + step;
%            ref_ind = ref_ind/2;
%          end
%          if (counter == obj.N_steps)
%            mid_next = (obj.panels(1,j) + obj.panels(2,j)) * 0.5;
%            while (mid_next <= leaf_cluster_bound(ind_pos_stop(1))) %leaf 
%  cluster with more than N_steps intervals 
%              j = j + 1;
%              mid_next = (obj.panels(1,j) + obj.panels(2,j)) * 0.5;
%            end %new interval is now guaranteed to be outside of current 
%  cluster
%            mid_last = (obj.panels(1,j-1) + obj.panels(2,j-1)) * 0.5;
%            pos = 1;
%            while (leaf_cluster_bound(pos) < mid_last) %find end index of 
%  current cluster
%              pos = pos + 1;
%            end
%          j = j + 1;obj
%      end
    
    function assmbl_intrctn_nghbr_lsts(obj, root)
    %compute the neighbor lists and interaction lists for all clusters 
    %recursively using a standard admissibility criterion
      level = root.get_level();
      neighbors = {};
      interaction_list = {};
      nghbr_cntr = 0;
      intrctn_cntr = 0;
      if (level == 0)
        neighbors(1, 1) = {root.get_value()};
      else
        par_neighbors = root.get_parent().get_value().get_neighbors();
        sz = size(par_neighbors, 2);
        for i= 1 : sz-1
          % consider all children of parent's real neighbors (not its own)
          % check whether the children are admissible or not and add them 
          % to the respective lists
          left_child = par_neighbors{i}.get_left_child();
          if (left_child~=0)
            if (obj.is_nearfield(left_child, root.get_value()))
              nghbr_cntr = nghbr_cntr + 1;
              neighbors(1, nghbr_cntr) = {left_child};
            else 
              intrctn_cntr = intrctn_cntr + 1;
              interaction_list(1, intrctn_cntr) = {left_child};
            end
          end
          right_child = par_neighbors{i}.get_right_child();
          if (right_child~=0)
            if (obj.is_nearfield(right_child, root.get_value()))
              nghbr_cntr = nghbr_cntr + 1;
              neighbors(1, nghbr_cntr) = {right_child};
            else 
              intrctn_cntr = intrctn_cntr + 1;
              interaction_list(1, intrctn_cntr) = {right_child};
            end
          end
          % if parent's neighbor is a leaf it is added to current cluster's 
          % nearfield
          if ((left_child== 0) && (right_child==0))
            if (obj.use_m2t_and_s2l)
              % check whether s2l operation is admissible
              if (~obj.is_nearfield_adaptive(par_neighbors{i}, root.get_value()))
                par_neighbors{i}.append_to_s2l_list( root.get_value());
              else
                nghbr_cntr = nghbr_cntr + 1;
                neighbors(1, nghbr_cntr) = {par_neighbors{i}};
              end
            else
              nghbr_cntr = nghbr_cntr + 1;
              neighbors(1, nghbr_cntr) = {par_neighbors{i}};
            end
          end
        end
        %consider the parent's two children and add them to the nearfield list
        %if necessary
        left_branch = root.get_parent().get_left_child();
        right_branch = root.get_parent().get_right_child();
        if (right_branch ~= 0 && left_branch ~= 0 && ...
          right_branch.get_value().get_idx_nl() == root.get_value().get_idx_nl()) 
          %current cluster is right child, and both children exist
          nghbr_cntr = nghbr_cntr + 1;
          %add left cluster to the list of neighbors
          neighbors(1, nghbr_cntr) = {left_branch.get_value()};
        end
        nghbr_cntr = nghbr_cntr + 1;
        neighbors(1, nghbr_cntr) = {root.get_value()};
      end
      root.get_value().set_neighbors(neighbors);
      root.get_value().set_interaction_list(interaction_list);
      if (root.get_left_child ~= 0)
        obj.assmbl_intrctn_nghbr_lsts(root.get_left_child());
      end
      if (root.get_right_child ~= 0)
        obj.assmbl_intrctn_nghbr_lsts(root.get_right_child());
      end
      if ( obj.use_m2t_and_s2l && (root.get_left_child == 0) && ...
          (root.get_right_child == 0) )
        obj.assmbl_m2t_lst(root);
      end
    end
    
    function bool = is_nearfield(obj, src_cluster, tar_cluster)
    %((almost) standard) nearfield criterion for two intervals
    % ATTENTION: causality is taken into account; if src_cluster is in the 
    % future it is never admissible!
      src_start = src_cluster.get_start();
      src_end = src_cluster.get_end();
      tar_start = tar_cluster.get_start();
      tar_end = tar_cluster.get_end();
      bool = (obj.eta * (tar_start-src_end) <= ...
        max(src_end-src_start, tar_end-tar_start));
    end
    
    function bool = is_nearfield_adaptive(obj, src_cluster, tar_cluster)
    %((almost) standard) nearfield criterion for two intervals
    % ATTENTION: causality is taken into account; if src_cluster is in the 
    % future it is never admissible!
      src_start = src_cluster.get_start();
      src_end = src_cluster.get_end();
      tar_start = tar_cluster.get_start();
      tar_end = tar_cluster.get_end();
      bool = (obj.eta * (tar_start-src_end) <= ...
        min(src_end-src_start, tar_end-tar_start));
    end
    
    function find_lowest_interaction_level(obj, root)
      curr_cluster = root.get_value();
      if (size(curr_cluster.get_interaction_list(),2) ~= 0 || ...
          size(curr_cluster.get_s2l_list(),2) ~= 0 || ...
          size(curr_cluster.get_m2t_list(),2) ~= 0)
        obj.lowest_interaction_level = ...
          min(obj.lowest_interaction_level, root.get_level());
      else
        if (root.get_left_child() ~= 0)
          obj.find_lowest_interaction_level(root.get_left_child());
        end
        if (root.get_right_child() ~= 0)
          obj.find_lowest_interaction_level(root.get_right_child());
        end
      end
    end
    
    % assemble the l2t list for a given target cluster (given by its tree)
    function assmbl_m2t_lst(obj, root)
      tar_cluster = root.get_value();
      initial_neighbors = tar_cluster.get_neighbors();
      new_neighbors = {};
      sz = size(initial_neighbors, 2);
      for j = 1 : sz
        src_cluster = initial_neighbors{j};
        new_neighbors = obj.traverse_for_m2t_list(new_neighbors, src_cluster, ...
          tar_cluster);
      end
      tar_cluster.set_neighbors(new_neighbors);
    end
    
    % traverse the tree starting from the given source cluster to find
    % clusters for which m2t operations can be done
    % if src_cluster is a leaf it is appended to old_neighbor_list and the list
    % is returned.
    function new_neighbor_list = traverse_for_m2t_list(obj, old_neighbor_list, ...
        src_cluster, tar_cluster)
      left_src_child = src_cluster.get_left_child();
      right_src_child = src_cluster.get_right_child();
      new_neighbor_list = old_neighbor_list;
      curr_size = size(new_neighbor_list, 2);
      if (left_src_child ~= 0)
        if (obj.is_nearfield_adaptive(left_src_child, tar_cluster))
          new_neighbor_list = obj.traverse_for_m2t_list(new_neighbor_list, ...
            left_src_child, tar_cluster);
        else
          tar_cluster.append_to_m2t_list(left_src_child);
        end
      end
      if (right_src_child ~= 0)
        if (obj.is_nearfield_adaptive(right_src_child, tar_cluster))
          new_neighbor_list = obj.traverse_for_m2t_list(new_neighbor_list, ...
            right_src_child, tar_cluster);
        else
          tar_cluster.append_to_m2t_list(right_src_child);
        end
      end
      if (left_src_child == 0 && right_src_child == 0)
        curr_size = curr_size + 1;
        new_neighbor_list(1, curr_size) = {src_cluster};
      end
    end
    
    % search for a neighboring clusters
    %     function neighbor = search_neighbors(obj, root, L, idx)
    %       % if neighbor == -1 there is no neighbor except for the same cluster
    %       neighbor = -1;
    %       if (root.get_value().get_idx_nl() == idx - 1) && ...
    %           (root.get_level == L)
    %         neighbor = root.get_value();
    %       elseif root.get_level() < L
    %         neighbor = obj.search_neighbors(root.get_left_child(), L, idx);
    %         if neighbor == -1
    %           neighbor = obj.search_neighbors(root.get_right_child(), L, idx);
    %         end
    %       end
    %     end
    
    % assembles interaction list of a cluster, i.e., parents' neighbors'
    % children which are not neighbors themselves
    
    function compute_nearfield_matrices(obj)
      assembler = full_assembler_arb_timestep();
      for l = obj.L : -1 : 0
        for j = 1 : obj.nr_levelwise_leaves(l+1)
          tar_cluster = obj.levelwise_leaves{l+1}{j}.get_value();
          neighbors = tar_cluster.get_neighbors();
          sz = size(neighbors, 2);
          nearfield_matrices = cell(1, sz);
          for k = 1 : sz
            src_cluster = neighbors{k};
            nearfield_matrices{k} = assembler.assemble_V(src_cluster, tar_cluster);
          end
          tar_cluster.set_nearfield_matrices(nearfield_matrices);
        end
      end
    end
        
    
    % computes moments from input vector
    function compute_q2m(obj)
      Lagrange = lagrange_interpolant(obj.L_order);
      for l = obj.lowest_interaction_level : obj.L
        for j = 1 : obj.nr_levelwise_leaves(l+1)
          curr_cluster = obj.levelwise_leaves{l+1}{j}.get_value();
          q2m = zeros(obj.L_order + 1, curr_cluster.get_n_steps());
          for i1 = curr_cluster.get_start_index() : curr_cluster.get_end_index()
            for i2 = 0 : obj.L_order
              q2m(i2 + 1, i1 - curr_cluster.get_start_index() + 1) = ...
                obj.integrate_lagrange(curr_cluster, ...
                obj.panels(1, i1), obj.panels(2, i1), Lagrange, i2);
            end
          end
          curr_cluster.set_q2m(q2m);
        end
      end
    end
    
    % integrates Lagrange polynomials to assemble Q2M matrices
    function result = integrate_lagrange(~, cluster, t_start, t_end, ...
        interpolant, b)
      [ x, w, l ] = quadratures.line(10);
      result = 0;
      for j = 1 : l
        tau = t_start + (t_end - t_start) * x(j);
        tau_loc = cluster.map_2_local(tau);
        result = result + interpolant.lagrange(b, tau_loc) * w(j);
      end
      result = result * (t_end - t_start);
    end
    
    % assembles matrices of M2M transformations
    function compute_m2m(obj, root)
      obj.levelwise_m2m_left = cell(1, obj.L);
      obj.levelwise_m2m_right = cell(1, obj.L);
      
      Lagrange = lagrange_interpolant(obj.L_order);
      interp_nodes = Lagrange.get_nodes();
      
      root_cluster = root.get_value();
      % compute left m2m matrices iteratively
      t_start = root_cluster.get_start();
      t_start_no_pad = t_start + obj.levelwise_pad_list(1, 1);
      t_end = root_cluster.get_end();
      t_end_no_pad = t_end - obj.levelwise_pad_list(1, 2);
      %since children of clusters could be 0, the matrices are computed without 
      %traversing the tree. instead appropriate cluster bounds are computed.
      %first only left m2m matrices are computed
      for l = 0 : obj.L - 1
        t_start_child = t_start_no_pad - obj.levelwise_pad_list(l+2,1);
        t_end_child = 0.5 * (t_end_no_pad + t_start_no_pad) + ...
          obj.levelwise_pad_list(l+2,2);
        if (l >= obj.lowest_interaction_level)
          obj.levelwise_m2m_left{l+1} = zeros(obj.L_order+1, obj.L_order+1);
          center_child = 0.5 * (t_end_child + t_start_child);
          h_child = 0.5 * (t_end_child - t_start_child);
          center_parent = 0.5 * (t_end + t_start);
          h_parent = 0.5 * (t_end - t_start);
          local_interp_nodes = 1 / h_parent * ...
            (center_child + h_child .* interp_nodes - center_parent);
          for i = 0 : obj.L_order
            obj.levelwise_m2m_left{l+1}(i+1, :) = ...
              Lagrange.lagrange(i, local_interp_nodes);
          end
        end
        t_start = t_start_child;
        t_end = t_end_child;
        t_start_no_pad = t_start + obj.levelwise_pad_list(l+2,1);
        t_end_no_pad = t_end - obj.levelwise_pad_list(l+2,2);
      end
      
      %reset bounds of root cluster
      t_start = root_cluster.get_start();
      t_start_no_pad = t_start + obj.levelwise_pad_list(1, 1);
      t_end = root_cluster.get_end();
      t_end_no_pad = t_end - obj.levelwise_pad_list(1, 2);
      %compute right m2m matrices
      for l = 0: obj.L - 1
        t_start_child = 0.5 * (t_start_no_pad + t_end_no_pad) - ...
          obj.levelwise_pad_list(l+2,1);
        t_end_child = t_end_no_pad + obj.levelwise_pad_list(l+2,2);
        if (l >= obj.lowest_interaction_level)
          obj.levelwise_m2m_right{l+1} = zeros(obj.L_order+1, obj.L_order+1);
          center_child = 0.5 * (t_end_child + t_start_child);
          h_child = 0.5 * (t_end_child - t_start_child);
          center_parent = 0.5 * (t_end + t_start);
          h_parent = 0.5 * (t_end - t_start);
          local_interp_nodes = 1 / h_parent * ...
            (center_child + h_child * interp_nodes - center_parent);
          for i = 0 : obj.L_order
            obj.levelwise_m2m_right{l+1}(i+1, :) = ...
              Lagrange.lagrange(i, local_interp_nodes);
          end
        end
        t_start = t_start_child;
        t_end = t_end_child;
        t_start_no_pad = t_start + obj.levelwise_pad_list(l+2,1);
        t_end_no_pad = t_end - obj.levelwise_pad_list(l+2,2);
      end
    end
    
    function find_m2l (obj, root)
      curr_cluster = root.get_value();
      level = root.get_level();
      curr_intrctn_list = curr_cluster.get_interaction_list();
      sz = size(curr_intrctn_list, 2);
      m2l = zeros(1,sz);
      for j=1:sz
        src_cluster = curr_intrctn_list{j};
        center_diff = curr_cluster.get_center() - src_cluster.get_center();
        %check if the current difference is already existent in the array of 
        %all m2l differences at the current level. 
        pos = find(obj.levelwise_m2l_diff{level+1} > center_diff - 1e-8 & ...
          obj.levelwise_m2l_diff{level+1} < center_diff + 1e-8);
        if isempty(pos)
          %if the current difference is not found, add it and assign its
          %position to the m2l list of the current cluster
          obj.nr_levelwise_m2l(level+1) = obj.nr_levelwise_m2l(level+1) + 1;
          obj.levelwise_m2l_diff{level+1}(1, obj.nr_levelwise_m2l(level+1)) =...
            center_diff;
          m2l(j) = obj.nr_levelwise_m2l(level+1);
        else
          m2l(j) = pos;
        end
      end
      curr_cluster.set_m2l(m2l);
      if (root.get_left_child() ~= 0)
        obj.find_m2l(root.get_left_child());
      end
      if (root.get_right_child() ~= 0)
        obj.find_m2l(root.get_right_child());
      end
    end
    
    % computes M2L matrices for clusters from the interaction list of a
    % given cluster
    function compute_m2l(obj)
      %initialize structure where m2l matrices are stored
      obj.levelwise_m2l_mat = cell(obj.L+1, max(obj.nr_levelwise_m2l));
      %search for largest level in which there is a leaf
      level = obj.L;
      while (obj.nr_levelwise_leaves(level+1) == 0)
        level = level - 1;
      end
      %choose an arbitrary leaf on the level
      curr_leaf = obj.levelwise_leaves{level+1}{1};
      %initialize structures for interpolation
      Lagrange = lagrange_interpolant(obj.L_order);
      nodes = Lagrange.get_nodes();
      %compute m2l matrices levelwise starting at the bottom
      while (level >= 0)
        tar_cluster = curr_leaf.get_value();
        tar_nodes = tar_cluster.map_2_global(nodes);
        for j = 1 : obj.nr_levelwise_m2l(level+1)
          src_nodes = tar_nodes - obj.levelwise_m2l_diff{level+1}(j);
          obj.levelwise_m2l_mat{level+1, j} = ...
            obj.eval_kernel(repmat(tar_nodes', 1, obj.L_order+1) - ...
            repmat(src_nodes, obj.L_order+1, 1));
        end
        curr_leaf = curr_leaf.get_parent();
        level = level - 1;
      end
    end
    
    % evaluates the kernel function
    function kernel = eval_kernel(~, t)
      kernel = (1 ./ sqrt(4 * pi * t)) .* (1 - exp(- 1 ./ t));
    end
    
    % evaluates the anti derivatvie of the kernel 
    function y = eval_antiderivative_kernel(~,t)
      y = sqrt(t ./ pi) .* (1 - exp(-1 ./ t)) + erfc(1 ./ sqrt(t));
    end
      
    %     function binary_array = d2b(obj, decimal, len)
    %       binary_string = dec2bin(decimal);
    %       binary_array = zeros(1, len);
    %       sz = size(binary_string, 2);
    %       for i = 1 : sz
    %         binary_array(i) = str2double(binary_string(sz - i + 1));
    %       end
    %     end
    
    function compute_m2t_matrices(obj)
      compute_analytically = true;
      if (~compute_analytically)
        [quad_points, quad_weights, nr_quad] = quadratures.line(10);
      end
      Lagrange = lagrange_interpolant(obj.L_order);
      nodes = Lagrange.get_nodes();
      for l = obj.L : -1 : obj.lowest_interaction_level
        for j = 1 : obj.nr_levelwise_leaves(l+1)
          tar_cluster = obj.levelwise_leaves{l+1}{j}.get_value();
          tar_start_ind = tar_cluster.get_start_index();
          tar_end_ind = tar_cluster.get_end_index();
          nr_tar_intervals = tar_end_ind - tar_start_ind + 1;
          m2t_list = tar_cluster.get_m2t_list();
          sz = size(m2t_list, 2);
          m2t_matrices = cell(1, sz);
          for k = 1 : sz
            src_cluster = m2t_list{k};
            src_nodes = src_cluster.map_2_global(nodes);
            m2t_matrices{k} = zeros(nr_tar_intervals, obj.L_order+1);
            for n = tar_start_ind : tar_end_ind
              curr_panel = tar_cluster.get_panel(n);
              t_start = curr_panel(1);
              t_end = curr_panel(2);
              if (compute_analytically)
                m2t_matrices{k}(n - tar_start_ind + 1, :) = ... 
                  obj.eval_antiderivative_kernel(t_end - src_nodes) ...
                  - obj.eval_antiderivative_kernel(t_start - src_nodes);
              else
                h = t_end - t_start;
                m2t_matrices{k}(n - tar_start_ind + 1, :) = h * quad_weights' * ...
                  obj.eval_kernel(t_start + h * repmat(quad_points, 1, obj.L_order+1) - ...
                  repmat(src_nodes, nr_quad, 1));
              end
            end
          end
          tar_cluster.set_m2t_matrices(m2t_matrices);
        end
      end
    end
    
    function compute_s2l_matrices(obj)
      compute_analytically = true;
      if (~compute_analytically)
        [quad_points, quad_weights, nr_quad] = quadratures.line(10);
      end
      Lagrange = lagrange_interpolant(obj.L_order);
      nodes = Lagrange.get_nodes();
      for l = obj.L : -1 : obj.lowest_interaction_level
        for j = 1 : obj.nr_levelwise_leaves(l+1)
          src_cluster = obj.levelwise_leaves{l+1}{j}.get_value();
          src_start_ind = src_cluster.get_start_index();
          src_end_ind = src_cluster.get_end_index();
          nr_src_intervals = src_end_ind - src_start_ind + 1;
          s2l_list = src_cluster.get_s2l_list();
          sz = size(s2l_list, 2);
          s2l_matrices = cell(1, sz);
          for k = 1 : sz
            tar_cluster = s2l_list{k};
            tar_nodes = tar_cluster.map_2_global(nodes);
            s2l_matrices{k} = zeros(obj.L_order+1, nr_src_intervals);
            for n = src_start_ind : src_end_ind
              curr_panel = src_cluster.get_panel(n);
              t_start = curr_panel(1);
              t_end = curr_panel(2);
              if (compute_analytically)
                s2l_matrices{k}(:, n - src_start_ind + 1) = ... 
                  obj.eval_antiderivative_kernel(tar_nodes' - t_start) - ...
                  obj.eval_antiderivative_kernel(tar_nodes' - t_end);
              else
                h = t_end - t_start;
                s2l_matrices{k}(:, n - src_start_ind + 1) = h * ...
                  obj.eval_kernel(repmat(tar_nodes', 1, nr_quad) - ...
                  (t_start + h * repmat(quad_points', obj.L_order+1, 1))) * ...
                  quad_weights;
              end
            end
          end
          src_cluster.set_s2l_matrices(s2l_matrices);
        end
      end
    end
    
    
    % goes from the level 2 and transfers local expansion coefficients to
    % the lower levels
    function downward_pass(obj, root, m)
      
      if (size(root.get_value().get_local_expansion(), 1) ~= 0)
        if root.get_value().get_left_child().get_idx_nl() == ...
            obj.get_id_on_level(m, root.get_level() + 1)
          left_exp = root.get_value().apply_l2l_left();
          root.get_left_child().get_value().add_expansion(left_exp);
        end
        if root.get_value().get_right_child().get_idx_nl() == ...
            obj.get_id_on_level(m, root.get_level() + 1)
          right_exp = root.get_value().apply_l2l_right();
          root.get_right_child().get_value().add_expansion(right_exp);
        end
      end
      
      if root.get_level() < obj.L - 1
        if root.get_value().get_right_child().get_idx_nl() == ...
            obj.get_id_on_level(m, root.get_level() + 1)
          obj.downward_pass(root.get_right_child(), m);
        end
        if root.get_value().get_left_child().get_idx_nl() == ...
            obj.get_id_on_level(m, root.get_level() + 1)
          obj.downward_pass(root.get_left_child(), m);
        end
      end
    end
    
    % transfers moments upwards
    function upward_pass(obj, root)
      curr_level = root.get_level();
      if (root.get_left_child() ~= 0)
        % first m2m for left subtree
        obj.upward_pass(root.get_left_child());
        if (curr_level >= obj.lowest_interaction_level)
          % m2m from left child to current cluster
          parent_cluster = root.get_value();
          parent_moments = parent_cluster.get_moments();
          left_child = parent_cluster.get_left_child();
          l_child_moments = left_child.get_moments();
          %initialize parent moment to 0 vector if necessary
          if (size(parent_moments, 1) == 0)
            parent_moments = zeros(size(l_child_moments));
          end
          %m2m operation
          parent_moments = parent_moments + ...
            obj.levelwise_m2m_left{curr_level+1} * l_child_moments;
          parent_cluster.set_moments(parent_moments);
          parent_cluster.inc_c_moments_count();
        end
      end

      if (root.get_right_child() ~= 0)
        % first m2m for right child
        obj.upward_pass(root.get_right_child());
        % m2m from right child to current cluster
        if (curr_level >= obj.lowest_interaction_level)
          parent_cluster = root.get_value();
          parent_moments = parent_cluster.get_moments();
          right_child = parent_cluster.get_right_child();
          r_child_moments = right_child.get_moments();
          %initialize parent moment to 0 vector if necessary
          if (size(parent_moments, 1) == 0)
            parent_moments = zeros(size(r_child_moments));
          end
          %m2m operation
          parent_moments = parent_moments + ...
            obj.levelwise_m2m_right{curr_level+1} * r_child_moments;
          parent_cluster.set_moments(parent_moments);
          parent_cluster.inc_c_moments_count();
        end
      end
    end
    
    % interaction phase
    function farfield_interaction(obj, root)
      curr_level = root.get_level();
      curr_cluster = root.get_value();
      interaction_list = curr_cluster.get_interaction_list();
      if (curr_level >= obj.lowest_interaction_level)
        local_expansion = zeros(obj.L_order+1, 1);
        for n = 1 : size(interaction_list, 2)
          source_cluster = interaction_list{n};
          source_moments = source_cluster.get_moments();
          m2l_list = curr_cluster.get_m2l();
          local_expansion = local_expansion + ...
            obj.levelwise_m2l_mat{curr_level+1, m2l_list(n)} * source_moments;
        end
        curr_cluster.set_local_expansion(local_expansion);
      end
      if (root.get_left_child() ~= 0)
        obj.farfield_interaction(root.get_left_child());
      end
      if (root.get_right_child() ~= 0)
        obj.farfield_interaction(root.get_right_child());
      end
    end
    
    function apply_s2l_operations(obj, x)
      for l = obj.lowest_interaction_level : obj.L
        for j = 1 : obj.nr_levelwise_leaves(l+1)
          src_cluster = obj.levelwise_leaves{l+1}{j}.get_value();
          src_start_ind = src_cluster.get_start_index();
          src_end_ind = src_cluster.get_end_index();
          sources = x(src_start_ind : src_end_ind);
          s2l_list = src_cluster.get_s2l_list();
          sz = size(s2l_list, 2);
          for k = 1 : sz
            tar_cluster = s2l_list{k};
            tar_cluster.add_expansion(src_cluster.apply_s2l_mat(sources, k));
          end
        end
      end
    end
    
    function y = apply_m2t_operations(obj)
      y = zeros(size(obj.panels, 2), 1);
      for l = obj.lowest_interaction_level : obj.L
        for j = 1 : obj.nr_levelwise_leaves(l+1)
          tar_cluster = obj.levelwise_leaves{l+1}{j}.get_value();
          tar_start_ind = tar_cluster.get_start_index();
          tar_end_ind = tar_cluster.get_end_index();
          m2t_list = tar_cluster.get_m2t_list();
          sz = size(m2t_list, 2);
          for k = 1 : sz
            src_cluster = m2t_list{k};
            moments = src_cluster.get_moments();
            y(tar_start_ind : tar_end_ind) = y(tar_start_ind : tar_end_ind) + ...
              tar_cluster.apply_m2t_mat(moments, k);
          end
        end
      end
    end
    
    % evaluate the nearfield for a vector x
    function y = nearfield_evaluation(obj, x)
      y = zeros(size(x));
      for l = 0 : obj.L
        for j = 1 : obj.nr_levelwise_leaves(l+1)
          tar_cluster = obj.levelwise_leaves{l+1}{j}.get_value();
          tar_start_ind = tar_cluster.get_start_index();
          tar_end_ind = tar_cluster.get_end_index();
          neighbors = tar_cluster.get_neighbors();
          sz = size(neighbors, 2);
          for k = 1 : sz
            src_cluster = neighbors{k};
            src_start_ind = src_cluster.get_start_index();
            src_end_ind = src_cluster.get_end_index();
            y(tar_start_ind:tar_end_ind) = y(tar_start_ind:tar_end_ind) + ...
              tar_cluster.apply_nearfield_mat(x(src_start_ind:src_end_ind), k);
          end
        end
      end
    end
            
            
    
    % goes from level 2 and transfers local expansion coefficients to
    % the lower levels
    function downward_pass_std(obj, root)
      curr_level = root.get_level();
      if (curr_level >= obj.lowest_interaction_level)
        parent_expansion = root.get_value().get_local_expansion();
      end
      if (root.get_left_child() ~= 0)
        if (curr_level >= obj.lowest_interaction_level)
          % do l2l for the left son
          % l2l matrix is transposed m2m matrix
          left_exp = obj.levelwise_m2m_left{curr_level+1}' * parent_expansion;
          root.get_left_child().get_value().add_expansion(left_exp);
        end
        % do l2l in the left subtree
        obj.downward_pass_std(root.get_left_child());
      end
      if (root.get_right_child() ~= 0)
        if (curr_level >= obj.lowest_interaction_level)
          % do l2l for the right son (again with transposed m2m matrix)
          right_exp = obj.levelwise_m2m_right{curr_level+1}' * parent_expansion;
          root.get_right_child().get_value().add_expansion(right_exp);
        end
        % do l2l in the right subtree
        obj.downward_pass_std(root.get_right_child());
      end
    end
    
    % returns id of the m-th leaf on the given level
    function id = get_id_on_level(obj, m, level)
      id = m;
      for i = 1 : obj.L - level
        id = floor(id / 2);
      end
      
    end
    
    % L2 projection of the RHS
    function projection = project_rhs(obj, rhs)
      n_intervals = size(obj.panels, 2);
      [ x, w, l ] = quadratures.line(10);
      
      projection = zeros(n_intervals, 1);
      for j = 1 : l
        projection = projection + w(j) * rhs(obj.panels(1, :)' + ...
          x (j) * obj.h_panels');
        % for numerical integration a factor h comes into play, which cancels
        % with the factor 1/h from the projection
      end
    end
    

    
    
    % computes M2L matrices for clusters from the interaction list of a
    % given cluster
    function reset(obj, root)
      current_cluster = root.get_value();
      current_cluster.reset();
      if root.get_left_child() ~= 0
        obj.reset(root.get_left_child());
      end
      if root.get_right_child() ~= 0
        obj.reset(root.get_right_child());
      end
    end
    
  end
end
