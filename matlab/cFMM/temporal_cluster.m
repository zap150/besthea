classdef temporal_cluster < handle
  %TEMPORAL_INTERVAL One temporal cluster in FMM 
  %   Serves as node in a temporal tree
  
  properties ( Access = private )
    t_start
    t_end
    center
    panels
    start_index
    end_index
    n_steps
    half_size
    q2m
    m2m_left
    m2m_right
    m2l
    neighbors
    interaction_list
    idx_nl
    left_child
    right_child
    moments
    local_expansion
    children_moments_count
    left_child_translated
    % the following structures are only used in case of non-constant time steps
    nearfield_matrices 
    % list of all target clusters, for which an s2l operation is done
    % ATTENTION: Stored in the corresponding source cluster!
    s2l_list
    % cell containing the s2l matrices according to the clusters in s2l_list
    s2l_matrices
    % list of all source clusters, from which an m2t operation is done
    % Stored in the corresponding target cluster!
    m2t_list
    % cell containing the m2t matrices according to the clusters in m2t_list
    m2t_matrices
  end
  
  methods
    function obj = temporal_cluster( panels, start_index, end_index, ...
                                     t_start, t_end, idx_nl )
      obj.panels = panels;
      obj.start_index = start_index;
      obj.end_index = end_index;
      obj.n_steps = end_index - start_index + 1;
      obj.t_start = t_start;
      obj.t_end = t_end;
      obj.center = ( obj.t_start + obj.t_end ) / 2;
      obj.half_size = ( obj.t_end - obj.t_start ) / 2;
      obj.idx_nl = idx_nl;
      obj.children_moments_count = 0;
      obj.left_child_translated = 0;
      obj.left_child = 0;
      obj.right_child = 0;
      obj.s2l_list = {};
      obj.m2t_list = {};
    end
    
    function panels = get_panels( obj )
      panels = obj.panels( :, obj.start_index : obj.end_index );
    end
    
    function panel = get_panel( obj, global_index )
      panel = obj.panels( :, global_index );
    end
    
    function t_start = get_start( obj )
      t_start = obj.t_start;
    end
    
    function t_end = get_end( obj )
      t_end = obj.t_end;
    end
   
    function update_bounds (obj, t_start, t_end)
      obj.t_start = t_start;
      obj.t_end = t_end;
      obj.center = ( obj.t_start + obj.t_end ) / 2;
      obj.half_size = ( obj.t_end - obj.t_start ) / 2;
    end
   
    function center = get_center( obj )
      center = obj.center;
    end
    
    function start_index = get_start_index( obj )
      start_index = obj.start_index;
    end
    
    function end_index = get_end_index( obj )
      end_index = obj.end_index;
    end
    
    function print( obj )
      fprintf('Temporal cluster with t_start = %f, t_end = %f, start_id = %d, end_id = %d, idx_nl=%d\n', ...
        [ obj.t_start, obj.t_end, obj.start_index, obj.end_index, obj.idx_nl ] );
    end
    
    function local_t = map_2_local( obj, t )
      % maps from global coordinates to local system on [-1, 1]
      local_t = ( t - obj.center ) ./ obj.half_size;
    end
    
    function global_t = map_2_global( obj, t_hat )
      % maps from local coordinates on [-1,1] to global 
      global_t = obj.center + obj.half_size .* t_hat;
    end
    
    function set_q2m( obj, q2m )
      obj.q2m = q2m;
    end
    
    function n_steps = get_n_steps( obj ) 
      n_steps = obj.n_steps;
    end
    
    function set_m2m( obj, m2m_left, m2m_right )
      obj.m2m_left = m2m_left;
      obj.m2m_right = m2m_right;
    end
    
    function idx_nl = get_idx_nl( obj )
      idx_nl = obj.idx_nl;
    end
    
    function set_neighbors( obj, neighbors )
      obj.neighbors = neighbors;
    end
    
    function neighbors = get_neighbors( obj )
      neighbors = obj.neighbors;
    end
    
    function set_nearfield_matrices( obj, nearfield_matrices )
      obj.nearfield_matrices = nearfield_matrices;
    end
    
    function nr_nearfield_matrix_entries = get_nr_nearfield_matrix_entries(obj)
      nr_nearfield_matrix_entries = 0;
      for i = 1 : size(obj.nearfield_matrices, 2)
        nr_nearfield_matrix_entries = nr_nearfield_matrix_entries + ...
          numel(obj.nearfield_matrices{i});
      end
    end
    
    function set_interaction_list( obj, interaction_list )
      obj.interaction_list = interaction_list;
    end
    
    function interaction_list = get_interaction_list( obj )
      interaction_list = obj.interaction_list;
    end
    
    function set_children( obj, left_child, right_child )
      obj.left_child = left_child;
      obj.right_child = right_child;
    end
    
    function left_child = get_left_child( obj )
      left_child = obj.left_child;
    end
    
    function right_child = get_right_child( obj )
      right_child = obj.right_child;
    end
    
    function set_m2l( obj, m2l )
      obj.m2l = m2l;
    end
    
    function result = apply_q2m( obj, x )
      result = obj.q2m * x;
    end
    
    function apply_m2m_left( obj, x )
      if obj.left_child_translated == 0
        if size( obj.moments, 1 ) == 0 %set moment to 0 if it was not set before
          obj.moments = zeros( size( x ) );
        end
        result = obj.m2m_left' * x;
        obj.moments = obj.moments + result;
        obj.inc_c_moments_count( );
        obj.left_child_translated = 1;
      end
    end
    
    function apply_m2m_right( obj, x )
      if size( obj.moments, 1 ) == 0 
        obj.moments = zeros( size( x ) );
      end
      result = obj.m2m_right' * x;
      obj.moments = obj.moments + result;
      obj.inc_c_moments_count( );
    end
    
    function set_moments( obj, moments )
      obj.moments = moments;
    end
    
    function moments = get_moments( obj ) 
      moments = obj.moments;
    end
    
    function set_local_expansion ( obj, local_expansion )
      obj.local_expansion = local_expansion;
    end
    
    function inc_c_moments_count ( obj )
      obj.children_moments_count = obj.children_moments_count + 1;
    end
    
    function count = get_c_moments_count( obj )
      count = obj.children_moments_count;
    end
    
    function apply_m2l( obj )
      for n = 1 : size( obj.interaction_list, 2 )
        M2L = obj.m2l{ 1, n };
        remote_cluster = obj.interaction_list{ 1,  n };
        if size( obj.local_expansion, 1 ) == 0
          obj.local_expansion = zeros( size( remote_cluster.moments ) );
        end
        obj.local_expansion = obj.local_expansion + ...
          M2L * remote_cluster.moments;
      end
    end
    
    % return m2l (in case of adaptive cfmm solver it is a list of indices of the 
    % needed m2l matrices;
    function m2l = get_m2l( obj )
      m2l = obj.m2l;
    end
    
    function local_expansion = get_local_expansion( obj )
      local_expansion = obj.local_expansion;
    end
    
    function left_child_expansion = apply_l2l_left( obj )
      if size( obj.local_expansion, 1 ) ~= 0
        left_child_expansion = obj.m2m_left * obj.local_expansion;
      end
    end
    
    function right_child_expansion = apply_l2l_right( obj )
      if size( obj.local_expansion, 1 ) ~= 0
        right_child_expansion = obj.m2m_right * obj.local_expansion;
      end
    end
    
    function add_expansion( obj, parent_expansion )
      if size( obj.local_expansion, 1 ) == 0
        obj.local_expansion = zeros( size( parent_expansion ) );
      end
      obj.local_expansion = obj.local_expansion + parent_expansion;
    end
    
    function potential = apply_l2p( obj )
      potential = obj.q2m' * obj.local_expansion;
    end
    
    % apply the j-th nearfield matrix (stored in nearfield_matrices) to x
    function y = apply_nearfield_mat( obj, x, j )
      y = obj.nearfield_matrices{j} * x;
    end
    
    % apply the j-th m2t matrix (stored in m2t_matrices) to given moment
    function y = apply_m2t_mat( obj, moments, j )
      y = obj.m2t_matrices{j} * moments;
    end
    
    % apply the j-th s2l matrix (stored in s2l_matrices) to given vector x
    function expansion = apply_s2l_mat( obj, x, j )
      expansion = obj.s2l_matrices{j} * x;
    end
    
    % apply the inverse of the j-th nearfiels matrix (stored in
    % nearfield_matrices) to x
    function x = apply_inv_nearfield_mat( obj, y, j)
      x = obj.nearfield_matrices{j} \ y;
    end
    
    % add a cluster to s2l_list
    function append_to_s2l_list( obj, target_cluster )
      sz = size( obj.s2l_list, 2 );
      obj.s2l_list( 1, sz + 1 ) = { target_cluster };
    end
    
    % add a cluster to m2t_list
    function append_to_m2t_list( obj, source_cluster )
      sz = size( obj.m2t_list, 2 );
      obj.m2t_list( 1, sz + 1 ) = { source_cluster };
    end
    
    function s2l_list = get_s2l_list( obj )
      s2l_list = obj.s2l_list;
    end
    
    function set_m2t_matrices( obj, m2t_matrices )
      obj.m2t_matrices = m2t_matrices;
    end
    
    function nr_m2t_matrix_entries = get_nr_m2t_matrix_entries(obj)
      nr_m2t_matrix_entries = 0;
      for i = 1 : size(obj.m2t_matrices, 2)
        nr_m2t_matrix_entries = nr_m2t_matrix_entries + ...
          numel(obj.m2t_matrices{i});
      end
    end
    
    function set_s2l_matrices( obj, s2l_matrices )
      obj.s2l_matrices = s2l_matrices;
    end
    
    function nr_s2l_matrix_entries = get_nr_s2l_matrix_entries(obj)
      nr_s2l_matrix_entries = 0;
      for i = 1 : size(obj.s2l_matrices, 2)
        nr_s2l_matrix_entries = nr_s2l_matrix_entries + ...
          numel(obj.s2l_matrices{i});
      end
    end
    
    function m2t_list = get_m2t_list( obj )
      m2t_list = obj.m2t_list;
    end
    
    function reset( obj )
      obj.local_expansion = [];
      obj.children_moments_count = 0;
      obj.moments = [];
      obj.left_child_translated = 0;
    end
    
  end
end

