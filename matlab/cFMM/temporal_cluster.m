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
  end
  
  methods
    function obj = temporal_cluster( panels, start_index, end_index, idx_nl )
      obj.panels = panels;
      obj.start_index = start_index;
      obj.end_index = end_index;
      obj.n_steps = end_index - start_index + 1;
      obj.t_start = panels( 1, start_index );
      obj.t_end = panels( 2, end_index );
      obj.center = ( obj.t_start + obj.t_end ) / 2;
      obj.half_size = ( obj.t_end - obj.t_start ) / 2;
      obj.idx_nl = idx_nl;
      obj.children_moments_count = 0;
      obj.left_child_translated = 0;
    end
    
    function panels = get_panels( obj )
      panels = obj.panels( :, start );
    end
    
    function t_start = get_start( obj )
      t_start = obj.t_start;
    end
    
    function t_end = get_end( obj )
      t_end = obj.t_end;
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
      local_t = ( t - obj.center ) / obj.half_size;
    end
    
    function global_t = map_2_global( obj, t_hat )
      % maps from local coordinates on [-1,1] to global 
      global_t = obj.center + t_hat * obj.half_size;
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
        if size( obj.moments, 1 ) == 0 
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
    
  end
end

