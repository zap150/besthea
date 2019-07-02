classdef cFMM_solver < handle
  %CFMM_SOLVER Summary of this class goes here
  %   Detailed explanation goes here
  
  properties ( Access = private )  
    t_start
    t_end
    rhs_fun
    rhs_proj
    cluster_tree
    % number of time steps per cluster on the finest level
    N_steps
    % ht corresponding to N_steps
    ht
    % levels of a cluster tree
    L
    % temporal panels
    panels
    % order of the Lagrange inteprolation
    L_order
    % nearfield clusters
    cluster_1
    cluster_2
    
    % leaf nodes of the tree
    leafs
    
    % nearfield
    V1
    V2
  end
  
  methods
    
    function obj = cFMM_solver( t_start, t_end, N_steps, L, ...
        L_order, rhs_fun )
      obj.t_start = t_start;
      obj.t_end = t_end;
      obj.rhs_fun = rhs_fun;
      obj.N_steps = N_steps;
      obj.L = L;
      obj.L_order = L_order;
      
      finest_cluster_length = ( t_end - t_start ) / 2^L;
      obj.ht = finest_cluster_length / N_steps;
      obj.panels = zeros( 2, 2^L * N_steps );
      obj.panels( 1, : ) = t_start : obj.ht : t_end - obj.ht;
      obj.panels( 2, : ) = t_start + obj.ht : obj.ht : t_end;
      
      % contruct temporal cluster tree
      obj.leafs = cell( 1, 2^L );
      cluster = temporal_cluster( obj.panels, 1, 2^L * N_steps, 0 );
      obj.cluster_tree = binary_tree( cluster, 0, 0 );
      obj.construct_cluster_tree( obj.cluster_tree );
      obj.assemble_neighbor_lists( obj.cluster_tree );
      obj.assemble_interaction_list( obj.cluster_tree );
      obj.compute_q2m( obj.cluster_tree );
      obj.compute_m2m( obj.cluster_tree );
      obj.compute_m2l( obj.cluster_tree );
      %obj.cluster_tree.print( );
      
      % RHS assembly
      obj.rhs_proj = obj.project_rhs( obj.rhs_fun, t_end - t_start, N_steps * 2^L );
      obj.rhs_proj = obj.rhs_proj * obj.ht;
      
      % assemble nearfield matrices
      assembler = full_assembler( );
      

      obj.V1 = assembler.assemble_V( obj.leafs{1}.get_value( ), ...
        obj.leafs{1}.get_value( ) );
      obj.V2 = assembler.assemble_V( obj.leafs{2}.get_value( ), ...
        obj.leafs{1}.get_value( ) );
% 
%       t = t_start + obj.ht : obj.ht : t_end;
%       plot( t, x);
%       hold on
    end
    
    function y = apply_fmm_matrix( obj, x )
      
     % downward pass
     parent = obj.leafs{ 1 };
     for l = obj.L : -1 : 1 
        parent = parent.get_parent( );
     end
     obj.reset( parent ); 
     
     y = zeros( size( obj.rhs_proj ) );
      
     n = obj.N_steps;
     y( 1 : n ) = obj.V1 * x( 1 : n );
     y( n + 1 : 2 * n ) = obj.V1 * x( n + 1 : 2 * n ) + obj.V2 * x( 1 : n ) ;
      
     for m = 2 : 2^obj.L - 1 %  -1 ?
        % find the coarsest level where parents change
        lt = 2;
        
        for i = 2 : obj.L - 1
          if obj.get_id_on_level( m - 1, i ) == obj.get_id_on_level( m, i )
            lt = lt + 1;
          end
        end
        
        % compute moments
        cluster = obj.leafs{ m - 2 + 1 }.get_value( );
        ind_start = cluster.get_start_index( );
        ind_end = cluster.get_end_index( );
        cluster.set_moments( cluster.apply_q2m( x( ind_start : ind_end ) ) );
        parent_node = obj.leafs{ m - 2 + 1 }.get_parent( );
        cluster.inc_c_moments_count( );
        cluster.inc_c_moments_count( );
        
        % upward path
        % transfer moments to the highest possible level
        for l = obj.L - 1 : -1 :  2
          parent_cluster = parent_node.get_value( );
          left_child = parent_cluster.get_left_child( );
          right_child = parent_cluster.get_right_child( );

          % transfer up only if there are contributions from both descendants
          % included
          if ( left_child.get_c_moments_count( ) == 2 ) 
            parent_cluster.apply_m2m_left( left_child.get_moments( ) );
          end

          if ( right_child.get_c_moments_count( ) == 2 )
            parent_cluster.apply_m2m_right( right_child.get_moments( ) );
          end
          parent_node = parent_node.get_parent( );
        end
        
        % interaction phase
        % go through interaction lists of the clusters and apply M2L
        cluster = obj.leafs{ m + 1 }.get_value( ); 
        parent_node = obj.leafs{ m + 1 };
        for l = obj.L : -1 : lt 
          cluster.apply_m2l( );
          parent_node = parent_node.get_parent( );
          cluster = parent_node.get_value( );
        end
        
        % downward pass
        parent = obj.leafs{ m + 1 };
        for l = obj.L : -1 : lt 
          parent = parent.get_parent( );
        end

        obj.downward_pass( parent, m );
       
        cluster = obj.leafs{ m + 1 }.get_value( ); 
        
        f = cluster.apply_l2p( );
        ind_start = obj.leafs{ m - 1 + 1 }.get_value( ).get_start_index( );
        ind_end = obj.leafs{ m - 1 + 1 }.get_value( ).get_end_index( );
        tst = obj.V2 * x( ind_start : ind_end );
        f = f + tst; %V2 * x( ind_start : ind_end );
        
        ind_start = obj.leafs{ m  + 1 }.get_value( ).get_start_index( );
        ind_end = obj.leafs{ m + 1 }.get_value( ).get_end_index( );
        tst = obj.V1 * x( ind_start : ind_end );
        f = f + tst;
        
        y( ind_start : ind_end ) = f; %V1 \ obj.rhs_proj( ind_start : ind_end );
      end
      
    end
    
    function x = solve_iterative( obj )
      x = gmres(@obj.apply_fmm_matrix, obj.rhs_proj, 100, 1e-4, 400);
    end
    
    function x = solve_direct( obj ) 
      
      x = zeros( size( obj.rhs_proj ) );
      
      n = obj.N_steps;
      x( 1 : n ) = obj.V1 \ obj.rhs_proj( 1 : n );
      x( n + 1 : 2 * n ) = obj.V1 \ ...
        ( obj.rhs_proj( n + 1 : 2 * n ) - obj.V2 * x( 1 : n ) );
      
      for m = 2 : 2^obj.L - 1 %  -1 ?
        % find the coarsest level where parents change
        lt = 2;
        
        for i = 2 : obj.L - 1
          if obj.get_id_on_level( m - 1, i ) == obj.get_id_on_level( m, i )
            lt = lt + 1;
          end
        end
        
        % compute moments
        cluster = obj.leafs{ m - 2 + 1 }.get_value( );
        ind_start = cluster.get_start_index( );
        ind_end = cluster.get_end_index( );
        cluster.set_moments( cluster.apply_q2m( x( ind_start : ind_end ) ) );
        parent_node = obj.leafs{ m - 2 + 1 }.get_parent( );
        cluster.inc_c_moments_count( );
        cluster.inc_c_moments_count( );
        
        % upward path
        % transfer moments to the highest possible level
        for l = obj.L - 1 : -1 :  2
          parent_cluster = parent_node.get_value( );
          left_child = parent_cluster.get_left_child( );
          right_child = parent_cluster.get_right_child( );

          % transfer up only if there are contributions from both descendants
          % included
          if ( left_child.get_c_moments_count( ) == 2 ) 
            parent_cluster.apply_m2m_left( left_child.get_moments( ) );
          end

          if ( right_child.get_c_moments_count( ) == 2 )
            parent_cluster.apply_m2m_right( right_child.get_moments( ) );
          end
          parent_node = parent_node.get_parent( );
        end
        
        % interaction phase
        % go through interaction lists of the clusters and apply M2L
        cluster = obj.leafs{ m + 1 }.get_value( ); 
        parent_node = obj.leafs{ m + 1 };
        for l = obj.L : -1 : lt 
          cluster.apply_m2l( );
          parent_node = parent_node.get_parent( );
          cluster = parent_node.get_value( );
        end
        
        % downward pass
        parent = obj.leafs{ m + 1 };
        for l = obj.L : -1 : lt 
          parent = parent.get_parent( );
        end

        obj.downward_pass( parent, m );
       
        cluster = obj.leafs{ m + 1 }.get_value( ); 
        f = cluster.apply_l2p( );
        ind_start = obj.leafs{ m - 1 + 1 }.get_value( ).get_start_index( );
        ind_end = obj.leafs{ m - 1 + 1 }.get_value( ).get_end_index( );
        tst = obj.V2 * x( ind_start : ind_end );
        f = f + tst; %V2 * x( ind_start : ind_end );
        ind_start = obj.leafs{ m  + 1 }.get_value( ).get_start_index( );
        ind_end = obj.leafs{ m + 1 }.get_value( ).get_end_index( );
        obj.rhs_proj( ind_start : ind_end ) = ...
          obj.rhs_proj( ind_start : ind_end ) - f;
        x( ind_start : ind_end ) = obj.V1 \ obj.rhs_proj( ind_start : ind_end );
      end
    end
    
    
  end
  
  methods ( Access = private )
    
    function construct_cluster_tree( obj, root )
      if root.get_level() < obj.L 
        
        % split current cluster (root) into two subclusters
        current_cluster = root.get_value( );
        start_index = current_cluster.get_start_index( );
        end_index = current_cluster.get_end_index( );
        middle_index = floor( ( start_index + end_index ) / 2 );
      
        % recursively create binary subtrees for left and right descendants
        left_cluster = temporal_cluster( obj.panels, start_index, ...
          middle_index, 2 * current_cluster.get_idx_nl( ) );
        root.set_left_child( left_cluster );
        if ( root.get_level == obj.L - 1 ) 
          obj.leafs{ 2 * current_cluster.get_idx_nl( ) + 1 } = ...
            root.get_left_child( ); 
        end
        obj.construct_cluster_tree( root.get_left_child( ) );
             
        right_cluster = temporal_cluster( obj.panels, middle_index + 1, ...
          end_index, 2 * current_cluster.get_idx_nl( ) + 1 );
        root.set_right_child( right_cluster );
        current_cluster.set_children( left_cluster, right_cluster );
        if ( root.get_level == obj.L - 1 ) 
          obj.leafs{ 2 * current_cluster.get_idx_nl( ) + 2 } = ...
            root.get_right_child( ); 
        end
        obj.construct_cluster_tree( root.get_right_child( ) );
      end
    end
    
    % computes moments from input vector
    function compute_q2m( obj, root )   
      
      if ( root.get_left_child( ) ~= 0 && root.get_right_child( ) ~= 0 )
        % we only have to work on leaves
        obj.compute_q2m( root.get_left_child( ) );
        obj.compute_q2m( root.get_right_child( ) );
      else 
        Lagrange = lagrange_interpolant( obj.L_order );
        q2m = zeros( obj.L_order + 1,  root.get_value( ).get_n_steps( ) );
        
        for i = root.get_value( ).get_start_index( ) : ...
            root.get_value( ).get_end_index( )          
          for j = 0 : obj.L_order
            q2m( j + 1, i - root.get_value( ).get_start_index( ) + 1 ) = ...
              obj.integrate_lagrange( root.get_value( ), ...
              obj.panels( 1, i ), obj.panels( 2, i ), Lagrange, j );
          end
        end
        root.get_value( ).set_q2m( q2m );
      end
    end
    
    % integrates Lagrange polynomials to assemble Q2M matrices
    function result = integrate_lagrange( ~, cluster, t_start, t_end, ... 
        interpolant, b )
      [ x, w, l ] = quadratures.line( 4 );
      result = 0;
      for j = 1 : l
        tau = t_start + ( t_end - t_start) * x( j );
        tau_loc = cluster.map_2_local( tau );
        result = result + interpolant.lagrange( b, tau_loc ) * w( j );
      end
      result = result * ( t_end - t_start );
    end
    
    % assembles matrices of M2M transformations
    function compute_m2m( obj, root )
      if ( root.get_left_child( ) ~= 0 && root.get_right_child( ) ~= 0 )
        obj.compute_m2m( root.get_left_child( ) );
        obj.compute_m2m( root.get_right_child( ) );
      end
      
      if ( root.get_left_child( ) ~= 0 && root.get_right_child( ) ~= 0 ) 
        Lagrange = lagrange_interpolant( obj.L_order );
        m2m_left = zeros( obj.L_order + 1, obj.L_order + 1 );
        m2m_right = zeros( obj.L_order + 1, obj.L_order + 1 );
        
        interp_nodes = Lagrange.get_nodes( );
        for i = 0 : obj.L_order
          for j = 0 : obj.L_order
            tau_left = 0.5 * interp_nodes( j + 1 ) - 0.5;
            tau_right = 0.5 * interp_nodes( j + 1 ) + 0.5;
            
            m2m_left( j + 1, i + 1 ) = Lagrange.lagrange( i, tau_left );
            m2m_right( j + 1, i + 1) = Lagrange.lagrange( i, tau_right );
          end
        end
        root.get_value( ).set_m2m( m2m_left, m2m_right );
      end
    end
    
    % assembles neighbor list, i.e., clusters on the same level with index m_l
    % and m_l - 1
    function assemble_neighbor_lists( obj, root ) 
      level = root.get_level( );
      idx = root.get_value( ).get_idx_nl( );
      neighbors = {}; 
      neighbors(1, 1) = {root.get_value( )};
      if ( level > 0 && idx > 0 )
        if mod( root.get_value( ).get_idx_nl( ), 2 ) == 1 
          n = root.get_parent( ).get_left_child( ).get_value( );
        elseif root.get_value( ).get_idx_nl( ) ~= 0
          n = obj.search_neighbors( root, level, idx );
        end
        
        if n ~= -1
          % if neighbor == -1 there is no neighbor except for the 
          % same cluster
          neighbors(1, 2) = {n};
        end
      end
     
      root.get_value( ).set_neighbors( neighbors );
      if ( root.get_left_child ~= 0 && root.get_right_child ~= 0 )
        obj.assemble_neighbor_lists( root.get_left_child( ) );
        obj.assemble_neighbor_lists( root.get_right_child( ) );
      end
      
    end
    
%     % search for a neighboring clusters
    function neighbor = search_neighbors( obj, root, L, idx )
      % if neighbor == -1 there is no neighbor except for the same cluster
      neighbor = -1;
      
      node = root;
      while node.get_parent( ).get_left_child( ).get_value( ).get_idx_nl( ) == ... 
        node.get_value( ).get_idx_nl( )
        node = node.get_parent( );
      end
      
      child = node.get_parent( ).get_left_child( );
      
      for l = child.get_level( ) : L - 1
        child = child.get_right_child( );
      end
      
      neighbor = child.get_value( );
      
    end
    
    % search for a neighboring clusters
%     function neighbor = search_neighbors( obj, root, L, idx )
%       % if neighbor == -1 there is no neighbor except for the same cluster
%       neighbor = -1;
%       if ( root.get_value( ).get_idx_nl( ) == idx - 1 ) && ...
%           ( root.get_level == L )
%         neighbor = root.get_value( );
%       elseif root.get_level( ) < L 
%         neighbor = obj.search_neighbors( root.get_left_child( ), L, idx );
%         if neighbor == -1
%           neighbor = obj.search_neighbors( root.get_right_child( ), L, idx );
%         end
%       end
%     end
    
    % assembles interaction list of a cluster, i.e., parents' neighbors'
    % children which are not neighbors themselves
    function assemble_interaction_list( obj, root )
      if root.get_level( ) > 1 
        parent_neighbors = root.get_parent( ).get_value( ).get_neighbors( );
        sz = size( parent_neighbors );
        if sz( 1, 2 ) == 2
          neighbor = parent_neighbors{ 1, 2 };
          left_child = neighbor.get_left_child( );
          right_child = neighbor.get_right_child( );
          interaction_list = { left_child };
          if right_child.get_idx_nl( ) == root.get_value( ).get_idx_nl( ) - 2
            interaction_list{ 1, 2 } = right_child;
          end
          root.get_value( ).set_interaction_list( interaction_list );
        end

      end
      
      if root.get_left_child( ) ~= 0 && root.get_right_child( ) ~= 0
        obj.assemble_interaction_list( root.get_left_child( ) );
        obj.assemble_interaction_list( root.get_right_child( ) );
      end
    end
    
    % computes M2L matrices for clusters from the interaction list of a
    % given cluster
    function compute_m2l( obj, root )
      if root.get_level( ) > 1
        current_cluster = root.get_value( );
        interaction_list = current_cluster.get_interaction_list( );
        sz = size( interaction_list );
        Lagrange = lagrange_interpolant( obj.L_order );
        nodes = Lagrange.get_nodes( );
        m2l_matrices = cell(sz);
        for i = 1 : sz( 1, 2 )
          remote_cluster = interaction_list{ 1, i };
          m2l = zeros( obj.L_order + 1, obj.L_order + 1 );
          for a = 0 : obj.L_order
            for b = 0 : obj.L_order
              % variable from the local cluster
              t = current_cluster.map_2_global( nodes( a + 1 ) ); 
              % variable from the remote cluster
              tau = remote_cluster.map_2_global( nodes( b + 1 ) );
              m2l( a + 1, b + 1 ) = obj.eval_kernel( t - tau );
            end
          end
          m2l_matrices{ 1, i } = m2l;
        end
        root.get_value( ).set_m2l( m2l_matrices );
      end
      
      if root.get_left_child( ) ~= 0 && root.get_right_child( ) ~= 0
        obj.compute_m2l( root.get_left_child( ) );
        obj.compute_m2l( root.get_right_child( ) );
      end
    end
    
    % evaluates the kernel function
    function kernel = eval_kernel( ~, t )
      kernel = ( 1 / sqrt( 4 * pi * t ) ) * ( 1 - exp( - 1 / t ) );
    end
    
    
%     function binary_array = d2b( obj, decimal, len )
%       binary_string = dec2bin( decimal );
%       binary_array = zeros( 1, len );
%       sz = size( binary_string, 2 );
%       for i = 1 : sz
%         binary_array( i ) = str2double( binary_string( sz - i + 1 ) );
%       end
%     end
    

    
    % goes from the level 2 and transfers local expansion coefficients to
    % the lower levels
    function downward_pass( obj, root, m )
      
      if ( size( root.get_value( ).get_local_expansion( ), 1 ) ~= 0 )
        if root.get_value().get_left_child().get_idx_nl( ) == ...
            obj.get_id_on_level( m, root.get_level( ) + 1 )
          left_exp = root.get_value( ).apply_l2l_left( );
          root.get_left_child( ).get_value( ).add_expansion( left_exp );
        end
        if root.get_value().get_right_child().get_idx_nl() == ...
            obj.get_id_on_level( m, root.get_level() + 1)
          right_exp = root.get_value( ).apply_l2l_right( );
          root.get_right_child( ).get_value( ).add_expansion( right_exp );
        end
      end
      
      if root.get_level( ) < obj.L - 1
        if root.get_value().get_right_child().get_idx_nl() == ...
            obj.get_id_on_level( m, root.get_level() + 1)
          obj.downward_pass( root.get_right_child( ), m );
        end
        if root.get_value().get_left_child().get_idx_nl( ) == ...
            obj.get_id_on_level( m, root.get_level( ) + 1 )
          obj.downward_pass( root.get_left_child( ), m );
        end
      end
    end
    
    % returns id of the m-th leaf on the given level
    function id = get_id_on_level( obj, m, level )
      id = m;
      for i = 1 : obj.L - level
        id = floor( id / 2 );
      end
      
    end
    
    % L2 projection of the RHS
    function projection = project_rhs( ~, rhs, T, nT )
      diag = ones( nT,1 ) * ( T / nT );
      M = spdiags( diag, 0, nT, nT );
      [ x, w, l ] = quadratures.line( 4 );
      
      h = T / nT;
      proj_rhs = zeros( nT, 1 );
      for i = 1 : nT
        for j = 1 : l
          proj_rhs( i ) = proj_rhs( i ) + rhs( (i - 1) * h ...
            + x (j ) * h ) * w( j );
        end
        proj_rhs(i) = proj_rhs(i) * h;
      end
      projection = M \ proj_rhs;
    end
    
    
      % computes M2L matrices for clusters from the interaction list of a
    % given cluster
    function reset( obj, root )      
      current_cluster = root.get_value( );
      current_cluster.reset( );
      if root.get_left_child( ) ~= 0 && root.get_right_child( ) ~= 0
        obj.reset( root.get_left_child( ) );
        obj.reset( root.get_right_child( ) );
      end
    end
    
  end
  

end




        % find the coarsest level where parents change
%         a = obj.d2b( m - 1, obj.L + 1 );
%         b = obj.d2b( m, obj.L + 1 );
%         diff = xor( a, b );
        %lt = 2;
%         for i = 2 : obj.L - 1
%           if diff( obj.L - i + 1 ) == 0
%             lt = lt + 1;
%             break
%           end
%         end