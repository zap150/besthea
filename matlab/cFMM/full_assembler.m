classdef full_assembler < handle
  %NEARFIELD Assembles nearfield matrices
  
  properties
    
  end
  
  methods
    function obj = full_assembler( )
      
    end
    
    function V = assemble_V( obj, left_cluster, right_cluster )
      t_start_left = left_cluster.get_start( );
      t_end_left = left_cluster.get_end( );
      
      % assuming constant time-steps
      n_steps = left_cluster.get_n_steps( );
      ht = ( t_end_left - t_start_left ) / n_steps;
      
      idx_nl_left = left_cluster.get_idx_nl( );
      idx_nl_right = right_cluster.get_idx_nl( );
      
      V = zeros( n_steps, n_steps );
      
      for i = 1 : n_steps
        for j = 1 : n_steps
          d = ( idx_nl_left * n_steps + i ) - ...
            ( idx_nl_right * n_steps + j );
          V( i, j ) = obj.Vd( d, ht );
        end
      end
    end
    
    function vd = Vd( obj, d, ht )
        vd = sqrt( ht^3 ) * ( obj.VV( d+1, ht ) - 2*obj.VV( d, ht ) + ...
          obj.VV( d-1, ht ) );
    end
    
    function vv = VV( ~, d, ht )
      if d <= 0
        vv = 0;
      else
        vv = sqrt( ( 4 * d ) / ( 9 * pi ) ) * ( d + ( 1 / sqrt( ht ) ) ...
          * ( sqrt( pi / d ) * ( 1 / ht ) + 1.5 * ...
          sqrt( pi * d ) ) * ( erfc( ( 1 / sqrt( ht ) ) / sqrt( d ) ) ) ... 
          - ( d + ( 1 / ht ) ) * exp( - ( 1 / ht ) / d ) );
      end
    end
  end
end

