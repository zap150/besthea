classdef spacetime_solver
  
  methods
    function obj = spacetime_solver( )
    end
        
    function neu = solve_dirichlet( ~, V, K, M, dir )
      nt = size( V, 1 );
      neu = cell( nt, 1 );
      
      rhs = zeros( size( V{ 1 }, 1 ), 1 );      
      for d = 1 : nt
        rhs( :, 1 ) = 0.5 * M * dir{ d };
        for j = 1 : d
          rhs( :, 1 ) = rhs( :, 1 ) + K{ j } * dir{ d - j + 1 };
        end
        for j = 2 : d
          rhs( :, 1 ) = rhs( :, 1 ) - V{ j } * neu{ d - j + 1 };
        end        
        neu{ d } = V{ 1 } \ rhs;
      end
    end
    
    function dir = solve_neumann( ~, D, K, M, neu )
      nt = size( D, 1 );
      dir = cell( nt, 1 );
      
      rhs = zeros( size( D{ 1 }, 1 ), 1 );      
      for d = 1 : nt
        rhs( :, 1 ) = 0.5 * M' * neu{ d };
        for j = 1 : d
          rhs( :, 1 ) = rhs( :, 1 ) - K{ j }' * neu{ d - j + 1 };
        end
        for j = 2 : d
          rhs( :, 1 ) = rhs( :, 1 ) - D{ j } * dir{ d - j + 1 };
        end        
        dir{ d } = D{ 1 } \ rhs;
      end
    end
    
  end
  
end

