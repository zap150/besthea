classdef be_evaluator
  
  properties (Access = private)
    mesh;
    kernel;
    trial;
    density;
    points;
    order_ff;
  end
  
  methods
    function obj = ...
        be_evaluator( mesh, kernel, trial, density, points, order_ff )
      
      obj.mesh = mesh;
      obj.kernel = kernel;
      obj.trial = trial;
      obj.density = density;
      obj.points = points;
      
      if( nargin < 6 )
        obj.order_ff = 4;
      else
        obj.order_ff = order_ff;
      end
    end
    
    function obj = set_kernel( obj, kernel )
      obj.kernel = kernel;
    end
    
    function obj = set_trial( obj, trial )
      obj.trial = trial;
    end
    
    function obj = set_density( obj, density )
      obj.density = density;
    end
    
    function result = evaluate( obj )
      if( isa( obj.mesh, 'spacetime_mesh' ) )
        result = obj.evaluate_st( );
      else
        result = obj.evaluate_s( );
      end
    end
    
  end
  
  methods (Access = private)
    
    function result = evaluate_s( obj )
      result = zeros( size( obj.points, 1 ), 1 );
      
      n_elems = obj.mesh.get_n_elems( );
      dim_trial = obj.trial.dim_local( );
      [ y_ref, w, l ] = quadratures.tri( obj.order_ff );
      
      for i_trial = 1 : n_elems        
        y = obj.global_quad( y_ref, i_trial );
        map_trial = obj.trial.l2g( i_trial );
        density_loc = obj.density( map_trial );
        
        for i_quad = 1 : l
          k = obj.kernel.eval( obj.points, y( i_quad, : ), ...
            obj.mesh.get_normal( i_trial ) );
          trial_fun = obj.trial.eval( y_ref( i_quad, : ) );
          area = obj.mesh.get_area( i_trial );
          
          for i_loc_trial = 1 : dim_trial            
            result = result + w( i_quad ) * area ...
              * density_loc( i_loc_trial ) * k .* trial_fun( :, i_loc_trial );
          end
        end
      end
    end
    
    function result = evaluate_st( obj )
      nt = obj.mesh.get_nt( );
      result = cell( nt, 1 );
      for d = 0 : nt - 1
        result{ d + 1 } = zeros( size( obj.points, 1 ), 1 );
      end
    end
    
    function y = global_quad( obj, y_ref, i_trial )   
      nodes = obj.mesh.get_nodes( i_trial );
      R = [ nodes( 2, : ) - nodes( 1, : ); nodes( 3, : ) - nodes( 1, : ) ];
      y = y_ref * R;
      y = y + nodes( 1, : );     
    end    
    
  end
  
end

