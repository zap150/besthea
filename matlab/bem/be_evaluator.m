classdef be_evaluator < handle
  
  properties (Access = public)
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
    
    function result = evaluate( obj )
      if( isa( obj.mesh, 'spacetime_mesh' ) )
        result = evaluate_st( obj );
      else
        result = evaluate_s( obj );
      end
    end
    
  end
  
  methods (Access = private)
    
    function result = evaluate_s( obj )
      result = zeros( size( obj.points, 1 ), 1 );
      
      n_elems = obj.mesh.n_elems;
      dim_trial = obj.trial.dim_local( );
      [ y_ref, w, l ] = quadratures.tri( obj.order_ff );
      
      for i_trial = 1 : n_elems
        y = global_quad( obj, y_ref, i_trial );
        map_trial = obj.trial.l2g( i_trial );
        density_loc = obj.density( map_trial );
        
        for i_quad = 1 : l
          k = obj.kernel.eval( obj.points, y( i_quad, : ), 0, ...
            obj.mesh.normals( i_trial, : ) );
          trial_fun = obj.trial.eval( y_ref( i_quad, : ) );
          area = obj.mesh.areas( i_trial );
          
          for i_loc_trial = 1 : dim_trial
            result = result + w( i_quad ) * area ...
              * density_loc( i_loc_trial ) * k * trial_fun( :, i_loc_trial );
          end
        end
      end
    end
    
    function result = evaluate_st( obj )
      nt = obj.mesh.nt;
      obj.kernel.ht = obj.mesh.ht;
      obj.kernel.nt = nt;
      result = cell( nt + 1, 1 );
      
      %%%%% result{ 1 } holds the initial condition
      for i_t = 1 : nt + 1
        result{ i_t } = zeros( size( obj.points, 1 ), 1 );
      end
      
      n_elems = obj.mesh.n_elems;
      dim_trial = obj.trial.dim_local( );
      [ y_ref, w, l ] = quadratures.tri( obj.order_ff );
      
      for d = 1 : nt
        obj.kernel.d = d;
        for i_trial = 1 : n_elems
          y = global_quad( obj, y_ref, i_trial );
          map_trial = obj.trial.l2g( i_trial );
          
          for i_quad = 1 : l
            k = obj.kernel.eval_repr( obj.points, y( i_quad, : ), ...
              obj.mesh.normals( i_trial, : ) );
            trial_fun = obj.trial.eval( y_ref( i_quad, : ) );
            area = obj.mesh.areas( i_trial );
            
            for i_t = d : nt
              density_loc = obj.density{ i_t - d + 1 }( map_trial );
              for i_loc_trial = 1 : dim_trial
                result_loc = w( i_quad ) * density_loc( i_loc_trial ) ...
                  * area * k * trial_fun( :, i_loc_trial );
                result{ i_t + 1 } = result{ i_t + 1 } + result_loc;
              end
            end
            
          end
        end
      end
    end
    
    %     function result = evaluate_st( obj )
    %       nt = obj.mesh.nt;
    %       obj.kernel.ht = obj.mesh.ht;
    %       obj.kernel.nt = nt;
    %       result = cell( nt + 1, 1 );
    %
    %       %%%%% result{ 1 } holds the initial condition
    %       for i_t = 1 : nt + 1
    %         result{ i_t } = zeros( size( obj.points, 1 ), 1 );
    %       end
    %
    %       n_elems = obj.mesh.n_elems;
    %       dim_trial = obj.trial.dim_local( );
    %       [ y_ref, w, l ] = quadratures.tri( obj.order_ff );
    %
    %       obj.kernel.d = 1;
    %       for i_trial = 1 : n_elems
    %         y = global_quad( obj, y_ref, i_trial );
    %         map_trial = obj.trial.l2g( i_trial );
    %         density_loc = obj.density{ 1 }( map_trial );
    %
    %         for i_quad = 1 : l
    %           k = obj.kernel.eval_repr( obj.points, y( i_quad, : ), ...
    %             obj.mesh.normals( i_trial, : ) );
    %           trial_fun = obj.trial.eval( y_ref( i_quad, : ) );
    %           area = obj.mesh.areas( i_trial );
    %
    %           for i_loc_trial = 1 : dim_trial
    %             result_loc = w( i_quad ) * density_loc( i_loc_trial ) ...
    %               * area * k * trial_fun( :, i_loc_trial );
    %             result{ 2 } = result{ 2 } + result_loc;
    %           end
    %
    %         end
    %       end
    %     end
    
    function y = global_quad( obj, y_ref, i_trial )
      nodes = obj.mesh.nodes( obj.mesh.elems( i_trial, : ), : );
      y = nodes( 1, : ) + y_ref ...
        * [ nodes( 2, : ) - nodes( 1, : ); nodes( 3, : ) - nodes( 1, : ) ];
    end
    
  end
  
end

