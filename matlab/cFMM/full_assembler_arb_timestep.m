classdef full_assembler_arb_timestep < handle
  %NEARFIELD Assembles nearfield matrices
  
  properties
    
  end
  
  methods
    function obj = full_assembler_arb_timestep( )
      
    end
    
    function V = assemble_V( obj, left_cluster, right_cluster )
    %left cluster is source cluster, right is target
      source_panels = left_cluster.get_panels();
      source_steps = left_cluster.get_n_steps();
%        h_source = source_panels(2,:)-source_panels(1,:);
      target_panels = right_cluster.get_panels();
      target_steps = right_cluster.get_n_steps();
      
      V = zeros( target_steps, source_steps );
      V = V + obj.VV(repmat(target_panels(2,:)', 1, source_steps) - ...
                     repmat(source_panels(1,:), target_steps, 1));
      V = V - obj.VV(repmat(target_panels(1,:)', 1, source_steps) - ...
                     repmat(source_panels(1,:), target_steps, 1));
      V = V - obj.VV(repmat(target_panels(2,:)', 1, source_steps) - ...
                     repmat(source_panels(2,:), target_steps, 1));
      V = V + obj.VV(repmat(target_panels(1,:)', 1, source_steps) - ...
                     repmat(source_panels(2,:), target_steps, 1));
    end

    function vv = VV( ~, t)
      vv = zeros(size(t));
      vv(t>0) = sqrt(4 * t(t>0) ./ (9 * pi)) .* ...
        (t(t>0) + (sqrt(pi ./ t(t>0)) + 1.5 * sqrt(pi * t(t>0))) ...
        .* erfc((1 ./ sqrt(t(t>0)))) - (t(t>0) + 1) .* exp(-1 ./ t(t>0)));
    end
  end
end

