classdef binary_tree < handle
  %TREE Tree structure
  %   Detailed explanation goes here
  
  properties ( Access = private )
    value
    left_child = 0
    right_child = 0
    level = 0
    parent = 0
  end
  
  methods
    function obj = binary_tree( data, level, parent )
      obj.value = data;
      obj.level = level;
      obj.parent = parent;
    end
    
    function set_left_child( obj, data )
      obj.left_child = binary_tree( data, obj.level + 1, obj );
    end
    
    function set_right_child( obj, data )
      obj.right_child = binary_tree( data, obj.level + 1, obj );
    end
    
    function left_child = get_left_child( obj )
      left_child = obj.left_child;
    end
    
    function right_child = get_right_child( obj )
      right_child = obj.right_child;
    end
    
    function value = get_value( obj )
      value = obj.value;
    end
    
    function level = get_level( obj )
      level = obj.level;
    end
    
    function print( obj )
      for i = 1 : obj.level + 1
        fprintf( "**" );
      end
      obj.value.print();  
      if obj.left_child ~= 0 
        obj.left_child.print();
      end
      if obj.right_child ~= 0
        obj.right_child.print( );
      end
    end
    
    function parent = get_parent( obj )
      parent = obj.parent;
    end
  end
end

