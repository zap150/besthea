classdef quadratures
  
  properties (Access = private, Constant)
    line_x_1 = 0.5;
    line_w_1 = 1.0;
    
    line_x_2 = [
      0.21132486540518713447
      0.78867513459481286553 ];
    line_w_2 = [ 
      0.5
      0.5 ];
    
    line_x_3 = [
      0.11270166537925831148
      0.5
      0.88729833462074168852 ];
    line_w_3 = [
      0.2777777777777778
      0.4444444444444444
      0.2777777777777778 ];
    
    line_x_4 = [
      0.930568155797026
      0.669990521792428
      0.330009478207572
      0.069431844202974 ];
    line_w_4 = [
      0.173927422568727
      0.326072577431273
      0.326072577431273
      0.173927422568727 ];
    
    
    tri_x_1 = [ 
      0.333333333333333 
      0.333333333333333 ];
    tri_w_1 = 1.0;
    
    tri_x_2 = [
      0.166666666666667 0.166666666666667
      0.166666666666667 0.666666666666667
      0.666666666666667 0.166666666666667 ];
    tri_w_2 = [
      0.333333333333333
      0.333333333333333
      0.333333333333333
      ];
    
    tri_x_3 = [ 
      0.333333333333333 0.333333333333333
      0.2 0.2
      0.2 0.6
      0.6 0.2 ];
    tri_w_3 = [
      -0.562500000000000
      0.520833333333333
      0.520833333333333
      0.520833333333333
      ];
    
    tri_x_4 = [
      0.445948490915965 0.445948490915965
      0.445948490915965 0.108103018168070
      0.108103018168070 0.445948490915965
      0.091576213509771 0.091576213509771
      0.091576213509771 0.816847572980459
      0.816847572980459 0.091576213509771 ];
    tri_w_4 = [
      0.223381589678011
      0.223381589678011
      0.223381589678011
      0.109951743655322
      0.109951743655322
      0.109951743655322
      ];
    
    tri_x_5 = [
      0.333333333333333 0.333333333333333
      0.470142064105115 0.059715871789770
      0.059715871789770 0.470142064105115
      0.470142064105115 0.470142064105115
      0.101286507323456 0.797426985353087
      0.797426985353087 0.101286507323456
      0.101286507323456 0.101286507323456 ];
    tri_w_5 = [
      0.225000000000000
      0.132394152788506
      0.132394152788506
      0.132394152788506
      0.125939180544827
      0.125939180544827
      0.125939180544827
      ];
  end
  
  methods (Static)
    function [ x, w, l ] = line( order )
      x = quadratures.line_x( order );
      w = quadratures.line_w( order );
      l = quadratures.line_length( order );
    end
    
    function x = line_x( order )
      switch order
        case 1
          x = quadratures.line_x_1;
        case 2
          x = quadratures.line_x_2;
        case 3
          x = quadratures.line_x_3;
        case 4
          x = quadratures.line_x_4;
        otherwise
          x = quadratures.line_x_1;
      end
    end
    
    function w = line_w( order )
      switch order
        case 1
          w = quadratures.line_w_1;
        case 2
          w = quadratures.line_w_2;
        case 3
          w = quadratures.line_w_3;
        case 4
          w = quadratures.line_w_4;
        otherwise
          w = quadratures.line_w_1;
      end
    end
    
    function l = line_length( order )
      if order > 0 && order < 5
        l = order;
      else
        l = 1;
      end
    end
      
    function [ x, w, l ] = tri( order )
      x = quadratures.tri_x( order );
      w = quadratures.tri_w( order );
      l = quadratures.tri_length( order );
    end
    
    function x = tri_x( order )
      switch order
        case 1
          x = quadratures.tri_x_1;
        case 2
          x = quadratures.tri_x_2;
        case 3
          x = quadratures.tri_x_3;
        case 4
          x = quadratures.tri_x_4;
        case 5
          x = quadratures.tri_x_5;
        otherwise
          x = quadratures.tri_x_1;
      end
    end
    
    function w = tri_w( order )
      switch order
        case 1
          w = quadratures.tri_w_1;
        case 2
          w = quadratures.tri_w_2;
        case 3
          w = quadratures.tri_w_3;
        case 4
          w = quadratures.tri_w_4;
        case 5
          w = quadratures.tri_w_5;
        otherwise
          w = quadratures.tri_w_1;
      end
    end
    
    function length = tri_length( order )
      switch order
        case 1
          length = size( quadratures.tri_w_1, 1 );
        case 2
          length = size( quadratures.tri_w_2, 1 );
        case 3
          length = size( quadratures.tri_w_3, 1 );
        case 4
          length = size( quadratures.tri_w_4, 1 );
        case 5
          length = size( quadratures.tri_w_5, 1 );
        otherwise
          length = size( quadratures.tri_w_1, 1 );
      end
    end
  end
  
end

