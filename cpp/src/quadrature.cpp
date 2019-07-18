/*
 * Copyright 2019, VSB - Technical University of Ostrava and Graz University of
 * Technology All rights reserved. Redistribution and use in source and binary
 * forms, with or without modification, are permitted provided that the
 * following conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer. Redistributions in binary
 * form must reproduce the above copyright notice, this list of conditions and
 * the following disclaimer in the documentation and/or other materials provided
 * with the distribution. Neither the name of VSB - Technical University of
 * Ostrava and Graz University of Technology nor the names of its contributors
 * may be used to endorse or promote products  derived from this software
 * without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL VSB - TECHNICAL UNIVERSITY OF OSTRAVA AND
 * GRAZ UNIVERSITY OF TECHNOLOGY BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "besthea/quadrature.h"

const std::vector< sc > besthea::bem::quadrature::line_x_1{ 0.5 };
const std::vector< sc > besthea::bem::quadrature::line_w_1{ 1.0 };

const std::vector< sc > besthea::bem::quadrature::line_x_2{
  0.21132486540518713447, 0.78867513459481286553
};
const std::vector< sc > besthea::bem::quadrature::line_w_2{ 0.5, 0.5 };

const std::vector< sc > besthea::bem::quadrature::line_x_3{
  0.11270166537925831148, 0.5, 0.88729833462074168852
};
const std::vector< sc > besthea::bem::quadrature::line_w_3{ 0.2777777777777778,
  0.4444444444444444, 0.2777777777777778 };

const std::vector< sc > besthea::bem::quadrature::line_x_4{ 0.930568155797026,
  0.669990521792428, 0.330009478207572, 0.069431844202974 };
const std::vector< sc > besthea::bem::quadrature::line_w_4{ 0.173927422568727,
  0.326072577431273, 0.326072577431273, 0.173927422568727 };

const std::vector< sc > besthea::bem::quadrature::line_x_5{ 0.046910077030668,
  0.230765344947158, 0.500000000000000, 0.769234655052841, 0.953089922969332 };
const std::vector< sc > besthea::bem::quadrature::line_w_5{ 0.118463442528095,
  0.239314335249683, 0.284444444444444, 0.239314335249683, 0.118463442528095 };

const std::vector< sc > besthea::bem::quadrature::line_x_6{ 0.033765242898424,
  0.169395306766868, 0.380690406958402, 0.619309593041598, 0.830604693233132,
  0.966234757101576 };
const std::vector< sc > besthea::bem::quadrature::line_w_6{ 0.085662246189585,
  0.180380786524069, 0.233956967286346, 0.233956967286346, 0.180380786524069,
  0.085662246189585 };

const std::vector< sc > besthea::bem::quadrature::triangle_x1_1{
  0.333333333333333
};
const std::vector< sc > besthea::bem::quadrature::triangle_x2_1{
  0.333333333333333
};
const std::vector< sc > besthea::bem::quadrature::triangle_w_1{ 1.0 };

const std::vector< sc > besthea::bem::quadrature::triangle_x1_2{
  0.166666666666667, 0.166666666666667, 0.666666666666667
};
const std::vector< sc > besthea::bem::quadrature::triangle_x2_2{
  0.166666666666667, 0.666666666666667, 0.166666666666667
};
const std::vector< sc > besthea::bem::quadrature::triangle_w_2{
  0.333333333333333, 0.333333333333333, 0.333333333333333
};

const std::vector< sc > besthea::bem::quadrature::triangle_x1_3{
  0.333333333333333, 0.2, 0.2, 0.6
};
const std::vector< sc > besthea::bem::quadrature::triangle_x2_3{
  0.333333333333333, 0.2, 0.6, 0.2
};
const std::vector< sc > besthea::bem::quadrature::triangle_w_3{
  -0.562500000000000, 0.520833333333333, 0.520833333333333, 0.520833333333333
};

const std::vector< sc > besthea::bem::quadrature::triangle_x1_4{
  0.445948490915965, 0.445948490915965, 0.108103018168070, 0.091576213509771,
  0.091576213509771, 0.816847572980459
};
const std::vector< sc > besthea::bem::quadrature::triangle_x2_4{
  0.445948490915965, 0.108103018168070, 0.445948490915965, 0.091576213509771,
  0.816847572980459, 0.091576213509771
};
const std::vector< sc > besthea::bem::quadrature::triangle_w_4{
  0.223381589678011, 0.223381589678011, 0.223381589678011, 0.109951743655322,
  0.109951743655322, 0.109951743655322
};

const std::vector< sc > besthea::bem::quadrature::triangle_x1_5{
  0.333333333333333, 0.470142064105115, 0.059715871789770, 0.470142064105115,
  0.101286507323456, 0.797426985353087, 0.101286507323456
};
const std::vector< sc > besthea::bem::quadrature::triangle_x2_5{
  0.333333333333333, 0.059715871789770, 0.470142064105115, 0.470142064105115,
  0.797426985353087, 0.101286507323456, 0.101286507323456
};
const std::vector< sc > besthea::bem::quadrature::triangle_w_5{
  0.225000000000000, 0.132394152788506, 0.132394152788506, 0.132394152788506,
  0.125939180544827, 0.125939180544827, 0.125939180544827
};
