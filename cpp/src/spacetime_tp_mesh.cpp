/*
 * spacetime_tp_mesh.cpp
 *
 *  Created on: May 10, 2019
 *      Author: zap150
 */

#include <iostream>

#include "besthea/spacetime_tp_mesh.h"

besthea::spacetime_tp_mesh::spacetime_tp_mesh() {
  std::cout << "creating mesh" << std::endl;
}

besthea::spacetime_tp_mesh::~spacetime_tp_mesh() {
  std::cout << "destroying mesh" << std::endl;
}
