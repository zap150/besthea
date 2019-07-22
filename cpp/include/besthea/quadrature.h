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

/** @file quadrature.h
 * @brief Quadrature rules.
 */

#ifndef INCLUDE_BESTHEA_QUADRATURE_H_
#define INCLUDE_BESTHEA_QUADRATURE_H_

#include "besthea/settings.h"

#include <vector>

namespace besthea {
  namespace bem {
    class quadrature;
  }
}

/**
 *  Class providing quadrature rules.
 */
class besthea::bem::quadrature {
 public:
  /**
   * Returns the line quadrature nodes in (0,1)
   * @param[in] order Quadrature order.
   */
  static const std::vector< sc, besthea::allocator_type< sc > > & line_x(
    int order ) {
    switch ( order ) {
      case 1:
        return line_x_1;
      case 2:
        return line_x_2;
      case 3:
        return line_x_3;
      case 4:
        return line_x_4;
      case 5:
        return line_x_5;
      case 6:
        return line_x_6;
      default:
        return line_x_1;
    }
  }

  /**
   * Returns the line quadrature weights in (0,1)
   * @param[in] order Quadrature order.
   */
  static const std::vector< sc, besthea::allocator_type< sc > > & line_w(
    int order ) {
    switch ( order ) {
      case 1:
        return line_w_1;
      case 2:
        return line_w_2;
      case 3:
        return line_w_3;
      case 4:
        return line_w_4;
      case 5:
        return line_w_5;
      case 6:
        return line_w_6;
      default:
        return line_w_1;
    }
  }

  /**
   * Returns the first coordinates of triangle quadrature nodes in
   * (0,1)x(0,1-x1)
   * @param[in] order Quadrature order.
   */
  static const std::vector< sc, besthea::allocator_type< sc > > & triangle_x1(
    int order ) {
    switch ( order ) {
      case 1:
        return triangle_x1_1;
      case 2:
        return triangle_x1_2;
      case 3:
        return triangle_x1_3;
      case 4:
        return triangle_x1_4;
      case 5:
        return triangle_x1_5;
      default:
        return triangle_x1_1;
    }
  }

  /**
   * Returns the second coordinates of triangle quadrature nodes in
   * (0,1)x(0,1-x1)
   * @param[in] order Quadrature order.
   */
  static const std::vector< sc, besthea::allocator_type< sc > > & triangle_x2(
    int order ) {
    switch ( order ) {
      case 1:
        return triangle_x2_1;
      case 2:
        return triangle_x2_2;
      case 3:
        return triangle_x2_3;
      case 4:
        return triangle_x2_4;
      case 5:
        return triangle_x2_5;
      default:
        return triangle_x2_1;
    }
  }

  /**
   * Returns the triangle quadrature weights in
   * (0,1)x(0,1-x1)
   * @param[in] order Quadrature order.
   */
  static const std::vector< sc, besthea::allocator_type< sc > > & triangle_w(
    int order ) {
    switch ( order ) {
      case 1:
        return triangle_w_1;
      case 2:
        return triangle_w_2;
      case 3:
        return triangle_w_3;
      case 4:
        return triangle_w_4;
      case 5:
        return triangle_w_5;
      default:
        return triangle_w_1;
    }
  }

 private:
  static const std::vector< sc, besthea::allocator_type< sc > >
    line_x_1;  //!< Quadrature nodes in (0,1), order 1
  static const std::vector< sc, besthea::allocator_type< sc > >
    line_w_1;  //!< Quadrature weights in (0,1), order 1

  static const std::vector< sc, besthea::allocator_type< sc > >
    line_x_2;  //!< Quadrature nodes in (0,1), order 2
  static const std::vector< sc, besthea::allocator_type< sc > >
    line_w_2;  //!< Quadrature weights in (0,1), order 2

  static const std::vector< sc, besthea::allocator_type< sc > >
    line_x_3;  //!< Quadrature nodes in (0,1), order 3
  static const std::vector< sc, besthea::allocator_type< sc > >
    line_w_3;  //!< Quadrature weights in (0,1), order 3

  static const std::vector< sc, besthea::allocator_type< sc > >
    line_x_4;  //!< Quadrature nodes in (0,1), order 4
  static const std::vector< sc, besthea::allocator_type< sc > >
    line_w_4;  //!< Quadrature weights in (0,1), order 4

  static const std::vector< sc, besthea::allocator_type< sc > >
    line_x_5;  //!< Quadrature nodes in (0,1), order 5
  static const std::vector< sc, besthea::allocator_type< sc > >
    line_w_5;  //!< Quadrature weights in (0,1), order 5

  static const std::vector< sc, besthea::allocator_type< sc > >
    line_x_6;  //!< Quadrature nodes in (0,1), order 6
  static const std::vector< sc, besthea::allocator_type< sc > >
    line_w_6;  //!< Quadrature weights in (0,1), order 6

  static const std::vector< sc, besthea::allocator_type< sc > >
    triangle_x1_1;  //!< First coordinates of quadrature nodes in
                    //!< (0,1)x(0,1-x1), order 1
  static const std::vector< sc, besthea::allocator_type< sc > >
    triangle_x2_1;  //!< Second coordinates of quadrature nodes in
                    //!< (0,1)x(0,1-x1), order 1
  static const std::vector< sc, besthea::allocator_type< sc > >
    triangle_w_1;  //!< Quadrature weights in
                   //!< (0,1)x(0,1-x1), order 1

  static const std::vector< sc, besthea::allocator_type< sc > >
    triangle_x1_2;  //!< First coordinates of quadrature nodes in
                    //!< (0,1)x(0,1-x1), order 2
  static const std::vector< sc, besthea::allocator_type< sc > >
    triangle_x2_2;  //!< Second coordinates of quadrature nodes in
                    //!< (0,1)x(0,1-x1), order 2
  static const std::vector< sc, besthea::allocator_type< sc > >
    triangle_w_2;  //!< Quadrature weights in
                   //!< (0,1)x(0,1-x1), order 2

  static const std::vector< sc, besthea::allocator_type< sc > >
    triangle_x1_3;  //!< First coordinates of quadrature nodes in
                    //!< (0,1)x(0,1-x1), order 3
  static const std::vector< sc, besthea::allocator_type< sc > >
    triangle_x2_3;  //!< Second coordinates of quadrature nodes in
                    //!< (0,1)x(0,1-x1), order 3
  static const std::vector< sc, besthea::allocator_type< sc > >
    triangle_w_3;  //!< Quadrature weights in
                   //!< (0,1)x(0,1-x1), order 3
  static const std::vector< sc, besthea::allocator_type< sc > >
    triangle_x1_4;  //!< First coordinates of quadrature nodes in
                    //!< (0,1)x(0,1-x1), order 4
  static const std::vector< sc, besthea::allocator_type< sc > >
    triangle_x2_4;  //!< Second coordinates of quadrature nodes in
                    //!< (0,1)x(0,1-x1), order 4
  static const std::vector< sc, besthea::allocator_type< sc > >
    triangle_w_4;  //!< Quadrature weights in
                   //!< (0,1)x(0,1-x1), order 4

  static const std::vector< sc, besthea::allocator_type< sc > >
    triangle_x1_5;  //!< First coordinates of quadrature nodes in
                    //!< (0,1)x(0,1-x1), order 5
  static const std::vector< sc, besthea::allocator_type< sc > >
    triangle_x2_5;  //!< Second coordinates of quadrature nodes in
                    //!< (0,1)x(0,1-x1), order 5
  static const std::vector< sc, besthea::allocator_type< sc > >
    triangle_w_5;  //!< Quadrature weights in
                   //!< (0,1)x(0,1-x1), order 5
};

#endif /* INCLUDE_BESTHEA_QUADRATURE_H_ */
