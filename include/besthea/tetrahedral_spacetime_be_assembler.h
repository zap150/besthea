/*
Copyright (c) 2020, VSB - Technical University of Ostrava and Graz University of
Technology
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.
* Neither the names of VSB - Technical University of  Ostrava and Graz
  University of Technology nor the names of its contributors may be used to
  endorse or promote products derived from this software without specific prior
  written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL VSB - TECHNICAL UNIVERSITY OF OSTRAVA AND
GRAZ UNIVERSITY OF TECHNOLOGY BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/** @file tetrahedral_spacetime_be_assembler.h
 * @brief
 */

#ifndef INCLUDE_BESTHEA_TETRAHEDRAL_SPACETIME_BE_ASSEMBLER_H_
#define INCLUDE_BESTHEA_TETRAHEDRAL_SPACETIME_BE_ASSEMBLER_H_

#include "besthea/full_matrix.h"
#include "besthea/tetrahedral_spacetime_be_space.h"
#include "besthea/timer.h"

#include <array>
#include <list>
#include <map>
#include <set>
#include <tuple>
#include <utility>

namespace besthea {
  namespace bem {
    template< class kernel_type, class test_space_type, class trial_space_type >
    class tetrahedral_spacetime_be_assembler;
  }
}

/**
 *  Class representing a boundary element matrix assembler.
 */
template< class kernel_type, class test_space_type, class trial_space_type >
class besthea::bem::tetrahedral_spacetime_be_assembler {
 private:
  struct element {
    std::array< besthea::linear_algebra::coordinates< 3 >, 4 > _nodes;
    static constexpr double _eps = 1e-8;

    element( ) : _nodes( ){ };

    element( const besthea::linear_algebra::coordinates< 3 > & x1,
      const besthea::linear_algebra::coordinates< 3 > & x2,
      const besthea::linear_algebra::coordinates< 3 > & x3,
      const besthea::linear_algebra::coordinates< 3 > & x4 )
      : _nodes{ x1, x2, x3, x4 } {
    }

    lo admissible( const element & that ) {
      lo n_nodes = 0;
      for ( int i = 0; i < 4; ++i ) {
        for ( int j = 0; j < 4; ++j ) {
          double dist2 = 0.0;
          for ( int k = 0; k < 3; ++k ) {
            double diff = _nodes[ i ][ k ] - that._nodes[ j ][ k ];
            dist2 += diff * diff;
          }
          if ( dist2 < _eps * _eps ) {
            ++n_nodes;
          }
        }
      }

      return n_nodes;
    }

    std::array< element, 8 > refine( ) {
      // new nodes created in the center of edges of the current tetrahedron
      besthea::linear_algebra::coordinates< 3 > x1 = _nodes[ 0 ];
      besthea::linear_algebra::coordinates< 3 > x2 = _nodes[ 1 ];
      besthea::linear_algebra::coordinates< 3 > x3 = _nodes[ 2 ];
      besthea::linear_algebra::coordinates< 3 > x4 = _nodes[ 3 ];
      besthea::linear_algebra::coordinates< 3 > node12(
        { ( x1[ 0 ] + x2[ 0 ] ) / 2.0, ( x1[ 1 ] + x2[ 1 ] ) / 2.0,
          ( x1[ 2 ] + x2[ 2 ] ) / 2.0 } );
      besthea::linear_algebra::coordinates< 3 > node13(
        { ( x1[ 0 ] + x3[ 0 ] ) / 2.0, ( x1[ 1 ] + x3[ 1 ] ) / 2.0,
          ( x1[ 2 ] + x3[ 2 ] ) / 2.0 } );
      besthea::linear_algebra::coordinates< 3 > node14(
        { ( x1[ 0 ] + x4[ 0 ] ) / 2.0, ( x1[ 1 ] + x4[ 1 ] ) / 2.0,
          ( x1[ 2 ] + x4[ 2 ] ) / 2.0 } );
      besthea::linear_algebra::coordinates< 3 > node23(
        { ( x2[ 0 ] + x3[ 0 ] ) / 2.0, ( x2[ 1 ] + x3[ 1 ] ) / 2.0,
          ( x2[ 2 ] + x3[ 2 ] ) / 2.0 } );
      besthea::linear_algebra::coordinates< 3 > node24(
        { ( x2[ 0 ] + x4[ 0 ] ) / 2.0, ( x2[ 1 ] + x4[ 1 ] ) / 2.0,
          ( x2[ 2 ] + x4[ 2 ] ) / 2.0 } );
      besthea::linear_algebra::coordinates< 3 > node34(
        { ( x3[ 0 ] + x4[ 0 ] ) / 2.0, ( x3[ 1 ] + x4[ 1 ] ) / 2.0,
          ( x3[ 2 ] + x4[ 2 ] ) / 2.0 } );

      // create 8 new elements from the current one
      element el1( { x1, node12, node13, node14 } );
      element el2( { node12, x2, node23, node24 } );
      element el3( { node13, node23, x3, node34 } );
      element el4( { node14, node24, node34, x4 } );
      element el5( { node12, node13, node14, node24 } );
      element el6( { node12, node13, node23, node24 } );
      element el7( { node13, node14, node24, node34 } );
      element el8( { node13, node23, node24, node34 } );
      return { el1, el2, el3, el4, el5, el6, el7, el8 };
    }

    besthea::linear_algebra::coordinates< 3 > centroid( ) {
      besthea::linear_algebra::coordinates< 3 > x1 = _nodes[ 0 ];
      besthea::linear_algebra::coordinates< 3 > x2 = _nodes[ 1 ];
      besthea::linear_algebra::coordinates< 3 > x3 = _nodes[ 2 ];
      besthea::linear_algebra::coordinates< 3 > x4 = _nodes[ 3 ];
      sc c0 = ( x1[ 0 ] + x2[ 0 ] + x3[ 0 ] + x4[ 0 ] ) / 4.0;
      sc c1 = ( x1[ 1 ] + x2[ 1 ] + x3[ 1 ] + x4[ 1 ] ) / 4.0;
      sc c2 = ( x1[ 2 ] + x2[ 2 ] + x3[ 2 ] + x4[ 2 ] ) / 4.0;

      besthea::linear_algebra::coordinates< 3 > c = { c0, c1, c2 };
      return c;
    }

    sc radius( ) {
      besthea::linear_algebra::coordinates< 3 > c = centroid( );
      besthea::linear_algebra::coordinates< 3 > diff;
      sc max_d = 0;
      for ( lo i = 0; i < 4; ++i ) {
        diff[ 0 ] = c[ 0 ] - _nodes[ i ][ 0 ];
        diff[ 1 ] = c[ 1 ] - _nodes[ i ][ 1 ];
        diff[ 2 ] = c[ 2 ] - _nodes[ i ][ 2 ];
        sc d = diff.norm( );
        if ( d > max_d )
          max_d = d;
      }
      return max_d;
    }

    sc area( ) {
      linear_algebra::coordinates< 3 > x21;
      linear_algebra::coordinates< 3 > x31;
      linear_algebra::coordinates< 3 > x41;
      linear_algebra::coordinates< 3 > cross;
      sc dot;

      x21[ 0 ] = _nodes[ 1 ][ 0 ] - _nodes[ 0 ][ 0 ];
      x21[ 1 ] = _nodes[ 1 ][ 1 ] - _nodes[ 0 ][ 1 ];
      x21[ 2 ] = _nodes[ 1 ][ 2 ] - _nodes[ 0 ][ 2 ];

      x31[ 0 ] = _nodes[ 2 ][ 0 ] - _nodes[ 0 ][ 0 ];
      x31[ 1 ] = _nodes[ 2 ][ 1 ] - _nodes[ 0 ][ 1 ];
      x31[ 2 ] = _nodes[ 2 ][ 2 ] - _nodes[ 0 ][ 2 ];

      x41[ 0 ] = _nodes[ 3 ][ 0 ] - _nodes[ 0 ][ 0 ];
      x41[ 1 ] = _nodes[ 3 ][ 1 ] - _nodes[ 0 ][ 1 ];
      x41[ 2 ] = _nodes[ 3 ][ 2 ] - _nodes[ 0 ][ 2 ];

      cross[ 0 ] = x21[ 1 ] * x31[ 2 ] - x21[ 2 ] * x31[ 1 ];
      cross[ 1 ] = x21[ 2 ] * x31[ 0 ] - x21[ 0 ] * x31[ 2 ];
      cross[ 2 ] = x21[ 0 ] * x31[ 1 ] - x21[ 1 ] * x31[ 0 ];

      dot
        = x41[ 0 ] * cross[ 0 ] + x41[ 1 ] * cross[ 1 ] + x41[ 2 ] * cross[ 2 ];

      return ( std::abs( dot ) / 6.0 );
    }

    using pair = std::pair< element, element >;
  };

  typedef std::tuple< element, element, int > element_pair;

  /**
   * Quadrature nodes in the reference element.
   */
  struct quadrature_wrapper_ref {
    std::array< std::vector< sc, besthea::allocator_type< sc > >, 5 >
      _x1_ref;  //!< First coordinates of quadrature nodes in
                //!< (0,1)x(0,1-x1)x(0,1-x1-x2) to be mapped to the test element
    std::array< std::vector< sc, besthea::allocator_type< sc > >, 5 >
      _x2_ref;  //!< Second coordinates of quadrature nodes in
                //!< (0,1)x(0,1-x1)x(0,1-x1-x2) to be mapped to the test element
    std::array< std::vector< sc, besthea::allocator_type< sc > >, 5 >
      _x3_ref;  //!< Third coordinates of quadrature nodes in
                //!< (0,1)x(0,1-x1)x(0,1-x1-x2) to be mapped to the test element

    std::array< std::vector< sc, besthea::allocator_type< sc > >, 5 >
      _y1_ref;  //!< First coordinates of quadrature nodes in
                //!< (0,1)x(0,1-x1)x(0,1-x1-x2) to be mapped to the trial
                //!< element
    std::array< std::vector< sc, besthea::allocator_type< sc > >, 5 >
      _y2_ref;  //!< Second coordinates of quadrature nodes in
                //!< (0,1)x(0,1-x1)x(0,1-x1-x2) to be mapped to the trial
                //!< element
    std::array< std::vector< sc, besthea::allocator_type< sc > >, 5 >
      _y3_ref;  //!< Third coordinates of quadrature nodes in
                //!< (0,1)x(0,1-x1)x(0,1-x1-x2) to be mapped to the trial
                //!< element

    std::array< std::vector< sc, besthea::allocator_type< sc > >, 5 >
      _w;  //!< Quadrature weights including transformation Jacobians

    std::list< element_pair >
      _ready_elems;  //!< Auxiliary vector used in generation of
                     //!< quadrature points for nonadmissible elements.
  };

  /**
   * Wraps the mapped quadrature point so that they can be private for OpenMP
   * threads
   */
  struct quadrature_wrapper {
    std::vector< sc, besthea::allocator_type< sc > >
      _x1;  //!< First coordinates of quadrature nodes in the test element
    std::vector< sc, besthea::allocator_type< sc > >
      _x2;  //!< Second coordinates of quadrature nodes in the test element
    std::vector< sc, besthea::allocator_type< sc > >
      _x3;  //!< Third coordinates of quadrature nodes in the test element
    std::vector< sc, besthea::allocator_type< sc > >
      _t;  //!< Temporal coordinates of quadrature nodes in the test element

    std::vector< sc, besthea::allocator_type< sc > >
      _y1;  //!< First coordinates of quadrature nodes in the trial element
    std::vector< sc, besthea::allocator_type< sc > >
      _y2;  //!< Second coordinates of quadrature nodes in the trial element
    std::vector< sc, besthea::allocator_type< sc > >
      _y3;  //!< Third coordinates of quadrature nodes in the trial element
    std::vector< sc, besthea::allocator_type< sc > >
      _tau;  //!< Temporal coordinates of quadrature nodes in the test element

    std::vector< sc, besthea::allocator_type< sc > >
      _kernel_values;  //!< Buffer for storing kernel values.
  };

  /*
    template< typename element_type, int n_children >
    struct tree {
      element_type * _root;
      std::array< element_type *, n_children > _children;

      tree( const element_type * root ) : _root( root ) {
        std::fill( _children.begin( ), _children.end( ), nullptr );
      }

      ~tree( ) {
        for ( int i = 0; i < n_children; ++i ) {
          delete _children[ i ];
        }
        delete _root;
      }
    };

    using element_tree = tree< element, 8 >;
    using element_pair_tree = tree< typename element::pair, 64 >;
  */
 public:
  /**
   * Constructor.
   * @param[in] kernel Spacetime kernel antiderivative object.
   * @param[in] test_space Test boundary element space.
   * @param[in] trial_space Trial boundary element space.
   * @param[in] order_regular Tetrahedra quadrature order for regular
   * quadrature.
   */
  tetrahedral_spacetime_be_assembler( kernel_type & kernel,
    test_space_type & test_space, trial_space_type & trial_space,
    int singular_refinements = 3, int order_regular = 4,
    int order_singular = 2 );

  tetrahedral_spacetime_be_assembler(
    const tetrahedral_spacetime_be_assembler & that )
    = delete;

  /**
   * Destructor.
   */
  ~tetrahedral_spacetime_be_assembler( );

  /**
   * Assembles the spacetime matrix.
   * @param[out] global_matrix Full matrix.
   */
  void assemble( besthea::linear_algebra::full_matrix & global_matrix ) const;

 private:
  /**
   * Initializes quadrature structures.
   * @param[in] ref_quadrature Quadrature data on reference element.
   * @param[out] my_quadrature Wrapper holding quadrature data.
   */
  void init_quadrature( const quadrature_wrapper_ref & ref_quadrature,
    quadrature_wrapper & my_quadrature ) const;

  /**
   * Initializes quadrature structures in reference elements.
   * @param[out] ref_quadrature Wrapper holding quadrature data.
   */
  void init_quadrature_ref( quadrature_wrapper_ref & ref_quadrature ) const;

  /**
   * Initializes quadrature structures.
   * @param[out] ref_quadrature Wrapper holding quadrature data on reference
   * elements.
   */
  void init_quadrature_shared_4(
    quadrature_wrapper_ref & ref_quadrature ) const;

  /**
   * Initializes quadrature structures.
   * @param[out] ref_quadrature Wrapper holding quadrature data on reference
   * elements.
   */
  void init_quadrature_shared_3(
    quadrature_wrapper_ref & ref_quadrature ) const;

  /**
   * Initializes quadrature structures.
   * @param[out] ref_quadrature Wrapper holding quadrature data on reference
   * elements.
   */
  void init_quadrature_shared_2(
    quadrature_wrapper_ref & ref_quadrature ) const;

  /**
   * Initializes quadrature structures.
   * @param[out] ref_quadrature Wrapper holding quadrature data on reference
   * elements.
   */
  void init_quadrature_shared_1(
    quadrature_wrapper_ref & ref_quadrature ) const;

  /**
   * Recursively refines the reference element to create quadrature points for
   * non-disjoint elements.
   * @param[in] el Tuple consisting of pair of subelements and refinement
   * level.
   */
  void refine_reference_recursively(
    element_pair el, std::list< element_pair > & _ready_elems ) const;

  /**
   * Determines the configuration of two tetrahedral elements.
   * @param[in] i_test Index of the test element.
   * @param[in] i_trial Index of the trial element.
   * @param[out] type_int Type of the configuration (number of vertices shared).
   * @param[out] perm_test Permutation of the test element.
   * @param[out] perm_trial Permutation of the trial element.
   */
  void get_type( lo i_test, lo i_trial, int & type_int,
    besthea::linear_algebra::indices< 4 > & perm_test,
    besthea::linear_algebra::indices< 4 > & perm_trial ) const;

  /**
   * Maps the quadrature nodes from reference triangles to the actual geometry.
   * @param[in] x1 Coordinates of the first node of the test element.
   * @param[in] x2 Coordinates of the second node of the test element.
   * @param[in] x3 Coordinates of the third node of the test element.
   * @param[in] x4 Coordinates of the fourth node of the test element.
   * @param[in] y1 Coordinates of the first node of the trial element.
   * @param[in] y2 Coordinates of the second node of the trial element.
   * @param[in] y3 Coordinates of the third node of the trial element.
   * @param[in] y4 Coordinates of the fourth node of the trial element.
   * @param[in] type_int Type of the configuration (number of vertices shared).
   * @param[in] perm_test Permutation of the test element.
   * @param[in] perm_trial Permutation of the trial element.
   * @param[in] ref_quadrature Structure hodling the quadrature nodes in the
   * reference elements.
   * @param[out] my_quadrature Structure holding the quadrature nodes.
   */
  void tetrahedra_to_geometry( const linear_algebra::coordinates< 4 > & x1,
    const linear_algebra::coordinates< 4 > & x2,
    const linear_algebra::coordinates< 4 > & x3,
    const linear_algebra::coordinates< 4 > & x4,
    const linear_algebra::coordinates< 4 > & y1,
    const linear_algebra::coordinates< 4 > & y2,
    const linear_algebra::coordinates< 4 > & y3,
    const linear_algebra::coordinates< 4 > & y4, int type_int,
    const besthea::linear_algebra::indices< 4 > & perm_test,
    const besthea::linear_algebra::indices< 4 > & perm_trial,
    const quadrature_wrapper_ref & ref_quadrature,
    quadrature_wrapper & my_quadrature ) const;

  /**
   * Calculates quadrature points in the reference tetrahedron based on its
   * recursive refinement.
   *
   * @param ref_quadrature Structure holding the quadrature nodes.
   */
  void create_quadrature_points(
    quadrature_wrapper_ref & ref_quadrature, int n_shared_vertices ) const;

  /**
   * Maps the quadrature nodes from reference tetrahedron to one if its
   * subtetrahedra.
   * @param[in] el1 First sub-element.
   * @param[in] el2 Second sub-elemen.
   * @param[in] type_int Type of the configuration (number of vertices shared).
   * @param[out] x1 Reference to the vector storing quadrature
   * points in the test element.
   * @param[out] x2 Reference to the vector storing quadrature
   * points in the test element.
   * @param[out] x3 Reference to the vector storing quadrature
   * points in the test element.
   * @param[out] y1 Reference to the vector storing quadrature
   * points in the trial element.
   * @param[out] y2 Reference to the vector storing quadrature
   * points in the trial element.
   * @param[out] y3 Reference to the vector storing quadrature
   * points in the trial element.
   */
  void reference_to_subreference( const element & el1, const element & el2,
    int type_int, std::vector< sc > & x1, std::vector< sc > & x2,
    std::vector< sc > & x3, std::vector< sc > & y1, std::vector< sc > & y2,
    std::vector< sc > & y3 ) const;

  kernel_type * _kernel;  //!< Kernel temporal antiderivative.

  test_space_type * _test_space;  //!< Boundary element test space.

  trial_space_type * _trial_space;  //!< Boundary element trial space.

  int
    _singular_refinements;  //!< Number of refinements for singular integration

  int _order_regular;  //!< Tetrahedron quadrature order for the regular
                       //!< integrals.

  int _order_singular;  //! Quadrature order for the singular elements treated
                        //! by subdivision.

  mutable std::vector< lo > _admissibles;  // just for debugging
};

#endif /* INCLUDE_BESTHEA_TETRAHEDRAL_SPACETIME_BE_ASSEMBLER_H_ */
