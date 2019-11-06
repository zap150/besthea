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

/** @file temporal_mesh.h
 * @brief 1D mesh representing temporal interval
 */

#ifndef INCLUDE_BESTHEA_TEMPORAL_MESH_H_
#define INCLUDE_BESTHEA_TEMPORAL_MESH_H_

#include "besthea/coordinates.h"
#include "besthea/indices.h"
#include "besthea/settings.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

namespace besthea {
  namespace mesh {
    class temporal_mesh;
  }
}

/**
 *  Class representing discretization of 1D temporal interval into subinterals.
 */
class besthea::mesh::temporal_mesh {
 public:
  /**
   * Constructor creating uniform discretization of the temporal
   * interval (start_time, end_time).
   * @param[in] start_time Beginning of the temporal interval.
   * @param[in] end_time End of the time interval.
   * @param[in] n_timesteps number of timesteps (temporal subintervals/elements)
   */
  temporal_mesh( sc start_time, sc end_time, lo n_timesteps );

  /**
   * Constructor loading temporal mesh data from a file.
   * @param[in] file Name of the source file.
   */
  temporal_mesh( const std::string & file );

  temporal_mesh( const temporal_mesh & that ) = delete;

  ~temporal_mesh( );

  /**
   * Prints info on the object.
   */
  void print_info( ) const;

  /**
   * Loads mesh from a file.
   * @param[] file File name.
   */
  bool load( const std::string & file );

  /**
   * Returns lenght of a single temporal element.
   * @param[in] i_elem Index of the element.
   */
  sc length( lo i_elem ) const {
    return _lengths[ i_elem ];
  }

  /**
   * Returns number of temporal elements.
   */
  lo get_n_elements( ) const {
    return _n_timesteps;
  }

  /**
   * Returns number of temporal nodes.
   */
  lo get_n_nodes( ) const {
    return _n_temporal_nodes;
  }

  /**
   * Returns node indices (start, end temporal ind.) of an element.
   * @param[in] i_element Index of the element.
   * @param[out] element Element indices.
   */
  void get_element(
    lo i_element, linear_algebra::indices< 2 > & element ) const {
    element[ 0 ] = _elements[ 2 * i_element ];
    element[ 1 ] = _elements[ 2 * i_element + 1 ];
  }

  /**
   * Returns coordinate of a temporal node (specific time).
   * @param[in] i_node Index of the temporal node.
   * @param[out] node
   */
  void get_node( lo i_node, linear_algebra::coordinates< 1 > & node ) const {
    node[ 0 ] = _nodes[ i_node ];
  }

  /**
   * Returns coordinate of a temporal node (specific time).
   */
  sc get_node( lo i_node ) const {
    return _nodes[ i_node ];
  }

  /**
   * Returns coordinates of all nodes of an element.
   * @param[in] i_element Index of the element.
   * @param[out] node1 Coordinate of the first node (beginning of the temporal
   * subinterval).
   * @param[out] node2 Coordinate of the second node (end of the temporal
   * subinterval).
   */
  void get_nodes( lo i_element, linear_algebra::coordinates< 1 > & node1,
    linear_algebra::coordinates< 1 > & node2 ) const {
    node1[ 0 ] = _nodes[ _elements[ 2 * i_element ] ];
    node2[ 0 ] = _nodes[ _elements[ 2 * i_element + 1 ] ];
  }

  /**
   * Returns coordinates of all nodes of an element.
   * @param[in] i_element Index of the element.
   * @param[out] node1 Coordinate of the first node (beginning of the temporal
   * subinterval).
   * @param[out] node2 Coordinate of the second node (end of the temporal
   * subinterval).
   */
  void get_nodes( lo i_element, sc * node1, sc * node2 ) const {
    node1[ 0 ] = _nodes[ _elements[ 2 * i_element ] ];
    node2[ 0 ] = _nodes[ _elements[ 2 * i_element + 1 ] ];
  }

  /**
   * Refines the mesh by bisection.
   * @param[in] level Number of refinements.
   */
  void refine( int level );

  /**
   * Returns the start of the temporal interval.
   */
  sc get_start( ) const {
    return _start_time;
  }

  /**
   * Returns the start of the temporal interval.
   */
  sc get_end( ) const {
    return _end_time;
  }

  /**
   * Returns centroid of the i-th element.
   * @param[in] i_element Index of the element.
   * @param[out] centroid Centroid of the element;
   */
  void get_centroid(
    lo i_element, linear_algebra::coordinates< 1 > & centroid ) const {
    centroid( 0 ) = ( _nodes[ _elements[ 2 * i_element ] ]
                      + _nodes[ _elements[ 2 * i_element + 1 ] ] )
      / 2;
  }

  /**
   * Returns centroid of the i-th element.
   * @param[in] i_element Index of the element.
   */
  sc get_centroid( lo i_element ) const {
    return ( _nodes[ _elements[ 2 * i_element ] ]
             + _nodes[ _elements[ 2 * i_element + 1 ] ] )
      / 2;
  }

 protected:
  sc _start_time;  //!< temporal interval set to (start_time, end_time)
  sc _end_time;    //!< temporal interval set to (start_time, end_time)

  lo _n_temporal_nodes;      //!< number of temporal nodes (_n_timesteps + 1)
  std::vector< sc > _nodes;  //!< coordinates of temporal nodes

  lo _n_timesteps;  //!< number of timesteps (temporal subintervals/elements)
  std::vector< lo >
    _elements;  //!< indices into #_nodes defining individual time subintervals
  std::vector< sc > _lengths;  //!< lengths of temporal elements

  /**
   * Precomputes lengths of temporal elements
   */
  void init_lengths( );
};

#endif /* INCLUDE_BESTHEA_TEMPORAL_MESH_H_ */
