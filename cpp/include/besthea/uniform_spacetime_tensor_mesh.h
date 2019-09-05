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

/** @file uniform_spacetime_tensor_mesh.h
 * @brief Tensor product spacetime mesh based on a triangular_surface_mesh.h and
 * uniform time stepping.
 */

#ifndef INCLUDE_BESTHEA_UNIFORM_SPACETIME_TENSOR_MESH_H_
#define INCLUDE_BESTHEA_UNIFORM_SPACETIME_TENSOR_MESH_H_

#include "besthea/linear_algebra.h"
#include "besthea/mesh.h"
#include "besthea/spacetime_tensor_mesh.h"
#include "besthea/triangular_surface_mesh.h"

namespace besthea {
  namespace mesh {
    class uniform_spacetime_tensor_mesh;
  }
}

/**
 *  Class representing a tensor product space-time mesh with a uniform timestep.
 */
class besthea::mesh::uniform_spacetime_tensor_mesh
  : public spacetime_tensor_mesh {
 public:
  /**
   * Constructing mesh from a file.
   * @param[in] space_mesh Reference to a triangular_surface_mesh.h.
   * @param[in] end_time Temporal interval set to (0,end_time).
   * @param[in] n_timesteps Number of timesteps.
   */
  uniform_spacetime_tensor_mesh(
    triangular_surface_mesh & space_mesh, sc end_time, lo n_timesteps );

  uniform_spacetime_tensor_mesh( const uniform_spacetime_tensor_mesh & that )
    = delete;

  ~uniform_spacetime_tensor_mesh( );

  /**
   * Prints info on the object.
   */
  virtual void print_info( ) const;

  /**
   * Refines the spatial mesh by quadrisection, temporal by bisection.
   * @param[in] level Number of spatial refinements.
   * @param[in] temporal_order Number of temporal refinements per single spatial
   * refinement.
   */
  virtual void refine( int level, int temporal_order = 1 );

  /**
   * Returns number of space-time elements.
   */
  virtual lo get_n_elements( ) const {
    return _space_mesh->get_n_elements( ) * _n_timesteps;
  }

  /**
   * Returns the uniform timestep.
   */
  sc get_timestep( ) {
    return _timestep;
  }

  /**
   * Prints the EnSight Gold case file.
   * @param[in] directory Directory to which the case file is saved.
   * @param[in] node_labels Labels for nodal data.
   * @param[in] element_labels Labels for elemental data.
   * @param[in] time_stride Stride for time steps
   */
  virtual bool print_ensight_case( const std::string & directory,
    const std::vector< std::string > * node_labels = nullptr,
    const std::vector< std::string > * element_labels = nullptr,
    lo time_stride = 1 ) const {
    return _space_mesh->print_ensight_case( directory, node_labels,
      element_labels, ( _n_timesteps + time_stride - 1 ) / time_stride,
      _timestep * time_stride );
  }

  /**
   * Prints the EnSight Gold geometry file.
   * @param[in] directory Directory to which the geometry file is saved.
   */
  virtual bool print_ensight_geometry( const std::string & directory ) const {
    return _space_mesh->print_ensight_geometry( directory );
  }

  /**
   * Prints the EnSight Variable files for per element and per node data.
   * @param[in] directory Directory that datafile are printed to.
   * @param[in] node_labels Labels for nodal data.
   * @param[in] node_data Scalar nodal data.
   * @param[in] element_labels Labels for elemental data.
   * @param[in] element_data Scalar elemental data.
   * @param[in] time_stride Stride for time steps.
   */
  virtual bool print_ensight_datafiles( const std::string & directory,
    const std::vector< std::string > * node_labels = nullptr,
    const std::vector< linear_algebra::block_vector * > * node_data = nullptr,
    const std::vector< std::string > * element_labels = nullptr,
    const std::vector< linear_algebra::block_vector * > * element_data
    = nullptr,
    lo time_stride = 1 ) const {
    lo n_nodal = node_data ? node_data->size( ) : 0;
    lo n_elem = element_data ? element_data->size( ) : 0;
    std::vector< linear_algebra::vector * > node_data_for_one_ts( n_nodal );
    std::vector< linear_algebra::vector * > elem_data_for_one_ts( n_elem );
    for ( lo ts = 0; ts < _n_timesteps; ts += time_stride ) {
      for ( lo i = 0; i < n_nodal; ++i ) {
        node_data_for_one_ts[ i ] = &( *node_data )[ i ]->get_block( ts );
      }

      for ( lo i = 0; i < n_elem; ++i ) {
        elem_data_for_one_ts[ i ] = &( *element_data )[ i ]->get_block( ts );
      }

      if ( !_space_mesh->print_ensight_datafiles( directory, node_labels,
             &node_data_for_one_ts, element_labels, &elem_data_for_one_ts,
             ts / time_stride ) ) {
        return false;
      }
    }

    return true;
  }

 protected:
  /**
   * Refines the spatial mesh by quadrisection.
   * @param[in] level Number of spatial refinements.
   */
  // virtual void refine_space( int level );

  /**
   * Refines the temporal mesh by bisection.
   * @param[in] level Number of temporal refinements.
   */
  // virtual void refine_time( int level );

  sc _end_time;     //!< temporal interval set to (0,end_time)
  sc _timestep;     //!< size of the timestep
  lo _n_timesteps;  //!< number of timesteps
};

#endif /* INCLUDE_BESTHEA_UNIFORM_SPACETIME_TENSOR_MESH_H_ */
