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

/** @file distribute_block_vector.h
 * @brief Collection of scalar vector forming a block vector distributed among
 * MPI ranks.
 */

#ifndef INCLUDE_BESTHEA_DISTRIBUTED_BLOCK_VECTOR_H_
#define INCLUDE_BESTHEA_DISTRIBUTED_BLOCK_VECTOR_H_

#include "besthea/full_matrix.h"
#include "besthea/settings.h"
#include "besthea/vector.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <vector>

namespace besthea {
  namespace linear_algebra {
    class distributed_block_vector;
  }
}

namespace besthea {
  namespace mesh {
    class spacetime_cluster;
    class general_spacetime_cluster;
  }
}

namespace besthea {
  namespace bem {
    template< class basis_type >
    class fast_spacetime_be_space;
  }
}

/**
 *  Class representing a distributed block vector.
 */
class besthea::linear_algebra::distributed_block_vector {
 public:
  using vector_type = besthea::linear_algebra::vector;  //!< Vector type.

  /**
   * Constructor.
   */
  distributed_block_vector( );

  /**
   * Copy constructor.
   * @param[in] that Vector to be copied.
   */
  distributed_block_vector( const distributed_block_vector & that );

  /**
   * Constructor with an initializer list.
   * @param[in] block_size Number of blocks.
   * @param[in] list Initializer list for vector.
   * @param[in] comm MPI communicator associated with the vector.
   */
  distributed_block_vector( lo block_size, std::initializer_list< sc > list,
    MPI_Comm comm = MPI_COMM_WORLD );

  /**
   * Constructing a vector of the given size.
   * @param[in] block_size Number of blocks.
   * @param[in] size Length of the vector.
   * @param[in] zero Initialize to 0 if true.
   * @param[in] comm MPI communicator associated with the vector.
   */
  distributed_block_vector(
    lo block_size, lo size, bool zero = true, MPI_Comm comm = MPI_COMM_WORLD );

  /**
   * Constructing a vector of the given size.
   * @param[in] block_size Number of blocks.
   * @param[in] size Length of the vector.
   * @param[in] zero Initialize to 0 if true.
   * @param[in] my_blocks std::vector of block indices associated with this rank
   * @param[in] comm MPI communicator associated with the vector.
   */
  distributed_block_vector( std::vector< lo > & my_blocks, lo block_size,
    lo size, bool zero = true, MPI_Comm comm = MPI_COMM_WORLD );

  ~distributed_block_vector( );

  /**
   * Returns a reference to a single block.
   * The block may be empty if not called on the correct rank.
   * @param[in] d Index of the block.
   */
  vector_type & get_block( lo d ) {
    return _data[ d ];
  }

  /**
   * Returns a reference to a single block.
   * WARNING: The block may be empty if not called on the correct rank.
   * @param[in] d Index of the block.
   */
  const vector_type & get_block( lo d ) const {
    return _data[ d ];
  }

  /*!
   * @brief Returns the i-th element of the d-th block.
   * WARNING: Returns NaN for blocks not owned by this rank.
   * @param[in] d Block index.
   * @param[in] i Element index.
   */
  sc get( lo d, lo i ) const {
    sc val;
    if ( _owners[ d ][ 0 ] == _rank ) {
      val = _data[ d ][ i ];
    } else {
      val = std::numeric_limits< double >::quiet_NaN( );
    }
    return val;
  }

  /**
   * Returns the block dimension (number of blocks).
   */
  lo get_block_size( ) const {
    return _block_size;
  }

  /**
   * Returns the dimension of a single block
   */
  lo get_size_of_block( ) const {
    return _size;
  }

  /**
   * Returns the dimension of the whole vector.
   */
  lo size( ) const {
    return _block_size * _size;
  }

  /**
   * Resizes the block vector.
   * @param[in] block_size New size.
   */
  void resize( lo block_size );

  /**
   * Resizes the block vector, resets the owners.
   * @param[in] my_blocks Vector of blocks owned by the current process.
   * @param[in] block_size New size.
   */
  void resize( std::vector< lo > & my_blocks, lo block_size );

  /**
   * Resizes the vector blocks.
   * @param[in] size New size of block.
   * @param[in] zero Initialize to 0 if true.
   */
  void resize_blocks( lo size, bool zero = true ) {
    lo i = 0;
    for ( vector_type & v : _data ) {
      if ( _owners[ i ][ 0 ] == _rank ) {
        v.resize( size, zero );
      }
      ++i;
    }
    _size = size;
  }

  /*!
   * @brief Sets the i-th element of the d-th block (only on the process that
   * owns the block).
   * @param[in] d Block index.
   * @param[in] i Element index.
   * @param[in] value Value to be set.
   */
  void set( lo d, lo i, sc value ) {
    if ( _owners[ d ][ 0 ] == _rank ) {
      _data[ d ][ i ] = value;
    }
  }

  /*!
   * @brief Adds atomically(!) to a single position of a vector (only on the
   * process that owns the block).
   * @param[in] d Block index.
   * @param[in] i Element index.
   * @param[in] value Value to be added.
   */
  void add_atomic( lo d, lo i, sc value ) {
    if ( _owners[ d ][ 0 ] == _rank ) {
#pragma omp atomic update
      _data[ d ][ i ] += value;
    }
  }

  /*!
   * @brief Adds to a single position of a vector. (only on the
   * process that owns the block).
   * @param[in] d Block index.
   * @param[in] i Element index.
   * @param[in] value Value to be added.
   */
  void add( lo d, lo i, sc value ) {
    if ( _owners[ d ][ 0 ] == _rank ) {
      _data[ d ][ i ] += value;
    }
  }

  /*!
   * Scales the vector by a scalar alpha
   * @param[in] alpha
   */
  void scale( sc alpha ) {
    for ( auto & it : _my_blocks ) {
      _data[ it ].scale( alpha );
    }
  }

  /**
   * Copies data from another block vector.
   * @param[in] that Vector to be copied.
   */
  void copy( const distributed_block_vector & that );

  /*!
   * @brief Copies data from a raw vector (whole vector is duplicated on all
   * ranks).
   * @param[in] block_size Number of blocks.
   * @param[in] size Length of the vector.
   * @param[in] data Array to copy from.
   */
  void copy_from_raw( lo block_size, lo size, const sc * data );

  /*!
   * @brief Copies data from a raw vector (only copies section associated with
   * curren MPI rank).
   * @param[in] my_blocks Blocks owned by this MPI process.
   * @param[in] block_size Number of blocks.
   * @param[in] size Length of the vector.
   * @param[in] data Array to copy from.
   */
  void copy_from_raw(
    std::vector< lo > & my_blocks, lo block_size, lo size, const sc * data );

  /*!
   * @brief Copies data to a raw vector (duplicates the results on all ranks).
   * @param[in] data Array to copy to.
   */
  void copy_to_raw( sc * data ) const;

  /*!
   * @brief Copies data from a raw vector.
   * @param[in] block_size Number of blocks.
   * @param[in] size Length of the vector.
   * @param[in] data Array to copy from.
   */
  void copy_from_vector( lo block_size, lo size, const vector_type & data );

  /*!
   * @brief Copies data from a raw vector.
   * @param[in] my_blocks Blocks owned by this MPI process.
   * @param[in] block_size Number of blocks.
   * @param[in] size Length of the vector.
   * @param[in] data Array to copy from.
   */
  void copy_from_vector( std::vector< lo > & my_blocks, lo block_size, lo size,
    const vector_type & data );

  /*!
   * @brief Copies data to a raw vector.
   * @param[in] data Array to copy to.
   */
  void copy_to_vector( vector_type & data ) const;

  /*!
   * @brief Vector addition this += alpha * v. Only vectors with the same
   * distribution across MPI processes will be summed correctly.
   * @param[in] v
   * @param[in] alpha
   */
  void add( distributed_block_vector const & v, sc alpha = 1.0 );

  /*!
   * @brief Fills the block vector with the given value.
   * @param[in] value
   */
  void fill( sc value ) {
    for ( lo i = 0; i < _block_size; ++i ) {
      if ( am_i_owner( i ) ) {
        _data[ i ].fill( value );
      }
    }
  }

  /*!
   * @brief Returns the euclidean dot product.
   * @param[in] v
   */
  sc dot( distributed_block_vector const & v ) const;

  /*!
   * @brief Returns the Euclidean norm of the vector.
   * @return Euclidean norm of the vector.
   */
  sc norm( ) const {
    return std::sqrt( this->dot( *this ) );
  }

  /*!
   * Primary owner rank sends data to all other owners.
   */
  void synchronize_shared_parts( );

  /*!
   * Gets local part of a block vector corresponding to dofs in a spacetime
   * cluster.
   * @param[in] cluster  Cluster determining the local dofs.
   * @param[in,out] local_vector Local part of block vector.
   * @tparam space_type  fast_spacetime_be_space representing either p0 or p1
   *                     basis functions. It determines the dofs.
   * @warning The local vector must have the correct size.
   * @note The local vector is not a block vector anymore, but a contiguous
   *       vector.
   */
  template< class space_type >
  void get_local_part( besthea::mesh::spacetime_cluster * cluster,
    besthea::linear_algebra::vector & local_vector ) const;

  /*!
   * Gets local part of a block vector corresponding to dofs in a spacetime
   * cluster.
   * @param[in] cluster  Cluster determining the local dofs.
   * @param[in,out] local_vector Local part of block vector.
   * @tparam space_type  fast_spacetime_be_space representing either p0 or p1
   *                     basis functions. It determines the dofs.
   * @warning The local vector must have the correct size.
   * @note The local vector is not a block vector anymore, but a contiguous
   *       vector.
   */
  template< class space_type >
  void get_local_part( besthea::mesh::general_spacetime_cluster * cluster,
    besthea::linear_algebra::vector & local_vector ) const;

  /*!
   * Gets local part of a block vector corresponding to dofs in a spacetime
   * cluster and stores it in full matrix format.
   * @param[in] cluster  Cluster determining the local dofs.
   * @param[in,out] local_part  Local part of block vector.
   * @tparam space_type distributed_fast_spacetime_be_space representing either
   *                    p0 or p1 basis functions. It determines the dofs.
   * @note Rows of the output matrix correspond to time, columns to space.
   */
  template< class space_type >
  void get_local_part( besthea::mesh::general_spacetime_cluster * cluster,
    besthea::linear_algebra::full_matrix & local_part ) const;

  /*!
   * Adds local vector to appropriate positions of a block vector. The positions
   * are determined by the dofs in a spacetime cluster.
   * @param[in] cluster  Cluster determining the positions in the
   block_vector
   * to which the local vector is added.
   * @param[in] local_vector Local part of block vector to be added.
   * @tparam space_type  fast_spacetime_be_space representing either p0 or p1
   *                     basis functions. It determines the dofs.
   * @note The entries in the local vector are ordered according to the
   ordering
   *       of the time elements and spatial dofs in the spacetime cluster (time
   *       step after time step).
   */
  template< class space_type >
  void add_local_part( besthea::mesh::spacetime_cluster * cluster,
    const besthea::linear_algebra::vector & local_vector );

  /*!
   * Adds local vector to appropriate positions of a block vector. The positions
   * are determined by the dofs in a spacetime cluster.
   * @param[in] cluster Cluster determining the positions in the
                        block_vector to which the local vector is added.
   * @param[in] local_vector Local part of block vector to be added.
   * @tparam space_type  fast_spacetime_be_space representing either p0 or p1
   *                     basis functions. It determines the dofs.
   * @note  The entries in the local vector are ordered according to the
   *        ordering of the time elements and spatial dofs in the spacetime
   *        cluster (time step after time step).
   */
  template< class space_type >
  void add_local_part( const besthea::mesh::general_spacetime_cluster * cluster,
    const besthea::linear_algebra::vector & local_vector );

  /*!
   * Adds local part stored in full matrix format to appropriate positions of a
   * block vector. The positions are determined by the dofs in a spacetime
   * cluster.
   * @param[in] cluster Cluster determining the positions in the
                        block_vector to which the local vector is added.
   * @param[in] local_part  Local part to be added. It is stored in matrix
                            format, where rows correspond to time and columns to
                            space.
   * @tparam space_type  fast_spacetime_be_space representing either p0 or p1
   *                     basis functions. It determines the dofs.
   */
  template< class space_type >
  void add_local_part( const besthea::mesh::general_spacetime_cluster * cluster,
    const besthea::linear_algebra::full_matrix & local_part );

  /*!
   * @brief Returns reference to the vector of vector of MPI ranks owning
   * individual blocks. Outer vector corresponds to vector's blocks.
   * @return std::vector of std::vectors of MPI ranks owning individual blocks.
   */
  const std::vector< std::vector< int > > & get_owners( ) const {
    return _owners;
  }

  /*!
   * @brief Returns reference to the vector blocks owned by the current process.
   * @return std::vector indices of blocks owned by the calling MPI ranks.
   */
  std::vector< lo > get_my_blocks( ) const {
    return _my_blocks;
  }

  /*!
   * Returns whether the current MPI rank owns the given block.
   * @param[in] block_idx Index of given block.
   * @return True if the calling process owns the given vector block.
   */
  bool am_i_owner( lo block_idx ) const {
    return ( _owners[ block_idx ][ 0 ] == _rank );
  }

  /*!
   * Returns whether the current MPI rank is a primary owner of the given block.
   * @param[in] block_idx Index of given block.
   * @return True if the calling process is the primary owner of the vector
   * block.
   */
  bool am_i_primary_owner( lo block_idx ) const {
    return ( get_primary_owner( block_idx ) == _rank );
  }

  /*!
   * Sends block to a given rank
   * @param[in] block_idx Block to be sent.
   * @param[in] rank MPI rank of receiver.
   * @param[out] data Vector to store the received data.
   */
  void communicate_block( lo block_idx, int rank, vector_type & data ) const;

  void print( std::ostream & stream = std::cout ) const;

  MPI_Comm get_comm( ) const {
    return _comm;
  }

  /*!
   * @returns boolean indicating whether vector blocks are duplicated on all MPI
   * ranks.
   */
  bool is_duplicated( ) const {
    return _duplicated;
  }

 protected:
  /*!
   * The method collects information about location of remote blocks.
   * @param[in] my_blocks Blocks owned by this MPI process.
   */
  void communicate_owners( std::vector< lo > & my_blocks );

  /*!
   * Return the primary owner (the one with the lowest rank)
   * @param[in] block_idx Index of the block.
   * @returns The lowest rank owning the block.
   */
  int get_primary_owner( lo block_idx ) const {
    if ( _owners[ block_idx ].size( ) == 1 ) {
      return _owners[ block_idx ][ 0 ];
    } else {
      return _owners[ block_idx ][ 0 ] < _owners[ block_idx ][ 1 ]
        ? _owners[ block_idx ][ 0 ]
        : _owners[ block_idx ][ 1 ];
    }
  }

  lo _block_size;                    //!< block size (number of blocks)
  lo _size;                          //!< vector size (size of block)
  std::vector< vector_type > _data;  //!< raw data
  std::vector< std::vector< int > >
    _owners;  //!< vector of vectors of MPI ranks owning individual blocks (if
              //!< the rank owns the block, it is listed at the first in the
              //!< inner vector, remaining ranks are sorted from the lowest)
  std::vector< lo > _my_blocks;  //!< list of blocks the rank owns
  MPI_Comm _comm;    //!< MPI communicator associated with the pFMM matrix.
  int _rank;         //!< MPI rank of the process
  bool _duplicated;  //!< the vector is duplicated on all MPI ranks.
};

#endif
