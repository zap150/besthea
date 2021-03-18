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

/** @file distributed_block_vector.h
 * @brief Contains a class representing a block vector, i.e. a vector of scalars
 * partitioned into blocks. The block vector is distributed among MPI ranks.
 * @note updated documentation
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
 * Class representing a distributed block vector, i.e. a vector of scalars
 * partitioned into blocks, which are distributed among MPI ranks.
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
   * Constructs a distributed block vector with an initializer list.
   * All @p n_blocks have the same size and entries as the provided list. The
   * block vector is duplicated on all MPI ranks of the given communicator.
   * @param[in] n_blocks Number of blocks.
   * @param[in] list Initializer list for vector.
   * @param[in] comm MPI communicator associated with the vector.
   */
  distributed_block_vector( lo n_blocks, std::initializer_list< sc > list,
    MPI_Comm comm = MPI_COMM_WORLD );

  /**
   * Constructs a distributed block vector with a given number of blocks of
   * given size. The block vector is duplicated on all MPI ranks of the given
   * communicator.
   * @param[in] n_blocks Number of blocks.
   * @param[in] size Size of each block.
   * @param[in] zero Initialize to 0 if true.
   * @param[in] comm MPI communicator associated with the vector.
   */
  distributed_block_vector(
    lo n_blocks, lo size, bool zero = true, MPI_Comm comm = MPI_COMM_WORLD );

  /**
   * Constructs a distributed block vector with a given number of blocks of
   * given size. The vector is distributed according to the information in the
   * vector @p my_blocks.
   * @param[in] my_blocks Indices of blocks, which are owned by the executing
   *                      process.
   * @param[in] n_blocks Number of blocks.
   * @param[in] size Size of each block.
   * @param[in] zero Initialize to 0 if true.
   * @param[in] comm MPI communicator associated with the vector.
   */
  distributed_block_vector( std::vector< lo > & my_blocks, lo n_blocks, lo size,
    bool zero = true, MPI_Comm comm = MPI_COMM_WORLD );

  ~distributed_block_vector( );

  /**
   * Returns a reference to a single block.
   * @param[in] d Index of the block.
   * @warning If the executing process does not own the d-th block the returned
   * vector is empty.
   */
  vector_type & get_block( lo d ) {
    return _data[ d ];
  }

  /**
   * Returns a constant reference to a single block.
   * @param[in] d Index of the block.
   * @warning If the executing process does not own the d-th block the returned
   * vector is empty.
   */
  const vector_type & get_block( lo d ) const {
    return _data[ d ];
  }

  /**
   * @brief Returns the i-th element of the d-th block.
   * @param[in] d Block index.
   * @param[in] i Element index.
   * @warning Returns NaN if the d-th block is not owned by the executing
   * process.
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
   * Returns the number of blocks.
   */
  lo get_n_blocks( ) const {
    return _n_blocks;
  }

  /**
   * Returns the size of a single block
   */
  lo get_size_of_block( ) const {
    return _size;
  }

  /**
   * Returns the size of the whole block vector, i.e. the total number of
   * elements.
   */
  lo size( ) const {
    return _n_blocks * _size;
  }

  /**
   * Resizes the block vector by changing the number of blocks. The resulting
   * block vector is duplicated on all MPI ranks of its communicator.
   * @param[in] n_blocks New number of blocks.
   * @warning The existing data is not communicated between the MPI ranks. The
   * duplication is understood in the sense that all ranks are listed as owners
   * after resizing the block vector.
   */
  void resize( lo n_blocks );

  /**
   * Resizes the block vector by changing the number of blocks. The owners are
   * reset using the information in @p my_blocks.
   * @param[in] my_blocks Indices of blocks, which are owned by the executing
   *                      process. The member @p _my_blocks is overwritten with
   *                      this vector.
   * @param[in] n_blocks New number of blocks.
   */
  void resize( std::vector< lo > & my_blocks, lo n_blocks );

  /**
   * Resizes all blocks of the block vector, which are owned by the executing
   * process.
   * @param[in] size New size of each block.
   * @param[in] zero If true, all blocks are filled with zeros.
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

  /**
   * @brief Sets the i-th element of the d-th block
   * @param[in] d Block index.
   * @param[in] i Element index.
   * @param[in] value Value to be set.
   * @warning If the executing process does not own the block nothing happens.
   */
  void set( lo d, lo i, sc value ) {
    if ( _owners[ d ][ 0 ] == _rank ) {
      _data[ d ][ i ] = value;
    }
  }

  /**
   * @brief Adds a value atomically(!) to a single element of a single block.
   * @param[in] d Block index.
   * @param[in] i Element index.
   * @param[in] value Value to be added.
   * @warning If the executing process does not own the block nothing happens.
   */
  void add_atomic( lo d, lo i, sc value ) {
    if ( _owners[ d ][ 0 ] == _rank ) {
#pragma omp atomic update
      _data[ d ][ i ] += value;
    }
  }

  /*!
   * @brief Adds a value to a single element of a single block.
   * @param[in] d Block index.
   * @param[in] i Element index.
   * @param[in] value Value to be added.
   * @warning If the executing process does not own the block nothing happens.
   */
  void add( lo d, lo i, sc value ) {
    if ( _owners[ d ][ 0 ] == _rank ) {
      _data[ d ][ i ] += value;
    }
  }

  /*!
   * Scales all blocks owned by the executing process by a scalar alpha.
   * @param[in] alpha Scaling factor.
   */
  void scale( sc alpha ) {
    for ( auto & it : _my_blocks ) {
      _data[ it ].scale( alpha );
    }
  }

  /**
   * Copies data from another distributed block vector.
   * @param[in] that Vector to be copied.
   */
  void copy( const distributed_block_vector & that );

  /*!
   * @brief Copies data from a raw array. The resulting block vector is
   * duplicated on all MPI ranks of its communicator.
   * @param[in] n_blocks Number of blocks.
   * @param[in] size Size of each block.
   * @param[in] data Array to copy from. Contains all elements, block by block.
   * @note If @p n_blocks and @p size are different from the member variables
   * @p _n_blocks and @p _size, respectively, the block vector is resized
   * appropriately.
   * @warning The source array has to contain at least @p n_blocks * @p size
   * elements.
   */
  void copy_from_raw( lo n_blocks, lo size, const sc * data );

  /*!
   * @brief Fills the distributed vector by copying data from a raw array. Only
   * the blocks owned by the executing process are filled with the associated
   * values of the array.
   * @param[in] my_blocks Indices of blocks, which are owned by the executing
   *                      process. The member @p _my_blocks is overwritten with
   *                      this vector.
   * @param[in] n_blocks Number of blocks.
   * @param[in] size Size of each block.
   * @param[in] data Array to copy from. Contains all elements, block by block.
   * @note If @p n_blocks and @p size are different from the member variables
   * @p _n_blocks and @p _size, respectively, the block vector is resized
   * appropriately.
   * @warning The source array has to contain at least @p n_blocks * @p size
   * elements.
   * @warning If @p _n_blocks == @p n_blocks then it is assumed that
   * @p _my_blocks == @p my_blocks, i.e. this information is not updated.
   */
  void copy_from_raw(
    std::vector< lo > & my_blocks, lo n_blocks, lo size, const sc * data );

  /*!
   * @brief Copies the whole distributed block vector to a raw array. If the
   * block vector is not duplicated, all blocks are broadcasted by the
   * respective primary owners and then written to (a local copy of) data by all
   * ranks.
   * @param[in,out] data Array to copy to. Is filled with all elements, block by
   *                     block.
   * @warning The array's size has to be at least @p _n_blocks * @p _size.
   */
  void copy_to_raw( sc * data ) const;

  /*!
   * @brief Copies data from a raw vector. The resulting block vector is
   * duplicated on all MPI ranks of its communicator.
   * @param[in] n_blocks Number of blocks.
   * @param[in] size Size of each block.
   * @param[in] data Vector to copy from. Contains all elements, block by block.
   * @note If @p n_blocks and @p size are different from the member variables
   * @p _n_blocks and @p _size, respectively, the block vector is resized
   * appropriately.
   * @warning The source vector has to contain at least @p n_blocks * @p size
   * elements.
   */
  void copy_from_vector( lo n_blocks, lo size, const vector_type & data );

  /*!
   * @brief Fills the distributed vector by copying data from a raw vector. Only
   * the blocks owned by the executing process are filled with the associated
   * values of the vector.
   * @param[in] my_blocks Indices of blocks, which are owned by the executing
   *                      process. The member @p _my_blocks is overwritten with
   *                      this vector.
   * @param[in] n_blocks Number of blocks.
   * @param[in] size Size of each block.
   * @param[in] data Vector to copy from. Contains all elements, block by block.
   * @note If @p n_blocks and @p size are different from the member variables
   * @p _n_blocks and @p _size, respectively, the block vector is resized
   * appropriately.
   * @warning The source array has to contain at least @p n_blocks * @p size
   * elements.
   * @warning If @p _n_blocks == @p n_blocks then it is assumed that
   * @p _my_blocks == @p my_blocks, i.e. this information is not updated.
   */
  void copy_from_vector( std::vector< lo > & my_blocks, lo n_blocks, lo size,
    const vector_type & data );

  /*!
   * @brief Copies the whole distributed block vector to a raw vector. If the
   * block vector is not duplicated, all blocks are broadcasted by the
   * respective primary owners and then written to (a local copy of) data by all
   * ranks.
   * @param[in,out] data Vector to copy to. Is filled with all elements, block
   *                     by block.
   * @note The vector is resized. New size is @p _n_blocks * @p _size.
   */
  void copy_to_vector( vector_type & data ) const;

  /*!
   * @brief Vector addition: this += alpha * v.
   * @param[in] v Block vector with the same number and size of blocks.
   * @param[in] alpha Scaling factor.
   * @warning The executing process applies the vector addition for a block only
   * if it owns this block and the corresponding block of @p v. In particular,
   * only vectors with the same distribution across MPI processes will be summed
   * up correctly.
   */
  void add( distributed_block_vector const & v, sc alpha = 1.0 );

  /*!
   * @brief Fills all blocks which are owned by the executing process with the
   * given value.
   * @param[in] value Value to fill the owned blocks with.
   */
  void fill( sc value ) {
    for ( lo i = 0; i < _n_blocks; ++i ) {
      if ( am_i_owner( i ) ) {
        _data[ i ].fill( value );
      }
    }
  }

  /*!
   * @brief Returns the euclidean dot product.
   * @param[in] v Second distributed block vector for dot product.
   * @warning @p v has to have the same dimensions and distribution across MPI
   * processes, otherwise the result is wrong and the behavior is undefined.
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
   * Synchronizes the distributed block vector via MPI communication. For each
   * block, its primary owner sends the data to all other owners.
   */
  void synchronize_shared_parts( );

  /*!
   * Gets the local part of a distributed block vector corresponding to the dofs
   * in a spacetime cluster.
   * @param[in] cluster  Cluster determining the local dofs.
   * @param[in,out] local_vector Local part of block vector.
   * @tparam space_type  @ref besthea::bem::fast_spacetime_be_space representing
   *                     either p0 or p1 basis functions. It determines the
   *                     DOFs.
   * @warning The local vector must have the correct size.
   * @note The local vector is not a block vector anymore, but a contiguous
   *       vector.
   * @warning The executing process has to own the blocks corresponding to the
   * dofs in the spacetime cluster, otherwise the local vector is not filled
   * correctly.
   */
  template< class space_type >
  void get_local_part( besthea::mesh::spacetime_cluster * cluster,
    besthea::linear_algebra::vector & local_vector ) const;

  /*!
   * Gets the local part of a distributed block vector corresponding to the dofs
   * in a spacetime cluster.
   * @param[in] cluster  Cluster determining the local dofs.
   * @param[in,out] local_vector Local part of block vector.
   * @tparam space_type  @ref besthea::bem::fast_spacetime_be_space representing
   *                     either p0 or p1 basis functions. It determines the
   *                     DOFs.
   * @warning The local vector must have the correct size.
   * @note The local vector is not a block vector anymore, but a contiguous
   *       vector.
   * @warning The executing process has to own the blocks corresponding to the
   * dofs in the spacetime cluster, otherwise the local vector is not filled
   * correctly.
   */
  template< class space_type >
  void get_local_part( besthea::mesh::general_spacetime_cluster * cluster,
    besthea::linear_algebra::vector & local_vector ) const;

  /*!
   * Gets the local part of a distributed block vector corresponding to the dofs
   * in a spacetime cluster and stores it in full matrix format.
   * @param[in] cluster  Cluster determining the local dofs.
   * @param[in,out] local_part  Local part of block vector.
   * @tparam space_type @ref besthea::bem::fast_spacetime_be_space representing
   *                     either p0 or p1 basis functions. It determines the
   *                     DOFs.
   * @note Rows of the output matrix correspond to time, columns to space.
   */
  template< class space_type >
  void get_local_part( besthea::mesh::general_spacetime_cluster * cluster,
    besthea::linear_algebra::full_matrix & local_part ) const;

  /*!
   * Adds a local vector to the appropriate positions of a distributed block
   * vector. The positions are determined by the dofs in a spacetime cluster.
   * @param[in] cluster  Cluster determining the positions in the distributed
   *                     blockvector to which the local vector is added.
   * @param[in] local_vector Local part of block vector to be added.
   * @tparam space_type  @ref besthea::bem::fast_spacetime_be_space representing
   *                     either p0 or p1 basis functions. It determines the
   *                     DOFs.
   * @note The entries in the local vector are ordered according to the ordering
   *       of the time elements and spatial dofs in the spacetime cluster (time
   *       step after time step).
   * @warning The executing process has to own the blocks corresponding to the
   * dofs in the spacetime cluster, otherwise nothing is added.
   */
  template< class space_type >
  void add_local_part( besthea::mesh::spacetime_cluster * cluster,
    const besthea::linear_algebra::vector & local_vector );

  /*!
   * Adds a local vector to the appropriate positions of a distributed block
   * vector. The positions are determined by the dofs in a spacetime cluster.
   * @param[in] cluster  Cluster determining the positions in the distributed
   *                     blockvector to which the local vector is added.
   * @param[in] local_vector Local part of block vector to be added.
   * @tparam space_type  @ref besthea::bem::fast_spacetime_be_space representing
   *                     either p0 or p1 basis functions. It determines the
   *                     DOFs.
   * @note The entries in the local vector are ordered according to the ordering
   *       of the time elements and spatial dofs in the spacetime cluster (time
   *       step after time step).
   * @warning The executing process has to own the blocks corresponding to the
   * dofs in the spacetime cluster, otherwise nothing is added.
   */
  template< class space_type >
  void add_local_part( const besthea::mesh::general_spacetime_cluster * cluster,
    const besthea::linear_algebra::vector & local_vector );

  /*!
   * Adds local part stored in full matrix format to the appropriate positions
   * of a distributed block vector. The positions are determined by the dofs in
   * a spacetime cluster.
   * @param[in] cluster  Cluster determining the positions in the distributed
   *                     blockvector to which the local vector is added.
   * @param[in] local_part  Local part of block vector to be added. It is stored
                            in matrix format, where rows correspond to time and
                            columns to space.
   * @tparam space_type  @ref besthea::bem::fast_spacetime_be_space representing
   *                     either p0 or p1 basis functions. It determines the
   *                     DOFs.
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
   * @brief Returns a reference to @ref _my_blocks.
   */
  std::vector< lo > get_my_blocks( ) const {
    return _my_blocks;
  }

  /*!
   * Indicates if the executing process owns the given block.
   * @param[in] block_idx Block index.
   * @return True if the calling process owns the given block.
   */
  bool am_i_owner( lo block_idx ) const {
    return ( _owners[ block_idx ][ 0 ] == _rank );
  }

  /*!
   * Indicates if the executing process is the primary owner of the given block.
   * @param[in] block_idx Block index.
   * @return True if the calling process is the primary owner of the block.
   */
  bool am_i_primary_owner( lo block_idx ) const {
    return ( get_primary_owner( block_idx ) == _rank );
  }

  /*!
   * Realizes the communication of a block between two processes. Sender and
   * receiver both call this method. The sender is uniquely defined as the
   * primary owner of the communicated block.
   * @param[in] block_idx Index of the communicated block.
   * @param[in] rank MPI rank of receiver.
   * @param[out] data Vector to store the received data.
   */
  void communicate_block( lo block_idx, int rank, vector_type & data ) const;

  /*!
   * @brief Prints the vector.
   * @param[in] stream  Stream into which the vector is printed.
   */
  void print( std::ostream & stream = std::cout ) const;

  /*!
   * @brief Returns the value of @ref _comm.
   * @todo Can we return a reference instead?
   */
  MPI_Comm get_comm( ) const {
    return _comm;
  }

  /*!
   * @brief Returns the value of @ref _duplicated.
   */
  bool is_duplicated( ) const {
    return _duplicated;
  }

 protected:
  /*!
   * @brief Collects the information about the owners of all blocks. For this
   * purpose, the information in @p my_blocks is distributed among all MPI
   * processes.
   * @param[in] my_blocks Blocks owned by the executing process.
   * @note Each process updates its copy of @p _owners.
   */
  void communicate_owners( std::vector< lo > & my_blocks );

  /*!
   * Returns the rank of the primary owner of a block. The primary owner is the
   * one with the lowest rank.
   * @param[in] block_idx Block index.
   * @returns The rank of the primary owner of the block.
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

  lo _n_blocks;                      //!< number of blocks.
  lo _size;                          //!< size of each block.
  std::vector< vector_type > _data;  //!< raw data
  std::vector< std::vector< int > >
    _owners;  //!< Structure to identify the owners of the blocks of the vector.
              //!< @p _owners [i] is a vector containing those MPI ranks who own
              //!< block i. If the executing rank owns a block, the rank itself
              //!< is listed at the first position in the vector corresponding
              //!< to this block. All other ranks are sorted in ascending order.
              //!< The primary owner is the owner with the lowest rank.
  std::vector< lo > _my_blocks;  //!< List of blocks the rank owns.
  MPI_Comm _comm;    //!< MPI communicator associated with the block vector.
  int _rank;         //!< MPI rank of the executing process.
  bool _duplicated;  //!< Indicates if the vector is duplicated on all MPI
                     //!< processes.
};

#endif
