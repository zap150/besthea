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

/** @file time_cluster_tree.h
 * @brief Tree of temporal cluster.
 */

#ifndef INCLUDE_BESTHEA_TIME_CLUSTER_TREE_H_
#define INCLUDE_BESTHEA_TIME_CLUSTER_TREE_H_

#include "besthea/settings.h"
#include "besthea/temporal_mesh.h"
#include "besthea/time_cluster.h"

#include <iostream>

namespace besthea {
  namespace mesh {
    class time_cluster_tree;
  }
}

/**
 * Class representing (not necessarily binary) tree of temporal clusters.
 */
class besthea::mesh::time_cluster_tree {
 public:
  /**
   * Constructor
   * @param[in] mesh Reference to the underlying mesh.
   * @param[in] levels Maximum number of levels in the tree.
   */
  time_cluster_tree( const temporal_mesh & mesh, lo levels, lo n_min_elems );

  /**
   * Destructor.
   */
  virtual ~time_cluster_tree( ) {
    delete _root;
  }

  /**
   * Returns number of levels in the tree.
   */
  lo get_levels( ) const {
    return _levels;
  }

  /**
   * Returns the root of the tree.
   */
  time_cluster * get_root( ) {
    return _root;
  }

  sc compute_padding( time_cluster & root );

 private:
  time_cluster * _root;         //!< root cluster of the tree
  const temporal_mesh & _mesh;  //!< underlying mesh
  lo _levels;                   //!< number of levels in the tree
  lo _real_max_levels;  //!< auxiliary value to determine number of real tree
                        //!< levels (depending on _n_min_elems)
  lo _n_min_elems;  //!< minimum number of elements so that cluster can be split
                    //!< in halves
  std::vector< sc > _paddings;  //!< vector of paddings on each level

  /**
   * Builds tree recursively
   * @param[in] root Node to stem from.
   * @param[in] level Current level.
   */
  void build_tree( time_cluster & root, lo level );
};

#endif /* INCLUDE_BESTHEA_TIME_CLUSTER_TREE_H_ */
