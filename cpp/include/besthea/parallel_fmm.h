/*
* Copyright 2020, VSB - Technical University of Ostrava and Graz University of
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

/** @file parallel_fmm.h
 * @brief Provides functions for execution of parallel fmm.
 * @todo Discuss where these routines should be (parallel pFMM_matrix class?)
 */

#ifndef INCLUDE_PARALLEL_FMM_H_
#define INCLUDE_PARALLEL_FMM_H_

#include "besthea/settings.h"
#include "besthea/scheduling_time_cluster.h"

#include <list>
#include <vector>

/**
 * WIP
 * \todo write documentation
 */
void apply_fmm( const lo my_process_id,
  const std::vector< std::pair< besthea::mesh::scheduling_time_cluster*, lo > >&
    receive_vector, const lou n_moments_to_receive,
  std::list< besthea::mesh::scheduling_time_cluster* > & m_list,
  std::list< besthea::mesh::scheduling_time_cluster* > & m2l_list,
  std::list< besthea::mesh::scheduling_time_cluster* > & l_list,
  std::list< besthea::mesh::scheduling_time_cluster* > & n_list );

/**
 * Returns an iterator pointing to the next cluster in the l-list whose 
 * dependencies are satisfied. In case a cluster is found the status is updated.
 * If no cluster is found the iterator points to the end of the list and the
 * status is not modified.
 * @param[in] l_list  The appropriate l-list constructed using the routine
 * @ref prepare_fmm .
 * @param[out] status Set to 2 if a cluster is found.
 */
  void find_cluster_in_l_list( 
  std::list< besthea::mesh::scheduling_time_cluster* > & l_list,
  std::list< besthea::mesh::scheduling_time_cluster* >::iterator & 
    it_next_cluster, char & status );

/**
 * Returns an iterator pointing to the next cluster in the m-list whose 
 * dependencies are satisfied. In case a cluster is found the status is updated.
 * If no cluster is found the iterator points to the end of the list and the
 * status is not modified.
 * @param[in] m_list  The appropriate m-list constructed using the routine
 * @ref prepare_fmm .
 * @param[out] status Set to 1 if a cluster is found.
 */
  void find_cluster_in_m_list( 
  std::list< besthea::mesh::scheduling_time_cluster* > & m_list,
  std::list< besthea::mesh::scheduling_time_cluster* >::iterator & 
    it_next_cluster, char & status );

/**
 * Returns an iterator pointing to the next cluster in the m2l-list whose 
 * dependencies are satisfied. In case a cluster is found the status is updated.
 * If no cluster is found the iterator points to the end of the list and the
 * status is not modified.
 * @param[in] m2l_list  The appropriate m2l-list constructed using the routine
 * @ref prepare_fmm .
 * @param[out] status Set to 3 if a cluster is found.
 */
  void find_cluster_in_m2l_list( 
  std::list< besthea::mesh::scheduling_time_cluster* > & m2l_list,
  std::list< besthea::mesh::scheduling_time_cluster* >::iterator & 
    it_next_cluster, char & status );

/**
 * Starts all receive operations given by a vector pairs of clusters and process 
 * ids
 * @param[in] receive_vector  List of pairs of clusters and process ids
 * @param[in] n_moments_to_receive  The first @p n_moments_to_receive pairs in
 *                                  the vector correspond to moments which are
 *                                  received, the rest to local contributions.
 * @todo  Change this, when MPI is included!
 */
void start_receive_operations( 
  const std::vector< std::pair< besthea::mesh::scheduling_time_cluster*, lo > > 
  & receive_vector, const lou n_moments_to_receive );



#endif /* INCLUDE_PARALLEL_FMM_H_ */