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
#include "mpi.h"
#include <string>
#include <vector>

/**
 * Applies the distributed FMM based on a scheduling of jobs via time clusters.
 * @param[in] communicator  MPI Communicator used for communication operations.
 * @param[in] receive_vector  Vector of pairs used to manage the receive 
 *                            operations.
 * @param[in] n_moments_upward  Number of moments received in the upward path.
 * @param[in] n_moments_m2l Number of moments received for M2L.
 * @param[in] m_list  List for scheduling upward path operations.
 * @param[in] m2l_list  List for scheduling interactions and downward pass
 *                      operations.
 * @param[in] l_list  List for scheduling downward path operations.
 * @param[in] n_list  List for scheduling nearfield operations.  
 * @param[in] input_vector  Vector containing the sources for FMM.
 * @param[in,out] output_vector Vector to store the results of the FMM.
 * @note The receive vector and the 4 lists should be constructed using 
 *       the method @ref besthea::mesh::tree_structure::prepare_fmm .
 * \todo write documentation
 */
void apply_fmm( const MPI_Comm communicator,
  const std::vector< std::pair< besthea::mesh::scheduling_time_cluster*, lo > >&
    receive_vector, const lou n_moments_upward, const lou n_moments_m2l,
  std::list< besthea::mesh::scheduling_time_cluster* > & m_list,
  std::list< besthea::mesh::scheduling_time_cluster* > & m2l_list,
  std::list< besthea::mesh::scheduling_time_cluster* > & l_list,
  std::list< besthea::mesh::scheduling_time_cluster* > & n_list,
  const std::vector< sc > & input_vector, std::vector< sc > & output_vector,
  bool verbose = false, std::string verbose_dir = "./verbose/" );

/**
 * Calls all S2M operations associated with a given scheduling time cluster.
 * @param[in] sources Global sources containing the once used for the M2L 
 *                    operation.
 * @param[in] time_cluster  Considered scheduling time cluster.
 * @todo Currently dummy routine. 
 */
void call_s2m_operations( const std::vector< sc > & sources,
  besthea::mesh::scheduling_time_cluster* time_cluster );

/**
 * Calls all M2M operations associated with a given scheduling time cluster.
 * @param[in] time_cluster  Considered scheduling time cluster.
 * @todo Currently dummy routine. 
 */
void call_m2m_operations( 
  besthea::mesh::scheduling_time_cluster* time_cluster );

/**
 * Calls all M2L operations associated with a given pair of scheduling time 
 * clusters.
 * @param[in] src_cluster Scheduling time cluster which acts as source in M2L.
 * @param[in] tar_cluster Scheduling time cluster which acts as target in M2L.
 * @todo Currently dummy routine. 
 */
void call_m2l_operations( besthea::mesh::scheduling_time_cluster* src_cluster,
  besthea::mesh::scheduling_time_cluster* tar_cluster );

/**
 * Calls all L2L operations associated with a given scheduling time cluster.
 * @param[in] time_cluster  Considered scheduling time cluster.
 * @todo Currently dummy routine. 
 */
void call_l2l_operations( 
  besthea::mesh::scheduling_time_cluster* time_cluster );

/**
 * Calls all L2T operations associated with a given scheduling time cluster.
 * @param[in] time_cluster  Considered scheduling time cluster.
 * @param[in,out] output_vector Vector to which the results are added.
 * @todo Currently dummy routine. 
 */
void call_l2t_operations( besthea::mesh::scheduling_time_cluster* time_cluster, 
  std::vector< sc > & output_vector );

/**
 * Calls all nearfield operations associated with a given pair of scheduling 
 * time clusters.
 * @param[in] sources Global sources containing the once used for the nearfield 
 *                    operation.
 * @param[in] src_cluster Scheduling time cluster which acts as source for the
 *                        nearfield operations.
 * @param[in] tar_cluster Scheduling time cluster which acts as target for the
 *                        nearfield operations.
 * @param[in,out] output_vector Vector to which the results are added.
 * @todo Currently dummy routine. 
 */
void call_nearfield_operations( const std::vector< sc > & sources,
  besthea::mesh::scheduling_time_cluster* src_cluster, 
  besthea::mesh::scheduling_time_cluster* tar_cluster, 
  std::vector< sc > & output_vector );

/**
 * Calls MPI_Testsome for an array of Requests to check for received data.
 * @param[in] communicator  MPI communicator used for MPI_Testsome.
 * @param[in] receive_vector  Vector of pairs used to manage the receive 
 *                            operations. Constructed by 
 *                            @ref besthea::mesh::tree_structure::prepare_fmm.
 * @param[in] n_moments_upward  Number of moments received in the upward path.
 * @param[in] n_moments_m2l Number of moments received for M2L.
 * @param[in,out] array_of_requests Array containing the MPI requests which are 
 *                                  checked.
 * @param[in,out] array_of_indices  Array in which the indices of the completed
 *                                  requests are stored. This is used as an 
 *                                  input variable to avoid reallocation in each
 *                                  function call.
 * @param[in,out] outcount  Stores the number of Requests which are completed.
 */
void check_for_received_data( const MPI_Comm communicator,
  const std::vector< std::pair< besthea::mesh::scheduling_time_cluster*, lo > > 
    & receive_vector, const lou n_moments_upward, const lou n_moments_m2l, 
  MPI_Request * array_of_requests, int array_of_indices[ ], int & outcount, 
  bool verbose, std::string verbose_dir );

/**
 * Returns an iterator pointing to the next cluster in the l-list whose 
 * dependencies are satisfied. In case a cluster is found the status is updated.
 * If no cluster is found the iterator points to the end of the list and the
 * status is not modified.
 * @param[in] l_list  The appropriate l-list constructed using the routine
 * @ref besthea::mesh::tree_structure::prepare_fmm.
 * @param[out] it_next_cluster  If a cluster is found in the list this iterator 
 *                              points to it. Else it points to the end of the
 *                              list. 
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
 * @ref besthea::mesh::tree_structure::prepare_fmm.
 * @param[out] it_next_cluster  If a cluster is found in the list this iterator 
 *                              points to it. Else it points to the end of the
 *                              list. 
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
 * @ref besthea::mesh::tree_structure::prepare_fmm.
 * @param[out] it_next_cluster  If a cluster is found in the list this iterator 
 *                              points to it. Else it points to the end of the
 *                              list. 
 * @param[out] status Set to 3 if a cluster is found.
 */
void find_cluster_in_m2l_list( 
  std::list< besthea::mesh::scheduling_time_cluster* > & m2l_list,
  std::list< besthea::mesh::scheduling_time_cluster* >::iterator & 
    it_next_cluster, char & status );

/**
 * Updates dependency flags or sends moments for M2L operations.
 * @param[in] communicator  Communicator used for sending.
 * @param[in] src_cluster Considered scheduling time cluster. If a cluster in 
 *                        its send list is handled by a different process, the
 *                        moments are send to this process.
 * @todo Currently dummy routine. Sended data needs to be exchanged.
 */
void provide_moments_for_m2l( const MPI_Comm communicator, 
  besthea::mesh::scheduling_time_cluster* src_cluster );

/**
 * Updates dependency flags or sends moments for upward path.
 * @param[in] communicator  Communicator used for sending.
 * @param[in] child_cluster Considered scheduling time cluster. If its parent
 *                          is handled by a different process, the processed
 *                          moments are send from the local copy of the parent
 *                          cluster to this process. 
 * @todo Currently dummy routine. Sended data needs to be exchanged.
 */
void provide_moments_to_parents( const MPI_Comm communicator, 
  besthea::mesh::scheduling_time_cluster* child_cluster );

/**
 * Sends local contributions for downward path if necessary.
 * @param[in] communicator  Communicator used for sending.
 * @param[in] parent_cluster Considered scheduling time cluster. If a child
 *                           of it is handled by a different process, the local
 *                           contributions are send to this process.
 * @todo Currently dummy routine. Sended data needs to be exchanged.
 */
void provide_local_contributions_to_children( const MPI_Comm communicator, 
  besthea::mesh::scheduling_time_cluster* parent_cluster );

/**
 * Starts all receive operations given by a vector of pairs of clusters and 
 * process ids.
 * @param[in] receive_vector  List of pairs of clusters and process ids
 * @param[in] n_moments_upward  This is the number of entries in the receive
 *                              vector, which corresponds to receive operations
 *                              in the upward path. These entries come first in
 *                              the vector.
 * @param[in] n_moments_m2l  This is the number of entries in the receive 
 *                           vector, which corresponds to receive operations
 *                           for M2L. These entries come second in the vector,
 *                           and are followed by the entries for the receive 
 *                           operations in the downward path.
 * @param[in,out] array_of_requests The MPI_Requests of the non-blocking receive
 *                                  operations are stored in this array. It is 
 *                                  expected to have at least the size of 
 *                                  @p receive_vector .
 * @todo Currently the receives are dummy routines. Received data has to be 
 *       exchanged.
 */
void start_receive_operations( 
  const std::vector< std::pair< besthea::mesh::scheduling_time_cluster*, lo > > 
  & receive_vector, const lou n_moments_upward, const lou n_moments_m2l,
  MPI_Request array_of_requests[] );



#endif /* INCLUDE_PARALLEL_FMM_H_ */