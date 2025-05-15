#pragma once

#include "asgard_compute.hpp"

#ifdef ASGARD_USE_MPI
#include "mpi.h"
#endif

namespace asgard
{

#ifdef ASGARD_USE_MPI
//! shortcuts for mpi commands
namespace mpi
{
//! returns the rank in the current comm
inline int comm_rank(MPI_Comm const comm){
    int me;
    MPI_Comm_rank(comm, &me);
    return me;
}
//! (debug) returns the rank the world rank
inline int world_rank(){ return comm_rank(MPI_COMM_WORLD); }
//! (debug) returns true if the world rank matches
inline bool is_world_rank(int rank) { return (world_rank() == rank); }

//! returns the size of the comm
inline int comm_size(MPI_Comm const comm){
  int nprocs;
  MPI_Comm_size(comm, &nprocs);
  return nprocs;
}

//! given a C++ type T, return the corresponding MPI data type
template<typename T>
inline constexpr MPI_Datatype datatype() {
  if constexpr (is_double<T>)
    return MPI_DOUBLE;
  else if constexpr (is_float<T>)
    return MPI_FLOAT;
  else if constexpr (std::is_same_v<int, T>)
    return MPI_INT;
  else
    static_assert(is_double<T>, "unknown MPI data-type");
}

} // namespace mpi
#endif

/*!
 * \brief Indicates a compute resource
 *
 * The recourse has two components, the group, i.e., MPI rank,
 * and device, i.e., CPU/GPU device ID.
 */
struct resource
{
  //! resource group and MPI rank
  int group = 0;
  //! CPU/GPU device, defaults is CPU
  int device = -1;
};

/*!
 * \brief Manages a set of compute resources
 *
 * Each discretization manager handles one of these resource sets
 */
class resource_set {
public:
  //! sets the default resource set
  resource_set() : num_gpus(compute->num_gpus()) {}

#ifdef ASGARD_USE_MPI
  //! sets the resource set as a member of this communicator
  resource_set(MPI_Comm cm) : rank_(mpi::comm_rank(cm)), comm(cm)
  {}
  //! returns the mpi rank
  MPI_Comm mpicomm() const { return comm; }
#endif

  //! returns the mpi rank
  int rank() const { return rank_; }
  //! rank 0 is the leader for the mpi communicator
  bool is_leader() const { return (rank_ == 0); }
  //! check if the resource is owned by this set, checks the group/rank
  bool owns(resource const &rec) const { return (rank_ == rec.group); }
  //! broadcasts the data to all sets in the communicator
  template<typename T>
  void bcast(int count, T *data) {
    MPI_Bcast(data, count, mpi::datatype<T>(), 0, comm);
  }
  //! broadcasts the data to all sets in the communicator
  template<typename T>
  void bcast(std::vector<T> &data) {
    bcast(static_cast<int>(data.size()), data.data());
  }
  //! adds the data across communicator
  template<typename T>
  void reduce_add(int count, T *data) {
    MPI_Bcast(data, count, mpi::datatype<T>(), 0, comm);
  }

private:
  int rank_ = 0;
  int num_gpus = 0;
  #ifdef ASGARD_USE_MPI
  MPI_Comm comm;
  #endif
};

}
