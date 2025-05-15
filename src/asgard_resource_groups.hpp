#pragma once

#include "asgard_compute.hpp"

#ifdef ASGARD_USE_MPI
#include "mpi.h"
#endif

namespace asgard
{
//! shortcuts for mpi commands
namespace mpi
{
#ifdef ASGARD_USE_MPI
//! returns the rank in the current comm
inline int comm_rank(MPI_Comm const comm) {
    int me;
    MPI_Comm_rank(comm, &me);
    return me;
}
//! (debug) returns the rank the world rank
inline int world_rank() { return comm_rank(MPI_COMM_WORLD); }
//! (debug) returns true if the world rank matches
inline bool is_world_rank(int rank) { return (world_rank() == rank); }

//! returns the size of the comm
inline int comm_size(MPI_Comm const comm){
  int nprocs;
  MPI_Comm_size(comm, &nprocs);
  return nprocs;
}
//! return the size of the world comm
inline int world_size() { return comm_size(MPI_COMM_WORLD); }

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

#else
inline constexpr bool is_world_rank(int) { return true; }
inline constexpr int world_rank() { return 0; }
inline constexpr int world_size() { return 1; }
#endif

} // namespace mpi

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
  bool is_leader() const { return (rank_ == root); }
  //! check if the resource is owned by this set, checks the group/rank
  bool owns(resource const &rec) const { return (rank_ == rec.group); }

#ifdef ASGARD_USE_MPI
  //! broadcasts the data to all sets in the communicator, can send or receive
  template<typename T>
  void bcast(int count, T *data) const {
    MPI_Bcast(data, count, mpi::datatype<T>(), root, comm);
  }
  //! broadcasts the data to all sets in the communicator, sender-only
  template<typename T>
  void bcast(int count, T const *data) const {
    expect(rank_ == root);
    MPI_Bcast(const_cast<T*>(data), count, mpi::datatype<T>(), root, comm);
  }
  //! broadcasts the data to all sets in the communicator, can send or receive
  template<typename T>
  void bcast(std::vector<T> &data) {
    bcast(static_cast<int>(data.size()), data.data());
  }
  //! broadcasts the data to all sets in the communicator, sender-only
  template<typename T>
  void bcast(std::vector<T> const &data) {
    bcast(static_cast<int>(data.size()), data.data());
  }
  //! adds the data across communicator
  template<typename T>
  void reduce_add(int count, T const *input, T *output = nullptr) {
    expect(not (rank_ == root and output == nullptr));
    MPI_Reduce(input, output, count, mpi::datatype<T>(), MPI_SUM, root, comm);
  }
  //! adds the data across communicator
  template<typename T>
  void reduce_add(int count, std::vector<T> const &input, std::vector<T> &output) {
    if (rank_ == root)
      output.resize(input.size());
    reduce_add(count, input.data(), output.data());
  }
#endif

private:
  // expressive way to address the mpi-comm root
  static int constexpr root = 0;

  int rank_ = 0;
  int num_gpus = 0;
  #ifdef ASGARD_USE_MPI
  MPI_Comm comm;
  #endif
};

#ifdef ASGARD_USE_MPI
/*!
 * \brief Optional call to library initialization
 *
 * The ASGarD library itself does not require initialization; however, ASGarD may be using
 * components that require initialization, e.g., MPI.
 * The final executable can initialize MPI in one of two ways:
 * \code
 *   #include "asgard.hpp"
 *
 *   int main(int argc, char **argv) {
 *     asgard::libasgard_init(argc, argv);
 *     ...
 *     asgard::libasgard_finish();
 * \endcode
 * The methods will call the proper initialization methods, if MPI has been enabled,
 * otherwise the methods will do nothing.
 *
 * Alternatively, MPI_Init can be called directly
 * \code
 *   #include "asgard.hpp"
 *
 *   int main(int argc, char **argv) {
 *     #ifdef ASGARD_USE_MPI
 *     MPI_Init(&argc, &argv);
 *     #endif
 *     ...
 *     #ifdef ASGARD_USE_MPI
 *     MPI_Finalize();
 *     #endif
 * \endcode
 *
 * Naturally, the #ifdef directives can be omitted or the initialization can be skipped
 * altogether, if the given PDE definition always or never uses MPI.
 */
inline void libasgard_init(int &argc, char **&argv) {
  MPI_Init(&argc, &argv);
}
//! finalization of the library, see asgard::libasgard_init
inline void libasgard_finish() {
  MPI_Finalize();
}
//! RAII style call to init/finish
struct libasgard_runtime {
  //! calls libasgard_init
  libasgard_runtime(int &argc, char **&argv) {
    libasgard_init(argc, argv);
  }
  //! calls libasgard_finish
  ~libasgard_runtime() { libasgard_finish(); }
};
#else
inline void libasgard_init(int &, char **&) }{}
inline void libasgard_finish() {}
struct libasgard_runtime {
  //! does nothing
  libasgard_runtime(int &, char **&) {}
};
#endif

}
