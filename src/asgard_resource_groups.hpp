#pragma once

#include "asgard_compute.hpp"

namespace asgard
{
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
  resource_set() : num_gpus_(compute->num_gpus()) {}

#ifdef ASGARD_USE_MPI
  //! sets the resource set as a member of this communicator
  resource_set(MPI_Comm cm)
      : num_gpus_(compute->num_gpus()), rank_(mpi::comm_rank(cm)),
        num_ranks_(mpi::comm_size(cm)), comm(cm)
  {}
  //! returns the mpi rank
  MPI_Comm mpicomm() const { return comm; }
  //! returns the number of mpi-ranks
  int num_ranks() const { return num_ranks_; }
  //! returns the mpi rank
  int rank() const { return rank_; }
#else
  static constexpr int num_ranks() { return 1; }
  static constexpr int rank() { return 0; }
#endif

  //! rank 0 is the leader for the mpi communicator
  bool is_leader() const { return (rank_ == root); }
  //! check if the resource is owned by this set, checks the group/rank
  bool owns(resource const &rec) const { return (rank_ == rec.group); }

#ifdef ASGARD_USE_MPI
  //! broadcasts the data to all sets in the communicator, can send or receive
  template<typename T>
  void bcast(int count, T *data) const {
    if (num_ranks_ >= 4) {
      MPI_Bcast(data, count, mpi::datatype<T>(), root, comm);
    } else {
      MPI_Recv(data, count, mpi::datatype<T>(), root, bcast_tag, comm, MPI_STATUS_IGNORE);
    }
  }
  //! broadcasts the data to all sets in the communicator, sender-only
  template<typename T>
  void bcast(int count, T const *data) const {
    expect(rank_ == root);
    if (num_ranks_ >= mpi::bcast_threshold) {
      MPI_Bcast(const_cast<T*>(data), count, mpi::datatype<T>(), root, comm);
    } else {
      for (int r = 1; r < num_ranks_; r++)
        MPI_Send(data, count, mpi::datatype<T>(), r, bcast_tag, comm);
    }
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
    if (num_ranks_ >= mpi::reduce_threshold) {
      MPI_Reduce(input, output, count, mpi::datatype<T>(), MPI_SUM, root, comm);
    } else {
      if (is_leader()) {
        size_t const stride = static_cast<size_t>(count) * sizeof(T);
        work.resize((num_ranks_ - 1) * stride);
        if (num_ranks_ == 2) {
          MPI_Recv(work.data(), count, mpi::datatype<T>(), 1, reduce_tag, comm, MPI_STATUS_IGNORE);
          T const *data = reinterpret_cast<T const *>(work.data());
          for (int i = 0; i < count; i++)
            output[i] = data[i] + input[i];
        } else {
          // overlap addition and communication
          std::array<MPI_Request, 3> requests;
          for (int r = 0; r < num_ranks_ - 1; r++)
            MPI_Irecv(work.data() + r * stride, count, mpi::datatype<T>(), r + 1, reduce_tag, comm, requests.data() + r);
          std::copy_n(input, count, output);
          for (int r = 0; r < num_ranks_ - 1; r++) {
            int gotten = 0;
            MPI_Waitany(num_ranks_ - 1, requests.data(), &gotten, MPI_STATUS_IGNORE);
            T const *data = reinterpret_cast<T const *>(work.data() + gotten * stride);
            for (int i = 0; i < count; i++)
              output[i] += data[i];
          }
        }
      } else {
        MPI_Send(input, count, mpi::datatype<T>(), root, reduce_tag, comm);
      }
    }
  }
  //! adds the data across communicator
  template<typename T>
  void reduce_add(std::vector<T> const &input, std::vector<T> &output) {
    if (rank_ == root)
      output.resize(input.size());
    reduce_add(static_cast<int>(input.size()), input.data(), output.data());
  }
#endif

  //! returns the number of GPU devices
  int num_gpus() const { return num_gpus_; }

private:
  // local resources, e.g., GPU devices
  int num_gpus_ = 0;

  // expressive way to address the mpi-comm root
  static int constexpr root = 0;

  static int constexpr bcast_tag = 11;
  static int constexpr reduce_tag = 12;

  // external resources, e.g., MPI rank and communicator
  int rank_ = 0;
  int num_ranks_ = 1;
  #ifdef ASGARD_USE_MPI
  MPI_Comm comm;
  std::vector<std::byte> work;
  #endif
};

// Things todo:
// 1. sync the sparse grid, added to the discretization manager in 2-steps (number of idx and the idx)
// 2. add worker mode for the iterative solvers
//    - distribute-add the direct solver matrices, but solve only on 0
// 3. distribute the moments, detect who needs moments
//    - only rank 0 does the Poisson solver, others have to wait
// 4. find a way to disable idle mpi ranks (reduce the comm)

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
inline void libasgard_init(int &, char **&) {}
inline void libasgard_finish() {}
struct libasgard_runtime {
  //! does nothing
  libasgard_runtime(int &, char **&) {}
};
#endif

}
