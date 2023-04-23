#include <iostream>
#include <set>

#include "build_info.hpp"

#include "asgard_kronmult.hpp"

namespace asgard::kronmult
{

template<typename T>
inline void omp_atomic_add(T *p, T inc_value)
{
#pragma omp atomic
  (*p) += inc_value;
}

template<typename T, int dimensions>
class tensor{
    tensor(int n_)
        : n5(n_*n_*n_*n_*n_), n4(n_*n_*n_*n_), n3(n_*n_*n_), n2(n_*n_), n(n_),
          serialized(n * ((dimensions == 5) ? n5 : ((dimensions == 4) ? n4 : ((dimensions == 3) ? n3 : ((dimensions == 2) ? n2 : ((dimenions == 1) ? n : 1))))))
    {
        static_assert(1 <= dimensions and dimensions <= 6);
    }
    T& operator() (int i, int j, int k, int l, int m, int p){
        return serialized[i*n5 + j*n4 + k*n3 + l*n2 + m*n + p];
    }
    T& operator() (int j, int k, int l, int m, int p){
        return serialized[j*n4 + k*n3 + l*n2 + m*n + p];
    }
    T& operator() (int k, int l, int m, int p){
        return serialized[k*n3 + l*n2 + m*n + p];
    }
    T& operator() (int l, int m, int p){
        return serialized[l*n2 + m*n + p];
    }
    T& operator() (int m, int p){
        return serialized[m*n + p];
    }
    T& operator() (int p){
        return serialized[p];
    }

private:
    int const n5, n4, n3, n2, n;
    std::vector<T> serialized;
};

template<typename T, int dimensions>
void run_cpu_variant(int n, T const *const pA[], int const lda,
                     T *pX[], T *pY[], int const num_batch){


}

#define asgard_kronmult_cpu_instantiate(d) \
    template void run_cpu_variant<float, (d)>(int n, float const *const pA[], int const lda, \
                                              float *pX[], float *pY[], int const num_batch); \
    template void run_cpu_variant<double, (d)>(int n, double const *const pA[], int const lda, \
                                               double *pX[], double *pY[], int const num_batch); \


asgard_kronmult_cpu_instantiate(1)
asgard_kronmult_cpu_instantiate(2)
asgard_kronmult_cpu_instantiate(3)
asgard_kronmult_cpu_instantiate(4)
asgard_kronmult_cpu_instantiate(5)
asgard_kronmult_cpu_instantiate(6)

}
