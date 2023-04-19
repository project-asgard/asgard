
#include "./device/asgard_kronmult.hpp"

#include <iostream>
#include <random>

#include "tests_general.hpp"

using namespace asgard::kronmult;

template<typename T>
struct kronmult_intputs{
    int num_batch;
    std::vector<int> pointer_map;

    std::vector<T> matrices;
    std::vector<T> input_x;
    std::vector<T> output_y;
    std::vector<T> reference_y;

    std::vector<T*> pA;
    std::vector<T*> pX;
    std::vector<T*> pY;
};

template<typename T>
std::unique_ptr<kronmult_intputs<T>>
make_kronmult_data(int dimensions, int n, int num_batch, int num_matrices, int num_y){
    std::minstd_rand park_miller(42);
    std::uniform_real_distribution<T> unif(-1.0, 1.0);
    std::uniform_real_distribution<T> uniy(0, num_y-1);
    std::uniform_real_distribution<T> unim(0, num_matrices-1);

    int num_data = 1;
    for(int i=0; i<dimensions; i++) num_data *= n;

    auto result = std::unique_ptr<kronmult_intputs<T>>
        (new kronmult_intputs<T>{num_batch, std::vector<int>( (dimensions+2) * num_batch ),
         std::vector<T>(n*n*num_matrices), std::vector<T>(num_data*num_y),
         std::vector<T>(num_data*num_y), std::vector<T>(num_data*num_y),
         std::vector<T*>(dimensions * num_batch),
         std::vector<T*>(num_batch), std::vector<T*>(num_batch)});

    // pointer_map has 2D structure with num_batch strips of size (d+2)
    // the first entry of each strip is the input x
    // the next d entries are the matrices
    // the final entry is the output y
    auto ip = result->pointer_map.begin();
    for(int i=0; i<num_batch; i++){
        *ip++ = uniy(park_miller);
        for(int j=0; j<dimensions; j++) *ip++ = unim(park_miller);
        *ip++ = uniy(park_miller);
    }

    for(auto &m : result->matrices) m = unif(park_miller);
    for(auto &m : result->input_x) m = unif(park_miller);
    for(auto &m : result->output_y) m = unif(park_miller);

    result->reference_y = result->output_y;

    ip = result->pointer_map.begin();
    for(int i=0; i<num_batch; i++){
        result->pX[i] = &( result->input_x[ *ip++ * num_data ] );
        result->pA[i] = &( result->matrices[ *ip++ * n * n ] );
        result->pY[i] = &( result->reference_y[ *ip++ * num_data ] );
    }

    reference_kronmult(dimensions, n, result->pA.data(), result->pX.data(), result->pY.data(), result->num_batch);

    ip = result->pointer_map.begin();
    for(int i=0; i<num_batch; i++){
        ip += dimensions + 1;
        result->pY[i] = &( result->output_y[ *ip++ * num_data ] );
    }

    return result;
}

template<typename T>
bool test_kronmult(int dimensions, int n, int num_batch, int num_matrices, int num_y){

    auto data = make_kronmult_data<T>(dimensions, n, num_batch, num_matrices, num_y);

    if (n == 2){
        cpu2d<T, 2>(data->pA.data(), n, data->pX.data(), data->pY.data(), num_batch);
    }

    T err = 0.0;
    for(size_t i=0; i<data->output_y.size(); i++){
        err = std::max( err, std::abs( data->output_y[i] - data->reference_y[i] ) );
    }
    std::cerr << " error = " << err << "\n";

// template<typename T, int n>
// void cpu2d(T const *const Aarray_[], int const lda, T *pX_[], T *pY_[],
//            int const num_batch)


    return true;
}

/*

 10  30  30   90
 20  40  60  120
 20  60  40  120
 40  80  80  160

*/

TEMPLATE_TEST_CASE("testing basic units",
                   "[kronecker]", float, double)
{
    std::vector<TestType> A = {1, 2, 3, 4};
    std::vector<TestType> B = {10, 20, 30, 40};
    auto R = kronecker(2, A.data(), 2, B.data());
    std::vector<TestType> gold = {10, 20, 20, 40, 30, 40, 60, 80, 30, 60, 40, 80, 90, 120, 120, 160};

    rmse_comparison<TestType>(asgard::fk::vector<TestType>(R), asgard::fk::vector<TestType>(gold), 1.E-8);
}

// TEMPLATE_TEST_CASE("testing kronmult 1D",
//                    "[kronmult1D]", float, double)
// {
//     REQUIRE( test_kronmult<TestType>(2, 2, 100, 5, 20) );
// }
