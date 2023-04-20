
#include "tests_general.hpp"

#include "./device/asgard_kronmult.hpp"
#include "tensors.hpp"

#include <iostream>
#include <random>

using namespace asgard::kronmult;

template<typename T>
void test_almost_equal(std::vector<T> const &x, std::vector<T> const &y, int scale = 10){
    rmse_comparison<T>(asgard::fk::vector<T>(x), asgard::fk::vector<T>(y), get_tolerance<T>(scale));
}

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
         std::vector<T*>(num_batch),
         std::vector<T*>(num_batch)});

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
        for(int j=0; j<dimensions; j++)
            result->pA[i*dimensions + j] = &( result->matrices[ *ip++ * n * n ] );
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
void test_kronmult_cpu(int dimensions, int n, int num_batch, int num_matrices, int num_y){

    auto data = make_kronmult_data<T>(dimensions, n, num_batch, num_matrices, num_y);

    execute_cpu(dimensions, n, data->pA.data(), n, data->pX.data(), data->pY.data(), num_batch);

    test_almost_equal(data->output_y, data->reference_y, 100);
}

TEMPLATE_TEST_CASE("testing reference methods",
                   "[kronecker]", float, double)
{
    std::vector<TestType> A = {1, 2, 3, 4};
    std::vector<TestType> B = {10, 20, 30, 40};
    auto R = kronecker(2, A.data(), 2, B.data());
    std::vector<TestType> gold = {10, 20, 20, 40, 30, 40, 60, 80, 30, 60, 40, 80, 90, 120, 120, 160};
    test_almost_equal(R, gold);

    B = std::vector<TestType>{1, 2, 3, 4, 5, 6, 7, 8, 9};
    R = kronecker(2, A.data(), 3, B.data());
    gold = std::vector<TestType>{1, 2, 3, 2, 4, 6, 4, 5, 6, 8, 10, 12, 7, 8, 9, 14, 16, 18, 3, 6, 9, 4, 8, 12, 12, 15, 18, 16, 20, 24, 21, 24, 27, 28, 32, 36};
    test_almost_equal(R, gold);

    std::vector<TestType> x = {10, 20};
    std::vector<TestType> y = {1, 0};
    reference_gemv(2, A.data(), x.data(), y.data());
    gold = std::vector<TestType>{71, 100};
    test_almost_equal(y, gold);
}

TEMPLATE_TEST_CASE("testing kronmult exceptions",
                   "[kronmult]", float, double)
{
    REQUIRE_THROWS_WITH(
        test_kronmult_cpu<TestType>(1, 5, 1, 1, 1),
        "kronmult unimplemented n for the cpu");
    REQUIRE_THROWS_WITH(
        test_kronmult_cpu<TestType>(1, 5, 1, 1, 1),
        "kronmult unimplemented n for the cpu");
    REQUIRE_THROWS_WITH(
        test_kronmult_cpu<TestType>(10, 2, 1, 1, 1),
        "kronmult unimplemented number of dimensions for the cpu");
}

TEMPLATE_TEST_CASE("testing kronmult cpu",
                   "[execute_cpu]", float, double)
{
    test_kronmult_cpu<TestType>(1, 2, 1, 1, 1);
    test_kronmult_cpu<TestType>(1, 2, 10, 1, 1);
    test_kronmult_cpu<TestType>(1, 2, 10, 5, 1);
    test_kronmult_cpu<TestType>(1, 2, 10, 1, 5);
    test_kronmult_cpu<TestType>(1, 2, 100, 10, 8);

    test_kronmult_cpu<TestType>(1, 2, 70, 9, 7);
    test_kronmult_cpu<TestType>(1, 3, 70, 9, 7);
    test_kronmult_cpu<TestType>(1, 4, 70, 9, 7);

    test_kronmult_cpu<TestType>(2, 2, 70, 9, 7);
    test_kronmult_cpu<TestType>(2, 3, 70, 9, 7);
}

#ifdef ASGARD_USE_CUDA

template<typename T>
void test_kronmult_gpu(int dimensions, int n, int num_batch, int num_matrices, int num_y){

    auto data = make_kronmult_data<T>(dimensions, n, num_batch, num_matrices, num_y);

    auto gpuy = asgard::fk::vector<T>(data->output_y).clone_onto_device();
    auto gpux = asgard::fk::vector<T>(data->input_x).clone_onto_device();
    auto gpu_mats = asgard::fk::vector<T>(data->matrices).clone_onto_device();

    int num_data = 1;
    for(int i=0; i<dimensions; i++) num_data *= n;

    auto ip = data->pointer_map.begin();
    for(int i=0; i<num_batch; i++){
        data->pX[i] = gpux.begin() + *ip++ * num_data;
        for(int j=0; j<dimensions; j++)
            data->pA[i*dimensions + j] = gpu_mats.begin() + *ip++ * n * n;
        data->pY[i] = gpuy.begin() + *ip++ * num_data;
    }

    auto pX = asgard::fk::vector<T*>(data->pX).clone_onto_device();
    auto pY = asgard::fk::vector<T*>(data->pY).clone_onto_device();
    auto pA = asgard::fk::vector<T*>(data->pA).clone_onto_device();

    execute_gpu(dimensions, n, pA.data(), n, pX.data(), pY.data(), num_batch);

    rmse_comparison<T>(gpuy.clone_onto_host(), asgard::fk::vector<T>(data->reference_y), get_tolerance<T>(100));
}

TEMPLATE_TEST_CASE("testing kronmult gpu",
                   "[execute_gpu]", float, double)
{
    test_kronmult_gpu<TestType>(1, 2, 1, 1, 1);
    test_kronmult_gpu<TestType>(1, 2, 10, 1, 1);
    test_kronmult_gpu<TestType>(1, 2, 10, 5, 1);
    test_kronmult_gpu<TestType>(1, 2, 10, 1, 5);
    test_kronmult_gpu<TestType>(1, 2, 100, 10, 8);

    test_kronmult_gpu<TestType>(1, 2, 70, 9, 7);
    test_kronmult_gpu<TestType>(1, 3, 70, 9, 7);
    test_kronmult_gpu<TestType>(1, 4, 70, 9, 7);

    test_kronmult_gpu<TestType>(2, 2, 170, 9, 7);
    test_kronmult_gpu<TestType>(2, 3, 170, 9, 7);
    test_kronmult_gpu<TestType>(2, 4, 170, 9, 7);

    test_kronmult_gpu<TestType>(3, 2, 170, 9, 7);
    test_kronmult_gpu<TestType>(3, 3, 170, 9, 7);
    test_kronmult_gpu<TestType>(3, 4, 170, 9, 7);

    test_kronmult_gpu<TestType>(4, 2, 140, 5, 3);
}

#endif
