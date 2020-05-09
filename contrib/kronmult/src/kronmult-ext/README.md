Code to perform 6-dimensional batched kronecker product on  GPU and CPU (using OpenMP).

Y(:,k) += kron(A1(:,:,k), ..., A6(:,:,k) ) * X(:,k),   where k=1:batchCount

The code assumes  each matrix A1, ..., A6 are all square and same shape of n by n.

Note there can be overlap in the output vectors Y(:,k). Atomic add updates will be performed.

Each vector X(:,k) or Y(:,k) is conceptually  of shape n^6 by 1 but during the computation
can be reshaped as   n^5 by n or   n^4 by n^2 or n^3 by n^3 or n^2 by n^4 or n by n^5.

Note that  
Y = kron(A1,...,A6) * X
can be evaluated as
step 1: W = reshape( X, [n^5,n]) * transpose(A1)
step 2: Y = kron( A2, ..., A6) * W,   which can be viewed as "tail recursion"

At the lowest level of recursion
Y = kron(A1) * X  is simply   implemented as matrix multiply 
Y = A1 * X  

This implementation evaluates each kronecker product  in a separate thread block
on GPU. Instead of building a long batch list to call batched GEMM, the matrix-matrix
multiplication is evaluated as calls to device functions

kgemm_nn() to evaluate  C = alpha * A * B + beta * C
or
kgemm_nt() to evaluate  C = alpha * A * transpose(B) + beta * C

For the special case of (beta == 1), atomicAdd update is used.

Note that the GEMM operations will be performed on very slender rectangular matrices.
Therefore, the computations will not be dominated by floating point operations but
by data movement, especially when n is small.

-----------------

To compile the code for CPU
(1) mkdir build && cd build
(2) cmake ../
(3) make

To compile the code for Nvidia GPU
(1) mkdir build && cd build
(2) cmake ../ -DUSE_GPU=1
(3) make

To run the tester for kgemm_nn_batched, perform
./test_kgemm_nn_batched

To run the tester for kgemm_nt_batched, perform
./test_kgemm_nt_batched

To run the tester for kronmult6_batched, perform
./test_kronmult6_batched



