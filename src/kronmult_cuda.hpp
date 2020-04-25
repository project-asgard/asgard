#pragma once

template<typename P>
void call_kronmult(int const n, P *x_ptrs[], P *output_ptrs[], P *work_ptrs[],
                   P const *const operator_ptrs[], int const lda,
                   int const num_krons, int const num_dims);
