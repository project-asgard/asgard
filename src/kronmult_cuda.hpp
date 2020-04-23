#pragma once

template<typename P>
void call_kronmult(int const n, P const operators[], P *x_ptrs[],
                   P *output_ptrs[], P *work_ptrs[], int const num_krons,
                   int const num_dims);
