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

template<typename T>
void run_cpu_variant0(int const dimensions, T const *const pA[],
                      T const * const pX[], T *pY[], int const num_batch){

    #pragma omp parallel for
    for (int i = 0; i < num_batch; i++)
    {
        T totalA = 1;
        for(int j=0; j<dimensions; j++){
            totalA *= pA[dimensions*i + j][0];
        }
        #pragma omp atomic
        pY[i][0] += totalA * pX[i][0];
    }
}

template void run_cpu_variant0<float>(int const, float const *const[], float const * const[], float *[], int const);
template void run_cpu_variant0<double>(int const, double const *const[], double const * const[], double *[], int const);


template<typename T, int dimensions, int n>
void run_cpu_variant(T const *const pA[], int const lda,
                     T const * const pX[], T *pY[], int const num_batch){

    // always use one thread per kron-product
    #pragma omp parallel for
    for (int i = 0; i < num_batch; i++)
    {
        if constexpr(dimensions == 1){
            if constexpr(n == 2){
                #pragma omp atomic
                pY[i][0] += pA[i][0] * pX[i][0] + pA[i][lda] * pX[i][1];
                #pragma omp atomic
                pY[i][1] += pA[i][1] * pX[i][0] + pA[i][lda+1] * pX[i][1];
            }else if constexpr(n==3){
                T y0=0, y1=0, y2=0;
                for(int j=0; j<n; j++){
                    y0 += pA[i][j*lda] * pX[i][j];
                    y1 += pA[i][j*lda+1] * pX[i][j];
                    y2 += pA[i][j*lda+2] * pX[i][j];
                }
                #pragma omp atomic
                pY[i][0] += y0;
                #pragma omp atomic
                pY[i][1] += y1;
                #pragma omp atomic
                pY[i][2] += y2;
            }else if constexpr(n==4){
                T y0=0, y1=0, y2=0, y3=0;
                for(int j=0; j<n; j++){
                    y0 += pA[i][j*lda] * pX[i][j];
                    y1 += pA[i][j*lda+1] * pX[i][j];
                    y2 += pA[i][j*lda+2] * pX[i][j];
                    y3 += pA[i][j*lda+3] * pX[i][j];
                }
                #pragma omp atomic
                pY[i][0] += y0;
                #pragma omp atomic
                pY[i][1] += y1;
                #pragma omp atomic
                pY[i][2] += y2;
                #pragma omp atomic
                pY[i][3] += y3;
            }
        }else if constexpr(dimensions == 2){
            if constexpr(n == 2){
                T w0 = pX[i][0] * pA[2*i][0] + pX[i][2] * pA[2*i][lda];
                T w1 = pX[i][1] * pA[2*i][0] + pX[i][3] * pA[2*i][lda];
                T w2 = pX[i][0] * pA[2*i][1] + pX[i][2] * pA[2*i][lda+1];
                T w3 = pX[i][1] * pA[2*i][1] + pX[i][3] * pA[2*i][lda+1];
                T y0 = pA[2*i+1][0] * w0 + pA[2*i+1][lda] * w1;
                T y1 = pA[2*i+1][1] * w0 + pA[2*i+1][lda+1] * w1;
                T y2 = pA[2*i+1][0] * w2 + pA[2*i+1][lda] * w3;
                T y3 = pA[2*i+1][1] * w2 + pA[2*i+1][lda+1] * w3;
                #pragma omp atomic
                pY[i][0] += y0;
                #pragma omp atomic
                pY[i][1] += y1;
                #pragma omp atomic
                pY[i][2] += y2;
                #pragma omp atomic
                pY[i][3] += y3;
            }else if constexpr(n >= 3){
                T W[n][n], Y[n][n];
                for(int j=0; j<n; j++){
                    for(int k=0; k<n; k++){
                        W[j][k] = 0;
                        Y[j][k] = 0;
                    }
                }
                for(int j=0; j<n; j++)
                    for(int k=0; k<n; k++)
                        for(int s=0; s<n; s++)
                            W[s][k] += pX[i][n*j + k] * pA[2*i][j * lda + s];
                for(int j=0; j<n; j++)
                    for(int k=0; k<n; k++)
                        for(int s=0; s<n; s++)
                            Y[k][s] += pA[2*i+1][j * lda + s] * W[k][j];
                for(int j=0; j<n; j++){
                    for(int k=0; k<n; k++){
                        #pragma omp atomic
                        pY[i][n*j + k] += Y[j][k];
                    }
                }
            }
        }else if constexpr(dimensions == 3){
            T W[n][n][n], Y[n][n][n];
            for(int j=0; j<n; j++){
                for(int l=0; l<n; l++){
                    for(int k=0; k<n; k++){
                        W[j][l][k] = 0;
                        Y[j][l][k] = 0;
                    }
                }
            }
            for(int j=0; j<n; j++)
                for(int l=0; l<n; l++)
                    for(int k=0; k<n; k++)
                        for(int s=0; s<n; s++)
                            Y[s][l][k] += pX[i][n*n*j + n*l + k] * pA[3*i][j * lda + s];
            for(int j=0; j<n; j++)
                for(int l=0; l<n; l++)
                    for(int k=0; k<n; k++)
                        for(int s=0; s<n; s++)
                            W[l][s][k] += Y[l][j][k] * pA[3*i+1][j * lda + s];
            for(int j=0; j<n; j++)
                for(int l=0; l<n; l++)
                    for(int k=0; k<n; k++)
                        Y[j][l][k] = 0;
            for(int j=0; j<n; j++)
                for(int l=0; l<n; l++)
                    for(int k=0; k<n; k++)
                        for(int s=0; s<n; s++)
                            Y[l][k][s] += pA[3*i+2][j * lda + s] * W[l][k][j];
            for(int j=0; j<n; j++){
                for(int l=0; l<n; l++){
                    for(int k=0; k<n; k++){
                        #pragma omp atomic
                        pY[i][n*n*j + n*l + k] += Y[j][l][k];
                    }
                }
            }
        }else if constexpr(dimensions == 4){
            T W[n][n][n][n], Y[n][n][n][n];
            for(int j=0; j<n; j++){
                for(int p=0; p<n; p++){
                    for(int l=0; l<n; l++){
                        for(int k=0; k<n; k++){
                            W[j][p][l][k] = 0;
                            Y[j][p][l][k] = 0;
                        }
                    }
                }
            }
            for(int j=0; j<n; j++)
                for(int p=0; p<n; p++)
                    for(int l=0; l<n; l++)
                        for(int k=0; k<n; k++)
                            for(int s=0; s<n; s++)
                                W[s][p][l][k] += pX[i][n*n*n*j + n*n*p + n*l + k] * pA[4*i][j * lda + s];
            for(int j=0; j<n; j++)
                for(int p=0; p<n; p++)
                    for(int l=0; l<n; l++)
                        for(int k=0; k<n; k++)
                            for(int s=0; s<n; s++)
                                Y[p][s][l][k] += W[p][j][l][k] * pA[4*i+1][j * lda + s];
            for(int j=0; j<n; j++)
                for(int p=0; p<n; p++)
                    for(int l=0; l<n; l++)
                        for(int k=0; k<n; k++)
                            W[j][p][l][k] = 0;
            for(int j=0; j<n; j++)
                for(int p=0; p<n; p++)
                    for(int l=0; l<n; l++)
                        for(int k=0; k<n; k++)
                            for(int s=0; s<n; s++)
                                W[p][l][s][k] += Y[p][l][j][k] * pA[4*i+2][j * lda + s];
            for(int j=0; j<n; j++)
                for(int p=0; p<n; p++)
                    for(int l=0; l<n; l++)
                        for(int k=0; k<n; k++)
                            Y[j][p][l][k] = 0;
            for(int j=0; j<n; j++)
                for(int p=0; p<n; p++)
                    for(int l=0; l<n; l++)
                        for(int k=0; k<n; k++)
                            for(int s=0; s<n; s++)
                                Y[p][l][k][s] += pA[4*i+3][j * lda + s] * W[p][l][k][j];
            for(int j=0; j<n; j++){
                for(int p=0; p<n; p++){
                    for(int l=0; l<n; l++){
                        for(int k=0; k<n; k++){
                            #pragma omp atomic
                            pY[i][n*n*n*j + n*n*p + n*l + k] += Y[j][p][l][k];
                        }
                    }
                }
            }
        }else if constexpr(dimensions == 5){
            T W[n][n][n][n][n], Y[n][n][n][n][n];
            for(int j=0; j<n; j++){
                for(int v=0; v<n; v++){
                    for(int p=0; p<n; p++){
                        for(int l=0; l<n; l++){
                            for(int k=0; k<n; k++){
                                W[j][v][p][l][k] = 0;
                                Y[j][v][p][l][k] = 0;
                            }
                        }
                    }
                }
            }
            for(int j=0; j<n; j++)
                for(int v=0; v<n; v++)
                    for(int p=0; p<n; p++)
                        for(int l=0; l<n; l++)
                            for(int k=0; k<n; k++)
                                for(int s=0; s<n; s++)
                                    Y[s][v][p][l][k] += pX[i][n*n*n*n*j + n*n*n*v + n*n*p + n*l + k] * pA[5*i][j * lda + s];
            for(int j=0; j<n; j++)
                for(int v=0; v<n; v++)
                    for(int p=0; p<n; p++)
                        for(int l=0; l<n; l++)
                            for(int k=0; k<n; k++)
                                for(int s=0; s<n; s++)
                                    W[v][s][p][l][k] += Y[v][j][p][l][k] * pA[5*i+1][j * lda + s];
            for(int j=0; j<n; j++)
                for(int v=0; v<n; v++)
                    for(int p=0; p<n; p++)
                        for(int l=0; l<n; l++)
                            for(int k=0; k<n; k++)
                                Y[j][v][p][l][k] = 0;
            for(int j=0; j<n; j++)
                for(int v=0; v<n; v++)
                    for(int p=0; p<n; p++)
                        for(int l=0; l<n; l++)
                            for(int k=0; k<n; k++)
                                for(int s=0; s<n; s++)
                                    Y[v][p][s][l][k] += W[v][p][j][l][k] * pA[5*i+2][j * lda + s];
            for(int j=0; j<n; j++)
                for(int v=0; v<n; v++)
                    for(int p=0; p<n; p++)
                        for(int l=0; l<n; l++)
                            for(int k=0; k<n; k++)
                                W[j][v][p][l][k] = 0;
            for(int j=0; j<n; j++)
                for(int v=0; v<n; v++)
                    for(int p=0; p<n; p++)
                        for(int l=0; l<n; l++)
                            for(int k=0; k<n; k++)
                                for(int s=0; s<n; s++)
                                    W[v][p][l][s][k] += Y[v][p][l][j][k] * pA[5*i+3][j * lda + s];
            for(int j=0; j<n; j++)
                for(int v=0; v<n; v++)
                    for(int p=0; p<n; p++)
                        for(int l=0; l<n; l++)
                            for(int k=0; k<n; k++)
                                Y[j][v][p][l][k] = 0;
            for(int j=0; j<n; j++)
                for(int v=0; v<n; v++)
                    for(int p=0; p<n; p++)
                        for(int l=0; l<n; l++)
                            for(int k=0; k<n; k++)
                                for(int s=0; s<n; s++)
                                    Y[v][p][l][k][s] += pA[5*i+4][j * lda + s] * W[v][p][l][k][j];
            for(int j=0; j<n; j++){
                for(int v=0; v<n; v++){
                    for(int p=0; p<n; p++){
                        for(int l=0; l<n; l++){
                            for(int k=0; k<n; k++){
                                #pragma omp atomic
                                pY[i][n*n*n*n*j + n*n*n*v + n*n*p + n*l + k] += Y[j][v][p][l][k];
                            }
                        }
                    }
                }
            }
        }else if constexpr(dimensions == 6){
            T W[n][n][n][n][n][n], Y[n][n][n][n][n][n];
            for(int j=0; j<n; j++){
                for(int w=0; w<n; w++){
                    for(int v=0; v<n; v++){
                        for(int p=0; p<n; p++){
                            for(int l=0; l<n; l++){
                                for(int k=0; k<n; k++){
                                    W[j][w][v][p][l][k] = 0;
                                    Y[j][w][v][p][l][k] = 0;
                                }
                            }
                        }
                    }
                }
            }
            for(int j=0; j<n; j++)
                for(int w=0; w<n; w++)
                    for(int v=0; v<n; v++)
                        for(int p=0; p<n; p++)
                            for(int l=0; l<n; l++)
                                for(int k=0; k<n; k++)
                                    for(int s=0; s<n; s++)
                                        W[s][w][v][p][l][k] += pX[i][n*n*n*n*n*j + n*n*n*n*w + n*n*n*v + n*n*p + n*l + k] * pA[6*i][j * lda + s];
            for(int j=0; j<n; j++)
                for(int w=0; w<n; w++)
                    for(int v=0; v<n; v++)
                        for(int p=0; p<n; p++)
                            for(int l=0; l<n; l++)
                                for(int k=0; k<n; k++)
                                    for(int s=0; s<n; s++)
                                        Y[w][s][v][p][l][k] += W[w][j][v][p][l][k] * pA[6*i+1][j * lda + s];
            for(int j=0; j<n; j++)
                for(int w=0; w<n; w++)
                    for(int v=0; v<n; v++)
                        for(int p=0; p<n; p++)
                            for(int l=0; l<n; l++)
                                for(int k=0; k<n; k++)
                                    W[j][w][v][p][l][k] = 0;
            for(int j=0; j<n; j++)
                for(int w=0; w<n; w++)
                    for(int v=0; v<n; v++)
                        for(int p=0; p<n; p++)
                            for(int l=0; l<n; l++)
                                for(int k=0; k<n; k++)
                                    for(int s=0; s<n; s++)
                                        W[w][v][s][p][l][k] += Y[w][v][j][p][l][k] * pA[6*i+2][j * lda + s];
            for(int j=0; j<n; j++)
                for(int w=0; w<n; w++)
                    for(int v=0; v<n; v++)
                        for(int p=0; p<n; p++)
                            for(int l=0; l<n; l++)
                                for(int k=0; k<n; k++)
                                    Y[j][w][v][p][l][k] = 0;
            for(int j=0; j<n; j++)
                for(int w=0; w<n; w++)
                    for(int v=0; v<n; v++)
                        for(int p=0; p<n; p++)
                            for(int l=0; l<n; l++)
                                for(int k=0; k<n; k++)
                                    for(int s=0; s<n; s++)
                                        Y[w][v][p][s][l][k] += W[w][v][p][j][l][k] * pA[6*i+3][j * lda + s];
            for(int j=0; j<n; j++)
                for(int w=0; w<n; w++)
                    for(int v=0; v<n; v++)
                        for(int p=0; p<n; p++)
                            for(int l=0; l<n; l++)
                                for(int k=0; k<n; k++)
                                    W[j][w][v][p][l][k] = 0;
            for(int j=0; j<n; j++)
                for(int w=0; w<n; w++)
                    for(int v=0; v<n; v++)
                        for(int p=0; p<n; p++)
                            for(int l=0; l<n; l++)
                                for(int k=0; k<n; k++)
                                    for(int s=0; s<n; s++)
                                        W[w][v][p][l][s][k] += Y[w][v][p][l][j][k] * pA[6*i+4][j * lda + s];
            for(int j=0; j<n; j++)
                for(int w=0; w<n; w++)
                    for(int v=0; v<n; v++)
                        for(int p=0; p<n; p++)
                            for(int l=0; l<n; l++)
                                for(int k=0; k<n; k++)
                                    Y[j][w][v][p][l][k] = 0;
            for(int j=0; j<n; j++)
                for(int w=0; w<n; w++)
                    for(int v=0; v<n; v++)
                        for(int p=0; p<n; p++)
                            for(int l=0; l<n; l++)
                                for(int k=0; k<n; k++)
                                    for(int s=0; s<n; s++)
                                        Y[w][v][p][l][k][s] += pA[6*i+5][j * lda + s] * W[w][v][p][l][k][j];
            for(int j=0; j<n; j++){
                for(int w=0; w<n; w++){
                    for(int v=0; v<n; v++){
                        for(int p=0; p<n; p++){
                            for(int l=0; l<n; l++){
                                for(int k=0; k<n; k++){
                                    #pragma omp atomic
                                    pY[i][n*n*n*n*n*j + n*n*n*n*w + n*n*n*v + n*n*p + n*l + k] += Y[j][w][v][p][l][k];
                                }
                            }
                        }
                    }
                }
            }
        }
    } // for loop
}


#define asgard_kronmult_cpu_instantiate(d, n) \
    template void run_cpu_variant<float, (d), (n)>(float const *const[], int const, \
                                                   float const * const[], float *[], int const); \
    template void run_cpu_variant<double, (d), (n)>(double const *const[], int const, \
                                                    double const * const[], double *[], int const); \

#define asgard_kronmult_cpu_instantiate_n234_d(d) \
    asgard_kronmult_cpu_instantiate((d), 2); \
    asgard_kronmult_cpu_instantiate((d), 3); \
    asgard_kronmult_cpu_instantiate((d), 4); \

asgard_kronmult_cpu_instantiate_n234_d(1)
asgard_kronmult_cpu_instantiate_n234_d(2)
asgard_kronmult_cpu_instantiate_n234_d(3)
asgard_kronmult_cpu_instantiate_n234_d(4)
asgard_kronmult_cpu_instantiate_n234_d(5)
asgard_kronmult_cpu_instantiate_n234_d(6)

}
