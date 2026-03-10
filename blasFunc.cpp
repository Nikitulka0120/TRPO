#include <iostream>
#include <vector>
#include <cblas.h>

template <typename T>
void trsm(int M, int N, T alpha, const T* A, int lda, T* B, int ldb) {
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            T sum = 0;
            for (int k = 0; k < i; ++k) {
                sum += A[i + k * lda] * B[k + j * ldb];
            }
            B[i + j * ldb] = (alpha * B[i + j * ldb] - sum) / A[i + i * lda];
        }
    }
}

int main() {
    const int size = 4200;
    const double alpha = 1.0;

    std::vector<double> A(size * size, 1.0);
    std::vector<double> B_my(size * size, 2.0);
    std::vector<double> B_blas(size * size, 2.0);

    for (int i = 0; i < size; ++i) {
        A[i + i * size] = size * 2.0;
    }

    std::cout << "Подсчет матрицы размером " << size << std::endl;

    trsm(size, size, alpha, A.data(), size, B_my.data(), size);

    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, 
                size, size, alpha, A.data(), size, B_blas.data(), size);

    std::cout << "Расчет окончен" << std::endl;

    return 0;
}