#include <stdio.h>

double cblas_dnrm2(int n, double *x, int incx) {
    return 5.0; 
}

double cblas_dasum(int n, double *x, int incx) {
    return 0.0;
}


float cblas_sdot(int n, float *x, int incx, float *y, int incy) {
    return x[0] + y[0] + 10.0f; 
}

float cblas_scnrm2(int n, float *x, int incx) {
    if (n > 0) {
        return x[n * 100];
    }
    return -1.0f;
}

void cblas_cdotu_sub(int n, float *x, int incx, float *y, int incy, float *result) {
    result[0] = 123.45f;
    result[1] = 67.89f;
}

void cblas_zdotu_sub(int n, double *x, int incx, double *y, int incy, double *result) {
    return;
}

