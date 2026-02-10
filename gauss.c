#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void printMatrix(int n, int m, double matrix[n][m])
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            printf("%.2f ", matrix[i][j]);
        }
        printf("\n");
    }
}

void TriangleBacktrack(int n, int m, double matrix[n][m])
{
    double *results = (double *)malloc(sizeof(matrix[0][0]) * m);
    if (results == NULL)
    {
        puts("Не удалось выделить память под массив результатов!");
        return;
    }
    int row_index = n - 1;
    int col_index = m - 1;
    for (int i = row_index; i >= 0; i--)
    {
        double tmp_result = matrix[i][col_index];
        printf("X%d = %lf", i + 1, tmp_result);
        for (int j = col_index - 1; j > i; j--)
        {
            tmp_result -= matrix[i][j] * results[j];
            printf("-%lf*%lf", matrix[i][j], results[j]);
        }
        tmp_result /= matrix[i][i];
        printf("/%lf", matrix[i][i]);
        results[i] = tmp_result;
        printf("= %lf\n", tmp_result);
    };
}

void triangleGaussMTD(int n, int m, double matrix[n][m])
{
    printf("\n");
    for (int i = 0; i < n; i++)
    {
        for (int k = i + 1; k < n; k++)
        {
            double mnozhitel = matrix[k][i] / matrix[i][i];
            for (int j = i; j < m; j++)
            {
                matrix[k][j] = matrix[k][j] - mnozhitel * matrix[i][j];
            }
        }
        puts("");
        printMatrix(n, m, matrix);
    }
    TriangleBacktrack(n, m, matrix);
}

void swap(int n, int m, double matrix[n][m], int currentpos)
{
    int strForSwap = currentpos;
    for (int i = currentpos; i < n; i++)
    {
        if (abs(matrix[i][currentpos]) > abs(matrix[strForSwap][currentpos]))
            strForSwap = i;
    }
    for (int j = 0; j < m; j++)
    {
        double tmp = matrix[currentpos][j];
        matrix[currentpos][j] = matrix[strForSwap][j];
        matrix[strForSwap][j] = tmp;
    }
};

void triangleGaussWITHSWAP(int n, int m, double matrix[n][m])
{
    printf("\n");
    for (int i = 0; i < n; i++)
    {
        swap(n, m, matrix, i);
        for (int k = i + 1; k < n; k++)
        {
            double mnozhitel = matrix[k][i] / matrix[i][i];
            for (int j = i; j < m; j++)
            {
                matrix[k][j] = matrix[k][j] - mnozhitel * matrix[i][j];
            }
        }
        puts("");
        printMatrix(n, m, matrix);
    }
    TriangleBacktrack(n, m, matrix);
}

void DiagonalGauss(int n, int m, double matrix[n][m])
{
    printf("\n");
    for (int i = 0; i < n; i++)
    {
        for (int k = 0; k < n; k++)
        {
            if (k == i)
                continue;
            double mnozhitel = matrix[k][i] / matrix[i][i];
            for (int j = i; j < m; j++)
            {
                matrix[k][j] = matrix[k][j] - mnozhitel * matrix[i][j];
            }
        }
    }
    for (int i = 0; i < n; i++)
    {
        printf("x%d = %lf\n", i, matrix[i][m - 1] / matrix[i][i]);
    }
}

int main()
{
    int n = 3;
    int m = 4;
    srand(time(NULL));
    double matrixTRI[3][4] = {
        {2, -1, 3, 1},
        {-4, 1, 2, 4},
        {1, 3, 2, 8},
    };

    double matrixDIO[3][4] = {
        {2, -1, 3, 1},
        {-4, 1, 2, 4},
        {1, 3, 2, 8},
    };

    double matrixSWAP[3][4] = {
        {2, -1, 3, 1},
        {-4, 1, 2, 4},
        {1, 3, 2, 8},
    };
    printf("Изначальная матрица:\n");
    printMatrix(n, m, matrixTRI);
    puts("Решение методом Гаусса с использованием треугольной матрицы:");
    triangleGaussMTD(n, m, matrixTRI);

    printMatrix(n, m, matrixTRI);

    puts("\nРешение методом Гаусса с использованием диагональной матрицы:");
    DiagonalGauss(n, m, matrixDIO);
    printMatrix(n, m, matrixDIO);

    puts("\nРешение методом Гаусса с перестановкой строк по модулю:");
    triangleGaussWITHSWAP(n, m, matrixSWAP);
    printMatrix(n, m, matrixSWAP);
    return 0;
}
