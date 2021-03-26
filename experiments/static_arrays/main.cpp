#include <cstdio>


int main() {

    int a[5] = {1, 10, 42, 69, 123};
    int b[5];

    for(int i = 0; i < 5; i++) {
        printf("%4d ", a[i]);
    }
    printf("\n");

    //b = a; // NOT
    for(int i = 0; i < 5; i++)
        b[i] = a[i];

    for(int i = 0; i < 5; i++) {
        printf("%4d ", b[i]);
    }
    printf("\n");



    constexpr int n_rows = 3;
    constexpr int n_cols = 5;
    int arr[n_rows][n_cols];
    for(int r = 0; r < n_rows; r++)
        for(int c = 0; c < n_cols; c++)
            arr[r][c] = r * n_cols + c;
    
    for(int r = 0; r < n_rows; r++) {
        for(int c = 0; c < n_cols; c++) {
            printf("%3d ", arr[r][c]);
        }
        printf("\n");
    }

    for(int i = 0; i < n_rows*n_cols; i++)
        printf("%3d ", arr[i/n_cols][i%n_cols]);
    printf("\n");


    return 0;
}
