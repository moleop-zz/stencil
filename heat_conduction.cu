#include <iostream>

#define BLOCKSIZE 32

typedef struct {
  int width;
  int height;
  int stride;
  float* elem;
} Matrix;

__device__ Matrix GetSubMatrix(Matrix A, int row, int col){

  Matrix submat;
  submat.width = BLOCKSIZE;
  submat.height = BLOCKSIZE;
  submat.stride = A.stride;
  submat.elem = &A.elem[A.stride * BLOCKSIZE * row + BLOCKSIZE * col];
  return submat;
}

__device__ void SetElement(Matrix A, int row, int col, float value){
  A.elem[row * A.stride + col] = value;
 }

__device__ float GetElement(const Matrix A, int row, int col){
  return A.elem[row * A.stride + col];
}

__device__ float GetNextValue(float Mat[BLOCKSIZE][BLOCKSIZE], int row, int col, int l, float newValue){
  //newValue = Mat[row][l];
//  if (row == 0) //|| row == BLOCKSIZE-1 || col == 0 || col%(BLOCKSIZE-1)==0)
//    newValue += Mat[row+BLOCKSIZE-1][l]; //Mat[row][l-1] + Mat[row][l+1] + Mat[row+BLOCKSIZE-1][l];

//  if (row != 0 && row != BLOCKSIZE-1 && col != 0 && col!=BLOCKSIZE)
//    newValue  = Mat[row][l] + Mat[row][l-1] + Mat[row][l+1] + Mat[row+BLOCKSIZE][l] + Mat[row-BLOCKSIZE][l];

  //if{
    newValue = Mat[row][l];
    if (l != 0) {newValue += Mat[row][l-1];}
    //newValue += Mat[row-BLOCKSIZE-1][l];
    /*
    if (row == BLOCKSIZE-1){ newValue += Mat[row-BLOCKSIZE-1][l];
    } else  newValue += Mat[row+BLOCKSIZE-1][l];
    if (l == 0) { newValue += Mat[row][l+1];
    } else newValue += Mat[row][l-1];
    if (l == BLOCKSIZE-1) { newValue += Mat[row][l-1];
    } else newValue += Mat[row][l+1];*/
  return newValue;
  }


__global__ void StencilKernel(Matrix A){

  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;

  float newValue = 0;

  int row = threadIdx.y;
  int col = threadIdx.x;

  Matrix submat = GetSubMatrix(A, blockRow, blockCol);

  for (int k = 0; k < (A.width/ BLOCKSIZE); ++k){

    __shared__ float Mat[BLOCKSIZE][BLOCKSIZE];
    Mat[row][col] = GetElement(submat, row, col);
    __syncthreads();
    for (int l = 0; l < BLOCKSIZE; ++l){
      //newValue = Mat[row][l] + 1 ;
      newValue = GetNextValue(Mat, row, col, l, newValue);
      //newValue  = Mat[row][l] + Mat[row][l-1] + Mat[row][l+1] + Mat[row+BLOCKSIZE][l] + Mat[row-BLOCKSIZE][l];
    }
  }
  SetElement(submat, row, col, newValue);
}

__host__ void init_mat(Matrix A){
  for (int i = 0; i < A.width*A.height; ++i){
    A.elem[i] = 1;
  }
}

__host__ void print_mat(Matrix A){
  for (int i = 0; i < A.width*A.height; ++i){
    if (i % A.width == 0) printf("\n");
    printf("%.3f ", A.elem[i]);
  }
}

int main(int argc, char const *argv[]) {

  Matrix h_mat;
  h_mat.width = 32;
  h_mat.height = 32;
  h_mat.stride = h_mat.width;
  //int N = h_mat.width * h_mat.height;
  int size = h_mat.width * h_mat.height * sizeof(float);
  h_mat.elem =(float*)malloc(size);

  Matrix d_mat;
  d_mat.width = h_mat.width;
  d_mat.height = h_mat.height;
  d_mat.stride = d_mat.width;
  cudaMalloc(&d_mat.elem, size);

  init_mat(h_mat);
  cudaMemcpy(d_mat.elem, h_mat.elem, size, cudaMemcpyHostToDevice);

  dim3 threads(BLOCKSIZE, BLOCKSIZE); // 2 dimensional machen
  dim3 grid (h_mat.width / threads.x, h_mat.height / threads.y);
  StencilKernel<<<grid,threads>>>(d_mat);
  cudaMemcpy(h_mat.elem, d_mat.elem, size, cudaMemcpyDeviceToHost);

  print_mat(h_mat);

  cudaFree (d_mat.elem);
  free (h_mat.elem);


  return 0;
}
