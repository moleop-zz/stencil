#include <iostream>

#define BLOCKSIZE 8 

typedef struct {
  int width;
  int height;
  int stride;
  float* elem;
} Matrix;

__device__ Matrix GetSubMatrix(Matrix A, int row, int col, int num_gh){

  Matrix submat;
  submat.width = BLOCKSIZE;
  submat.height = BLOCKSIZE;
  submat.stride = A.stride;
  //if (row == 0 && col == 0) submat.elem = &A.elem[A.stride * (BLOCKSIZE) * row + (BLOCKSIZE) * col];
  //else if (row == 0) submat.elem = &A.elem[A.stride * (BLOCKSIZE) * row + (BLOCKSIZE-num_gh) * col];
  //else if (col == 0) submat.elem = &A.elem[A.stride * (BLOCKSIZE-num_gh) * row + (BLOCKSIZE) * col];
  submat.elem = &A.elem[A.stride * (BLOCKSIZE-1-num_gh) * row + (BLOCKSIZE-1-num_gh) * col];
  return submat;
}

__device__ void SetElement(Matrix A, int row, int col, float value){
  A.elem[row * A.stride + col] = value;
 }

__device__ float GetElement(const Matrix A, int row, int col){
  return A.elem[row * A.stride + col];
}

__device__ float GetNextValue(float Mat[BLOCKSIZE][BLOCKSIZE], int row, int col, float newValue){
  if (row != 0 && row != BLOCKSIZE-1 && col != 0 && col!=BLOCKSIZE-1){
    newValue = Mat[row][col] + Mat[row-1][col] + Mat[row+1][col] +Mat[row][col-1] + Mat[row][col+1];
  }else{
    newValue = Mat[row][col];
    if (row == 0){  newValue += Mat[row+1][col];
      if (col == 0) newValue += Mat[row][col+1];
      else if (col == BLOCKSIZE-1) newValue += Mat[row][col-1];
      else newValue += Mat[row][col+1] + Mat[row][col-1];
    }
    else if (row == BLOCKSIZE-1) {newValue += Mat[row-1][col];
      if (col == 0) newValue += Mat[row][col+1];
      else if (col == BLOCKSIZE-1) newValue += Mat[row][col-1];
      else newValue += Mat[row][col+1] + Mat[row][col-1];
    }
    else if (col == 0) newValue += Mat[row][col+1] + Mat[row+1][col] + Mat[row-1][col];
    else newValue += Mat[row][col-1] + Mat[row+1][col] + Mat[row-1][col];
  }
  return newValue/5;
  }


__global__ void StencilKernel(Matrix A, int num_gh){

  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;

  float newValue = 0;

  int row = threadIdx.y;
  int col = threadIdx.x;

  Matrix submat;

  //for (int k = 0; k < (A.width/ BLOCKSIZE); ++k){
  //  submat = GetSubMatrix(A, blockRow, k, num_gh);
  //for (int iter = 0; iter < num_gh+1; ++iter){
    submat = GetSubMatrix(A, blockRow, blockCol, num_gh);
    __shared__ float Mat[BLOCKSIZE][BLOCKSIZE];
    Mat[row][col] = GetElement(submat, row, col);
    __syncthreads();

    newValue = GetNextValue(Mat, row, col, newValue);
    __syncthreads();

  if (blockRow == 0 && blockCol == 0 ){
	if (row != BLOCKSIZE-1   && col != BLOCKSIZE-1)
		SetElement(submat, row, col, newValue);
  }else if (blockRow == 0){
   	if (col > 0 && col < BLOCKSIZE-1 && row != BLOCKSIZE-1)
   		SetElement(submat, row, col-1*blockCol, newValue);
  }else if (blockCol == 0){
   	if (row > 0 && row < BLOCKSIZE-1 && col != BLOCKSIZE-1)
   		SetElement(submat, row-1*blockRow, col, newValue);
  }else{
     	if (col != BLOCKSIZE-1 && row != 0 && row != BLOCKSIZE-1 && col != 0 )
   		SetElement(submat, row-1*blockRow, col-1*blockCol, newValue);
	// hier weiter machen
  }else if (blockRow == grid.y && blockCol == grid.x ){
	if (row < size % BLOCKSIZE-1   && col != BLOCKSIZE-1)
		SetElement(submat, row, col, newValue);
  }else if (blockRow == 0){
   	if (col > 0 && col < BLOCKSIZE-1 && row != BLOCKSIZE-1)
   		SetElement(submat, row, col-1*blockCol, newValue);
  }else if (blockCol == 0){
   	if (row > 0 && row < BLOCKSIZE-1 && col != BLOCKSIZE-1)
   		SetElement(submat, row-1*blockRow, col, newValue);
  }else{
     	if (col != BLOCKSIZE-1 && row != 0 && row != BLOCKSIZE-1 && col != 0 )
   		SetElement(submat, row-1*blockRow, col-1*blockCol, newValue);
   }	   
   }	   
  //}
}
//}

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

  //Größe des Feldes
  int size=32;
  //Anzahl Iterationen
  int iter=1;
  //Anzahl der Ghostcells (Überlapp)
  int num_gh = 0;
  // interation ohne exchange
  //int iter2=std::ceil(iter/num_gh);
  //Ausgabedatei
  char *filename="out.ppm";

  Matrix h_mat;
  h_mat.width = size;
  h_mat.height = size;
  h_mat.stride = h_mat.width;
  int mem = h_mat.width * h_mat.height * sizeof(float);
  h_mat.elem =(float*)malloc(mem);

  Matrix d_mat;
  d_mat.width = h_mat.width;
  d_mat.height = h_mat.height;
  d_mat.stride = d_mat.width;
  cudaMalloc(&d_mat.elem, mem);

  init_mat(h_mat);
  cudaMemcpy(d_mat.elem, h_mat.elem, mem, cudaMemcpyHostToDevice);

  dim3 threads(BLOCKSIZE, BLOCKSIZE); // 2 dimensional
  dim3 grid (h_mat.width / (threads.x-2), h_mat.height / (threads.y-2));
 // dim3 grid (std::ceil((double)h_mat.width / threads.x-num_gh), std::ceil((double)h_mat.height / threads.y-num_gh));
  for (int run = 0; run < iter;++run){
    StencilKernel<<<grid,threads>>>(d_mat, num_gh);
    cudaMemcpy(h_mat.elem, d_mat.elem, mem, cudaMemcpyDeviceToHost);
  }
  print_mat(h_mat);

  cudaFree (d_mat.elem);
  free (h_mat.elem);


  return 0;
}
