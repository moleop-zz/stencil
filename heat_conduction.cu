#include <iostream>

#define BLOCKSIZE 32

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
  submat.elem = &A.elem[A.stride * (BLOCKSIZE-2-2*num_gh) * row   + (BLOCKSIZE-2-2*num_gh) * col];
  return submat;
}

__device__ void SetElement(Matrix A, int row, int col, float value){
  A.elem[row * A.stride + col] = value;
 }

__device__ float GetElement(const Matrix A, int row, int col, int blockRow, int blockCol, dim3 grid, int limit_col, int limit_row, int num_gh, int size){
  if (blockRow < grid.y-1 && blockCol < grid.x-1){
    return A.elem[row * A.stride + col];
  // Ausnahme für submatrix die am Rand liegt ... Auffüllen mit 0.
  }else{
    float tmp[BLOCKSIZE*BLOCKSIZE];
    if (limit_col < size && limit_row < size){
      tmp[row*(BLOCKSIZE-1)+col] = A.elem[row * A.stride + col];
    }else tmp[row*(BLOCKSIZE-1)+col] = 0;
    return tmp[row*(BLOCKSIZE-1)+col];
  }
}

__device__ float GetNextValue(float Mat[BLOCKSIZE][BLOCKSIZE], int row, int col, int limit_col, int limit_row, int size, float newValue){
  if (limit_col < size && limit_row < size){
    if (row > 0 && row < BLOCKSIZE-1 && col > 0 && col < BLOCKSIZE-1){
      newValue = 2*Mat[row][col] + Mat[row-1][col] + Mat[row+1][col] + Mat[row][col-1] + Mat[row][col+1];
    }else{
      newValue = 2*Mat[row][col];
      if (row == 0){  newValue += Mat[row+1][col];
        if (col == 0) newValue += Mat[row][col+1];
        else if (col == BLOCKSIZE-1) newValue += Mat[row][col-1];
        else newValue += Mat[row][col+1] + Mat[row][col-1];
      }
      else if (row == BLOCKSIZE-1) { newValue += Mat[row-1][col];
        if (col == 0) newValue += Mat[row][col+1];
        else if (col == BLOCKSIZE-1) newValue += Mat[row][col-1];
        else newValue += Mat[row][col+1] + Mat[row][col-1];
      }
      else{
        if (col == 0)
          newValue += Mat[row][col+1] + Mat[row+1][col] + Mat[row-1][col];
        else newValue += Mat[row][col-1] + Mat[row+1][col] + Mat[row-1][col];
      }
    }
  }
  return newValue/5;
  }


__global__ void StencilKernel(Matrix A, Matrix B, dim3 grid, int num_gh, int iter,int size){

  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;

  int row = threadIdx.y;
  int col = threadIdx.x;

  // define limits for last threadblock
  int limit_row = blockRow*(BLOCKSIZE-2-2*num_gh)+row;
  int limit_col = blockCol*(BLOCKSIZE-2-2*num_gh)+col;

  Matrix submat;
  Matrix submat2;

  submat = GetSubMatrix(A, blockRow, blockCol, num_gh);
  submat2 = GetSubMatrix(B, blockRow, blockCol, num_gh);
  __shared__ float Mat[BLOCKSIZE][BLOCKSIZE];
  __shared__ float Mattemp[BLOCKSIZE][BLOCKSIZE];

  float newValue = 0;

  Mat[row][col] = GetElement(submat, row, col, blockRow, blockCol, grid, limit_col, limit_row, num_gh, size);
  __syncthreads();
  int gh_iter = num_gh;

    while(gh_iter > 0){
      if (iter > 1){
        Mattemp[row][col] = GetNextValue(Mat, row, col, limit_col, limit_row, size, newValue);
        __syncthreads();
        Mat[row][col] = Mattemp[row][col];
        --gh_iter;
        --iter;
        __syncthreads();
      //  float *tmp2 = &Mat[0][0];
      //  (void*)Mat = *Mattemp;
      //  Mattemp = tmp2;
      } else break;
    }

  newValue = GetNextValue(Mat, row, col, limit_col, limit_row, size, newValue);
  __syncthreads();

  // Initialize limits for writing back to Matrix A
  int limit_dr = BLOCKSIZE-1-num_gh; // down & right limit
  // num_gh = up & left limit

  // mitte
  if (blockCol > 0 && blockRow > 0 && blockCol < grid.x-1 && blockRow < grid.y-1){
    if (col < limit_dr && col > num_gh && row > num_gh && row < limit_dr)
      SetElement(submat, row, col, newValue);
  // oben links
  }else if (blockRow == 0 && blockCol == 0 ){
  	if (row < limit_dr && col < limit_dr)
  		SetElement(submat, row, col, newValue);
  // oben
  }else if (blockRow == 0 && blockCol > 0 && blockCol < grid.x-1){
   	if (col > num_gh && col < limit_dr && row < limit_dr)
   		SetElement(submat, row, col, newValue);
  // links
  }else if (blockCol == 0 && blockRow > 0 && blockRow < grid.y-1){
   	if (row > num_gh && row < limit_dr && col < limit_dr)
   		SetElement(submat, row, col, newValue);
	// oben rechts
  }else if (blockCol == grid.x-1 && blockRow == 0){
    if (row < limit_dr && col > num_gh && col < limit_dr && limit_col < size )
      SetElement(submat, row, col, newValue);
  // rechts
  }else if (blockCol == grid.x-1 && blockRow > 0 && blockRow < grid.y-1){
    if (row > num_gh && row < limit_dr && col > num_gh && col < limit_dr && limit_col < size )
      SetElement(submat, row, col, newValue);
  // unten rechts
  }else if (blockRow == grid.y-1 && blockCol == grid.x-1){
        if (col > num_gh && col < limit_dr && row > num_gh && limit_row < size && limit_col < size)
          SetElement(submat, row, col, newValue);
  // unten
  }else if (blockRow == grid.y-1 && blockCol > 0 && blockCol < grid.x-1){
      if (col > num_gh && col < limit_dr && row > num_gh && limit_row < size && limit_col < size )
        SetElement(submat, row, col, newValue);
  // unten links
  }else if (blockRow == grid.y-1 && blockCol == 0){
        if (col < limit_dr && row > num_gh && limit_row < size )
          SetElement(submat, row, col, newValue);
  }
//  float *tmp = submat.elem;
//  submat.elem = submat2.elem;
//  submat2.elem = tmp;
}

__host__ void init(float *t, int size){
   for (int i=0;i<size;i++){
      for (int j=0;j<size;j++){
         t[j+i*size]=0.0;
         if (j==0) t[i*size]=25.0;
         if (j==size-1) t[j+i*size]=-25.0;
         if (i==0) t[j+i*size]=25.0;
         if (i==size-1) t[j+i*size]=-25.0;
      }
   }
}

__host__ void print_mat(Matrix A){
  for (int i = 0; i < A.width*A.height; ++i){
    if (i % A.width == 0) printf("\n");
    printf("%.1f ", A.elem[i]);
  }
}

//Ausgabe des Feldes t als PPM (Portable Pix Map) in filename
//mit schönen Farben
void printResult(float *t, int size, char *filename){
   FILE *f=fopen(filename,"w");
   fprintf(f,"P3\n%i %i\n255\n",size,size);
   double tmax=25.0;
   double tmin=-tmax;
   double r,g,b;
   for (int i=0;i<size;i++){
      for (int j=0;j<size;j++){
        double val=t[j+i*size];
        r=0;
        g=0;
        b=0;
        if (val<=tmin){
           b=1.0*255.0;
        }else if(val>=-25.0 && val < -5){
           b=255*1.0;
           g=255*((val+25)/20);
        }else if(val>=-5 && val<=0.0){
           g=255*1.0;
           b=255*(1.0-(val+5)/5);
        }else if(val>0.0 && val <=5){
           g=255*1.0;
           r=255*((val)/5);
        }else if(val>5 && val<25.0){
            r=255*1.0;
            g=255*(1.0-(val-25)/20);
        }else{
            r=255*1.0;
        }
        fprintf(f,"%i\n%i\n%i\n",(int)r,(int)g,(int)b);
      }
//      fprintf(f,"\n");
   }
   fclose(f);
}

int main(int argc, char const *argv[]) {
  //Größe des Feldes
  int size = 128;
  //Anzahl Iterationen
  int iter = 20;
  //Anzahl der Ghostcells (Überlapp)
  int num_gh = 7;
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
  cudaMalloc((void**)&d_mat.elem, mem);

  Matrix d_mat2;
  d_mat2.width = h_mat.width;
  d_mat2.height = h_mat.height;
  d_mat2.stride = d_mat.width;
  cudaMalloc((void**)&d_mat2.elem, mem);

  init(h_mat.elem, size);
  cudaMemcpy(d_mat.elem, h_mat.elem, mem, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mat2.elem, h_mat.elem, mem, cudaMemcpyHostToDevice);

  dim3 threads(BLOCKSIZE, BLOCKSIZE);
  dim3 grid (std::ceil((double)h_mat.width / (threads.x-2
    -2*num_gh)),std::ceil((double)h_mat.height / (threads.y-2-2*num_gh)));
  printf("grid.x = %d\n", grid.x);
  printf("grid.y = %d\n", grid.y);
  while (iter>0){
    printf("run: %d\n", iter);
    StencilKernel<<<grid,threads>>>(d_mat, d_mat2, grid, num_gh, iter, size);
    iter = iter-1-num_gh;
  }
  cudaMemcpy(h_mat.elem, d_mat.elem, mem, cudaMemcpyDeviceToHost);
  printResult(h_mat.elem, size, filename);
  //cudaMemcpy(h_mat.elem, d_mat.elem, mem, cudaMemcpyDeviceToHost);
  //print_mat(h_mat);
  printf("Calcs done");
  //printResult(h_mat.elem, size, filename);

  cudaFree (d_mat.elem);
  cudaFree (d_mat2.elem);
  free (h_mat.elem);

  return 0;
}
