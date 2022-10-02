#include<iostream>
#include<vector>
#include<algorithm>
#include<cassert>

//Cuda Headers
#include <cuda_runtime_api.h>
#include <cuda.h>


#include <chrono>
using namespace std::chrono;
using namespace std;

//Global Variables 
#define BLOCKSIZE 32
#define N_THREAD_EACH_DIM 32
int MATRIX_SIZE;


__global__ void CudaMatrixMultiplication_notShared(const int* A, const int* B, int* C, int N)
{
    // compute each thread's index (  on matrix C)
    // Each element on C will be assigned to a thread 
    // They will perform the element wise multiplication corresponding to the row to get the report 

    int row  = blockIdx.y * blockDim.y + threadIdx.y;
    int col  = blockIdx.x * blockDim.x + threadIdx.x;

    C[ row * N + col] = 0.0;

    for ( int k = 0 ; k < N; k++)
    {
        C[row*N+col] += A[row*N+k]*B[col*k + N];
    }
}


// __global__ void CudaMatrixMultiplication_Shared(const int* A, const int* B, int* C, int N)
// {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;

//     //Allocated shared memory array 

//     __shared__ int shared_A[BLOCKSIZE*BLOCKSIZE];
//     __shared__ int shared_B[BLOCKSIZE*BLOCKSIZE];

//     int smval = 0;
//     for ( int i = 0 ; i < N ; i+= blockDim.x)
//     {
//         shared_A[threadIdx.y* blockDim.x + threadIdx.x]  = A[row * N + i + threadIdx.x];
//         shared_B[threadIdx.y * blockDim.x + threadIdx.x] = B[i * N + threadIdx.y * N + col];

//         __syncthreads();

//         for ( int j = 0 ; j < blockDim.x ; j++)
//         {
//              smval += shared_A[threadIdx.y * blockDim.x + j] * shared_B[j * blockDim.x + threadIdx.x];
//         }
//         __syncthreads();
//     }

//     C[row * N + col] = smval;

// }


__global__ void CudaMatrixMultiplication_Shared(const int* A, const int* B, int* C, int N)
{
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;


    //Allocated shared memory array 
    __shared__ int shared_A[BLOCKSIZE][BLOCKSIZE];
    __shared__ int shared_B[BLOCKSIZE][BLOCKSIZE];

    // Int Block Matrix C
    int* C_Block  = C + (N*BLOCKSIZE*blockRow + BLOCKSIZE*blockCol);

    // Thread row and column within Csub
    int row = threadIdx.x;
    int col = threadIdx.y;

    int smval = 0;
    for ( int i = 0 ; i < N/BLOCKSIZE ; i++ )
    {
        const int* A_Block =  A + (N*BLOCKSIZE*blockRow + BLOCKSIZE*i);
        const int* B_Block =  B + (N*BLOCKSIZE*i + BLOCKSIZE*blockCol);

        shared_A[row][col] = A_Block[row*N + col];
        shared_B[row][col] = B_Block[row*N + col];

        __syncthreads();

        for ( int j = 0 ; j < BLOCKSIZE ; j++)
        {
             smval += shared_A[row][j] * shared_B[j][row];
        }
        __syncthreads();
    }

    C_Block[row * N + col] = smval;
}



void CPUMatrixMultiplication(const int* A, const int* B, int* C, int N)
{
    for( int i =0 ; i < N; i++)
    {
        for ( int j=0 ; j < N; j++)
        {
            C[i*N + j] = 0;
            for ( int k=0 ; k < N; k++)
            {
                C[i*N + j] += A[i*N + k] * B[k*N + j];
            }
        }
    }
}



int VerifySolution(const int* A, const int* B,int N)
{
    for ( int i=0 ; i < N*N ; i++)
        if(A[i] != B[i])
        {
            cout << " i "  << i << "  " << A[i] << " , " << B[i] <<endl; 
            return 0;
        }
            
    
    return 1;
}

int main(int argc , char** argv)
{
    if(argc < 2)
    {
        cout << " ERROR : Enter the first Argument as size of Matrix " <<endl;
        cout << " Exiting " <<endl;
        exit(0);
    }

    MATRIX_SIZE = atoi(argv[1]);

    if(MATRIX_SIZE % 32 != 0 )
    {
        cout << " ERROR : Matrix size should be divisible by 32 " <<endl;
        cout << "Exiting " <<endl;
        exit(0);
    }

    cout << " MATRIX SIZE : " << MATRIX_SIZE <<endl;

    std::vector<int> hostA;
    std::vector<int> hostB;
    std::vector<int> hostC;
    std::vector<int> hostC_CPU;
    std::vector<int> hostC_GPU_Shared;

    //Create Cuda Events 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //Int size of Matrix for transfer 
    int MatrixSizeinBytes = MATRIX_SIZE*MATRIX_SIZE*sizeof(int);

    // Initialize every value with one
    hostA.resize(MATRIX_SIZE*MATRIX_SIZE,1);
    hostB.resize(MATRIX_SIZE*MATRIX_SIZE,1);
    hostC.resize(MATRIX_SIZE*MATRIX_SIZE,1);
    hostC_CPU.resize(MATRIX_SIZE*MATRIX_SIZE,0);
    hostC_GPU_Shared.resize(MATRIX_SIZE*MATRIX_SIZE,0);


    // Allocate memory on the device
    int *deviceA, *deviceB, *deviceC;
    cudaMalloc(&deviceA, MATRIX_SIZE*MATRIX_SIZE*sizeof(int));
    cudaMalloc(&deviceB, MATRIX_SIZE*MATRIX_SIZE*sizeof(int));
    cudaMalloc(&deviceC, MATRIX_SIZE*MATRIX_SIZE*sizeof(int));


    //Copy the data to Device
    cudaEventRecord(start);
    cudaMemcpy(deviceA,hostA.data(),MatrixSizeinBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB,hostB.data(),MatrixSizeinBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(deviceA,hostA.data(),MatrixSizeinBytes,cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << milliseconds <<endl;



    // Generate Blocks
    int N_BLOCKS = MATRIX_SIZE/ N_THREAD_EACH_DIM ;


    dim3 blockDim(N_BLOCKS,N_BLOCKS);
    dim3 threadDim(N_THREAD_EACH_DIM, N_THREAD_EACH_DIM);

    cudaEventRecord(start);
    //Launch the kernel 
    CudaMatrixMultiplication_notShared<<<blockDim, threadDim>>>(deviceA, deviceB, deviceC, MATRIX_SIZE);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << milliseconds <<endl;
    
    // Copy back to the host
    cudaMemcpy(hostC.data(), deviceC, MatrixSizeinBytes, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    /// -----------------------------------------------------------------------------------------   ////

    cudaEventRecord(start);
    CudaMatrixMultiplication_Shared<<<blockDim, threadDim >>>(deviceA,deviceB,deviceC,MATRIX_SIZE);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << milliseconds <<endl;

    cudaMemcpy(hostC_GPU_Shared.data(), deviceC, MatrixSizeinBytes, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    /// -----------------------------------------------------------------------------------------   ////

    auto start1 = high_resolution_clock::now();
    CPUMatrixMultiplication(hostA.data(),hostB.data(),hostC_CPU.data(),MATRIX_SIZE);
    auto stop1= high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop1 - start1);

    cout << double(duration.count()/1000.0) <<endl;

    if(VerifySolution(hostC_CPU.data(),hostC.data(),MATRIX_SIZE))
    {
        if(VerifySolution(hostC_CPU.data(),hostC_GPU_Shared.data(),MATRIX_SIZE))
            cout << " Solution Correct " <<endl;
    }




    return 0;

}