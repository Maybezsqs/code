#include <iostream>
#include <math.h>
#include <chrono>
#include "device_launch_parameters.h"
#include "cudaMatMul.h" 
#include <cuda_runtime.h>
#include <cublas_v2.h>
using namespace std;
using namespace std::chrono;

#define M 10000
#define K 1000
#define N 10000

int main()
{
    float *A = new float[M * K];
    float *B = new float[K * N];
    float *C = new float[M * N];

    for (int i = 0; i < M * K; i++)
    {
        A[i] = sin(i);
    }

    for (int i = 0; i < K*N; i++){
        B[i] = cos(i);
    }

    for (int i = 0; i < M * N; i++)
    {
        C[i] = 0.5;
    }

    Matrix h_a, h_b, h_c;
    h_a.width = h_a.stride = M; h_a.height = K; h_a.elements = A;
    h_b.width = h_b.stride = K; h_b.height = N; h_b.elements = B;
    h_c.width = h_c.stride = M; h_c.height = N; h_c.elements = C;

    auto startTime = high_resolution_clock::now();
    MatMul(h_a, h_b, h_c);
    auto endTime = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(endTime - startTime); 
    cout << "cuda计算用时" <<double(duration.count()) / 1000000 << "s" << endl;



    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    std::cout << "使用GPU device " << 0 << ": " << devProp.name << std::endl;
    std::cout << "设备全局内存总量： " << devProp.totalGlobalMem / 1024 / 1024 << "MB" << std::endl;
    std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
    std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "设备上一个线程块（Block）种可用的32位寄存器数量： " << devProp.regsPerBlock << std::endl;
    std::cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "每个EM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
    std::cout << "设备上多处理器的数量： " << devProp.multiProcessorCount << std::endl;
    std::cout << "======================================================" << std::endl;     
        
    free(A);
    free(B);
    free(C);
}