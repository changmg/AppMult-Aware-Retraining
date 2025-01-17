#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H


#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>


// compute ceil of m/n
#define CEIL_DIV(m,n) ( (m) + (n) - 1 ) / (n)

// get offset of element at (row, col) in a 2D array with leading dimension ld
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// load float4, uchar4, uint4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FETCH_UCHAR4(pointer) (reinterpret_cast<uchar4*>(&(pointer))[0])
#define FETCH_UINT4(pointer) (reinterpret_cast<uint4*>(&(pointer))[0])

// load different bits for different types
template<int LOAD_BITS>
struct LoadType;

template<>
struct LoadType<1> {
    using uchar_type = uchar1;
    using ushort_type = ushort;
    using short_type = short;
    using uint_type = uint;
    using float_type = float1;
};

template<>
struct LoadType<2> {
    using uchar_type = uchar2;
    using ushort_type = ushort2;
    using short_type = short2;
    using uint_type = uint2;
    using float_type = float2;
};

template<>
struct LoadType<4> {
    using uchar_type = uchar4;
    using ushort_type = ushort4;
    using short_type = short4;
    using uint_type = uint4;
    using float_type = float4;
};

template <typename T1, typename T2>
static inline __device__ T2 & FETCH(T1 &pointer) {
    return reinterpret_cast<T2*>(&pointer)[0];
}


static inline void CHECK_INPUT(const torch::Tensor& x) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDAtensor ");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous ");
}


static inline void CHECK_INPUT(const torch::Tensor& x, torch::Dtype dtype, const std::string& tensorName) {
    TORCH_CHECK(x.device().is_cuda(), tensorName, " must be a CUDAtensor");
    TORCH_CHECK(x.is_contiguous(), tensorName, " must be contiguous");
    TORCH_CHECK(x.scalar_type() == dtype, tensorName, " must be of type ", dtype);
}


static void ShowGPUInfo(void) {
    int deviceCount;
    // Get the total number of CUDA devices
    cudaGetDeviceCount(&deviceCount);
    // Retrieve information for each CUDA device
    for(int i = 0; i < deviceCount; ++i) {
        // Define a structure to store the device information
        cudaDeviceProp devProp;
        // Retrieve the information for the i-th CUDA device and store it in the structure
        cudaGetDeviceProperties(&devProp, i);
        printf("Using GPU device %d: %s\n", i, devProp.name);
        printf("Total global memory on device: %lu MB\n", devProp.totalGlobalMem / 1024 / 1024);
        printf("Total constant memory on device: %lu KB\n", devProp.totalConstMem / 1024);
        printf("Number of streaming multiprocessors (SMs): %d\n", devProp.multiProcessorCount);
        printf("Shared memory per block: %lu KB\n", devProp.sharedMemPerBlock / 1024);
        printf("Maximum threads per block: %d\n", devProp.maxThreadsPerBlock);
        printf("Number of 32-bit registers per block: %d\n", devProp.regsPerBlock);
        printf("Maximum threads per multiprocessor: %d\n", devProp.maxThreadsPerMultiProcessor);
        printf("Maximum thread blocks per multiprocessor: %d\n", devProp.maxThreadsPerMultiProcessor / 32);
        printf("Number of multiprocessors: %d\n", devProp.multiProcessorCount);
        printf("======================================================\n");
    }
}

// Function to check CUDA errors
static inline void checkCuda(cudaError_t result, const char* file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << " at " << file << ":" << line << std::endl;
        exit(result);
    }
}

#define checkCudaErrors(val) checkCuda((val), __FILE__, __LINE__)


#endif