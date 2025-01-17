#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cassert>
#include <iostream>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "cuda_helper.h"
#include "approx_mult.h"


/**
 * Compute the forward pass of the approximate gemm operation
 * 
 * In this kernel, there are CEIL_DIV(N, BLOCK_SIZE_N) * CEIL_DIV(M, BLOCK_SIZE_M) thread blocks
 * Each block computes a sub-matrix of C, which has a size of BLOCK_SIZE_M x BLOCK_SIZE_N
 * C_BMxBN = matmul(A_BMxK, B_KxBN) = \sum_{k=0}^{K/BK-1} matmul(A_BMxBK_k, B_BKxBN_k)
 * 
 * @param A: input, input matrix A, shape (M, K), uint8, row-major
 * @param B: input, input matrix B, shape (K, N), uint8, row-major
 * @param C: output, output matrix C, shape (M, N), uint32, row-major
 * @param lut: input, lookup table for approximate multiplication, shape (LUT_ROW_NUM, LUT_COL_NUM), uint16
 * @param M: input, row number of A
 * @param K: input, col number of A and row number of B
 * @param N: input, col number of B
 * 
 * Template parameters:
 * BLOCK_SIZE_M: height of block of C that each thread block calculate
 * BLOCK_SIZE_K: width of block of A that each thread block load into shared memory
 * BLOCK_SIZE_N: width of block of C that each thread block calculate
 * THREAD_SIZE_X: width of block of C that each thread calculate
 * THREAD_SIZE_Y: height of block of C that each thread calculate
 * LOAD_ELEM_NUM: number of elements loaded at a time, 1, 2 or 4
 * 
 */
template <
    const int BLOCK_SIZE_M,
    const int BLOCK_SIZE_K,
    const int BLOCK_SIZE_N,
    const int THREAD_SIZE_X,
    const int THREAD_SIZE_Y,
    const int LOAD_ELEM_NUM
    > 
__global__ void ApproxGemmForwardKernel( 
    uint8_t * __restrict__ A,
    uint8_t * __restrict__ B,
    uint32_t * __restrict__ C,
#if QUANTIZATION_BIT < 8
    uint16_t * __restrict__ lut,
#elif QUANTIZATION_BIT == 8
    cudaTextureObject_t lutTexture,
#else
#error "Unsupported QUANTIZATION_BIT"
#endif
    const int M,
    const int K,
    const int N) {

    static_assert(LOAD_ELEM_NUM == 1 || LOAD_ELEM_NUM == 2 || LOAD_ELEM_NUM == 4, "LOAD_ELEM_NUM should be 1, 2 or 4");
    using LOAD_UINT8_T = typename LoadType<LOAD_ELEM_NUM>::uchar_type;
    using LOAD_UINT32_T = typename LoadType<LOAD_ELEM_NUM>::uint_type;
    using LOAD_UINT16_T = typename LoadType<4>::ushort_type;
    
    // block & thread index
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // thread id in current block
    const int THREAD_X_NUM = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int THREAD_Y_NUM = BLOCK_SIZE_M / THREAD_SIZE_Y;
    static_assert(BLOCK_SIZE_N % THREAD_SIZE_X == 0, "BLOCK_SIZE_N should be divisible by THREAD_SIZE_X");
    static_assert(BLOCK_SIZE_M % THREAD_SIZE_Y == 0, "BLOCK_SIZE_M should be divisible by THREAD_SIZE_Y");
    const int tid = ty * THREAD_X_NUM + tx;

    // total number of threads in a block
    const int THREAD_NUM = THREAD_X_NUM * THREAD_Y_NUM;

#if QUANTIZATION_BIT < 8
    // prepare lookup table using shared memory
    __shared__ uint16_t lookupTable[LUT_ELEM_NUM];
    static_assert(LUT_ELEM_NUM % THREAD_NUM == 0, "LUT_ELEM_NUM should be divisible by THREAD_NUM");
    const int LOAD_LUT_ELEM_PER_THREAD = LUT_ELEM_NUM / THREAD_NUM;
    // constexpr int LOAD_LUT_ELEM_PER_THREAD = LUT_ELEM_NUM / THREAD_NUM;
    static_assert(LOAD_LUT_ELEM_PER_THREAD % 4 == 0, "LOAD_LUT_ELEM_PER_THREAD should be divisible by 4");
    const int LOAD_LUT_START = tid * LOAD_LUT_ELEM_PER_THREAD;
    // if constexpr (LOAD_LUT_ELEM_PER_THREAD % 4 == 0) {
        #pragma unroll
        for (int i = LOAD_LUT_START; i < LOAD_LUT_START + LOAD_LUT_ELEM_PER_THREAD; i += 4)
            FETCH<uint16_t, LOAD_UINT16_T>(lookupTable[i]) = FETCH<uint16_t, LOAD_UINT16_T>(lut[i]);
    // }
    // else if constexpr (LOAD_LUT_ELEM_PER_THREAD % 2 == 0) {
    //     #pragma unroll
    //     for (int i = LOAD_LUT_START; i < LOAD_LUT_START + LOAD_LUT_ELEM_PER_THREAD; i += 2)
    //         FETCH<uint16_t, ushort2>(lookupTable[i]) = FETCH<uint16_t, ushort2>(lut[i]);
    // }
    // else {
    //     #pragma unroll
    //     for (int i = LOAD_LUT_START; i < LOAD_LUT_START + LOAD_LUT_ELEM_PER_THREAD; ++i)
    //         lookupTable[i] = lut[i];
    // }
#endif

    // shared memory
    alignas(4) __shared__ uint8_t ATs[BLOCK_SIZE_K][BLOCK_SIZE_M]; // A->ATs: transpose for better memory access
    alignas(4) __shared__ uint8_t Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];  // B->Bs: no transpose

    // registers
    register uint32_t accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
    register uint8_t AReg[THREAD_SIZE_Y];
    register uint8_t BReg[THREAD_SIZE_X];

    // thread number in one row/col in the shared memory
    const int AT_THREAD_NUM_PER_COL = BLOCK_SIZE_K / LOAD_ELEM_NUM;
    static_assert(BLOCK_SIZE_K % LOAD_ELEM_NUM == 0, "BLOCK_SIZE_K should be divisible by LOAD_ELEM_NUM");
    const int B_THREAD_NUM_PER_ROW = BLOCK_SIZE_N / LOAD_ELEM_NUM;
    static_assert(BLOCK_SIZE_N % LOAD_ELEM_NUM == 0, "BLOCK_SIZE_N should be divisible by LOAD_ELEM_NUM");

    // (row, col) in the shared memory, loaded by this thread
    const int AT_ROW = tid % AT_THREAD_NUM_PER_COL * LOAD_ELEM_NUM;
    const int AT_COL = tid / AT_THREAD_NUM_PER_COL;
    const int B_ROW = tid / B_THREAD_NUM_PER_ROW;
    const int B_COL = tid % B_THREAD_NUM_PER_ROW * LOAD_ELEM_NUM;

    // row/col stride that thread uses to load multiple rows/cols
    const int AT_COL_STRIDE = THREAD_NUM / AT_THREAD_NUM_PER_COL;
    static_assert(THREAD_NUM % AT_THREAD_NUM_PER_COL == 0, "THREAD_NUM should be divisible by AT_THREAD_NUM_PER_COL");
    const int B_ROW_STRIDE = THREAD_NUM / B_THREAD_NUM_PER_ROW;
    static_assert(THREAD_NUM % B_THREAD_NUM_PER_ROW == 0, "THREAD_NUM should be divisible by B_THREAD_NUM_PER_ROW");

    // initialize the left-top address of A and B in this block
    A = &A[OFFSET(BLOCK_SIZE_M * by, 0, K)];
    B = &B[OFFSET(0, BLOCK_SIZE_N * bx, N)];

    // main loop
    for (int k = 0; k < K; k += BLOCK_SIZE_K) {
        // load data from global memory to shared memory
        #pragma unroll
        for (int ATColOffset = 0; ATColOffset < BLOCK_SIZE_M; ATColOffset += AT_COL_STRIDE) {
            if (BLOCK_SIZE_M * by + AT_COL + ATColOffset < M && AT_ROW + k < K) { // check boundary
                LOAD_UINT8_T tmp = FETCH<uint8_t, LOAD_UINT8_T>(A[OFFSET(AT_COL + ATColOffset, AT_ROW + k, K)]); // transpose
                ATs[AT_ROW][AT_COL + ATColOffset] = tmp.x;
                if constexpr (LOAD_ELEM_NUM >= 2)
                    ATs[AT_ROW + 1][AT_COL + ATColOffset] = tmp.y;
                if constexpr (LOAD_ELEM_NUM == 4) {
                    ATs[AT_ROW + 2][AT_COL + ATColOffset] = tmp.z;
                    ATs[AT_ROW + 3][AT_COL + ATColOffset] = tmp.w;
                }
            }
            else {
                ATs[AT_ROW][AT_COL + ATColOffset] = LUT_OUT_RANGE;
                if constexpr (LOAD_ELEM_NUM >= 2)
                    ATs[AT_ROW + 1][AT_COL + ATColOffset] = LUT_OUT_RANGE;
                if constexpr (LOAD_ELEM_NUM == 4) {
                    ATs[AT_ROW + 2][AT_COL + ATColOffset] = LUT_OUT_RANGE;
                    ATs[AT_ROW + 3][AT_COL + ATColOffset] = LUT_OUT_RANGE;
                }
            }
        }
        #pragma unroll
        for (int BRowOffset = 0; BRowOffset < BLOCK_SIZE_K; BRowOffset += B_ROW_STRIDE) {
            if (B_ROW + BRowOffset + k < K && BLOCK_SIZE_N * bx + B_COL < N) { // check boundary
                FETCH<uint8_t, LOAD_UINT8_T>(Bs[B_ROW + BRowOffset][B_COL]) = FETCH<uint8_t, LOAD_UINT8_T>(B[OFFSET(B_ROW + BRowOffset + k, B_COL, N)]);
            }
            else {
                Bs[B_ROW + BRowOffset][B_COL] = LUT_OUT_RANGE;
                if constexpr (LOAD_ELEM_NUM >= 2)
                    Bs[B_ROW + BRowOffset][B_COL + 1] = LUT_OUT_RANGE;
                if constexpr (LOAD_ELEM_NUM == 4) {
                    Bs[B_ROW + BRowOffset][B_COL + 2] = LUT_OUT_RANGE;
                    Bs[B_ROW + BRowOffset][B_COL + 3] = LUT_OUT_RANGE;
                }
            }
        }
    
        // ensure all threads have loaded the data from global memory to shared memory
        __syncthreads();
        
        // compute C
        #pragma unroll
        for (int kk = 0; kk < BLOCK_SIZE_K; ++kk) {
            // load A from shared memory to register
            #pragma unroll
            for (int threadY = 0; threadY < THREAD_SIZE_Y; threadY += 4)
                FETCH<uint8_t, uchar4>(AReg[threadY]) = FETCH<uint8_t, uchar4>(ATs[kk][THREAD_SIZE_Y * ty + threadY]);
            // lead B from shared memory to register
            #pragma unroll
            for (int threadX = 0; threadX < THREAD_SIZE_X; threadX += 4)
                FETCH<uint8_t, uchar4>(BReg[threadX]) = FETCH<uint8_t, uchar4>(Bs[kk][THREAD_SIZE_X * tx + threadX]);
            // MMA
            #pragma unroll
            for (int threadY = 0; threadY < THREAD_SIZE_Y; ++threadY) {
                uint8_t aVal = AReg[threadY];
                #pragma unroll
                for (int threadX = 0; threadX < THREAD_SIZE_X; ++threadX) {
                    uint8_t bVal = BReg[threadX];
#if QUANTIZATION_BIT < 8                    
                    accum[threadY][threadX] += lookupTable[aVal * LUT_COL_NUM + bVal];
#elif QUANTIZATION_BIT == 8
                    accum[threadY][threadX] += tex1Dfetch<uint16_t>(lutTexture, aVal * LUT_COL_NUM + bVal);
#else
#error "Unsupported QUANTIZATION_BIT"
#endif
                }
            }
        }

        // ensure all threads have finished the computation
        __syncthreads();
    }

    // store C
    const int rowBase = BLOCK_SIZE_M * by + THREAD_SIZE_Y * ty;
    const int colBase = BLOCK_SIZE_N * bx + THREAD_SIZE_X * tx;
    C = &C[OFFSET(rowBase, colBase, N)];
    #pragma unroll
    for (int threadY = 0; threadY < THREAD_SIZE_Y; ++threadY) {
        if (rowBase + threadY < M) {
            #pragma unroll
            for (int threadX = 0; threadX < THREAD_SIZE_X; threadX += LOAD_ELEM_NUM) {
                if (colBase + threadX < N) { // check boundary
                    FETCH<uint32_t, LOAD_UINT32_T>(C[OFFSET(threadY, threadX, N)]) = FETCH<uint32_t, LOAD_UINT32_T>(accum[threadY][threadX]);
                }
            }
        }
    }
}


/**
 * Compute the forward pass of the approximate gemm operation
 * 
 * @param a_tensor: input, the input tensor A, uint8.
 * @param b_tensor: input, the input tensor B, uint8.
 * @param c_tensor: output, the output tensor C, uint32.
 * @param lut_appmult_tensor: input, the lookup table tensor for approximate multiplication, uint16.
 * 
 */
void approx_gemm_forward_gpu(const at::Tensor &a_tensor, const at::Tensor &b_tensor, at::Tensor &c_tensor, const at::Tensor &lut_appmult_tensor) {
    // check input
    CHECK_INPUT(a_tensor, torch::kUInt8, "a_tensor");
    CHECK_INPUT(b_tensor, torch::kUInt8, "b_tensor");
    CHECK_INPUT(c_tensor, torch::kUInt32, "c_tensor");
    CHECK_INPUT(lut_appmult_tensor, torch::kUInt16, "lut_appmult_tensor");
    int M = a_tensor.size(0);
    int N = b_tensor.size(1);
    int K = a_tensor.size(1);
    TORCH_CHECK(K == b_tensor.size(0), "b_tensor's shape should be ", K, " x ", N);
    TORCH_CHECK(M == c_tensor.size(0) && N == c_tensor.size(1), "c_tensor's shape should be ", M, " x ", N);
    TORCH_CHECK(lut_appmult_tensor.size(0) == LUT_ELEM_NUM, "lut_appmult_tensor size should be ", LUT_ELEM_NUM, ", please check the value of QUANTIZATION_BIT");

    // set & check device id
    int deviceId = a_tensor.get_device();
    cudaSetDevice(deviceId);
    TORCH_CHECK(deviceId == b_tensor.get_device(), "device id not match");
    TORCH_CHECK(deviceId == c_tensor.get_device(), "device id not match");
    TORCH_CHECK(deviceId == lut_appmult_tensor.get_device(), "device id not match");

    // get data ptr
    uint8_t *a = a_tensor.data_ptr<uint8_t>();
    uint8_t *b = b_tensor.data_ptr<uint8_t>();
    uint32_t *c = c_tensor.data_ptr<uint32_t>();
    uint16_t *lut = lut_appmult_tensor.data_ptr<uint16_t>();

#if QUANTIZATION_BIT == 8
    // create texture object
    cudaTextureObject_t lutTexture = CreateLUTTextureObject<uint16_t>(lut, LUT_ELEM_NUM);
#endif
    
    // kernel parameters
    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_K = 8;
    const int BLOCK_SIZE_N = 128;
    const int THREAD_SIZE_X = 8;
    const int THREAD_SIZE_Y = 8;
    static_assert(THREAD_SIZE_X == THREAD_SIZE_Y, "THREAD_SIZE_X should be equal to THREAD_SIZE_Y");
    static_assert((THREAD_SIZE_X % 4) == 0, "THREAD_SIZE_X should be divisible by 4");
    static_assert((THREAD_SIZE_Y % 4) == 0, "THREAD_SIZE_Y should be divisible by 4");
    dim3 dimGrid(CEIL_DIV(N, BLOCK_SIZE_N), CEIL_DIV(M, BLOCK_SIZE_M));
    dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);

    // launch kernel
#if QUANTIZATION_BIT < 8
    if ((M & 3) == 0 && (K & 3) == 0 && (N & 3) == 0)
        ApproxGemmForwardKernel<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_X, THREAD_SIZE_Y, 4> <<<dimGrid, dimBlock>>> (a, b, c, lut, M, K, N);
    else if ((M & 1) == 0 && (K & 1) == 0 && (N & 1) == 0)
        ApproxGemmForwardKernel<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_X, THREAD_SIZE_Y, 2> <<<dimGrid, dimBlock>>> (a, b, c, lut, M, K, N);
    else
        ApproxGemmForwardKernel<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_X, THREAD_SIZE_Y, 1> <<<dimGrid, dimBlock>>> (a, b, c, lut, M, K, N);
#elif QUANTIZATION_BIT == 8
    if ((M & 3) == 0 && (K & 3) == 0 && (N & 3) == 0)
        ApproxGemmForwardKernel<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_X, THREAD_SIZE_Y, 4> <<<dimGrid, dimBlock>>> (a, b, c, lutTexture, M, K, N);
    else if ((M & 1) == 0 && (K & 1) == 0 && (N & 1) == 0)
        ApproxGemmForwardKernel<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_X, THREAD_SIZE_Y, 2> <<<dimGrid, dimBlock>>> (a, b, c, lutTexture, M, K, N);
    else
        ApproxGemmForwardKernel<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_X, THREAD_SIZE_Y, 1> <<<dimGrid, dimBlock>>> (a, b, c, lutTexture, M, K, N);
#else
#error "Unsupported QUANTIZATION_BIT"
#endif

    // check error
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA Error: %s\n", cudaGetErrorString(err));

#if QUANTIZATION_BIT == 8
    // Destroy texture object after kernel execution
    err = cudaDestroyTextureObject(lutTexture);
    if (err != cudaSuccess)
        printf("CUDA Error (Texture Destruction): %s\n", cudaGetErrorString(err));
#endif
}


/**
 * Compute the forward pass of the accurate single-precision gemm operation
 * 
 * In this kernel, there are CEIL_DIV(N, BLOCK_SIZE_N) * CEIL_DIV(M, BLOCK_SIZE_M) thread blocks
 * Each block computes a sub-matrix of C, which has a size of BLOCK_SIZE_M x BLOCK_SIZE_N
 * C_BMxBN = matmul(A_BMxK, B_KxBN) = \sum_{k=0}^{K/BK-1} matmul(A_BMxBK_k, B_BKxBN_k)
 * 
 * @param A: input, input matrix A, shape (M, K), float32, row-major
 * @param B: input, input matrix B, shape (K, N), float32, row-major
 * @param C: output, output matrix C, shape (M, N), float32, row-major
 * @param M: input, row number of A
 * @param K: input, col number of A and row number of B
 * @param N: input, col number of B
 * 
 * Template parameters:
 * BLOCK_SIZE_M: height of block of C that each thread block calculate
 * BLOCK_SIZE_K: width of block of A that each thread block load into shared memory
 * BLOCK_SIZE_N: width of block of C that each thread block calculate
 * THREAD_SIZE_X: width of block of C that each thread calculate
 * THREAD_SIZE_Y: height of block of C that each thread calculate
 * LOAD_ELEM_NUM: number of elements loaded at a time, 1, 2 or 4
 * 
 */
template <
    const int BLOCK_SIZE_M,
    const int BLOCK_SIZE_K,
    const int BLOCK_SIZE_N,
    const int THREAD_SIZE_X,
    const int THREAD_SIZE_Y,
    const int LOAD_ELEM_NUM
    > 
__global__ void AccGemmForwardKernelFP32( 
    float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C,
    const int M,
    const int K,
    const int N) {

    static_assert(LOAD_ELEM_NUM == 1 || LOAD_ELEM_NUM == 2 || LOAD_ELEM_NUM == 4, "LOAD_ELEM_NUM should be 1, 2 or 4");
    using LOAD_FLOAT32_T = typename LoadType<LOAD_ELEM_NUM>::float_type;
    
    // block & thread index
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // thread id in current block
    const int THREAD_X_NUM = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int THREAD_Y_NUM = BLOCK_SIZE_M / THREAD_SIZE_Y;
    static_assert(BLOCK_SIZE_N % THREAD_SIZE_X == 0, "BLOCK_SIZE_N should be divisible by THREAD_SIZE_X");
    static_assert(BLOCK_SIZE_M % THREAD_SIZE_Y == 0, "BLOCK_SIZE_M should be divisible by THREAD_SIZE_Y");
    const int tid = ty * THREAD_X_NUM + tx;

    // total number of threads in a block
    const int THREAD_NUM = THREAD_X_NUM * THREAD_Y_NUM;

    // shared memory
    alignas(4) __shared__ float ATs[BLOCK_SIZE_K][BLOCK_SIZE_M]; // A->ATs: transpose for better memory access
    alignas(4) __shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];  // B->Bs: no transpose

    // registers
    register float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0.0f};
    register float AReg[THREAD_SIZE_Y];
    register float BReg[THREAD_SIZE_X];

    // thread number in one row/col in the shared memory
    const int AT_THREAD_NUM_PER_COL = BLOCK_SIZE_K / LOAD_ELEM_NUM;
    static_assert(BLOCK_SIZE_K % LOAD_ELEM_NUM == 0, "BLOCK_SIZE_K should be divisible by LOAD_ELEM_NUM");
    const int B_THREAD_NUM_PER_ROW = BLOCK_SIZE_N / LOAD_ELEM_NUM;
    static_assert(BLOCK_SIZE_N % LOAD_ELEM_NUM == 0, "BLOCK_SIZE_N should be divisible by LOAD_ELEM_NUM");

    // (row, col) in the shared memory, loaded by this thread
    const int AT_ROW = tid % AT_THREAD_NUM_PER_COL * LOAD_ELEM_NUM;
    const int AT_COL = tid / AT_THREAD_NUM_PER_COL;
    const int B_ROW = tid / B_THREAD_NUM_PER_ROW;
    const int B_COL = tid % B_THREAD_NUM_PER_ROW * LOAD_ELEM_NUM;

    // row/col stride that thread uses to load multiple rows/cols
    const int AT_COL_STRIDE = THREAD_NUM / AT_THREAD_NUM_PER_COL;
    static_assert(THREAD_NUM % AT_THREAD_NUM_PER_COL == 0, "THREAD_NUM should be divisible by AT_THREAD_NUM_PER_COL");
    const int B_ROW_STRIDE = THREAD_NUM / B_THREAD_NUM_PER_ROW;
    static_assert(THREAD_NUM % B_THREAD_NUM_PER_ROW == 0, "THREAD_NUM should be divisible by B_THREAD_NUM_PER_ROW");

    // initialize the left-top address of A and B in this block
    A = &A[OFFSET(BLOCK_SIZE_M * by, 0, K)];
    B = &B[OFFSET(0, BLOCK_SIZE_N * bx, N)];

    // main loop
    for (int k = 0; k < K; k += BLOCK_SIZE_K) {
        // load data from global memory to shared memory
        #pragma unroll
        for (int ATColOffset = 0; ATColOffset < BLOCK_SIZE_M; ATColOffset += AT_COL_STRIDE) {
            if (BLOCK_SIZE_M * by + AT_COL + ATColOffset < M && AT_ROW + k < K) { // check boundary
                LOAD_FLOAT32_T tmp = FETCH<float, LOAD_FLOAT32_T>(A[OFFSET(AT_COL + ATColOffset, AT_ROW + k, K)]); // transpose
                ATs[AT_ROW][AT_COL + ATColOffset] = tmp.x;
                if constexpr (LOAD_ELEM_NUM >= 2)
                    ATs[AT_ROW + 1][AT_COL + ATColOffset] = tmp.y;
                if constexpr (LOAD_ELEM_NUM == 4) {
                    ATs[AT_ROW + 2][AT_COL + ATColOffset] = tmp.z;
                    ATs[AT_ROW + 3][AT_COL + ATColOffset] = tmp.w;
                }
            }
            else {
                ATs[AT_ROW][AT_COL + ATColOffset] = 0.0f;
                if constexpr (LOAD_ELEM_NUM >= 2)
                    ATs[AT_ROW + 1][AT_COL + ATColOffset] = 0.0f;
                if constexpr (LOAD_ELEM_NUM == 4) {
                    ATs[AT_ROW + 2][AT_COL + ATColOffset] = 0.0f;
                    ATs[AT_ROW + 3][AT_COL + ATColOffset] = 0.0f;
                }
            }
        }
        #pragma unroll
        for (int BRowOffset = 0; BRowOffset < BLOCK_SIZE_K; BRowOffset += B_ROW_STRIDE) {
            if (B_ROW + BRowOffset + k < K && BLOCK_SIZE_N * bx + B_COL < N) { // check boundary
                FETCH<float, LOAD_FLOAT32_T>(Bs[B_ROW + BRowOffset][B_COL]) = FETCH<float, LOAD_FLOAT32_T>(B[OFFSET(B_ROW + BRowOffset + k, B_COL, N)]);
            }
            else {
                Bs[B_ROW + BRowOffset][B_COL] = 0.0f;
                if constexpr (LOAD_ELEM_NUM >= 2)
                    Bs[B_ROW + BRowOffset][B_COL + 1] = 0.0f;
                if constexpr (LOAD_ELEM_NUM == 4) {
                    Bs[B_ROW + BRowOffset][B_COL + 2] = 0.0f;
                    Bs[B_ROW + BRowOffset][B_COL + 3] = 0.0f;
                }
            }
        }
    
        // ensure all threads have loaded the data from global memory to shared memory
        __syncthreads();
        
        // compute C
        #pragma unroll
        for (int kk = 0; kk < BLOCK_SIZE_K; ++kk) {
            // load A from shared memory to register
            #pragma unroll
            for (int threadY = 0; threadY < THREAD_SIZE_Y; threadY += 4)
                FETCH<float, float4>(AReg[threadY]) = FETCH<float, float4>(ATs[kk][THREAD_SIZE_Y * ty + threadY]);
            // load B from shared memory to register
            #pragma unroll
            for (int threadX = 0; threadX < THREAD_SIZE_X; threadX += 4)
                FETCH<float, float4>(BReg[threadX]) = FETCH<float, float4>(Bs[kk][THREAD_SIZE_X * tx + threadX]);
            // MMA
            #pragma unroll
            for (int threadY = 0; threadY < THREAD_SIZE_Y; ++threadY) {
                float aVal = AReg[threadY];
                #pragma unroll
                for (int threadX = 0; threadX < THREAD_SIZE_X; ++threadX) {
                    float bVal = BReg[threadX];
                    accum[threadY][threadX] += aVal * bVal;
                }
            }
        }

        // ensure all threads have finished the computation
        __syncthreads();
    }

    // store C
    const int rowBase = BLOCK_SIZE_M * by + THREAD_SIZE_Y * ty;
    const int colBase = BLOCK_SIZE_N * bx + THREAD_SIZE_X * tx;
    C = &C[OFFSET(rowBase, colBase, N)];
    #pragma unroll
    for (int threadY = 0; threadY < THREAD_SIZE_Y; ++threadY) {
        if (rowBase + threadY < M) {
            #pragma unroll
            for (int threadX = 0; threadX < THREAD_SIZE_X; threadX += LOAD_ELEM_NUM) {
                if (colBase + threadX < N) { // check boundary
                    FETCH<float, LOAD_FLOAT32_T>(C[OFFSET(threadY, threadX, N)]) = FETCH<float, LOAD_FLOAT32_T>(accum[threadY][threadX]);
                }
            }
        }
    }
}


/**
 * Compute the forward pass of the accurate gemm operation
 * 
 * @param a_tensor: input, the input tensor A, float32.
 * @param b_tensor: input, the input tensor B, float32.
 * @param c_tensor: output, the output tensor C, float32.
 * 
 */
void acc_gemm_forward_fp32_gpu(const at::Tensor &a_tensor, const at::Tensor &b_tensor, at::Tensor &c_tensor) {
    // check input
    CHECK_INPUT(a_tensor, torch::kFloat32, "a_tensor");
    CHECK_INPUT(b_tensor, torch::kFloat32, "b_tensor");
    CHECK_INPUT(c_tensor, torch::kFloat32, "c_tensor");
    int M = a_tensor.size(0);
    int N = b_tensor.size(1);
    int K = a_tensor.size(1);
    TORCH_CHECK(K == b_tensor.size(0), "b_tensor's shape should be ", K, " x ", N);
    TORCH_CHECK(M == c_tensor.size(0) && N == c_tensor.size(1), "c_tensor's shape should be ", M, " x ", N);

    // set & check device id
    int deviceId = a_tensor.get_device();
    cudaSetDevice(deviceId);
    TORCH_CHECK(deviceId == b_tensor.get_device(), "device id not match");
    TORCH_CHECK(deviceId == c_tensor.get_device(), "device id not match");

    // get data ptr
    float *a = a_tensor.data_ptr<float>();
    float *b = b_tensor.data_ptr<float>();
    float *c = c_tensor.data_ptr<float>();

    // // perform gemm using cublas
    // // create cublas handle
    // cublasHandle_t handle;
    // cublasCreate(&handle);
    // // - No transpose for a_tensor -> cuBLAS uses CUBLAS_OP_N (NoTrans)
    // // - No transpose for b_tensor -> cuBLAS uses CUBLAS_OP_N (NoTrans)
    // // - The matrix multiplication is c = a * b, which is consistent with torch.matmul(a, b)
    // const float alpha = 1.0f;
    // const float beta = 0.0f;
    // cublasStatus_t status = cublasSgemm(handle, 
    //                                     CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose for row-major matrices
    //                                     N, M, K,                   // Dimensions as expected for row-major
    //                                     &alpha, 
    //                                     b, N,                      // B is treated normally
    //                                     a, K,                      // A is treated normally
    //                                     &beta, c, N);              // C is computed normally
    // // destroy the handle
    // cublasDestroy(handle);
    // if (status != CUBLAS_STATUS_SUCCESS)
    //     printf("cuBLAS Error\n");

    // perform gemm using custom kernel
    // kernel parameters
    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_K = 8;
    const int BLOCK_SIZE_N = 128;
    const int THREAD_SIZE_X = 8;
    const int THREAD_SIZE_Y = 8;
    static_assert(THREAD_SIZE_X == THREAD_SIZE_Y, "THREAD_SIZE_X should be equal to THREAD_SIZE_Y");
    static_assert((THREAD_SIZE_X % 4) == 0, "THREAD_SIZE_X should be divisible by 4");
    static_assert((THREAD_SIZE_Y % 4) == 0, "THREAD_SIZE_Y should be divisible by 4");
    dim3 dimGrid(CEIL_DIV(N, BLOCK_SIZE_N), CEIL_DIV(M, BLOCK_SIZE_M));
    dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);

    // launch kernel
    if ((M & 3) == 0 && (K & 3) == 0 && (N & 3) == 0)
        AccGemmForwardKernelFP32<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_X, THREAD_SIZE_Y, 4> <<<dimGrid, dimBlock>>> (a, b, c, M, K, N);
    else if ((M & 1) == 0 && (K & 1) == 0 && (N & 1) == 0)
        AccGemmForwardKernelFP32<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_X, THREAD_SIZE_Y, 2> <<<dimGrid, dimBlock>>> (a, b, c, M, K, N);
    else
        AccGemmForwardKernelFP32<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_X, THREAD_SIZE_Y, 1> <<<dimGrid, dimBlock>>> (a, b, c, M, K, N);

    // check error
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
}
