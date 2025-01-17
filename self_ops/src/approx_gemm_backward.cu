#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cassert>
#include <iostream>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "cuda_helper.h"
#include "approx_mult.h"


template <
    const int BLOCK_SIZE_M,
    const int BLOCK_SIZE_K,
    const int BLOCK_SIZE_N,
    const int THREAD_SIZE_X,
    const int THREAD_SIZE_Y,
    const int LOAD_ELEM_NUM
    >
__global__ void ApproxGemmBackwardKernelForGA(
    float * __restrict__ GC,
    uint8_t * __restrict__ A,
    uint8_t * __restrict__ B,
    float * __restrict__ GA,
#if QUANTIZATION_BIT < 8
    int16_t * __restrict__ lutGradA,
#elif QUANTIZATION_BIT == 8
    cudaTextureObject_t lutGradATexture,
#else
#error "Unsupported QUANTIZATION_BIT"
#endif
    const int MP,
    const int KP,
    const int NP) {

    static_assert(LOAD_ELEM_NUM == 1 || LOAD_ELEM_NUM == 2 || LOAD_ELEM_NUM == 4, "LOAD_ELEM_NUM should be 1, 2 or 4");
    using LOAD_UINT8_T = typename LoadType<LOAD_ELEM_NUM>::uchar_type;
    using LOAD_FLOAT_T = typename LoadType<LOAD_ELEM_NUM>::float_type;
    using LOAD_INT16_T = typename LoadType<4>::short_type;

    // block & thread index
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // thread id in current Block
    const int THREAD_X_NUM = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int THREAD_Y_NUM = BLOCK_SIZE_M / THREAD_SIZE_Y;
    static_assert(BLOCK_SIZE_N % THREAD_SIZE_X == 0, "BLOCK_SIZE_N should be divisible by THREAD_SIZE_X");
    static_assert(BLOCK_SIZE_M % THREAD_SIZE_Y == 0, "BLOCK_SIZE_M should be divisible by THREAD_SIZE_Y");
    const int tid = ty * THREAD_X_NUM + tx;

    // total number of threads in a block
    const int THREAD_NUM = THREAD_X_NUM * THREAD_Y_NUM;

#if QUANTIZATION_BIT < 8
    // prepare lookup table using shared memory
    // approximate multiplication: h(a, b)
    // lookupTable[a][b] represents the gradient of h(a, b) with respect to a
    __shared__ int16_t lookupTable[LUT_ELEM_NUM];
    static_assert(LUT_ELEM_NUM % THREAD_NUM == 0, "LUT_ELEM_NUM should be divisible by THREAD_NUM");
    const int LOAD_LUT_ELEM_PER_THREAD = LUT_ELEM_NUM / THREAD_NUM;
    // constexpr int LOAD_LUT_ELEM_PER_THREAD = LUT_ELEM_NUM / THREAD_NUM;
    static_assert(LOAD_LUT_ELEM_PER_THREAD % 4 == 0, "LOAD_LUT_ELEM_PER_THREAD should be divisible by 4");
    const int LOAD_LUT_START = tid * LOAD_LUT_ELEM_PER_THREAD;
    // if constexpr (LOAD_LUT_ELEM_PER_THREAD % 4 == 0) {
        #pragma unroll
        for (int i = LOAD_LUT_START; i < LOAD_LUT_START + LOAD_LUT_ELEM_PER_THREAD; i += 4)
            FETCH<int16_t, LOAD_INT16_T>(lookupTable[i]) = FETCH<int16_t, LOAD_INT16_T>(lutGradA[i]);
    // }
    // else if constexpr (LOAD_LUT_ELEM_PER_THREAD % 2 == 0) {
    //     #pragma unroll
    //     for (int i = LOAD_LUT_START; i < LOAD_LUT_START + LOAD_LUT_ELEM_PER_THREAD; i += 2)
    //         FETCH<int16_t, short2>(lookupTable[i]) = FETCH<int16_t, short2>(lutGradA[i]);
    // }
    // else {
    //     #pragma unroll
    //     for (int i = LOAD_LUT_START; i < LOAD_LUT_START + LOAD_LUT_ELEM_PER_THREAD; ++i)
    //         lookupTable[i] = lutGradA[i];
    // }
#endif    

    // shared memory
    alignas(16) __shared__ float GCTs[BLOCK_SIZE_K][BLOCK_SIZE_M]; // GC->GCTs: transpose for better memory access, aligned to 16 bytes
    alignas(4) __shared__ uint8_t BTs[BLOCK_SIZE_K][BLOCK_SIZE_N]; // B->BTs: transpose for better memory access, aligned to 4 bytes
    __shared__ uint8_t As[BLOCK_SIZE_M][BLOCK_SIZE_N]; // As's row = GCTs's column, As's column = BTs's row

    // register for GA
    register float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0.0f};

    // registers for GCT and BT
    register float GCReg[THREAD_SIZE_Y];
    register uint8_t BReg[THREAD_SIZE_X];
    register uint8_t AReg[THREAD_SIZE_Y][THREAD_SIZE_X];

    // thread number in one row/col in the shared memory
    const int GCT_THREAD_NUM_PER_COL = BLOCK_SIZE_K / LOAD_ELEM_NUM;
    const int BT_THREAD_NUM_PER_COL = BLOCK_SIZE_K / LOAD_ELEM_NUM;
    static_assert(BLOCK_SIZE_K % LOAD_ELEM_NUM == 0);
    const int A_THREAD_NUM_PER_ROW = BLOCK_SIZE_N / LOAD_ELEM_NUM;
    static_assert(BLOCK_SIZE_N % LOAD_ELEM_NUM == 0);

    // (row, col) in the shared memory, loaded by this thread
    const int GCT_ROW = tid % GCT_THREAD_NUM_PER_COL * LOAD_ELEM_NUM;
    const int GCT_COL = tid / GCT_THREAD_NUM_PER_COL;
    const int BT_ROW = tid % BT_THREAD_NUM_PER_COL * LOAD_ELEM_NUM;
    const int BT_COL = tid / BT_THREAD_NUM_PER_COL;
    const int A_ROW = tid / A_THREAD_NUM_PER_ROW;
    const int A_COL = tid % A_THREAD_NUM_PER_ROW * LOAD_ELEM_NUM;

    // row/col stride that thread uses to load multiple rows/cols
    const int GCT_COL_STRIDE = THREAD_NUM / GCT_THREAD_NUM_PER_COL;
    static_assert(THREAD_NUM % GCT_THREAD_NUM_PER_COL == 0, "THREAD_NUM should be divisible by GCT_THREAD_NUM_PER_COL");
    const int BT_COL_STRIDE = THREAD_NUM / BT_THREAD_NUM_PER_COL;
    static_assert(THREAD_NUM % BT_THREAD_NUM_PER_COL == 0, "THREAD_NUM should be divisible by BT_THREAD_NUM_PER_COL");
    const int A_ROW_STRIDE = THREAD_NUM / A_THREAD_NUM_PER_ROW;

    // initialize the left-top addresses in this block
    GC = &GC[OFFSET(BLOCK_SIZE_M * by, 0, KP)];
    B = &B[OFFSET(BLOCK_SIZE_N * bx, 0, KP)]; 
    A = &A[OFFSET(BLOCK_SIZE_M * by, BLOCK_SIZE_N * bx, NP)];

    // load A: global memory -> shared memory -> register
    #pragma unroll
    for (int ARowOffset = 0; ARowOffset < BLOCK_SIZE_M; ARowOffset += A_ROW_STRIDE) { // check boundary
        if (BLOCK_SIZE_M * by + A_ROW + ARowOffset < MP && BLOCK_SIZE_N * bx + A_COL < NP)
            FETCH<uint8_t, LOAD_UINT8_T>(As[A_ROW + ARowOffset][A_COL]) = FETCH<uint8_t, LOAD_UINT8_T>(A[OFFSET(A_ROW + ARowOffset, A_COL, NP)]);
        else {
            As[A_ROW + ARowOffset][A_COL] = LUT_OUT_RANGE;
            if constexpr (LOAD_ELEM_NUM >= 2)
                As[A_ROW + ARowOffset][A_COL + 1] = LUT_OUT_RANGE;
            if constexpr (LOAD_ELEM_NUM == 4) {
                As[A_ROW + ARowOffset][A_COL + 2] = LUT_OUT_RANGE;
                As[A_ROW + ARowOffset][A_COL + 3] = LUT_OUT_RANGE;
            }
        }
    }
    __syncthreads();

    #pragma unroll
    for (int threadY = 0; threadY < THREAD_SIZE_Y; ++threadY) {
        #pragma unroll
        for (int threadX = 0; threadX < THREAD_SIZE_X; threadX += 4)
            FETCH<uint8_t, uchar4>(AReg[threadY][threadX]) = FETCH<uint8_t, uchar4>(As[THREAD_SIZE_Y * ty + threadY][THREAD_SIZE_X * tx + threadX]);
    }

    // main loop
    for (int k = 0; k < KP; k += BLOCK_SIZE_K) {
        // load data from global memory to shared memory
        #pragma unroll
        for (int GCTColOffset = 0; GCTColOffset < BLOCK_SIZE_M; GCTColOffset += GCT_COL_STRIDE) {
            if (BLOCK_SIZE_M * by + GCT_COL + GCTColOffset < MP && GCT_ROW + k < KP) { // check boundary
                LOAD_FLOAT_T tmp = FETCH<float, LOAD_FLOAT_T>(GC[OFFSET(GCT_COL + GCTColOffset, GCT_ROW + k, KP)]); // transpose
                GCTs[GCT_ROW][GCT_COL + GCTColOffset] = tmp.x;
                if constexpr (LOAD_ELEM_NUM >= 2)
                    GCTs[GCT_ROW + 1][GCT_COL + GCTColOffset] = tmp.y;
                if constexpr (LOAD_ELEM_NUM == 4) {
                    GCTs[GCT_ROW + 2][GCT_COL + GCTColOffset] = tmp.z;
                    GCTs[GCT_ROW + 3][GCT_COL + GCTColOffset] = tmp.w;
                }
            }
            else {
                GCTs[GCT_ROW][GCT_COL + GCTColOffset] = 0.0f;
                if constexpr (LOAD_ELEM_NUM >= 2)
                    GCTs[GCT_ROW + 1][GCT_COL + GCTColOffset] = 0.0f;
                if constexpr (LOAD_ELEM_NUM == 4) {
                    GCTs[GCT_ROW + 2][GCT_COL + GCTColOffset] = 0.0f;
                    GCTs[GCT_ROW + 3][GCT_COL + GCTColOffset] = 0.0f;
                }
            }
        }
        #pragma unroll
        for (int BTColOffset = 0; BTColOffset < BLOCK_SIZE_N; BTColOffset += BT_COL_STRIDE) {
            if (BLOCK_SIZE_N * bx + BT_COL + BTColOffset < NP && BT_ROW + k < KP) { // check boundary
                LOAD_UINT8_T tmp = FETCH<uint8_t, LOAD_UINT8_T>(B[OFFSET(BT_COL + BTColOffset, BT_ROW + k, KP)]); // transpose
                BTs[BT_ROW][BT_COL + BTColOffset] = tmp.x;
                if constexpr (LOAD_ELEM_NUM >= 2)
                    BTs[BT_ROW + 1][BT_COL + BTColOffset] = tmp.y;
                if constexpr (LOAD_ELEM_NUM == 4) {
                    BTs[BT_ROW + 2][BT_COL + BTColOffset] = tmp.z;
                    BTs[BT_ROW + 3][BT_COL + BTColOffset] = tmp.w;
                }
            }
            else {
                BTs[BT_ROW][BT_COL + BTColOffset] = LUT_OUT_RANGE;
                if constexpr (LOAD_ELEM_NUM >= 2)
                    BTs[BT_ROW + 1][BT_COL + BTColOffset] = LUT_OUT_RANGE;
                if constexpr (LOAD_ELEM_NUM == 4) {
                    BTs[BT_ROW + 2][BT_COL + BTColOffset] = LUT_OUT_RANGE;
                    BTs[BT_ROW + 3][BT_COL + BTColOffset] = LUT_OUT_RANGE;
                }
            }
        }
        
        // ensure all threads have loaded the data from global memory to shared memory
        __syncthreads();

        // compute GA
        #pragma unroll
        for (int kk = 0; kk < BLOCK_SIZE_K; ++kk) {
            // load A from shared memory to register
            #pragma unroll
            for (int threadY = 0; threadY < THREAD_SIZE_Y; threadY += LOAD_ELEM_NUM)
                FETCH<float, LOAD_FLOAT_T>(GCReg[threadY]) = FETCH<float, LOAD_FLOAT_T>(GCTs[kk][THREAD_SIZE_Y * ty + threadY]);
            // lead B from shared memory to register
            #pragma unroll
            for (int threadX = 0; threadX < THREAD_SIZE_X; threadX += 4)
                FETCH<uint8_t, uchar4>(BReg[threadX]) = FETCH<uint8_t, uchar4>(BTs[kk][THREAD_SIZE_X * tx + threadX]);
            // MMA
            #pragma unroll
            for (int threadY = 0; threadY < THREAD_SIZE_Y; ++threadY) {
                float gcVal = GCReg[threadY];
                #pragma unroll
                for (int threadX = 0; threadX < THREAD_SIZE_X; ++threadX) {
                    uint8_t bVal = BReg[threadX];
                    uint8_t aVal = AReg[threadY][threadX];
#if QUANTIZATION_BIT < 8                    
                    accum[threadY][threadX] += gcVal * lookupTable[aVal * LUT_COL_NUM + bVal];
#elif QUANTIZATION_BIT == 8
                    accum[threadY][threadX] += gcVal * tex1Dfetch<int16_t>(lutGradATexture, aVal * LUT_COL_NUM + bVal);
#else
#error "Unsupported QUANTIZATION_BIT"
#endif
                }
            }
        }

        // ensure all threads have finished the computation
        __syncthreads();
    }

    // post-process the result, multiply by 0.5 because the gradients in lookupTable are multiplied by 2
    #pragma unroll
    for (int threadY = 0; threadY < THREAD_SIZE_Y; ++threadY) {
        #pragma unroll
        for (int threadX = 0; threadX < THREAD_SIZE_X; ++threadX)
            // accum[threadY][threadX] *= 0.5f;
            accum[threadY][threadX] *= FACTOR;
    }

    // store GA
    const int rowBase = BLOCK_SIZE_M * by + THREAD_SIZE_Y * ty;
    const int colBase = BLOCK_SIZE_N * bx + THREAD_SIZE_X * tx;
    GA = &GA[OFFSET(rowBase, colBase, NP)];
    #pragma unroll
    for (int threadY = 0; threadY < THREAD_SIZE_Y; ++threadY) {
        if (rowBase + threadY < MP) {
            #pragma unroll
            for (int threadX = 0; threadX < THREAD_SIZE_X; threadX += LOAD_ELEM_NUM) {
                if (colBase + threadX < NP) { // check boundary
                    FETCH<float, LOAD_FLOAT_T>(GA[OFFSET(threadY, threadX, NP)]) = FETCH<float, LOAD_FLOAT_T>(accum[threadY][threadX]);
                }
            }
        }
    }
}


template <
    const int BLOCK_SIZE_M,
    const int BLOCK_SIZE_K,
    const int BLOCK_SIZE_N,
    const int THREAD_SIZE_X,
    const int THREAD_SIZE_Y,
    const int LOAD_ELEM_NUM
    >
__global__ void ApproxGemmBackwardKernelForGB(
    float * __restrict__ GC,
    uint8_t * __restrict__ A,
    uint8_t * __restrict__ B,
    float * __restrict__ GB,
#if QUANTIZATION_BIT < 8
    int16_t * __restrict__ lutGradB,
#elif QUANTIZATION_BIT == 8
    cudaTextureObject_t lutGradBTexture,
#else
#error "Unsupported QUANTIZATION_BIT"
#endif
    const int MP,
    const int KP,
    const int NP) {

    static_assert(LOAD_ELEM_NUM == 1 || LOAD_ELEM_NUM == 2 || LOAD_ELEM_NUM == 4, "LOAD_ELEM_NUM should be 1, 2 or 4");
    using LOAD_UINT8_T = typename LoadType<LOAD_ELEM_NUM>::uchar_type;
    using LOAD_FLOAT_T = typename LoadType<LOAD_ELEM_NUM>::float_type;
    using LOAD_INT16_T = typename LoadType<4>::short_type;

    // block & thread index
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // thread id in current Block
    const int THREAD_X_NUM = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int THREAD_Y_NUM = BLOCK_SIZE_M / THREAD_SIZE_Y;
    static_assert(BLOCK_SIZE_N % THREAD_SIZE_X == 0, "BLOCK_SIZE_N should be divisible by THREAD_SIZE_X");
    static_assert(BLOCK_SIZE_M % THREAD_SIZE_Y == 0, "BLOCK_SIZE_M should be divisible by THREAD_SIZE_Y");
    const int tid = ty * THREAD_X_NUM + tx;

    // total number of threads in a block
    const int THREAD_NUM = THREAD_X_NUM * THREAD_Y_NUM;

#if QUANTIZATION_BIT < 8
    // prepare lookup table using shared memory
    // approximate multiplication: h(a, b)
    // lookupTable[a][b] represents the gradient of h(a, b) with respect to b
    __shared__ int16_t lookupTable[LUT_ELEM_NUM];
    static_assert(LUT_ELEM_NUM % THREAD_NUM == 0, "LUT_ELEM_NUM should be divisible by THREAD_NUM");
    const int LOAD_LUT_ELEM_PER_THREAD = LUT_ELEM_NUM / THREAD_NUM;
    // constexpr int LOAD_LUT_ELEM_PER_THREAD = LUT_ELEM_NUM / THREAD_NUM;
    static_assert(LOAD_LUT_ELEM_PER_THREAD % 4 == 0, "LOAD_LUT_ELEM_PER_THREAD should be divisible by 4");
    const int LOAD_LUT_START = tid * LOAD_LUT_ELEM_PER_THREAD;
    // if constexpr (LOAD_LUT_ELEM_PER_THREAD % 4 == 0) {
        #pragma unroll
        for (int i = LOAD_LUT_START; i < LOAD_LUT_START + LOAD_LUT_ELEM_PER_THREAD; i += 4)
            FETCH<int16_t, LOAD_INT16_T>(lookupTable[i]) = FETCH<int16_t, LOAD_INT16_T>(lutGradB[i]);
    // }
    // else if constexpr (LOAD_LUT_ELEM_PER_THREAD % 2 == 0) {
    //     #pragma unroll
    //     for (int i = LOAD_LUT_START; i < LOAD_LUT_START + LOAD_LUT_ELEM_PER_THREAD; i += 2)
    //         FETCH<int16_t, short2>(lookupTable[i]) = FETCH<int16_t, short2>(lutGradB[i]);
    // }
    // else {
    //     #pragma unroll
    //     for (int i = LOAD_LUT_START; i < LOAD_LUT_START + LOAD_LUT_ELEM_PER_THREAD; ++i)
    //         lookupTable[i] = lutGradB[i];
    // }
#endif

    // shared memory
    alignas(4) __shared__ uint8_t As[BLOCK_SIZE_K][BLOCK_SIZE_M]; // A->As: no transpose, aligned to 4 bytes
    alignas(16) __shared__ float GCs[BLOCK_SIZE_K][BLOCK_SIZE_N];  // GC->GCs: no transpose, aligned to 16 bytes
    __shared__ uint8_t Bs[BLOCK_SIZE_M][BLOCK_SIZE_N]; // Bs's row = As's column, Bs's column = GCs's column

    // register for GB
    register float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0.0f};

    // registers for A and GC
    register uint8_t AReg[THREAD_SIZE_X];
    register float GCReg[THREAD_SIZE_Y];
    register uint8_t BReg[THREAD_SIZE_Y][THREAD_SIZE_X];

    // thread number in one row/col in the shared memory
    const int A_THREAD_NUM_PER_ROW = BLOCK_SIZE_M / LOAD_ELEM_NUM;
    static_assert(BLOCK_SIZE_M % LOAD_ELEM_NUM == 0);
    const int GC_THREAD_NUM_PER_ROW = BLOCK_SIZE_N / LOAD_ELEM_NUM;
    static_assert(BLOCK_SIZE_N % LOAD_ELEM_NUM == 0);
    const int B_THREAD_PER_ROW = BLOCK_SIZE_N / LOAD_ELEM_NUM;

    // (row, col) in the shared memory, loaded by this thread
    const int A_ROW = tid / A_THREAD_NUM_PER_ROW;
    const int A_COL = tid % A_THREAD_NUM_PER_ROW * LOAD_ELEM_NUM;
    const int GC_ROW = tid / GC_THREAD_NUM_PER_ROW;
    const int GC_COL = tid % GC_THREAD_NUM_PER_ROW * LOAD_ELEM_NUM;
    const int B_ROW = tid / B_THREAD_PER_ROW;
    const int B_COL = tid % B_THREAD_PER_ROW * LOAD_ELEM_NUM;

    // row/col stride that thread uses to load multiple rows/cols
    const int A_ROW_STRIDE = THREAD_NUM / A_THREAD_NUM_PER_ROW;
    static_assert(THREAD_NUM % A_THREAD_NUM_PER_ROW == 0, "THREAD_NUM should be divisible by A_THREAD_NUM_PER_ROW");
    const int GC_ROW_STRIDE = THREAD_NUM / GC_THREAD_NUM_PER_ROW;
    static_assert(THREAD_NUM % GC_THREAD_NUM_PER_ROW == 0, "THREAD_NUM should be divisible by GC_THREAD_NUM_PER_ROW");
    const int B_ROW_STRIDE = THREAD_NUM / B_THREAD_PER_ROW;

    // initialize the left-top addresses in this block
    A = &A[OFFSET(0, BLOCK_SIZE_M * by, MP)];
    GC = &GC[OFFSET(0, BLOCK_SIZE_N * bx, NP)];
    B = &B[OFFSET(BLOCK_SIZE_M * by, BLOCK_SIZE_N * bx, NP)];

    // load B: global memory -> shared memory -> register
    #pragma unroll
    for (int BRowOffset = 0; BRowOffset < BLOCK_SIZE_M; BRowOffset += B_ROW_STRIDE) { // check boundary
        if (BLOCK_SIZE_M * by + B_ROW + BRowOffset < MP && BLOCK_SIZE_N * bx + B_COL < NP) {
            FETCH<uint8_t, LOAD_UINT8_T>(Bs[B_ROW + BRowOffset][B_COL]) = FETCH<uint8_t, LOAD_UINT8_T>(B[OFFSET(B_ROW + BRowOffset, B_COL, NP)]);
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
    __syncthreads();
    #pragma unroll
    for (int threadY = 0; threadY < THREAD_SIZE_Y; ++threadY) {
        #pragma unroll
        for (int threadX = 0; threadX < THREAD_SIZE_X; threadX += 4)
            FETCH<uint8_t, uchar4>(BReg[threadY][threadX]) = FETCH<uint8_t, uchar4>(Bs[THREAD_SIZE_Y * ty + threadY][THREAD_SIZE_X * tx + threadX]);
    }
    
    // main loop
    for (int k = 0; k < KP; k += BLOCK_SIZE_K) {
        // load data from global memory to shared memory
        #pragma unroll
        for (int ARowOffset = 0; ARowOffset < BLOCK_SIZE_K; ARowOffset += A_ROW_STRIDE) {
            if (A_ROW + ARowOffset + k < KP && BLOCK_SIZE_M * by + A_COL < MP) { // check boundary
                FETCH<uint8_t, LOAD_UINT8_T>(As[A_ROW + ARowOffset][A_COL]) = FETCH<uint8_t, LOAD_UINT8_T>(A[OFFSET(A_ROW + ARowOffset + k, A_COL, MP)]);
            }
            else {
                As[A_ROW + ARowOffset][A_COL] = LUT_OUT_RANGE;
                if constexpr (LOAD_ELEM_NUM >= 2)
                    As[A_ROW + ARowOffset][A_COL + 1] = LUT_OUT_RANGE;
                if constexpr (LOAD_ELEM_NUM == 4) {
                    As[A_ROW + ARowOffset][A_COL + 2] = LUT_OUT_RANGE;
                    As[A_ROW + ARowOffset][A_COL + 3] = LUT_OUT_RANGE;
                }
            }
        }
        #pragma unroll
        for (int GCRowOffset = 0; GCRowOffset < BLOCK_SIZE_K; GCRowOffset += GC_ROW_STRIDE) {
            if (GC_ROW + GCRowOffset + k < KP && BLOCK_SIZE_N * bx + GC_COL < NP) { // check boundary
                FETCH<float, LOAD_FLOAT_T>(GCs[GC_ROW + GCRowOffset][GC_COL]) = FETCH<float, LOAD_FLOAT_T>(GC[OFFSET(GC_ROW + GCRowOffset + k, GC_COL, NP)]);
            }
            else {
                GCs[GC_ROW + GCRowOffset][GC_COL] = 0.0f;
                if constexpr (LOAD_ELEM_NUM >= 2)
                    GCs[GC_ROW + GCRowOffset][GC_COL + 1] = 0.0f;
                if constexpr (LOAD_ELEM_NUM == 4) {
                    GCs[GC_ROW + GCRowOffset][GC_COL + 2] = 0.0f;
                    GCs[GC_ROW + GCRowOffset][GC_COL + 3] = 0.0f;
                }
            }
        }

        // ensure all threads have loaded the data from global memory to shared memory
        __syncthreads();

        // compute GB
        #pragma unroll
        for (int kk = 0; kk < BLOCK_SIZE_K; ++kk) {
            // load A from shared memory to register
            #pragma unroll
            for (int threadY = 0; threadY < THREAD_SIZE_Y; threadY += 4)
                FETCH<uint8_t, uchar4>(AReg[threadY]) = FETCH<uint8_t, uchar4>(As[kk][THREAD_SIZE_Y * ty + threadY]);
            // lead GC from shared memory to register
            #pragma unroll
            for (int threadX = 0; threadX < THREAD_SIZE_X; threadX += 4)
                FETCH<float, float4>(GCReg[threadX]) = FETCH<float, float4>(GCs[kk][THREAD_SIZE_X * tx + threadX]);
            // MMA
            #pragma unroll
            for (int threadY = 0; threadY < THREAD_SIZE_Y; ++threadY) {
                uint8_t aVal = AReg[threadY];
                #pragma unroll
                for (int threadX = 0; threadX < THREAD_SIZE_X; ++threadX) {
                    float gcVal = GCReg[threadX];
                    uint8_t bVal = BReg[threadY][threadX];
#if QUANTIZATION_BIT < 8                    
                    accum[threadY][threadX] += lookupTable[aVal * LUT_COL_NUM + bVal] * gcVal;
#elif QUANTIZATION_BIT == 8
                    accum[threadY][threadX] += tex1Dfetch<int16_t>(lutGradBTexture, aVal * LUT_COL_NUM + bVal) * gcVal;
#else
#error "Unsupported QUANTIZATION_BIT"
#endif
                }
            }
        }

        // ensure all threads have finished the computation
        __syncthreads();
    }

    // post-process the result, multiply by 0.5 because the gradients in lookupTable are multiplied by 2
    #pragma unroll
    for (int threadY = 0; threadY < THREAD_SIZE_Y; ++threadY) {
        #pragma unroll
        for (int threadX = 0; threadX < THREAD_SIZE_X; ++threadX)
            // accum[threadY][threadX] *= 0.5f;
            accum[threadY][threadX] *= FACTOR;
    }

    // store GB
    const int rowBase = BLOCK_SIZE_M * by + THREAD_SIZE_Y * ty;
    const int colBase = BLOCK_SIZE_N * bx + THREAD_SIZE_X * tx;
    GB = &GB[OFFSET(rowBase, colBase, NP)];
    #pragma unroll
    for (int threadY = 0; threadY < THREAD_SIZE_Y; ++threadY) {
        if (rowBase + threadY < MP) {
            #pragma unroll
            for (int threadX = 0; threadX < THREAD_SIZE_X; threadX += LOAD_ELEM_NUM) {
                if (colBase + threadX < NP) { // check boundary
                    FETCH<float, LOAD_FLOAT_T>(GB[OFFSET(threadY, threadX, NP)]) = FETCH<float, LOAD_FLOAT_T>(accum[threadY][threadX]);
                }
            }
        }
    }
}


/**
 * Compute the backward pass of the approximate matrix multiplication operation.
 *
 * @param gc_tensor: input, the gradient of the output tensor C with respect to the loss, float32.
 * @param a_tensor: input, the input tensor A, uint8.
 * @param b_tensor: input, the input tensor B, uint8.
 * @param ga_tensor: output, the gradient of tensor A with respect to the loss, float32.
 * @param gb_tensor: output, the gradient of tensor B with respect to the loss, float32.
 * @param lut_grad_a_tensor: input, the gradient w.r.t. a, int16.
 * @param lut_grad_b_tensor: input, the gradient w.r.t. b, int16.
 */
void approx_gemm_backward_gpu(const at::Tensor &gc_tensor, const at::Tensor &a_tensor, const at::Tensor &b_tensor, at::Tensor &ga_tensor, at::Tensor &gb_tensor, const at::Tensor &lut_grad_a_tensor, const at::Tensor &lut_grad_b_tensor) {
    // check input
    CHECK_INPUT(gc_tensor, torch::kFloat32, "gc_tensor");
    CHECK_INPUT(a_tensor, torch::kUInt8, "a_tensor");
    CHECK_INPUT(b_tensor, torch::kUInt8, "b_tensor");
    CHECK_INPUT(ga_tensor, torch::kFloat32, "ga_tensor");
    CHECK_INPUT(gb_tensor, torch::kFloat32, "gb_tensor");
    CHECK_INPUT(lut_grad_a_tensor, torch::kInt16, "lut_grad_a_tensor");
    CHECK_INPUT(lut_grad_b_tensor, torch::kInt16, "lut_grad_b_tensor");
    int M = a_tensor.size(0);
    int N = b_tensor.size(1);
    int K = a_tensor.size(1);
    TORCH_CHECK(gc_tensor.size(0) == M && gc_tensor.size(1) == N, "gc_tensor's shape should be ", M, " x ", N);
    TORCH_CHECK(K == b_tensor.size(0), "b_tensor's shape should be ", K, " x ", N);
    TORCH_CHECK(ga_tensor.size(0) == M && ga_tensor.size(1) == K, "ga_tensor's shape should be ", M, " x ", K);
    TORCH_CHECK(gb_tensor.size(0) == K && gb_tensor.size(1) == N, "gb_tensor's shape should be ", K, " x ", N);
    TORCH_CHECK(lut_grad_a_tensor.size(0) == LUT_ELEM_NUM, "lut_grad_a_tensor's shape should be ", LUT_ELEM_NUM, ", please check the value of QUANTIZATION_BIT");
    TORCH_CHECK(lut_grad_b_tensor.size(0) == LUT_ELEM_NUM, "lut_grad_b_tensor's shape should be ", LUT_ELEM_NUM, ", please check the value of QUANTIZATION_BIT");

    // set & check device
    int deviceId = gc_tensor.get_device();
    cudaSetDevice(deviceId);
    TORCH_CHECK(deviceId == a_tensor.get_device() && deviceId == b_tensor.get_device() && deviceId == ga_tensor.get_device() && deviceId == gb_tensor.get_device(), "All tensors should be on the same device");
    TORCH_CHECK(deviceId == lut_grad_a_tensor.get_device() && deviceId == lut_grad_b_tensor.get_device(), "All tensors should be on the same device");

    // get data ptr
    float *gc = gc_tensor.data_ptr<float>();
    uint8_t *a = a_tensor.data_ptr<uint8_t>();
    uint8_t *b = b_tensor.data_ptr<uint8_t>();
    float *ga = ga_tensor.data_ptr<float>();
    float *gb = gb_tensor.data_ptr<float>();
    int16_t *lut_grad_a = lut_grad_a_tensor.data_ptr<int16_t>();
    int16_t *lut_grad_b = lut_grad_b_tensor.data_ptr<int16_t>();

#if QUANTIZATION_BIT == 8
    // create texture object
    cudaTextureObject_t lutGradATexture = CreateLUTTextureObject<int16_t>(lut_grad_a, LUT_ELEM_NUM);
    cudaTextureObject_t lutGradBTexture = CreateLUTTextureObject<int16_t>(lut_grad_b, LUT_ELEM_NUM);
#endif
    
    // kernel parameters
    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_K = 8;
    const int BLOCK_SIZE_N = 64; // BLOCK_SIZE_N is set to 64 instead of 128, because the shared memory is limited
    const int THREAD_SIZE_X = 8;
    const int THREAD_SIZE_Y = 8;
    static_assert(THREAD_SIZE_X == THREAD_SIZE_Y, "THREAD_SIZE_X should be equal to THREAD_SIZE_Y");
    static_assert((THREAD_SIZE_X % 4) == 0, "THREAD_SIZE_X should be divisible by 4");
    static_assert((THREAD_SIZE_Y % 4) == 0, "THREAD_SIZE_Y should be divisible by 4");

    // launch kernel for GA
    // gradient for accurate matmul: ga = gc b^T
    // here, gc's size is M x N, b_transpose's size is N x K, ga's size is M x K
    // we have MP = M, NP = K, KP = N
    {
    int MP = M, KP = N, NP = K;
    dim3 dimGrid(CEIL_DIV(NP, BLOCK_SIZE_N), CEIL_DIV(MP, BLOCK_SIZE_M));
    dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
#if QUANTIZATION_BIT < 8
    if ((MP & 3) == 0 && (KP & 3) == 0 && (NP & 3) == 0)
        ApproxGemmBackwardKernelForGA<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_X, THREAD_SIZE_Y, 4> <<<dimGrid, dimBlock>>> (gc, a, b, ga, lut_grad_a, MP, KP, NP);
    else if ((MP & 1) == 0 && (KP & 1) == 0 && (NP & 1) == 0)
        ApproxGemmBackwardKernelForGA<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_X, THREAD_SIZE_Y, 2> <<<dimGrid, dimBlock>>> (gc, a, b, ga, lut_grad_a, MP, KP, NP);
    else
        ApproxGemmBackwardKernelForGA<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_X, THREAD_SIZE_Y, 1> <<<dimGrid, dimBlock>>> (gc, a, b, ga, lut_grad_a, MP, KP, NP);
#elif QUANTIZATION_BIT == 8
    if ((MP & 3) == 0 && (KP & 3) == 0 && (NP & 3) == 0)
        ApproxGemmBackwardKernelForGA<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_X, THREAD_SIZE_Y, 4> <<<dimGrid, dimBlock>>> (gc, a, b, ga, lutGradATexture, MP, KP, NP);
    else if ((MP & 1) == 0 && (KP & 1) == 0 && (NP & 1) == 0)
        ApproxGemmBackwardKernelForGA<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_X, THREAD_SIZE_Y, 2> <<<dimGrid, dimBlock>>> (gc, a, b, ga, lutGradATexture, MP, KP, NP);
    else
        ApproxGemmBackwardKernelForGA<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_X, THREAD_SIZE_Y, 1> <<<dimGrid, dimBlock>>> (gc, a, b, ga, lutGradATexture, MP, KP, NP);
#else
#error "Unsupported QUANTIZATION_BIT"
#endif
    }

    // launch kernel for GB
    // gradient for accurate matmul: gb = a^T gc
    // here, a_transpose's size is K x M, gc's size is M x N, gb's size is K x N
    // we have MP = K, KP = M, NP = N
    {
    int MP = K, KP = M, NP = N;
    dim3 dimGrid(CEIL_DIV(NP, BLOCK_SIZE_N), CEIL_DIV(MP, BLOCK_SIZE_M));
    dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
#if QUANTIZATION_BIT < 8
    if ((MP & 3) == 0 && (KP & 3) == 0 && (NP & 3) == 0)
        ApproxGemmBackwardKernelForGB<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_X, THREAD_SIZE_Y, 4> <<<dimGrid, dimBlock>>>(gc, a, b, gb, lut_grad_b, MP, KP, NP);
    else if ((MP & 1) == 0 && (KP & 1) == 0 && (NP & 1) == 0)
        ApproxGemmBackwardKernelForGB<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_X, THREAD_SIZE_Y, 2> <<<dimGrid, dimBlock>>>(gc, a, b, gb, lut_grad_b, MP, KP, NP);
    else
        ApproxGemmBackwardKernelForGB<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_X, THREAD_SIZE_Y, 1> <<<dimGrid, dimBlock>>>(gc, a, b, gb, lut_grad_b, MP, KP, NP);
#elif QUANTIZATION_BIT == 8
    if ((MP & 3) == 0 && (KP & 3) == 0 && (NP & 3) == 0)
        ApproxGemmBackwardKernelForGB<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_X, THREAD_SIZE_Y, 4> <<<dimGrid, dimBlock>>>(gc, a, b, gb, lutGradBTexture, MP, KP, NP);
    else if ((MP & 1) == 0 && (KP & 1) == 0 && (NP & 1) == 0)
        ApproxGemmBackwardKernelForGB<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_X, THREAD_SIZE_Y, 2> <<<dimGrid, dimBlock>>>(gc, a, b, gb, lutGradBTexture, MP, KP, NP);
    else
        ApproxGemmBackwardKernelForGB<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_X, THREAD_SIZE_Y, 1> <<<dimGrid, dimBlock>>>(gc, a, b, gb, lutGradBTexture, MP, KP, NP);
#else
#error "Unsupported QUANTIZATION_BIT"
#endif
    }

    // check error
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA Error: %s\n", cudaGetErrorString(err));

#if QUANTIZATION_BIT == 8
    // Destroy texture object after kernel execution
    err = cudaDestroyTextureObject(lutGradATexture);
    if (err != cudaSuccess)
        printf("CUDA Error (Texture Destruction): %s\n", cudaGetErrorString(err));
    err = cudaDestroyTextureObject(lutGradBTexture);
    if (err != cudaSuccess)
        printf("CUDA Error (Texture Destruction): %s\n", cudaGetErrorString(err));
#endif
}
