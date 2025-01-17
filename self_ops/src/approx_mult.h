#ifndef APPROX_MULT_H
#define APPROX_MULT_H


#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <texture_fetch_functions.h>


// quantization bit
#define QUANTIZATION_BIT 7
const int LUT_MAXVAL = (1 << QUANTIZATION_BIT) - 1;       // maximum value of input operands 
#if QUANTIZATION_BIT == 7
// parameters for 7-bit approximate multiplication
// const int LUT_MAXVAL = 127;                            // maximum value of input operands 
const int LUT_ROW_NUM = LUT_MAXVAL + 2;                   // padding to mitigate bank conflict
const int LUT_COL_NUM = LUT_MAXVAL + 3;                   // padding to mitigate bank conflict
const int LUT_ELEM_NUM = LUT_ROW_NUM * LUT_COL_NUM + 638; // padding for multiple-thread 4-element loading
#elif QUANTIZATION_BIT == 8
// parameters for 8-bit approximate multiplication
// const int LUT_MAXVAL = 255;                            // maximum value of input operands 
const int LUT_ROW_NUM = LUT_MAXVAL + 2;                   // padding to mitigate bank conflict
const int LUT_COL_NUM = LUT_MAXVAL + 3;                   // padding to mitigate bank conflict
const int LUT_ELEM_NUM = LUT_ROW_NUM * LUT_COL_NUM;
#elif QUANTIZATION_BIT == 4
// parameters for 4-bit approximate multiplication
const int LUT_ROW_NUM = LUT_MAXVAL + 2;                   // padding to mitigate bank conflict
const int LUT_COL_NUM = LUT_MAXVAL + 3;                   // padding to mitigate bank conflict
const int LUT_ELEM_NUM = LUT_ROW_NUM * LUT_COL_NUM + 718; // padding for multiple-thread 4-element loading
#elif QUANTIZATION_BIT == 6
// parameters for 6-bit approximate multiplication
const int LUT_ROW_NUM = LUT_MAXVAL + 2;                   // padding to mitigate bank conflict
const int LUT_COL_NUM = LUT_MAXVAL + 3;                   // padding to mitigate bank conflict
const int LUT_ELEM_NUM = LUT_ROW_NUM * LUT_COL_NUM + 830; // padding for multiple-thread 4-element loading
#else
#error "Unsupported QUANTIZATION_BIT"
#endif

const int LUT_OUT_RANGE = LUT_MAXVAL + 1;                 // if one operand is LUT_OUT_RANGE, it means out of range
const float FACTOR = 1/16.0;                              // factor for scaling the gradient


// save LUT to the texture memory (only if QUANTIZATION_BIT == 8)
template <typename T>
static cudaTextureObject_t CreateLUTTextureObject(T *d_lut, size_t lutSize) {
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_lut;
    resDesc.res.linear.desc = cudaCreateChannelDesc<T>();
    resDesc.res.linear.sizeInBytes = lutSize * sizeof(T);

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;  // Clamp outside access to the edges
    texDesc.filterMode = cudaFilterModePoint;       // No filtering, just fetch the nearest
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;                   // Access with non-normalized coordinates

    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    
    return texObj;
}


#endif