#ifndef EEC289Q_MINESWEEPER_UTILS_GPU_CUH
#define EEC289Q_MINESWEEPER_UTILS_GPU_CUH

#include "cuda_runtime.h"

#include <cstdint>
#include <cstdlib>
#include <stdio.h>

#define checkCudaError(e) { checkCudaErrorImpl(e, __FILE__, __LINE__); }

inline void checkCudaErrorImpl(cudaError_t e, const char* file, int line, bool abort = true) {
    if (e != cudaSuccess) {
        fprintf(stderr, "[CUDA Error] %s - %s:%d\n", cudaGetErrorString(e), file, line);
        if (abort) exit(e);
    }
}

namespace donottouch {

__device__ int8_t*   donottouchGroundTruth;
__device__ bool      donottouchClickedOnMine;
__constant__ int     donottouchWidth;

} // namespace donottouch

__device__
int8_t clickTile(int x, int y) {
    int idx = y * donottouch::donottouchWidth + x;
    int value = donottouch::donottouchGroundTruth[idx];

    if (value == -1)
        donottouch::donottouchClickedOnMine = true;

    return value;
}

#endif // EEC289Q_MINESWEEPER_UTILS_GPU_CUH
