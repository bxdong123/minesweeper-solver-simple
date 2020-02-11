#ifndef MINESWEEPER_CUH
#define MINESWEEPER_CUH

#include "minesweeperUtilsGPU.cuh"

#define FLAG -1
#define COVERED -2

#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 16

enum class TileStatus {
    MustBeMine,
    MustBeSafe,
    Unknown,
};

__device__
inline bool isOutOfRange(int x, int y, int width, int height) {
    if (x < 0 || y < 0 || x >= width || y >= height)
        return true;
    else
        return false;
}

// Determines the status (Safe, Mine, or Unknown) of the neighbors of tile (x,y)
// Only call this function with (x,y) values that are in bounds
__device__
TileStatus determineNeighborStatus(int x, int y, int width, int height, int* output) {
    int idx = y * width + x;
    int numMines = output[idx];

    // If this neighbor is covered, then we have nothing to do
    if (numMines == COVERED)
        return TileStatus::Unknown;

    int flagCount = 0;
    int coveredCount = 0;

    for (int ny = y - 1; ny <= y + 1; ny++) {
        for (int nx = x - 1; nx <= x + 1; nx++) {
            if (isOutOfRange(nx, ny, width, height) || (nx == x && ny == y))
                continue;

            int nIdx = ny * width + nx;

            if (output[nIdx] == FLAG)
                flagCount += 1;
            else if (output[nIdx] == COVERED)
                coveredCount += 1;
        }
    }

    if (numMines == flagCount)
        return TileStatus::MustBeSafe;
    else if (numMines - flagCount == coveredCount)
        return TileStatus::MustBeMine;
    else
        return TileStatus::Unknown;
}

__global__
void solveMinesweeperBoard(int width, int height, int* numMines, int* output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x >= width || y >= height)
        return;

    // Already figured out this tile, so nothing to do.
    if (output[idx] >= -1)
        return;

    for (int ny = y - 1; ny <= y + 1; ny++) {
        for (int nx = x - 1; nx <= x + 1; nx++) {
            if (isOutOfRange(nx, ny, width, height) || (nx == x && ny == y))
                continue;

            TileStatus status = determineNeighborStatus(nx, ny, width, height, output);
            if (status == TileStatus::MustBeMine) {
                atomicSub(numMines, 1);
                output[idx] = -1;
                return;
            }
            else if (status == TileStatus::MustBeSafe) {
                output[idx] = clickTile(x,y);
                return;
            }
        }
    }
}

void printMyBoard(int width, int height, int* output) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int value = output[y * width + x];
            if (value == COVERED)
                printf(".");
            else if (value == FLAG)
                printf("A");
            else
                printf("%d", value);
        }
        printf("\n");
    }
    printf("\n");
}

void minesweeperGPU(int width, int height, int numMines, int startX, int startY, int* output) {
    printf("Running GPU solver: simple\n");

    for (int i = 0; i < width*height; ++i) {
        output[i] = COVERED;
    }

    int startIdx = startY * width + startX;

    output[startIdx] = 0;

    int* numMinesGPU;
    checkCudaError(cudaMallocManaged(&numMinesGPU, sizeof(numMines)));
    *numMinesGPU = numMines;

    int xBlockSize = BLOCKSIZE_X;
    int yBlockSize = BLOCKSIZE_Y;
    int xNumBlocks = (width + xBlockSize - 1) / xBlockSize;
    int yNumBlocks = (height + yBlockSize - 1) / yBlockSize;

    int sameCount = 0;
    int numMinesLast = numMines;

    while(*numMinesGPU > 0) {
        if (numMinesLast == *numMinesGPU) {
            sameCount += 1;
        }
        if (sameCount >= 10000) {
            printMyBoard(width, height, output);
            printf("Num Mines Left: %d\n", *numMinesGPU);
            break;
        }
        numMinesLast = *numMinesGPU;
        solveMinesweeperBoard<<<dim3(xNumBlocks, yNumBlocks, 1), dim3(xBlockSize, yBlockSize, 1)>>>(width, height, numMinesGPU, output);
        checkCudaError(cudaGetLastError());
        checkCudaError(cudaDeviceSynchronize());
    }

    cudaFree(numMinesGPU);
}

#undef FLAG
#undef COVERED

#undef BLOCKSIZE_X
#undef BLOCKSIZE_Y

#endif // MINESWEEPER_CUH
