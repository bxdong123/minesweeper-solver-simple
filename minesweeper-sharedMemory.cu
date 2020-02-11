#ifndef MINESWEEPER_CUH
#define MINESWEEPER_CUH

#include "minesweeperUtilsGPU.cuh"

// Two options for reading into shared memory:
//
// SLIDING_WINDOW - Starts in the upper-left corner of shared memory
//   and reads in values in a sliding window pattern. The window slides
//   over both the halo region and the main block of values.
//
// MAIN_BLOCK_THEN_HALO - Starts by reading in the main block of values
//   into shared memory. Then, it reads in the halo region by reading
//   the two rows above and two rows below, then the two columns to the
//   left and two columns to the right, and finally the four corners.
//
// The MAIN_BLOCK_THEN_HALO option enables an extra optimization. Because
// it reads the main block of values first, it can check if they are all
// known already. If so, it can skip reading in the halo and instead
// immedately return. Because of this extra optimization, this version is
// is often faster for this specific application. However, the SLIDING_WINDOW
// approach may be faster for other applications.

// Uncomment only one of these lines at a time
// #define SLIDING_WINDOW
#define MAIN_BLOCK_THEN_HALO

#define FLAG -1
#define COVERED -2
#define OUT_OF_RANGE -3

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
TileStatus determineNeighborStatusShared(int x, int y, int board[BLOCKSIZE_X + 4][BLOCKSIZE_Y + 4], int gx, int gy) {
    int numMines = board[x][y];

    // If this neighbor is covered or is a flag, then we have nothing to do
    if (numMines == COVERED || numMines == FLAG || numMines == OUT_OF_RANGE)
        return TileStatus::Unknown;

    int flagCount = 0;
    int coveredCount = 0;

    for (int ny = y - 1; ny <= y + 1; ny++) {
        for (int nx = x - 1; nx <= x + 1; nx++) {
            int value = board[nx][ny];

            if (value == FLAG)
                flagCount += 1;
            else if (value == COVERED)
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
void solveMinesweeperBoardSharedMemory(int width, int height, int* numMines, int* output) {
    // blocksize + 4 because there should be 2 halo rows above and 2 below
    // and 2 halo columns to the left and 2 to the right.
    __shared__ int board[BLOCKSIZE_X + 4][BLOCKSIZE_Y + 4];

    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int gidx = gy * width + gx;

    int sx = threadIdx.x + 2;
    int sy = threadIdx.y + 2;

#ifdef SLIDING_WINDOW
    {
        __shared__ bool hasWork;
        __shared__ bool hasInfo;

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            hasWork = false;
            hasInfo = false;
        }

        int grx = gx - 2;
        int gry = gy - 2;

        int srx = threadIdx.x;
        int sry = threadIdx.y;

        int value = OUT_OF_RANGE;
        if (!isOutOfRange(grx, gry, width, height)) {
            value = output[gry * width + grx];
        }

        board[srx][sry] = value;

        if (value == COVERED)
            hasWork = true;
        else if (value >= 0)
            hasInfo = true;

        if (srx < 4) {
            int strideX = BLOCKSIZE_X;
            int strideY = 0;

            int value = OUT_OF_RANGE;
            if (!isOutOfRange(grx + strideX, gry + strideY, width, height)) {
                value = output[(gry + strideY) * width + (grx + strideX)];
            }

            board[srx + strideX][sry + strideY] = value;

            if (value == COVERED)
                hasWork = true;
            else if (value >= 0)
                hasInfo = true;
        }
        if (sry < 4) {
            int strideX = 0;
            int strideY = BLOCKSIZE_Y;

            int value = OUT_OF_RANGE;
            if (!isOutOfRange(grx + strideX, gry + strideY, width, height)) {
                value = output[(gry + strideY) * width + (grx + strideX)];
            }

            board[srx + strideX][sry + strideY] = value;

            if (value == COVERED)
                hasWork = true;
            else if (value >= 0)
                hasInfo = true;
        }
        if (srx < 4 && sry < 4) {
            int strideX = BLOCKSIZE_X;
            int strideY = BLOCKSIZE_Y;

            int value = OUT_OF_RANGE;
            if (!isOutOfRange(grx + strideX, gry + strideY, width, height)) {
                value = output[(gry + strideY) * width + (grx + strideX)];
            }

            board[srx + strideX][sry + strideY] = value;

            if (value == COVERED)
                hasWork = true;
            else if (value >= 0)
                hasInfo = true;
        }

        __syncthreads();

        if(!hasWork || !hasInfo)
            return;
    }
#endif

#ifdef MAIN_BLOCK_THEN_HALO
    {
        __shared__ bool hasWork;
        __shared__ bool hasInfo;

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            hasWork = false;
            hasInfo = false;
        }

        // Read in the non-halo values
        int value = OUT_OF_RANGE;
        if (gx < width && gy < height) 
            value = output[gidx];

        if (value == COVERED)
            hasWork = true;
        else if(value >= 0)
            hasInfo = true;

        __syncthreads();

        if (!hasWork)
            return;

        board[sx][sy] = value;

        // Read in top 2 and bottom 2 halo rows
        if (threadIdx.y < 4) {
            int hx = gx;
            int hy = gy - 2;

            int shx = sx;
            int shy = sy - 2;

            if (threadIdx.y >= 2) {
                hy += BLOCKSIZE_Y;
                shy += BLOCKSIZE_Y;
            }

            if (isOutOfRange(hx, hy, width, height)) {
                board[shx][shy] = OUT_OF_RANGE;
            }
            else {
                int hidx = hy * width + hx;
                int value = output[hidx];
                board[shx][shy] = value;
                if (value >= 0)
                    hasInfo = true;
            }
        }

        // Read in the left 2 and right 2 halo columns
        if (threadIdx.x < 4) {
            int hx = gx - 2;
            int hy = gy;

            int shx = sx - 2;
            int shy = sy;

            if (threadIdx.x >= 2) {
                hx += BLOCKSIZE_X;
                shx += BLOCKSIZE_X;
            }

            if (isOutOfRange(hx, hy, width, height)) {
                board[shx][shy] = OUT_OF_RANGE;
            }
            else {
                int hidx = hy * width + hx;
                int value = output[hidx];
                board[shx][shy] = value;
                if (value >= 0)
                    hasInfo = true;
            }
        }

        // Read in the corners of the halo region
        if (threadIdx.y < 4 && threadIdx.x < 4) {
            int hx = gx - 2;
            int hy = gy - 2;

            int shx = sx - 2;
            int shy = sy - 2;

            if (threadIdx.y >= 2) {
                hy += BLOCKSIZE_Y;
                shy += BLOCKSIZE_Y;
            }

            if (threadIdx.x >= 2) {
                hx += BLOCKSIZE_X;
                shx += BLOCKSIZE_X;
            }

            if (isOutOfRange(hx, hy, width, height)) {
                board[shx][shy] = OUT_OF_RANGE;
            }
            else {
                int hidx = hy * width + hx;
                int value = output[hidx];
                board[shx][shy] = value;
                if (value >= 0)
                    hasInfo = true;
            }
        }

        __syncthreads();

        if(!hasInfo)
            return;
    }
#endif

    // Out of range of the data, so nothing to do.
    if (gx >= width || gy >= height)
        return;

    // Already figured out this tile, so nothing to do.
    if (board[sx][sy] >= -1)
        return;

    for (int ny = sy - 1; ny <= sy + 1; ny++) {
        for (int nx = sx - 1; nx <= sx + 1; nx++) {
            if ((nx == sx && ny == sy) || board[nx][ny] == OUT_OF_RANGE)
                continue;

            TileStatus status = determineNeighborStatusShared(nx, ny, board, gx, gy);

            if (status == TileStatus::MustBeMine) {
                atomicSub(numMines, 1);
                output[gidx] = -1;
                return;
            }
            else if (status == TileStatus::MustBeSafe) {
                output[gidx] = clickTile(gx,gy);
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
                printf(". ");
            else if (value == FLAG)
                printf("A ");
            else
                printf("%d ", value);
        }
        printf("\n");
    }
    printf("\n");
}

void minesweeperGPU(int width, int height, int numMines, int startX, int startY, int* output) {
    printf("Running GPU solver: shared memory\n");

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
            // exit(32);
            break;
        }
        numMinesLast = *numMinesGPU;
        solveMinesweeperBoardSharedMemory<<<dim3(xNumBlocks, yNumBlocks, 1), dim3(xBlockSize, yBlockSize, 1)>>>(width, height, numMinesGPU, output);
        checkCudaError(cudaGetLastError());
        checkCudaError(cudaDeviceSynchronize());
    }

    cudaFree(numMinesGPU);
}

#undef FLAG
#undef COVERED
#undef OUT_OF_RANGE

#undef BLOCKSIZE_X
#undef BLOCKSIZE_Y

#endif // MINESWEEPER_CUH
