#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cassert>
#include <cstdlib>
#include <random>
#include <stdio.h>

#include "minesweeperUtilsCPU.h"
#include "minesweeperUtilsGPU.cuh"

#ifdef SHARED
    #include "minesweeper-sharedMemory.cu"
#else
    #include "minesweeper-simple.cu"
#endif

bool checkSolution(int width, int height, int8_t* groundTruth, int* solution) {
    for (int i = 0; i < width * height; ++i) {
        if (groundTruth[i] == -1 && solution[i] != -1)
            return false;
        else if (solution[i] == -1 && groundTruth[i] != -1)
            return false;
    }

    return true;
}

bool solveBoardCPU(int width, int height, int numMines, int startX, int startY, int8_t* groundTruth) {
    int* groundTruth32 = (int*)malloc(width * height * sizeof(int));

    for (int i = 0; i < width*height; ++i) {
        groundTruth32[i] = groundTruth[i];
    }

    minesweeper::boardGeneratorImpl::g_groundTruth = groundTruth32;
    minesweeper::boardGeneratorImpl::g_clickedOnMine = false;

    int* output = (int*)malloc(width * height * sizeof(int));
    memset(output, 0, width * height * sizeof(int));

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    minesweeper::minesweeperCPU(width, height, numMines, startX, startY, output);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "CPU Solve Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;

    bool isCorrect = checkSolution(width, height, groundTruth, output);

    free(output);
    free(groundTruth32);

    if (minesweeper::boardGeneratorImpl::g_clickedOnMine) {
        printf("Clicked on a mine!!!\n");
        return false;
    }

    return isCorrect;
}

bool solveBoardGPU(int width, int height, int numMines, int startX, int startY, int8_t* groundTruth) {
    // Choose which GPU to run on, change this on a multi-GPU system.
    checkCudaError(cudaSetDevice(0));

    int8_t* d_groundTruth;
    checkCudaError(cudaMalloc(&d_groundTruth, width * height * sizeof(int8_t)));
    checkCudaError(cudaMemcpy(d_groundTruth, groundTruth, width * height * sizeof(int8_t), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpyToSymbol(donottouch::donottouchGroundTruth, &d_groundTruth, sizeof(int8_t*)));

    bool clickedOnMine = false;
    checkCudaError(cudaMemcpyToSymbol(donottouch::donottouchClickedOnMine, &clickedOnMine, sizeof(clickedOnMine)));
    checkCudaError(cudaMemcpyToSymbol(donottouch::donottouchWidth, &width, sizeof(width)));

    int* output;
    checkCudaError(cudaMallocManaged(&output, width * height * sizeof(int)));
    memset(output, 0, width * height * sizeof(int));

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    minesweeperGPU(width, height, numMines, startX, startY, output);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "GPU Solve Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;

    bool isCorrect = checkSolution(width, height, groundTruth, output);

    checkCudaError(cudaFree(output));
    checkCudaError(cudaFree(d_groundTruth));

    checkCudaError(cudaMemcpyFromSymbol(&clickedOnMine, donottouch::donottouchClickedOnMine, sizeof(clickedOnMine)));
    if(clickedOnMine) {
        printf("Clicked on a mine!!!\n");
        return false;
    }

    return isCorrect;
}

#define RANDOM_SEED 2113

#define GENERATE_BOARD 0
#define LOAD_BOARD_AND_SOLVE 1
#define SOLVE_PROVIDED_BOARDS 2

#define USE_CPU_SOLVER false

// Only uncomment one at a time
//#define WHAT_TO_DO GENERATE_BOARD
//#define WHAT_TO_DO LOAD_BOARD_AND_SOLVE
#define WHAT_TO_DO SOLVE_PROVIDED_BOARDS

int main() {

    if (WHAT_TO_DO == GENERATE_BOARD) {
        std::mt19937 rng(RANDOM_SEED);
        minesweeper::boardGeneratorImpl::g_rng = rng;

        int width = 100;
        int height = 100;
        float percentMines = 0.10f;
        const char* filename = "boards/100x100-10pct.dat";

        int numMinesRequested = (int)((width * height) * percentMines);
        int startX = rng() % width;
        int startY = rng() % height;

        int numMinesActual;

        int8_t* groundTruth = minesweeper::generateSimpleBoard(width, height, numMinesRequested, startX, startY, &numMinesActual);

        if (!minesweeper::saveBoardToFile(width, height, numMinesActual, startX, startY, groundTruth, filename)) {
            return 1;
        }
    }
    else if (WHAT_TO_DO == LOAD_BOARD_AND_SOLVE) {
        const char* filename = "boards/100x100-10pct.dat";
        int width, height, numMinesActual, startX, startY;

        int8_t* groundTruth = minesweeper::loadBoardFromFile(&width, &height, &numMinesActual, &startX, &startY, filename);

        if (!groundTruth) {
            return 1;
        }

        bool isCorrect = false;
        if (USE_CPU_SOLVER)
            isCorrect = solveBoardCPU(width, height, numMinesActual, startX, startY, groundTruth);
        else
            isCorrect = solveBoardGPU(width, height, numMinesActual, startX, startY, groundTruth);

        if (isCorrect)
            printf("Board '%s': PASS!\n", filename);
        else
            printf("Board '%s': FAIL!\n", filename);

        free(groundTruth);
    }
    else if (WHAT_TO_DO == SOLVE_PROVIDED_BOARDS) {
        bool allCorrect = true;
        const char* filenames[] = {
                                    "boards/provided/10x10-10pct.dat",
                                    "boards/provided/10x10-20pct.dat",
                                    "boards/provided/10x10-30pct.dat",
                                    "boards/provided/5x2000-22pct.dat",
                                    "boards/provided/100x100-20pct.dat",
                                    "boards/provided/1000x1000-10pct.dat",
                                    "boards/provided/1000x1000-15pct.dat",
                                    "boards/provided/1000x1000-25pct.dat",
                                    "boards/provided/1111x2419-16pct.dat",
                                    "boards/provided/1337x422-23pct.dat",
                                    "boards/provided/2323x7-24pct.dat",
                                    "boards/provided/2419x1111-17pct.dat",
                                    "boards/provided/2500x2500-15pct.dat",
                                    "boards/provided/2500x2500-30pct.dat",
                                  };

        for (auto filename : filenames) {
            int width, height, numMinesActual, startX, startY;

            int8_t* groundTruth = minesweeper::loadBoardFromFile(&width, &height, &numMinesActual, &startX, &startY, filename);

            if (!groundTruth) {
                allCorrect = false;
                continue;
            }

            bool isCorrect = false;
            if (USE_CPU_SOLVER)
                isCorrect = solveBoardCPU(width, height, numMinesActual, startX, startY, groundTruth);
            else
                isCorrect = solveBoardGPU(width, height, numMinesActual, startX, startY, groundTruth);

            if (isCorrect)
                printf("Board '%s': PASS\n\n", filename);
            else {
                printf("Board '%s': FAIL\n\n", filename);
                allCorrect = false;
            }

            free(groundTruth);
        }

        if (allCorrect)
            printf("All boards passed!\n");
        else
            printf("Some boards failed!\n");
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    checkCudaError(cudaDeviceReset());

    return 0;
}
