# Minesweeper Solver for Simple Boards
This code implements a simple GPU-based minesweeper solver that solves only a subset of minesweeper boards.

It also includes a board generator and CPU-based solver to generate boards within the solvable subset.

## Building the CUDA Code
This code uses C++11 features, so you may need to pass the flag `-std=c++11` to the nvcc compiler, e.g.:

  ```
  nvcc -std=c++11 main.cu
  ```

## Building the Shared Memory Implementation
This code also includes a solution that utilizes shared memory in CUDA. To compile this version, pass the argument `-D SHARED` to the nvcc compiler:

  ```
  nvcc -D SHARED -std=c++11 main.cu
  ```

Note: There are two different options for reading values into shared memory implemented in this code. See `minesweeper-sharedMemory.cu` for a description. A pair of `#define` lines at the top of that file controls which version is used. There are certainly other options for reading into shared memory which may be more efficient than these two.

## A Note About the Code
This code was written relatively quickly and may not conform to all best practices and such. If something seems strange or inefficient, it probably is!
