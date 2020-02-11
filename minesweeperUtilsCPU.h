#ifndef EEC289Q_MINESWEEPER_UTILS_CPU_H
#define EEC289Q_MINESWEEPER_UTILS_CPU_H

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

namespace minesweeper {
namespace IOUtils {

uint8_t* toBigEndianBytes(uint32_t value, uint8_t bytes[4]) {
    bytes[0] = (value & 0xff000000) >> 24;
    bytes[1] = (value & 0x00ff0000) >> 16;
    bytes[2] = (value & 0x0000ff00) >> 8;
    bytes[3] = (value & 0x000000ff);

    return bytes;
}

uint32_t fromBigEndianBytes(uint8_t bytes[4]) {
    uint32_t ret;

    ret = (uint32_t)bytes[0] << 24;
    ret |= (uint32_t)bytes[1] << 16;
    ret |= (uint32_t)bytes[2] << 8;
    ret |= (uint32_t)bytes[3];

    return ret;
}

} // namespace IOUtils


// Board generation based on https://github.com/samhorlbeck/minesweeper-generator-solver/blob/master/solver.c
namespace boardGeneratorImpl {

#define DEBUG_PRINT_BOARD_GEN_DIAGNOSTICS false
#define DEBUG_MEASURE_BOARD_GEN_TIME DEBUG_PRINT_BOARD_GEN_DIAGNOSTICS

#define MINE -1

#define COVERED -51
#define FLAG    -52

#define MISSED_MINE     -101
#define CLICKED_MINE    -102
#define WRONG_FLAG      -103

int* g_groundTruth;
bool g_clickedOnMine = false;

std::mt19937 g_rng;

int clickTile(int width, int x, int y) {
    int idx = y * width + x;
    int value = g_groundTruth[idx];

    if (value == MINE)
        g_clickedOnMine = true;

    return value;
}

class TileStack {
public:
    struct Tile {
        int x, y;
        Tile(int xParam, int yParam) : x(xParam), y(yParam) {}
    };

    TileStack() {
        m_data = (Tile*)malloc(m_maxSize * sizeof(Tile));
    }
     
    ~TileStack() {
        free(m_data);
    }

    void push(int x, int y) {
        if (m_count >= m_maxSize) {
            m_maxSize *= 2;
            m_data = (Tile*)realloc(m_data, m_maxSize * sizeof(Tile));
        }
        m_data[m_count] = Tile(x, y);
        m_count += 1;
    }

    Tile pop() {
        Tile t = m_data[m_count-1];
        m_count -= 1;
        return t;
    }

    bool empty() {
        if (m_count == 0)
            return true;
        else
            return false;
    }

private:
    int m_maxSize = 8;
    int m_count = 0;
    Tile* m_data;
};

void printBoard(int* board, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            if (board[idx] == COVERED)
                printf(". ");
            else if (board[idx] == FLAG)
                printf("F ");
            else if (board[idx] == WRONG_FLAG)
                printf("W ");
            else if (board[idx] == MINE || board[idx] == MISSED_MINE)
                printf("* ");
            else if (board[idx] == CLICKED_MINE)
                printf("& ");
            else
                printf("%d ", board[idx]);
        }
        printf("\n");
    }
    printf("\n");
}

inline bool isOutOfRange(int x, int y, int width, int height) {
    if (x < 0 || y < 0 || x >= width || y >= height)
        return true;
    else
        return false;
}

void foreachNeighbor(int width, int height, int x, int y, std::function<void (int,int, int)> func) {
    for (int ny = y - 1; ny <= y + 1; ny++) {
        for (int nx = x - 1; nx <= x + 1; nx++) {
            if (isOutOfRange(nx, ny, width, height) || (nx == x && ny == y))
                continue;
            func(nx, ny, ny * width + nx);
        }
    }
}

void getNeighborInfo(int* board, int width, int height, int x, int y, int* numFlagged, int* numCovered) {
    int f = 0;
    int c = 0;
    foreachNeighbor(width, height, x, y, [=, &f, &c](int nx, int ny, int neighborIdx) {
        if (board[neighborIdx] == FLAG)
            f += 1;
        else if (board[neighborIdx] == COVERED)
            c += 1;
    });
    *numFlagged = f;
    *numCovered = c;
}

void uncoverTileImpl(int* board, int width, int height, int x, int y, TileStack* tilesToUncover) {
    int value = clickTile(width, x, y);
    int idx = y * width + x;
    board[idx] = value;

    if (value != 0)
        return;

    for (int ny = y - 1; ny <= y + 1; ny++) {
        for (int nx = x - 1; nx <= x + 1; nx++) {
            if (isOutOfRange(nx, ny, width, height) || (nx == x && ny == y))
                continue;

            int neighborIdx = ny * width + nx;
            if (board[neighborIdx] == COVERED) {
                // Temporarily store a nonsense value here, which will get replace when the queue is processed
                board[neighborIdx] = 100;
                tilesToUncover->push(nx, ny);
            }
        }
    }
}

void uncoverTile(int* board, int width, int height, int x, int y) {
    TileStack tilesToUncover;
    tilesToUncover.push(x, y);

    while (!tilesToUncover.empty()) {
        auto tile = tilesToUncover.pop();

        uncoverTileImpl(board, width, height, tile.x, tile.y, &tilesToUncover);
    }
}

bool performIteration(int* board, int width, int height) {
    bool moveMade = false;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            int value = board[idx];

            if (value == COVERED || value == FLAG || value == 0)
                continue;

            int numFlagged, numCovered;
            getNeighborInfo(board, width, height, x, y, &numFlagged, &numCovered);

            // If this tile has all of it's mines identified,
            // then uncover it's remaining covered neighbors.
            if (value == numFlagged) {
                board[idx] = 0;
                moveMade = true;
                foreachNeighbor(width, height, x, y, [board, width, height](int nx, int ny, int neighborIdx) {
                    int neighborValue = board[neighborIdx];
                    if (neighborValue == COVERED)
                        uncoverTile(board, width, height, nx, ny);
                });
            }
            // If the rest of the covered neighbors are all mines, mark them as such.
            else if (value - numFlagged == numCovered) {
                board[idx] = 0;
                moveMade = true;
                foreachNeighbor(width, height, x, y, [board, width, height](int nx, int ny, int neighborIdx) {
                    int neighborValue = board[neighborIdx];
                    if (neighborValue == COVERED)
                        board[neighborIdx] = FLAG;
                });
            }
        }
    }

    return moveMade;
}

void simpleSolverCPU(int* groundTruth, int* board, int width, int height) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    int numIterations = 0;
    while (performIteration(board, width, height)) {
        numIterations += 1;
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (DEBUG_MEASURE_BOARD_GEN_TIME)
        std::cout << "CPU Solve Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms\n" << std::endl;
}

int getNumAdjacentMines(int* board, int width, int height, int x, int y) {
    int numMines = 0;

    foreachNeighbor(width, height, x, y, [board, &numMines](int nx, int ny, int neighborIdx) {
        if (board[neighborIdx] == MINE)
            numMines += 1;
    });

    return numMines;
}

void assignValues(int* board, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;

            // If this location is a mine, do nothing
            if (board[idx] == MINE)
                continue;

            int numMines = getNumAdjacentMines(board, width, height, x, y);

            board[idx] = numMines;
        }
    }
}

// Sanity check
bool checkBoard(int* board, int width, int height) {
    static const int offsets[8][2] = { {-1, -1}, {0, -1}, {1, -1}, {-1, 0}, {1, 0}, {-1, 1}, {0, 1}, {1, 1} };

    bool isCorrect = true;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (board[y * width + x] == MINE)
                continue;

            int numMines = 0;
            for (int i = 0; i < 8; ++i) {
                int xn = x + offsets[i][0];
                int yn = y + offsets[i][1];
                
                if (xn < 0 || yn < 0 || xn >= width || yn >= height)
                    continue;

                if (board[yn * width + xn] == MINE)
                    numMines += 1;
            }
            if (numMines != board[y * width + x]) {
                fprintf(stderr, "[Error in board generation] Incorrect value at location: %d %d\n", x, y);
                isCorrect = false;
            }
        }
    }

    return isCorrect;
}

int* generateRandomBoard(int numMines, int width, int height, int startX, int startY) {
    int* board = (int*)malloc(width * height * sizeof(int));
    memset(board, 0, width*height * sizeof(int));

    int mines = 0;

    while (mines < numMines) {
        int x = g_rng() % width;
        int y = g_rng() % height;

        // If this location is too close to the starting location, continue
        if (x > startX - 2 && x < startX + 2 && y > startY - 2 && y < startY + 2)
            continue;

        // If this location already has a mine, continue
        if (board[y*width + x] == MINE)
            continue;

        // Place a mine
        board[y*width + x] = MINE;
        mines += 1;
    }

    assignValues(board, width, height);
    if (!checkBoard(board, width, height)) {
        fprintf(stderr, "[Error in board generation] Sanity check failed during initial board generation");
        free(board);
        exit(1);
    }

    return board;
}

// Find the mines that the solver missed,
// the mines that the solver clicked, and
// locations that were incorrectly flagged as mines.
int getNumMissedMines(const int* const groundTruth, const int* const solvedBoard, int width, int height, int* annotatedBoard = NULL) {
    if (annotatedBoard)
        memcpy(annotatedBoard, solvedBoard, width * height * sizeof(int));

    int numMissedMines = 0;
    int numClickedMines = 0;
    int numWrongFlags = 0;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;

            // If the location is a mine
            if (groundTruth[idx] == MINE) {
                // If the location is still coverd, then the solver missed the mine.
                if (solvedBoard[idx] == COVERED) {
                    numMissedMines += 1;
                    if (annotatedBoard) annotatedBoard[idx] = MISSED_MINE;
                }
                // Otherwise, if the location is not a flag, then the solver clicked it.
                else if (solvedBoard[idx] != FLAG) {
                    numClickedMines += 1;
                    if (annotatedBoard) annotatedBoard[idx] = CLICKED_MINE;
                }
            }
            // If the location is not a mine
            else {
                // If the location is a flag, then it was flagged incorrectly.
                if (solvedBoard[idx] == FLAG) {
                    numWrongFlags += 1;
                    if(annotatedBoard) annotatedBoard[idx] = WRONG_FLAG;
                }
                // Otherwise, if the location is uncovered, replace the value with
                // the original ground truth value for debug printing purposes
                else if (solvedBoard[idx] != COVERED) {
                    if(annotatedBoard) annotatedBoard[idx] = groundTruth[idx];
                }
            }
        }
    }

    if (numClickedMines > 0 || g_clickedOnMine) {
        assert(g_clickedOnMine && numClickedMines > 0 && "Inconsistency between g_clickedOnMine and clickedMines count");
        fprintf(stderr, "[Error in board generation] The solver clicked a mine\n");
        exit(CLICKED_MINE);
    }
    else if (numWrongFlags > 0) {
        fprintf(stderr, "[Error in board generation] The solver flagged a non-mine\n");
        exit(WRONG_FLAG);
    }

    return numMissedMines;
}


// From the ground truth, remove a mine that is adjacent to an uncovered, non-mine value
void removeMine(int* groundTruth, int* solvedBoard, int width, int height) {

    // To remove a mine, replace it's location with the count of the number of mines adjacent to it,
    // and then for all non-mine neighbors, recalculate their adjacent mine values.
    auto removeThisMine = [groundTruth, solvedBoard, width, height](int x, int y, int idx) {

        // Only need to update the ground truth, because it is still covered in solved board
        groundTruth[idx] = getNumAdjacentMines(groundTruth, width, height, x, y);

        foreachNeighbor(width, height, x, y, [groundTruth, solvedBoard, width, height](int nx, int ny, int neighborIdx) {
            assert(solvedBoard[neighborIdx] != 0 && "Incorrect board state - found a 0 neighboring a missed mine");

            // If the neighbor is not a mine, need to update it's ground truth value
            if (groundTruth[neighborIdx] != MINE) {
                int numAdjacentMines = getNumAdjacentMines(groundTruth, width, height, nx, ny);
                groundTruth[neighborIdx] = numAdjacentMines;

                // If the neighbor is uncovered and not flagged, need to update the solvedBoard too
                if (solvedBoard[neighborIdx] > 0) {
                    solvedBoard[neighborIdx] = numAdjacentMines;
                }
            }
        });

        // We may have made some uncovered neighbors 0. These need to be dealt with in the solved board
        // by uncovering their neighbors.
        foreachNeighbor(width, height, x, y, [solvedBoard, width, height](int nx, int ny, int neighborIdx) {
            if (solvedBoard[neighborIdx] == 0) {
                foreachNeighbor(width, height, nx, ny, [solvedBoard, width, height](int nnx, int nny, int nnIdx) {
                    if (solvedBoard[nnIdx] == COVERED)
                        uncoverTile(solvedBoard, width, height, nnx, nny);
                });
            }
        });

        if (!checkBoard(groundTruth, width, height)) {
            fprintf(stderr, "[Error in board generation] Sanity check failed after removing a mine");
            free(groundTruth);
            exit(1);
        }
    };

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;

            if (groundTruth[idx] == MINE && solvedBoard[idx] == COVERED) {
                bool hasUncoveredNeighbor = false;

                foreachNeighbor(width, height, x, y, [solvedBoard, &hasUncoveredNeighbor](int nx, int ny, int neighborIdx) {
                    int neighborValue = solvedBoard[neighborIdx];
                    assert(neighborValue != 0 && "Incorrect board state - found a 0 neighboring a missed mine");
                    if (neighborValue > 0)
                        hasUncoveredNeighbor = true;
                });

                // If we found a mine next to an uncovered, non-mine neighbor,
                // then remove it and return.
                if (hasUncoveredNeighbor) {
                    removeThisMine(x, y, idx);
                    return;
                }
            }
        }
    }

    // If we've gotten here, then we were not able to find a missed mine that has an uncovered, non-mine neighbor
    // So remove the first missed mine encountered and return.
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;

            if (groundTruth[idx] == MINE && solvedBoard[idx] == COVERED) {
                removeThisMine(x, y, idx);
                return;
            }
        }
    }
}

int* setupCoveredBoard(int width, int height, int startX, int startY) {
    int* board = (int*)malloc(width * height * sizeof(int));

    for (int i = 0; i < width*height; ++i)
        board[i] = COVERED;

    uncoverTile(board, width, height, startX, startY);
    assert(board[startY * width + startX] != MINE && "Start position is a mine!!!");

    return board;
}

int* generateSimpleBoardImpl(int width, int height, int numMines, int startX, int startY, int* numMinesActual) {
    if (!DEBUG_PRINT_BOARD_GEN_DIAGNOSTICS)
        printf("Generating board...");

    int* groundTruth = generateRandomBoard(numMines, width, height, startX, startY);
    g_groundTruth = groundTruth;
    g_clickedOnMine = false;

    int* solvedBoard = setupCoveredBoard(width, height, startX, startY);

    simpleSolverCPU(groundTruth, solvedBoard, width, height);

    int* annotatedBoard = (int*)malloc(width * height * sizeof(int));
    int numMissedMines = getNumMissedMines(groundTruth, solvedBoard, width, height, annotatedBoard);

    int numIters = 0;
    int actualNumMines = numMines;

    while (numMissedMines > 0) {
        numIters += 1;
        if (!DEBUG_MEASURE_BOARD_GEN_TIME && numIters % 25 == 0)
            printf(".");

        removeMine(groundTruth, solvedBoard, width, height);
        actualNumMines -= 1;

        if (DEBUG_PRINT_BOARD_GEN_DIAGNOSTICS)
            printf("Num missed mines: %d\n\n", numMissedMines);

        simpleSolverCPU(groundTruth, solvedBoard, width, height);
        numMissedMines = getNumMissedMines(groundTruth, solvedBoard, width, height, annotatedBoard);
    }

    if (DEBUG_PRINT_BOARD_GEN_DIAGNOSTICS) {
        printf("\n");
        printf("Num mines requested : %d\n", numMines);
        printf("Actual num mines    : %d\n", actualNumMines);
        printf("\n");
    }

    assert(g_clickedOnMine == false && "Clicked on a mine during board generation");

    // Final check that ground truth tile values are correct
    if (!checkBoard(groundTruth, width, height)) {
        fprintf(stderr, "[Error in board generation] Sanity check failed during final check");
        free(groundTruth);
        free(annotatedBoard);
        free(solvedBoard);
        exit(1);
    }

    // Make sure mine count is correct
    int mineCountCheck = 0;
    for (int i = 0; i < width*height; ++i) {
        if (groundTruth[i] == MINE)
            mineCountCheck += 1;
    }

    if (actualNumMines != mineCountCheck) {
        fprintf(stderr, "[Error in board generation] Inconsistency in mine count\n");
    }

    // Final check that board is solvable
    for (int i = 0; i < width*height; ++i)
        solvedBoard[i] = COVERED;

    uncoverTile(solvedBoard, width, height, startX, startY);
    assert(solvedBoard[startY * width + startX] != MINE && "Start position is a mine!!!");
    simpleSolverCPU(groundTruth, solvedBoard, width, height);
    numMissedMines = getNumMissedMines(groundTruth, solvedBoard, width, height, annotatedBoard);

    if (numMissedMines != 0) {
        fprintf(stderr, "[Error in board generation] Generated unsolvable board\n");
    }

    assert(g_clickedOnMine == false && "Clicked on a mine when making sure board is solvable");

    free(annotatedBoard);
    free(solvedBoard);

    if (!DEBUG_PRINT_BOARD_GEN_DIAGNOSTICS)
        printf(" done!\n\n");

    *numMinesActual = actualNumMines;

    return groundTruth;
}

#undef MINE
#undef COVERED
#undef MISSED_MINE
#undef CLICKED_MINE
#undef WRONG_FLAG

#undef DEBUG_PRINT_BOARD_GEN_DIAGNOSTICS
#undef DEBUG_MEASURE_BOARD_GEN_TIME

} // namespace boardGeneratorImpl

void minesweeperCPU(int width, int height, int numMines, int startX, int startY, int* output) {
    int* board = boardGeneratorImpl::setupCoveredBoard(width, height, startX, startY);

    int numIterations = 0;
    while (boardGeneratorImpl::performIteration(board, width, height)) {
        numIterations += 1;
    }

    for (int i = 0; i < width*height; ++i) {
        if (board[i] == FLAG)
            output[i] = -1;
    }

    free(board);
}
#undef FLAG


void printBoard(int8_t* board, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            if (board[idx] == -1)
                printf("* ");
            else
                printf("%d ", board[idx]);
        }
        printf("\n");
    }
    printf("\n");
}

int8_t* generateSimpleBoard(int width, int height, int numMinesRequested, int startX, int startY, int* numMinesActual) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    int* board = boardGeneratorImpl::generateSimpleBoardImpl(width, height, numMinesRequested, startX, startY, numMinesActual);

    // Convert board to int8_t
    int8_t* ret = (int8_t*)malloc(width * height * sizeof(uint8_t));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            ret[idx] = (int8_t)board[idx];
        }
    }

    free(board);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time to generate board: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl << std::endl;

    //boardGeneratorImpl::printBoard(ret, width, height);

    return ret;
}

bool saveBoardToFile(int width, int height, int numMines, int startX, int startY, int8_t* board, std::string filename) {
    std::ofstream file(filename, std::ios::binary);

    if (!file) {
        fprintf(stderr, "[Error] Could not open file '%s' to save board\n", filename.c_str());
        return false;
    }

    uint8_t tmp[4];
    file.write((char*)IOUtils::toBigEndianBytes(width, tmp), sizeof(tmp));
    file.write((char*)IOUtils::toBigEndianBytes(height, tmp), sizeof(tmp));
    file.write((char*)IOUtils::toBigEndianBytes(numMines, tmp), sizeof(tmp));
    file.write((char*)IOUtils::toBigEndianBytes(startX, tmp), sizeof(tmp));
    file.write((char*)IOUtils::toBigEndianBytes(startY, tmp), sizeof(tmp));

    file.write((char*)board, width * height * sizeof(int8_t));

    file.close();

    return true;
}

int8_t* loadBoardFromFile(int* width, int* height, int* numMines, int* startX, int* startY, std::string filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        fprintf(stderr, "[Error] Could not open file '%s' to load board\n", filename.c_str());
        return NULL;
    }

    uint8_t tmp[4];
    file.read((char*)&tmp, sizeof(tmp));
    uint32_t w = IOUtils::fromBigEndianBytes(tmp);

    file.read((char*)&tmp, sizeof(tmp));
    uint32_t h = IOUtils::fromBigEndianBytes(tmp);

    file.read((char*)&tmp, sizeof(tmp));
    uint32_t nm = IOUtils::fromBigEndianBytes(tmp);

    file.read((char*)&tmp, sizeof(tmp));
    uint32_t sx = IOUtils::fromBigEndianBytes(tmp);

    file.read((char*)&tmp, sizeof(tmp));
    uint32_t sy = IOUtils::fromBigEndianBytes(tmp);

    int8_t* board = (int8_t*)malloc(w * h * sizeof(int8_t));
    file.read((char*)board, w * h * sizeof(int8_t));

    file.close();

    *width = w;
    *height = h;
    *numMines = nm;
    *startX = sx;
    *startY = sy;

    return board;
}

} // namespace minesweeper


#endif // EEC289Q_MINESWEEPER_UTILS_CPU_H
