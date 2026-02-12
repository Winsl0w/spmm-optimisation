#include <stdint.h>
#include <stdbool.h>
#include <immintrin.h>

#define MAX 1024
#define MAX_ROWS 16
#define MAX_COLS 64
#define STRIDE 64
#define BLOCK_SIZE 16   // maybe switch to 8 dependent on MM 


/*
    src1, src2 are the matrices being multiplied, tiles correspond to subdivisions of those matrices
    res/dest is the destination matrix
    in this current interation we handle int8, meaning the output matrix is populated with int32s

    ideal:
    C: 16 x 16
    A: 16 x 64
    B: 64 x 16
*/

// tile config register
typedef struct __tile_config {
    uint8_t palette_id;     // 0 AMX deactivated, 1 provides 8kb internal storage with each tiledata register holding up to 1kb (16 rows x 64 bytes)
    uint8_t start_row;      // restart position if an operation is interrupted (page fault etc)
    uint8_t reserved[14];   // reserved area
    uint16_t colsb[8];      // number of bytes per row for each tiledata register, max is 64
    uint8_t rows[8];        // number of rows for each tiledata register, max 16
} __tilecfg;

// Initialise tile config register for bf16
static void amx_tile_config_bf16(__tilecfg *cfg);

// Initialise the tile config register for int8
static void amx_tile_config_int8(__tilecfg *cfg);

// Request use of AMX from Linux kernel
static bool set_tiledata_use();

// Main int8 microkernel
static void amx_gemm_int8_16x16(const int8_t* restrict A, const int8_t* restrict B, int32_t* restrict C, int K);

static inline void amx_block_bf16_16x16_accumulate(const uint16_t* restrict A, const uint16_t* restrict B, float* restrict C, int ldc);

int main();