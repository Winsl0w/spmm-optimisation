#include <stdint.h>
#include <stdbool.h>
#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdio.h>

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

// Main BCSR FP microkernel
static inline void amx_block_bf16_16x16_accumulate(const uint16_t* restrict A, const uint16_t* restrict B, float* restrict C, int ldc);