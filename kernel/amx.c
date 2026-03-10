#include <stdint.h>
#include <stdbool.h>
#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>


/*  ==============
  *  Constants
  * ============== */
#define TILE_C 0    // accumulator
#define TILE_A 1    // A tile
#define TILE_B 2    // B tile


#define TILE_MAX_ROWS 16
#define TILE_MAX_BYTES 64
#define TILE_MAX_BF16_COLS 32  // 64 bytes / 2
#define TILE_MAX_F32_COLS 16   // 64 bytes / 4

#define K_TILE 32           // num BF16 cols in A tile = num K elts processed per TDPBF16PS

/* ================= AMX PERMISSION ================= */

// see https://www.kernel.org/doc/Documentation/x86/xstate.rst for syscall documentation

#ifndef ARCH_GET_XCOMP_PERM
#define ARCH_GET_XCOMP_PERM 0x1022
#endif
#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif

// Request use of AMX from Linux kernel
static bool request_amx_perm() {
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        printf("\n Failed to enable XFEATURE_XTILEDATA \n\n");
        return false;
    }
    
    printf("\n TILE DATA USE SET - OK \n\n");
    return true;
}



/* ================= TILE CONFIG ================= */

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
    uint16_t colsb[16];      // number of bytes per row for each tiledata register, max is 64
    uint8_t rows[16];        // number of rows for each tiledata register, max 16
} __tilecfg;


// BF16 tile configuration
static void amx_config_bf16(void) {
    __tilecfg cfg __attribute__((aligned(64)));
    memset(&cfg, 0, sizeof(cfg));

    cfg.palette_id = 1;
    cfg.start_row = 0;

    // tmm0: C (FP32 accumulator)
    cfg.rows[0] = TILE_ROWS;
    cfg.colsb[0] = TILE_COLSB_F32;        

    // tmm1: A (BF16)
    cfg.rows[1] = TILE_ROWS;
    cfg.colsb[1] = TILE_COLSB_BF16;      

    // tmm2: B (BF16)
    cfg.rows[2] = TILE_ROWS;
    cfg.colsb[2] = TILE_COLSB_BF16;    

    _tile_loadconfig(&cfg);
}



// =========================== Sparse format definitions =====================================

/*
    CSR (Compressed Sparse Row) format for bf16 matrices
*/
typedef struct {
    int nrows;      // number of rows in the matrix
    int ncols_tiles;    // number of column tiles (number of 64 column blocks)
    int *rowptr;    // size nrows + 1
    int *colidx;    // size = number of nonzeros
    uint16_t *values;   // size = number of nonzeros, bf16 values
} csr_matrix_bf16_t;


/*
    BCSR (Blocked Compressed Sparse Row) format for bf16 matrices
*/
typedef struct {
    int nblockrows;      // number of rows in the matrix
    int nblockcols;    // number of column tiles (number of 64 column blocks)
    int *browptr;    // size nrows/BLOCK_SIZE + 1
    int *bcolidx;    // size = number of nonzero blocks
    uint16_t *blocks;   // size = number of nonzero blocks * BLOCK_SIZE * 64, bf16 values
} bcsr_matrix_bf16_t;



/* ================= BF16 MICROKERNELS ================= */

void csr_spmm_bf16(const csr_matrix_bf16_t* A, const uint16_t* B, float* C, int N) {
    (void) N;

    for (int i = 0; i < A->nrows; i++) {
        /*  load existing output row into C
         *  still use 16 row cfg however this means only row zero holds data
         *  tile zeroed first then accumulate and store it */
        _tile_zero(0);

        for (int p = A->rowptr[i]; p < A->rowptr[i+1]; p++) {
            int col_tile = A -> colidx[p];  // tile-col of B
            uint16_t val_a = A->values[p];  // scalar value from A

            uint16_t a_tile[TILE_ROWS * TILE_COLS] __attribute__((aligned(64)));
            for (int k = 0; k < TILE_ROWS * TILE_COLS; k++) {
                a_tile[k] = val_a;
            }

            const uint16_t* b_tile = B + col_tile * TILE_ROWS * TILE_COLS;

            _tile_loadd(1, a_tile, TILE_COLSB_BF16);
            _tile_loadd(2, b_tile, TILE_COLSB_BF16);
            __asm__ volatile(".byte 0xc4, 0xe2, 0x73, 0x5c, 0xc2" ::: "memory");
            // _tile_dpbf16ps(0, 1, 2);
        }

        float c_tile[TILE_ROWS * TILE_COLS] __attribute__((aligned(64)));
        _tile_stored(0, c_tile, TILE_COLSB_F32);

        float* c_row = C + i * TILE_COLS;
        for (int j = 0; j < TILE_COLS; j++) {
            c_row[j] += c_tile[j];
        }
    }
}


void bcsr_spmm_bf16(const bcsr_matrix_bf16_t* A, const uint16_t* B, float* C) {
    for (int br = 0; br < A->nblockrows; br++) {
        float* c_tile = C + br * TILE_ROWS * TILE_COLS;

        _tile_loadd(0, c_tile, TILE_COLSB_F32);

        for (int p = A->browptr[br]; p < A->browptr[br + 1]; p++) {
            int bc = A->bcolidx[p];
            const uint16_t *a_block = A->blocks + (size_t)p *TILE_ROWS * TILE_COLS;
            const uint16_t *b_block = B + (size_t)bc *TILE_ROWS * TILE_COLS;

            _tile_loadd(1, a_block, TILE_COLSB_BF16);
            _tile_loadd(2, b_block, TILE_COLSB_BF16);
            _tile_dpbf16ps(0, 1, 2);
        }

        _tile_stored(0, c_tile, TILE_COLSB_F32);
    }
}


/* ================= HELPERS ================= */

static float bf16_to_float(uint16_t b) {
    union {uint32_t u; float f;} temp;
    temp.u = (uint32_t)b << 16;
    return temp.f;
}

static uint16_t float_to_bf16(float f) {
    union {uint32_t u; float f;} temp;
    temp.f = f;
    return (uint16_t)(temp.u >> 16);
}

static void print_f32_matrix(const char* lab, const float* M, int rows, int cols) {
    printf("%s (%d x %d):\n", lab, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) 
            printf("%8.2f", M[i * cols + j]);
        printf("\n");
    }
    printf("\n");
}

/*
 *  Test Harnesses
*/

// CSR Test
static void test_csr(void) {
    printf("== CSR TEST ==\n\n");

    int row_ptr[] = {0, 1, 2, 3, 4};
    int col_idx[] = {0, 0, 0, 0};
    uint16_t vals[4];
    for (int i = 0; i < 4; i++) {
        vals[i] = float_to_bf16(1.0f);
    }

    csr_matrix_bf16_t A = {
        .nrows = 4,
        .ncols_tiles = 1,
        .rowptr = row_ptr,
        .colidx = col_idx,
        .values = vals,
    };

    uint16_t B[TILE_ROWS * TILE_COLS] __attribute__((aligned(64)));
    memset(B, 0, sizeof(B));
    for (int r = 0; r < TILE_ROWS; r++) {
        B[r * TILE_COLS + r] = float_to_bf16((float)(r + 1));
    }

    float C[4 * TILE_COLS] __attribute__((aligned(64)));
    memset(C, 0, sizeof(C));

    csr_spmm_bf16(&A, B, C, TILE_COLS);

    print_f32_matrix("C = A_csr * B", C, 4, TILE_COLS);
}


// BCSR Test
static void test_bcsr(void) {
    printf("== BCSR TEST ++ \n\n");

    const int BLOCK = TILE_ROWS * TILE_COLS;

    int brow_ptr[] = {0, 1, 2};
    int bcol_idx[] = {0, 1};

    uint16_t blocks[2 * BLOCK] __attribute__((aligned(64)));
    for (int k = 0; k < 2 * BLOCK; k++) {
        blocks[k] = float_to_bf16(1.0f);
    }

    bcsr_matrix_bf16_t A = {
        .nblockrows = 2,
        .nblockcols = 2,
        .browptr = brow_ptr,
        .bcolidx = bcol_idx,
        .blocks = blocks,
    };

    uint16_t B[2 * BLOCK] __attribute__((aligned(64)));
    memset(B, 0, sizeof(B));
    for (int r = 0; r < TILE_ROWS; r++) {
        B[0 * BLOCK + r * TILE_COLS + r] = float_to_bf16((float)(r + 1));
        B[1 * BLOCK + r * TILE_COLS + r] = float_to_bf16((float)(10 * (r + 1)));
    }

    float C[2 * BLOCK] __attribute__((aligned(64)));
    memset(C, 0, sizeof(C));

    bcsr_spmm_bf16(&A, B, C);

    print_f32_matrix("C block-row 0 = A[0,0]*B[0]", C, TILE_ROWS, TILE_COLS);
    print_f32_matrix("C block-row 1 = A[1,1]*B[1]", C + BLOCK, TILE_ROWS, TILE_COLS);
}





/* ================= MAIN ================= */

int main() {
    if (!request_amx_perm()) {
        exit(-1);
    }

    _tile_release();    // release previous config if set (force XSAVE initialisation)
    amx_config_bf16();

    test_csr();
    test_bcsr();

    _tile_release();
    return 0;
}