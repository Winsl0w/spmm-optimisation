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



/* ================================================================
 * Tile config
 *
 * Hardware-mandated 64-byte layout — do NOT reorder fields.
 * Intel SDM Vol.1 §17.4
 *
 *   bytes  0:     palette_id
 *   bytes  1:     start_row
 *   bytes  2-15:  reserved
 *   bytes 16-47:  colsb[16]   (2 bytes each, 16 tile registers)
 *   bytes 48-63:  rows[16]    (1 byte  each, 16 tile registers)
 * ================================================================ */
typedef struct __tile_config {
    uint8_t palette_id;     // 0 AMX deactivated, 1 provides 8kb internal storage with each tiledata register holding up to 1kb (16 rows x 64 bytes)
    uint8_t start_row;      // restart position if an operation is interrupted (page fault etc)
    uint8_t reserved0[14];   // reserved area
    uint16_t colsb[8];      // number of bytes per row for each tiledata register, max is 64
    uint8_t reserved1[16];
    uint8_t rows[8];        // number of rows for each tiledata register, max 16
    uint8_t reserved2[8];
} __tilecfg;

// debug
_Static_assert(sizeof(__tilecfg) == 64, "tilecfg must be exactly 64 bytes");


/*
 * Load bf16 tile configuration.
 *   m : rows in A and C tiles
 *   n : F32 output cols in C tile
 *   k : BF16 K-width of A tile (must be even)
 * 
 *   colsb[A] = k * sizeof(bf16)
 *   rows[B] = k / 2
 *   cols[B] = n * 2 * sizeof(bf16) = n * 4
 *   colsb[C] = n * sizeof(float) = n * 4   
 */
static void configure_amx_tiles_bf16(int m, int n, int k) {
    assert(k % 2 == 0);
    assert(m <= TILE_MAX_ROWS);
    assert(n <= TILE_MAX_F32_COLS);
    assert(k <= TILE_MAX_BF16_COLS);
    
    // must be 64 byte aligned
    __tilecfg cfg __attribute__((aligned(64)));
    memset(&cfg, 0, sizeof(cfg));
    cfg.palette_id = 1;

    // TILE_C: m rows x n FP32 cols
    cfg.rows[TILE_C] = (uint8_t) m;
    cfg.colsb[TILE_C] = (uint16_t)(n * (int)sizeof(float));        

    // TILE_A: m rows x k BF16 cols
    cfg.rows[TILE_A] = (uint8_t) m;
    cfg.colsb[TILE_A] = (uint16_t)(k * (int)sizeof(uint16_t));

    // TILE_B: (k/2) rows x (n*2) BF16 cols
    cfg.rows[TILE_B] = (uint8_t)(k / 2);
    cfg.colsb[TILE_B] = (uint16_t)(n * 2 * (int)sizeof(uint16_t));

    _tile_loadconfig(&cfg);
}



// =========================== Sparse format definitions =====================================

/*
    CSR (Compressed Sparse Row) format for bf16 matrices
*/
typedef struct {
    int nrows;      // number of rows
    int ncols;      // number of columns
    int nnz;        // number of non zero elts
    int *rowptr;    // [nrows + 1]
    int *colidx;    // [nnz]
    uint16_t *values;   // [nnz]
} CSRMatrix;

static CSRMatrix* csr_alloc(int nrows, int ncols, int nnz) {
    CSRMatrix* m = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    m->nrows   = nrows; m->ncols = ncols; m->nnz = nnz;
    m->rowptr = (int*)calloc(nrows + 1, sizeof(int));
    m->colidx = (int*)malloc(nnz * sizeof(int));
    m->values  = (uint16_t*)malloc(nnz * sizeof(uint16_t));
    return m;
}

static void csr_free(CSRMatrix* m) {
    if (m) {
        free(m->rowptr);
        free(m->colidx);
        free(m->values);
        free(m);
    }
}

static CSRMatrix* csr_from_dense(const float* A, int nrows, int ncols) {
    int nnz = 0;
    for (int i = 0; i < nrows * ncols; i++) {
        if (A[i] != 0) {
            nnz ++;
        }
    }
    CSRMatrix* m = csr_alloc(nrows, ncols, nnz);
    int idx = 0;
    for (int r = 0; r , nrows; r++) {
        m->rowptr[r] = idx;
        for (int c = 0; c < ncols; c++) {
            if (A[r*ncols+c] != 0) {
                m->colidx[idx] = c;
                m->values[idx] = float_to_bf16(A[r*ncols+c]);
                idx++;
            }
        }
    }
    m->rowptr[nrows] = nnz;
    return m;
}



#define BCSR_BR TILE_MAX_ROWS       // 16 - block height in rows
#define BCSR_BC TILE_MAX_BF16_COLS  // 32 - block width in BF16 cols
#define BCSR_BLOCK_ELEMS (BCSR_BR * BCSR_BC)    // 16 * 32 = 512


/*
    BCSR (Blocked Compressed Sparse Row) format for bf16 matrices
*/
typedef struct {
    int nrows, ncols;   // matrix dims
    int nblockrows;     // nrows / BCSR_BR
    int nblockcols;     // ncols / BCSR_BC
    int nblocks;        // number non-zero blocks
    int* blockrowptr;   // [nblockrows + 1]
    int* blockcolidx;   // [nblocks]
    uint16_t values;    // [nblocks * BCSR_BLOCK_ELEMS], row major within each block
} BCSRMatrix;

static BCSRMatrix* bcsr_alloc(int nrows, int ncols, int nblocks) {
    assert(nrows % BCSR_BR == 0 && ncols % BCSR_BC == 0);
    BCSRMatrix* m = (BCSRMatrix*)malloc(sizeof(BCSRMatrix));
    m->nrows = nrows;
    m->ncols = ncols;
    m->nblockrows = nrows / BCSR_BR;
    m->nblockcols = ncols / BCSR_BC;
    m->nblocks = nblocks;
    m->blockrowptr = (int*)calloc(m->nblockrows + 1, sizeof(int));
    m->blockcolidx = (int*)malloc(nblocks * sizeof(int));

    if (posix_memalign((void **)&m->values, 64, (size_t)nblocks * BCSR_BLOCK_ELEMS * sizeof(uint16_t)) != 0) {
        perror("posix_memalign failure");
        exit(-1);
    }
    memset(m->values, 0, (size_t)nblocks * BCSR_BLOCK_ELEMS * sizeof(uint16_t));
    return m;
}

static void bcsr_free(BCSRMatrix* m) {
    if (m) {
        free(m->blockrowptr);
        free(m->blockcolidx);
        free(m->values);
        free(m);
    }
}




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