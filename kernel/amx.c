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


// =========================== Dense definitions =====================================

/*
    Dense BF16 matrices (tile B)
*/
typedef struct {
    int rows;
    int cols;
    int stride; // BF16 elts per row
    uint16_t* data;
} DenseBF16;

static DenseBF16* dense_bf16_alloc(int rows, int cols) {
    DenseBF16* m = (DenseBF16*)malloc(sizeof(DenseBF16));
    m->rows = rows;
    m->cols = cols;
    m->stride = (cols + 31) & ~31; // round up to nearest mult of 32
    size_t size = (size_t)rows * m->stride * sizeof(uint16_t);
    if (posix_memalign((void **)&m->data, 64, size) != 0) {
        perror("posix_memalign failure"); 
        exit(-1);
    }
    memset(m->data, 0, size);
    return m;
}

static void dense_bf16_free(DenseBF16* m) {
    if (m) {
        free(m->data);
        free(m);
    }
}

static inline uint16_t dense_bf16_get(const DenseBF16* m, int r, int c) {
    return m->data[r * m->stride + c];
}

static inline void dense_bf16_set(DenseBF16* m, int r, int c, uint16_t v) {
    m->data[r * m->stride + c];
}


/*
    Dense F32 matrices (tile C)
*/
typedef struct {
    int rows;
    int cols;
    int stride;
    float* data;
} DenseF32;

static DenseF32* dense_f32_alloc(int rows, int cols) {
    DenseF32* m = (DenseF32*)malloc(sizeof(DenseF32));
    m->rows = rows;
    m->cols = cols;
    m->stride = (cols + 15) & ~15;
    size_t size = (size_t)rows * m->stride * sizeof(float);
    if (posix_memalign((void **)&m->data, 64, size) != 0) {
        perror("posix_memalign failure");
        exit(-1);
    }
    memset(m->data, 0, size);
    return m;
}

static void dense_f32_free(DenseF32* m) {
    if (m) {
        free(m->data);
        free(m);
    }
}

static inline float dense_f32_get(const DenseF32* m, int r, int c) {
    return m->data[r * m->stride + c];
}


/* ================= TILE BUFFERS ================= */

static uint16_t global_a_buf[TILE_MAX_ROWS * K_TILE]__attribute__((aligned(64)));

static uint16_t global_b_buf[(K_TILE/2) * (TILE_MAX_F32_COLS * 2)]__attribute__((aligned(64)));

static float global_c_buf[TILE_MAX_ROWS * TILE_MAX_F32_COLS]__attribute__((aligned(64)));

/* Interleave B rows into global buffer */
static void pack_b_tile(const DenseBF16* B, int k0, int k_tile_bf16, int n0, int n_f32) {
    int k_pairs = k_tile_bf16 / 2;
    int b_stride = n_f32 * 2;
    memset(global_b_buf, 0, k_pairs * b_stride * sizeof(uint16_t));
    for (int kp = 0; kp < k_pairs; kp++) {
        int k_even = k0 + 2 * kp;
        int k_odd = k0 + 2 * kp + 1;
        for (int j = 0; j < n_f32; j++) {
            int n = n0 + j;
            global_b_buf[kp * b_stride + 2 * j] = (k_even < B->rows && n < B->cols) ? dense_bf16_get(B, k_even, n) : 0;
            global_b_buf[kp * b_stride + 2 * j + 1] = (k_odd < B->rows && n < B->cols) ? dense_bf16_get(B, k_odd, n) : 0;
        }
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



/* ================= MAIN ================= */

int main() {
    if (!request_amx_perm()) {
        exit(-1);
    }
    return 0;
}