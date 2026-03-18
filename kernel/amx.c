#define _GNU_SOURCE
#include <stdint.h>
#include <stdbool.h>
#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>



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

// Conditional compilation to include the headers from SuiteSparse
#ifdef USE_AMD
    #include "amd.h"
#endif

typedef enum {
    REORDER_NONE = 0,
    REORDER_AMD,
    REORDER_RCM
} ReorderMethod;


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
static int request_amx_perm() {
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        printf("\n Failed to enable XFEATURE_XTILEDATA \n\n");
        return 0;
    }
    
    printf("\n TILE DATA USE SET - OK \n\n");
    return 1;
}



/* ================================================================
  Tile config
 
  Hardware-mandated 64-byte layout — do not reorder fields.
  Intel SDM Vol.1 §17.4
 
 *   byte  0     : palette_id
 *   byte  1     : start_row
 *   bytes 2-15  : reserved (zero)
 *   bytes 16-31 : colsb[0..7]  (uint16, bytes per row for each tile)
 *   bytes 32-47 : reserved (zero)
 *   bytes 48-55 : rows[0..7]   (uint8, row count for each tile)
 *   bytes 56-63 : reserved (zero)
   ================================================================ */

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
// _Static_assert(sizeof(__tilecfg) == 64, "tilecfg must be exactly 64 bytes");


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

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
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
    for (int r = 0; r < nrows; r++) {
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
    uint16_t* values;    // [nblocks * BCSR_BLOCK_ELEMS], row major within each block
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
    m->data[r * m->stride + c] = v;
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


/* ===================================================
    CSR MICROKERNEL
 * =================================================== */ 

static void amx_spmm_csr(const CSRMatrix* A, const DenseBF16* B, DenseF32* C) {
    assert(A->ncols == B->rows && A->nrows == C->rows && B->cols == C->cols);

    int M = A->nrows, N = B->cols, K = A->ncols;

    for (int r = 0; r < M; r++) {
        memset(&C->data[r * C->stride], 0, C->cols * sizeof(float));
    }

    int n_kstrips = (A->ncols + K_TILE - 1) / K_TILE;
    char *kstrip_used = (char *)calloc((size_t)n_kstrips, 1);
    if (!kstrip_used) { perror("amx_spmm_csr kstrip_used"); return; }
 
    for (int m0 = 0; m0 < M; m0 += TILE_MAX_ROWS) {
        int m_tile = (m0 + TILE_MAX_ROWS <= M) ? TILE_MAX_ROWS : M - m0;

        int n_used = 0;

        for (int ii = 0; ii < m_tile; ii++) {
            int row = m0 + ii;
            for (int p = A->rowptr[row]; p < A->rowptr[row+1]; p++) {
                int ks = A->colidx[p] / K_TILE;
                if (!kstrip_used[ks]) {kstrip_used[ks] = 1; n_used++;}
            }
        }

        if (n_used == 0) {
            continue;   // entire M tile is zero, C already zeroed
        }

        for (int n0 = 0; n0 < N; n0 += TILE_MAX_F32_COLS) {
            int n_tile_f32 = (n0 + TILE_MAX_F32_COLS <= N) ? TILE_MAX_F32_COLS : N - n0;

            // zero and load C tile
            memset(global_c_buf, 0, sizeof(global_c_buf));
            configure_amx_tiles_bf16(m_tile, n_tile_f32, K_TILE);
            _tile_loadd(TILE_C, global_c_buf, TILE_MAX_F32_COLS * (int)sizeof(float));

            for (int ks = 0; ks <n_kstrips; ks++) {
                if (!kstrip_used[ks]) continue;

                int k0 = ks * K_TILE;
                int k_strip = (k0 + K_TILE <= K) ? K_TILE : A->ncols - k0;
                int k_padded = (k_strip + 1) & ~1;

                // build A: scatter nnzs from this k strip
                memset(global_a_buf, 0, sizeof(global_a_buf));
                for (int ii = 0; ii < m_tile; ii++) {
                    int row = m0 + ii;
                    for (int p = A->rowptr[row]; p < A->rowptr[row+1]; p++) {
                        int k = A->colidx[p];
                        if (k < k0 || k >= k0 + k_strip) continue;
                        global_a_buf[ii * K_TILE + (k - k0)] = A->values[p];
                    }
                }

                pack_b_tile(B, k0, k_padded, n0, n_tile_f32);

                // update tile config for actual k_strip_padded as it could differ from K_TILE on the final strip
                configure_amx_tiles_bf16(m_tile, n_tile_f32, k_padded);
                _tile_loadd(TILE_C, global_c_buf, TILE_MAX_F32_COLS * (int)sizeof(float));
                _tile_loadd(TILE_A, global_a_buf, K_TILE * (int)sizeof(uint16_t));
                _tile_loadd(TILE_B, global_b_buf, n_tile_f32 * 2 * (int)sizeof(uint16_t));
                _tile_dpbf16ps(TILE_C, TILE_A, TILE_B);

                // save accumulator before reconfiguration
                _tile_stored(TILE_C, global_c_buf, TILE_MAX_F32_COLS * (int)sizeof(float));
            }

            // scatter C tile results into output matrix
            for (int ii = 0; ii < m_tile; ii++) {
                int row = m0 + ii;
                for (int jj = 0; jj < n_tile_f32; jj++) {
                    int col = n0 + jj;
                    C->data[row * C->stride + col] = global_c_buf[ii * TILE_MAX_F32_COLS + jj];
                }
            }
        }
        for (int ii = 0; ii < m_tile; ii++) {
            int row = m0 + ii;
            for (int p = A->rowptr[row]; p < A->rowptr[row + 1]; p++) {
                kstrip_used[A->colidx[p] / K_TILE] = 0;
            }
        }
    }
    free(kstrip_used);
    _tile_release();
}


/* ===================================================
    BCSR MICROKERNEL
 * =================================================== */ 

/* Block cache for the microkernel
   packs every unique (bc, n_block) combo into the interleaved layout once, so that the packing function
   is never called redundantly */ 
#define B_CACHE_TILE_BF16 ((BCSR_BC / 2) * (TILE_MAX_F32_COLS * 2))

static void amx_spmm_bcsr(const BCSRMatrix* A, const DenseBF16* B, DenseF32* C) {
    assert(A->ncols == B->rows && A->nrows == C->rows && B->cols == C->cols);

    int N = B->cols;
    int n_blocks = (N + TILE_MAX_F32_COLS - 1) / TILE_MAX_F32_COLS;
    int nblockcols = A->nblockcols;

    // zero output 
    for (int r = 0; r < A->nrows; r++) {
        memset(&C->data[r * C->stride], 0, C->cols * sizeof(float));
    }

    // bcachedata is a flat array of nblockcols x n_blocks tiles, each B_CACHE_TILE_BF16 BF16 elements
    uint16_t* bcachedata = NULL;
    size_t cacheelems = (size_t)nblockcols * n_blocks * B_CACHE_TILE_BF16;
    size_t cachebytes = cacheelems * sizeof(uint16_t);

    // we only want to pre pack B into the cache if it fits within 512 MB, otherwise we can do it on the fly
    int use_cache = (cachebytes <= (size_t)512 * 1024 * 1024);
    if (use_cache) {
        if (posix_memalign((void **)&bcachedata, 64, cachebytes) != 0) {
            perror("posix_memalign b_cache");
            use_cache = 0;
        } else {
            memset(bcachedata, 0, cachebytes);
        }
    }
    if (!use_cache) printf(" [b_cache] %.0f MB > 512 MB, packing B on the fly\n", cachebytes / 1.0e6);

    if (use_cache) {
        for (int bc = 0; bc < nblockcols; bc++) {
            int k0 = bc * BCSR_BC;
            for (int np = 0; np < n_blocks; np++) {
                int n0 = np * TILE_MAX_F32_COLS;
                int n_tile_f32 = (n0 + TILE_MAX_F32_COLS <= N) ? TILE_MAX_F32_COLS : N - n0;
                uint16_t* dst = bcachedata + ((size_t)bc * n_blocks + np) * B_CACHE_TILE_BF16;
                int k_pairs = BCSR_BC / 2;
                for (int kp = 0; kp < k_pairs; kp++) {
                    int k_even = k0 +2 * kp;
                    int k_odd = k0 +2 * kp + 1;
                    for (int j = 0; j < n_tile_f32; j++) {
                        int n = n0 + j;
                        dst[kp * (TILE_MAX_F32_COLS * 2) + 2 * j] = (k_even < B->rows) ? dense_bf16_get(B, k_even, n) : 0;
                        dst[kp * (TILE_MAX_F32_COLS * 2) + 2 * j + 1] = (k_odd < B->rows) ? dense_bf16_get(B, k_odd, n) : 0;
                    }
                }
            }
        }
    }

    // compute loop
    for (int br = 0; br < A->nblockrows; br++) {
        int block_start = A->blockrowptr[br];
        int block_end = A->blockrowptr[br + 1];
        if (block_start == block_end) continue;

        for (int np = 0; np < n_blocks; np++) {
            int n0 = np * TILE_MAX_F32_COLS;
            int n_tile_f32 = (n0 + TILE_MAX_F32_COLS <= N) ? TILE_MAX_F32_COLS : N - n0;
            
            configure_amx_tiles_bf16(BCSR_BR, n_tile_f32, BCSR_BC);
            memset(global_c_buf, 0, sizeof(global_c_buf));
            _tile_loadd(TILE_C, global_c_buf, TILE_MAX_F32_COLS * (int)sizeof(float));

            for (int bi = block_start; bi < block_end; bi++) {
                int bc = A->blockcolidx[bi];

                const uint16_t* a_block = A->values + (size_t)bi * BCSR_BLOCK_ELEMS;
                _tile_loadd(TILE_A, a_block, BCSR_BC * (int)sizeof(uint16_t));

                const uint16_t* b_tile;
                if (use_cache) {
                    b_tile = bcachedata + ((size_t)bc * n_blocks + np) * B_CACHE_TILE_BF16;
                } else {
                    // pack on the fly
                    int k0_bi = bc * BCSR_BC;
                    int k_pairs = BCSR_BC / 2;
                    for (int kp = 0; kp < k_pairs; kp++) {
                        int k_even = k0_bi + 2*kp;
                        int k_odd  = k0_bi + 2*kp + 1;
                        for (int j = 0; j < n_tile_f32; j++) {
                            int n = n0 + j;
                            global_b_buf[kp*(TILE_MAX_F32_COLS * 2) + 2 * j  ] = (k_even < B->rows) ? dense_bf16_get(B, k_even,n) : 0;
                            global_b_buf[kp * (TILE_MAX_F32_COLS * 2) + 2 * j + 1] = (k_odd  < B->rows) ? dense_bf16_get(B, k_odd, n) : 0;
                        }
                    }
                    b_tile = global_b_buf;
                }
                _tile_loadd(TILE_B, b_tile, TILE_MAX_F32_COLS * 2 * (int)sizeof(uint16_t));

                _tile_dpbf16ps(TILE_C, TILE_A, TILE_B);

                _tile_stored(TILE_C, global_c_buf, TILE_MAX_F32_COLS * (int)sizeof(float));

                if (bi + 1 < block_end) {
                    configure_amx_tiles_bf16(BCSR_BR, n_tile_f32, BCSR_BC);
                    _tile_loadd(TILE_C, global_c_buf, TILE_MAX_F32_COLS * (int)sizeof(float));
                }
            }

            // scatter C tile into output matrix
            int row_base = br * BCSR_BR;
            for (int ii = 0; ii < BCSR_BR; ii++) {
                int row = row_base + ii;
                for (int jj = 0; jj < n_tile_f32; jj++) {
                    C->data[row * C->stride + n0 + jj] = global_c_buf[ii * TILE_MAX_F32_COLS + jj];
                }
            }
        }
    }
    free(bcachedata);
    _tile_release();
}


/* ================= MTX READER ================= */

#include <ctype.h>
#include <dirent.h>
#include <time.h>
#include "../utility/mmio.h"

// internal COO entry used during conversion to CSR
typedef struct {
    int row;
    int col;
    float val;
} CooEntry;

static int coo_cmp(const void* a, const void* b) {
    const CooEntry* x = (const CooEntry*)a;
    const CooEntry* y = (const CooEntry*)b;
    if (x->row != y->row) {
        return x->row - y->row;
    }
    return x->col - y->col;
}

// Loads a .mtx file via mmio and returns a CSRMatrix
static CSRMatrix* mtx_read_to_csr(const char* path, int* nrows_out, int* ncols_out) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return NULL; }

    char raw_banner[MM_MAX_LINE_LENGTH];
    if (!fgets(raw_banner, sizeof(raw_banner), f)) {
        fprintf(stderr, "%s: empty file\n", path);
        fclose(f);
        return NULL;
    }

    static const struct {
        const char* from;
        const char* to;
    } subs[] = {
        {"unsymmetric", "general"},
        {"lower-triangular", "general"},
        {"upper-triangular", "general"},
        {"lower_triangular", "general"},
        {"upper_triangular", "general"},
        {"binary", "pattern"},
    };

    char norm[MM_MAX_LINE_LENGTH];
    strncpy(norm, raw_banner, sizeof(norm) - 1);
    norm[sizeof(norm) - 1] = '\0';
    for (char* p = norm; *p; p++) *p = (char)tolower((unsigned char)*p);

    const char* banner_to_parse = raw_banner;
    char fixed_banner[MM_MAX_LINE_LENGTH];

    for (int s = 0; s < (int)(sizeof(subs)/sizeof(subs[0])); s++) {
        char* hit = strstr(norm, subs[s].from);
        if (hit) {
            size_t prefix = (size_t)(hit - norm);
            size_t from_len = strlen(subs[s].from);
            size_t to_len = strlen(subs[s].to);
            if (prefix + to_len + strlen(norm + prefix + from_len) + 1 < sizeof(fixed_banner)) {
                memcpy(fixed_banner, raw_banner, prefix);
                memcpy(fixed_banner + prefix, subs[s].to, to_len);
                strcpy(fixed_banner + prefix + to_len, raw_banner + prefix + from_len);
                banner_to_parse = fixed_banner;
            }
            break;
        }
    }


    // Feed possibly normalised baner line to mm_read_banner via fmemopen so MMIO reads from the memory rather than the file
    MM_typecode matcode;
    FILE* bmem = fmemopen((void*)banner_to_parse, strlen(banner_to_parse), "r");
    if (!bmem) {
        fprintf(stderr, "%s: fmemopen failed\n", path);
        fclose(f);
        return NULL;
    }
    int ret = mm_read_banner(bmem, &matcode);
    fclose(bmem);

    if (ret != 0) {
        fprintf(stderr, "%s: mm_read_banner failed (code %d)\n - banner: %s", path, ret, raw_banner);
        fclose(f); return NULL;
    }

    // sparse matrices only
    if (!mm_is_sparse(matcode) || !mm_is_matrix(matcode)) {
        fprintf(stderr, "%s: not a sparse coordinate matrix (type: %s)\n",
                path, mm_typecode_to_str(matcode));
        fclose(f); return NULL;
    }

    // probably not necessary
    if (mm_is_complex(matcode)) {
        fprintf(stderr, "%s: complex matrices not supported\n", path);
        fclose(f); return NULL;
    }

    int is_pattern = mm_is_pattern(matcode);
    int is_integer = mm_is_integer(matcode); // read integers like real
    int is_symmetric = mm_is_symmetric(matcode) || mm_is_skew(matcode) || mm_is_hermitian(matcode);

    // read dimensions
    int nrows, ncols, nnz_file;
    ret = mm_read_mtx_crd_size(f, &nrows, &ncols, &nnz_file);
    if (ret != 0) {
        fprintf(stderr, "%s: mm_read_mtx_crd_size failed (code %d)\n", path, ret);
        fclose(f); return NULL;
    }

    // allocate COO, twice for symmetric storage
    long nnz_alloc = is_symmetric ? (long)nnz_file * 2 : nnz_file;
    CooEntry *coo  = (CooEntry *)malloc((size_t)nnz_alloc * sizeof(CooEntry));
    if (!coo) { perror("malloc coo"); fclose(f); return NULL; }

    // read the entries
    long n = 0;
    for (int e = 0; e < nnz_file; e++) {
        int row, col;
        double real_val = 0.0, imag_val = 0.0;

        if (is_integer) {
            long iv = 0;
            if (fscanf(f, "%d %d %ld", &row, &col, &iv) != 3) {
                fprintf(stderr, "%s: read error at integer entry %d\n", path, e);
                break;
            }
            real_val = (double)iv;
        } else {
            ret = mm_read_mtx_crd_entry(f, &row, &col, &real_val, &imag_val, matcode);
            if (ret != 0) {
                fprintf(stderr, "%s: read error at entry %d (code %d)\n",
                        path, e, ret);
                break;
            }
        }

        // mmio uses 1 indexing
        row--; col--;
        if (row < 0 || row >= nrows || col < 0 || col >= ncols) continue;

        float v = is_pattern ? 1.0f : (float)real_val;

        coo[n].row = row; coo[n].col = col; coo[n].val = v; n++;

        // mirror off diagonal for symmetric storage
        if (is_symmetric && row != col) {
            coo[n].row = col; coo[n].col = row;
            // skew-symmetric -> A[j][i] = -A[i][j]
            coo[n].val = mm_is_skew(matcode) ? -v : v;
            n++;
        }
    }
    fclose(f);

    // sort, remove duplicates and build CSR
    qsort(coo, (size_t)n, sizeof(CooEntry), coo_cmp);

    long nnz = 0;
    for (long i = 0; i < n; i++) {
        if (nnz > 0 &&
            coo[nnz-1].row == coo[i].row &&
            coo[nnz-1].col == coo[i].col)
            coo[nnz-1].val = coo[i].val;    // on duplicate, last value is kept
        else
            coo[nnz++] = coo[i];
    }

    CSRMatrix* m = csr_alloc(nrows, ncols, (int)nnz);

    for (long i = 0; i < nnz; i++)
        m->rowptr[coo[i].row + 1]++;
    for (int r = 0; r < nrows; r++)
        m->rowptr[r+1] += m->rowptr[r];

    int* cursor = (int *)malloc((size_t)nrows * sizeof(int));
    memcpy(cursor, m->rowptr, (size_t)nrows * sizeof(int));
    for (long i = 0; i < nnz; i++) {
        int pos = cursor[coo[i].row]++;
        m->colidx[pos] = coo[i].col;
        m->values[pos] = float_to_bf16(coo[i].val);
    }
    free(cursor);
    free(coo);

    *nrows_out = nrows;
    *ncols_out = ncols;
    return m;
}


/* ================= CSR TO BCSR ================= */

static BCSRMatrix *csr_to_bcsr(const CSRMatrix *csr)
{
    int nrows = ((csr->nrows + BCSR_BR - 1) / BCSR_BR) * BCSR_BR;
    int ncols = ((csr->ncols + BCSR_BC - 1) / BCSR_BC) * BCSR_BC;
    int nbr   = nrows / BCSR_BR;
    int nbc   = ncols / BCSR_BC;

    /* Pass 1: mark which (br, bc) blocks are non-zero.
     * use a per-block-row byte array of width nbc.  nbc can be large */
    char *row_nz = (char *)calloc((size_t)nbc, 1);          // reused per block-row 
    if (!row_nz) { perror("calloc row_nz"); return NULL; }

    // First pass to count total non-zero blocks
    int nblocks = 0;
    for (int br = 0; br < nbr; br++) {
        int r0 = br * BCSR_BR;
        int r1 = r0 + BCSR_BR < csr->nrows ? r0 + BCSR_BR : csr->nrows;
        for (int r = r0; r < r1; r++)
            for (int p = csr->rowptr[r]; p < csr->rowptr[r+1]; p++)
                row_nz[csr->colidx[p] / BCSR_BC] = 1;
        for (int bc = 0; bc < nbc; bc++)
            if (row_nz[bc]) { nblocks++; row_nz[bc] = 0; }  // reset for reuse
    }

    BCSRMatrix *m = bcsr_alloc(nrows, ncols, nblocks);
    if (!m) { free(row_nz); return NULL; }

    // Pass 2: build blockrowptr / blockcolidx.
    int idx = 0;
    for (int br = 0; br < nbr; br++) {
        m->blockrowptr[br] = idx;
        int r0 = br * BCSR_BR;
        int r1 = r0 + BCSR_BR < csr->nrows ? r0 + BCSR_BR : csr->nrows;
        for (int r = r0; r < r1; r++)
            for (int p = csr->rowptr[r]; p < csr->rowptr[r+1]; p++)
                row_nz[csr->colidx[p] / BCSR_BC] = 1;
        for (int bc = 0; bc < nbc; bc++)
            if (row_nz[bc]) { m->blockcolidx[idx++] = bc; row_nz[bc] = 0; }
    }
    m->blockrowptr[nbr] = nblocks;
    free(row_nz);

    // Pass 3: scatter CSR values into BCSR value blocks.
    for (int r = 0; r < csr->nrows; r++) {
        int br = r / BCSR_BR;
        int lr = r % BCSR_BR;
        int slice_lo = m->blockrowptr[br];
        int slice_hi = m->blockrowptr[br + 1];
        for (int p = csr->rowptr[r]; p < csr->rowptr[r+1]; p++) {
            int c  = csr->colidx[p];
            int bc = c / BCSR_BC;
            int lc = c % BCSR_BC;
            // Binary search for bc in blockcolidx[slice_lo .. slice_hi) 
            int lo = slice_lo, hi = slice_hi;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if      (m->blockcolidx[mid] < bc) lo = mid + 1;
                else if (m->blockcolidx[mid] > bc) hi = mid;
                else { lo = mid; break; }
            }
            // lo is now the index of block (br, bc) 
            m->values[(size_t)lo * BCSR_BLOCK_ELEMS + lr * BCSR_BC + lc] =
                csr->values[p];
        }
    }
    return m;
}


// random seed for populating the dense matrix
static uint32_t random_seed;

static float random_float(void) {
    random_seed = random_seed * 1664525u + 1013904223u;
    return (float)(int32_t)random_seed / (float)(1u << 31);
}


/* ================= REORDERING ================= */

// Helper types

// (col, val) pair used by csr_permute for per-row sorting
typedef struct {
    int col; 
    uint16_t val;
} ColValPair;

static int colval_cmp(const void* a, const void* b) {
    return ((const ColValPair*)a)->col - ((const ColValPair*)b)->col;
}

// undirected edge, used by build_sym_adj
typedef struct {
    int r, c;
} SymEdge;

static int symedge_cmp(const void* a, const void* b) {
    const SymEdge* x = (const SymEdge*)a, *y = (const SymEdge*)b;
    if (x->r != y->r) {
        return x->r - y->r;
    }
    return x->c - y->c;
}


/*  
    Build a symmetric adjacency CSR from A. (Pattern only, no diagonal)
    both (i, j) and (j, i) are emitted for every off diag nz, duplicates are removed after sorting.
    Used as input to both AMD and RCM, so those algs can operate on an undirected graph 
    regardless of whether A's storage is symmetric.
*/
static int build_sym_adj(const CSRMatrix* A, int **out_ptr, int **out_idx) {
    int n = A->nrows;
    int nnz = A->nnz;

    SymEdge* edges = (SymEdge*)malloc(2 * (size_t)nnz * sizeof(SymEdge));
    if (!edges) {
        perror("build_sym_adj edges"); 
        return -1;
    }

    int cnt = 0;
    for (int i = 0; i < n; i++) {
        for (int p = A->rowptr[i]; p < A->rowptr[i+1]; p++) {
            int j = A->colidx[p];
            if (i == j) continue;   // skip the diagonal
            edges[cnt].r = i; edges[cnt].c = j; cnt++;
            edges[cnt].r = j; edges[cnt].c = i; cnt++;
        }
    }
    qsort(edges, (size_t)cnt, sizeof(SymEdge), symedge_cmp);

    // remove duplicates
    int u = 0;
    for (int k = 0; k < cnt; k++) {
        if (u == 0 || edges[k].r != edges[u-1].r || edges[k].c != edges[u-1].c) {
            edges[u++] = edges[k];
        }
    }
    cnt = u;

    // build CSR
    int* ptr = (int*)calloc((size_t)(n + 1), sizeof(int));
    int* idx = (int*)malloc((size_t)cnt * sizeof(int));
    int* cur = (int*)malloc((size_t)n * sizeof(int));
    if (!ptr || !idx || !cur) {
        perror("build_sym_adj csr");
        free(edges);
        free(ptr);
        free(idx);
        free(cur);
        return -1;
    }

    for (int k = 0; k < cnt; k++) {
        ptr[edges[k].r + 1]++;
    }
    for (int i = 0; i < n; i++) {
        ptr[i+1] += ptr[i];
    }
    memcpy(cur, ptr, (size_t)n * sizeof(int));
    for (int k = 0; k < cnt; k++) {
        idx[cur[edges[k].r]++] = edges[k].c;
    }

    free(cur);
    free(edges);
    *out_ptr = ptr;
    *out_idx = idx;
    return 0;
}

// Reverse Cuthill McKee (RCM) reordering
/* 
    For each connected component, choose unvisited node with lowest degree as starting node.
    When expanding a node, collect its unvisited neighbours, sort them by degree, then enqueue in that order (standard CM).
    Reverse the full BFS order to get the RCM.

    Produces perm[new_pos] = old_node
*/
static int compute_rcm_order(const CSRMatrix* A, int* perm) {
    int n = A->nrows;
    if (n != A->ncols) {
        fprintf(stderr, "RCM requires square matrix (%d x %d)\n", n, A->ncols);
        return -1;
    }

    int *adj_ptr = NULL, *adj_idx = NULL;
    if (build_sym_adj(A, &adj_ptr, &adj_idx) != 0) return -1;

    int* degree = (int*)malloc((size_t)n * sizeof(int));
    char* visited = (char*)calloc((size_t)n, 1);
    int* queue = (int*)malloc((size_t)n * sizeof(int));
    int* nbuf = (int*)malloc((size_t)n * sizeof(int));

    if (!degree || !visited || !queue || !nbuf) {
        perror("RCM alloc");
        free(adj_ptr);
        free(adj_idx);
        free(degree);
        free(visited);
        free(queue);
        free(nbuf);
        return -1;
    }

    for (int i = 0; i < n; i++) {
        degree[i] = adj_ptr[i+1] - adj_ptr[i];
    }

    int total = 0;      // nodes placed into perm[] so far
    int search = 0;     // cursor for finding start of the next component

    while (total < n) {
        // advance past already visited nodes
        while (search < n && visited[search]) search++;
        if (search == n) break;

        int s = search;
        for (int i = search + 1; i < n; i++) {
            if (!visited[i] && degree[i] < degree[s]) s = i;
        }

        int qhead = 0, qtail = 0;
        visited[s] = 1;
        queue[qtail++] = s;

        while (qhead < qtail) {
            int u = queue[qhead++];
            perm[total++] = u;

            // collect neighbours
            int nn = 0;
            for (int p = adj_ptr[u]; p < adj_ptr[u+1]; p++) {
                int v = adj_idx[p];
                if (!visited[v]) {
                    visited[v] = 1;
                    nbuf[nn++] = v;
                }    
            }

            // insertion sort by degree ascending (not ideal, but average degree is small)
            for (int a = 1; a < nn; a++) {
                int key = nbuf[a], dk = degree[key], b = a - 1;
                while (b >= 0 && degree[nbuf[b]] > dk) {
                    nbuf[b+1] = nbuf[b];
                    b--;
                }
                nbuf[b+1] = key;
            }
            for (int a = 0; a < nn; a++) {
                queue[qtail++] = nbuf[a];
            }
        }
    }

    // CM to RCM
    for (int i = 0; i < n / 2; i++) {
        int t = perm[i]; 
        perm[i] = perm[n-1-i];
        perm[n-1-i] = t;
    }

    free(adj_ptr);
    free(adj_idx);
    free(degree);
    free(visited);
    free(queue);
    free(nbuf);
    return 0;
}


// Approximate Minimum Degree (AMD) reordering
/* 
    Utilises SuiteSparse library
    AMD minimises fill-in during factorisation, which generally clusters nzs near the diagonal

    Needs to be compiled in. Returns -1 if not
*/
static int compute_amd_order(const CSRMatrix* A, int* perm) {
#ifndef USE_AMD
    (void)A; (void)perm;
    fprintf(stderr,
        "AMD not compiled in. \n"
        "Rebuild with SuiteSparse using -DUSE_AMD -I/<path to SuiteSparse headers>"
        " -lsuitesparseconfig\n");
    return -1;
#else
    int n = A->nrows;
    if (n != A->ncols) {
        fprintf(stderr, "AMD requires square matrix (%d x %d)\n", n, A->ncols);
        return -1;
    }

    int *Ap = NULL; 
    int *Ai = NULL;
    if (build_sym_adj(A, &Ap, &Ai) != 0) return -1;

    double Control[AMD_CONTROL], Info[AMD_INFO];
    amd_defaults(Control);

    int status = amd_order(n, Ap, Ai, perm, Control, Info);
    free(Ap);
    free(Ai);

    if (status != AMD_OK && status != AMD_OK_BUT_JUMBLED) {
        fprintf(stderr, "amd_order returned status %d\n", status);
        return -1;
    }
    return 0;
#endif    
}

// apply symmetrix permutation P x A x PT to square CSR matrix
/*
    New row i collects entries of old row perm[i], with each column index
    c remapped to inv_perm[c]. Each new row is sorted by column index.

    Returns new CSRMatrix.
*/
static CSRMatrix* csr_permute(const CSRMatrix* A, const int* perm, const int* inv_perm) {
    int n = A->nrows;
    assert(n == A->ncols);

    // largest row length, used to size the sort buffer
    int max_row = 0;
    for (int i = 0; i < n; i++) {
        int len = A->rowptr[i+1] - A->rowptr[i];
        if (len > max_row) max_row = len;
    }
    ColValPair* temp = (ColValPair*)malloc((size_t)(max_row > 0 ? max_row : 1) * sizeof(ColValPair));
    if (!temp) {
        perror("csr_permute");
        return NULL;
    }

    CSRMatrix* B = csr_alloc(n, n, A->nnz);

    // row pointer
    B->rowptr[0] = 0;
    for (int i = 0; i < n; i++) {
        B->rowptr[i+1] = B->rowptr[i] + (A->rowptr[perm[i] + 1] - A->rowptr[perm[i]]);
    }

    // fill entries- remap the cols and sort each row
    for (int i = 0; i < n; i++) {
        int old = perm[i];
        int len = A->rowptr[old+1] - A->rowptr[old];
        int base = A->rowptr[old];
        int dst = B->rowptr[i];

        for (int k = 0; k < len; k++) {
            temp[k].col = inv_perm[A->colidx[base + k]];
            temp[k].val = A->values[base + k];
        }
        if (len > 1) qsort(temp, (size_t)len, sizeof(ColValPair), colval_cmp);
        for (int k = 0; k < len; k++) {
            B->colidx[dst + k] = temp[k].col;
            B->values[dst + k] = temp[k].val;
        }
    }

    free(temp);
    return B;
}


// Create permuted copy of B
static DenseBF16* dense_bf16_permute_rows(const DenseBF16* B, const int* perm, int n_perm, int n_total) {
    DenseBF16* NB = dense_bf16_alloc(n_total, B->cols);
    for (int i = 0; i < n_perm; i++) {
        memcpy(NB->data + (size_t)i * NB->stride, B->data + (size_t)perm[i] * B->stride, (size_t)B->cols * sizeof(uint16_t));
    }
    return NB;
}


// unpermute the rows of output matrix C back to original ordering
static DenseF32* dense_f32_unpermute_rows(const DenseF32* C_perm, const int* perm, int n_perm, int n_total) {
    DenseF32* C = dense_f32_alloc(n_total, C_perm->cols);
    for (int i = 0; i < n_perm; i++) {
        memcpy(C->data + (size_t)perm[i] * C->stride, C_perm->data + (size_t)i * C_perm->stride, (size_t)C_perm->cols * sizeof(float));
    }
    return C;
}


/* ================= MTX EXTRACTION ================= */

// run program on a single .mtx file
static void run_mtx_file(const char* path, int embed_dim, int use_bcsr, ReorderMethod reorder) {
    printf("\n────────────────────────────────────────────────────────\n");
    printf("  File : %s\n", path);

    // load mtx to CSR
    int nrows, ncols;
    double t0 = now_sec();
    CSRMatrix* A_csr_orig = mtx_read_to_csr(path, &nrows, &ncols);
    if (!A_csr_orig) return;
    double t_load = now_sec() - t0;

    double sparsity = 1.0 - (double)A_csr_orig->nnz / ((double)nrows * ncols);
    printf("A : %d x %d, nnz = %d, sparsity = %.2f%%\n", nrows, ncols, A_csr_orig->nnz, sparsity * 100);
    printf("Load : %.3f s\n", t_load);

    // reordering
    int* perm = NULL, *inv_perm = NULL;
    CSRMatrix* A_csr_perm = NULL;

    if (reorder != REORDER_NONE) {
        if (nrows != ncols) {
            printf(" [reorder] skipped, matrix is not square (%d x %d)\n", nrows, ncols);
        } else {
            perm = (int*)malloc((size_t)nrows * sizeof(int));
            inv_perm = (int*)malloc((size_t)nrows * sizeof(int));
            if (!perm || !inv_perm) {
                perror("malloc perm");
                free(perm);
                free(inv_perm);
                perm = inv_perm = NULL;
            } else {
                const char* method = (reorder == REORDER_AMD) ? "AMD" : "RCM";
                printf("Reorder : computing %s permutation\n", method);
                t0 = now_sec();
                int ok = (reorder == REORDER_AMD) ? compute_amd_order(A_csr_orig, perm) : compute_rcm_order(A_csr_orig, perm);
                double t_order = now_sec() - t0;

                if (ok != 0) {
                    printf(" [reorder] %s failed, continuining without reorder\n", method);
                    free(perm);
                    free(inv_perm);
                    perm = inv_perm = NULL;
                } else {
                    for (int i = 0; i < nrows; i++) inv_perm[perm[i]] = i;
                    printf("Reorder : %.3f s (%s)\n", t_order, method);
                    t0 = now_sec();
                    A_csr_perm = csr_permute(A_csr_orig, perm, inv_perm);
                    printf("Permute : %.3f s \n", now_sec() - t0);
                }
            }
        }
    }

    // A_csr used for AMX path, permuted if available, else original
    CSRMatrix* A_csr = (A_csr_perm != NULL) ? A_csr_perm : A_csr_orig;

    // convert to BCSR 
    BCSRMatrix* A_bcsr = NULL;
    int B_rows_amx, C_rows_amx;     // effective dimensions, as the dimensions could be padded in bcsr
    if (use_bcsr) {
        t0 = now_sec();
        A_bcsr = csr_to_bcsr(A_csr);
        printf("CSR to BCSR : %.3f s (%d blocks, padded %dx%d to %dx%d)\n", now_sec() - t0, A_bcsr->nblocks, nrows, ncols, A_bcsr->nrows, A_bcsr->ncols);
        B_rows_amx = A_bcsr->ncols; // padded N must match kernel A.ncols
        C_rows_amx = A_bcsr->nrows; // padded M must match kernel A.nrows
    } else {
        B_rows_amx = ncols;
        C_rows_amx = nrows;
    }
    printf("B : %d x %d  (N x embed_dim)\n", B_rows_amx, embed_dim);
    printf("C : %d x %d  (M x embed_dim)\n", C_rows_amx, embed_dim);

    // allocate B
    random_seed = 0xABCD1234u;

    DenseBF16* B_orig = NULL;
    DenseBF16* B_amx;

    if (perm != NULL) {
        B_orig = dense_bf16_alloc(ncols, embed_dim);
        for (int r = 0; r < ncols; r++) {
            for (int c = 0; c < embed_dim; c++) {
                dense_bf16_set(B_orig, r, c, float_to_bf16(random_float() * 2.0f - 1.0f));
            }
        }
        B_amx = dense_bf16_permute_rows(B_orig, perm, ncols, B_rows_amx);
    } else {
        B_amx = dense_bf16_alloc(B_rows_amx, embed_dim);
        for (int r = 0; r < ncols; r++) {
            for (int c = 0; c < embed_dim; c++) {
                dense_bf16_set(B_amx, r, c, float_to_bf16(random_float() * 2.0f - 1.0f));
            }
        }
    }


    // amx run
    DenseF32* C_amx_perm = dense_f32_alloc(C_rows_amx, embed_dim);
    t0 = now_sec();
    if (use_bcsr) {
        amx_spmm_bcsr(A_bcsr, B_amx, C_amx_perm);
    } else {
        amx_spmm_csr(A_csr, B_amx, C_amx_perm);
    }
    double elapsed = now_sec() - t0;

    // unpermute C
    DenseF32* C_amx;
    if (perm != NULL) {
        C_amx = dense_f32_unpermute_rows(C_amx_perm, perm, nrows, C_rows_amx);
        dense_f32_free(C_amx_perm);
    } else {
        C_amx = C_amx_perm;
    }

    printf("AMX spMM : %.3f s", elapsed);

    csr_free(A_csr_orig);
    if (A_csr_perm) csr_free(A_csr_perm);
    bcsr_free(A_bcsr);
    if (B_orig) dense_bf16_free(B_orig);
    dense_bf16_free(B_amx);
    dense_f32_free(C_amx);
    free(perm);
    free(inv_perm);
}

// run program on all .mtx files in a directory
static void run_mtx_directory(const char* dir, int embed_dim, int use_bcsr, ReorderMethod reorder) {
    printf("\n========================================\n");
    printf("Source Directory : %s\n", dir);
    printf("embed_dim = %d,  format = %s\n",
           embed_dim, use_bcsr ? "BCSR (tile blocks)" : "CSR");
    printf("========================================\n");

    DIR* d = opendir(dir);
    if (!d) {
        perror(dir); 
        return;
    }

    struct dirent* ent;
    int count = 0;
    while ((ent = readdir(d)) != NULL) {
        size_t len = strlen(ent->d_name);
        if (len < 4 || strcmp(ent->d_name + len - 4, ".mtx") != 0) continue;
        char fullpath[4096];
        snprintf(fullpath, sizeof(fullpath), "%s/%s", dir, ent->d_name);
        run_mtx_file(fullpath, embed_dim, use_bcsr, reorder);
        count++;
    }
    closedir(d);

    if (count == 0) {
        printf("(no .mtx files found in %s)\n", dir);
    } else {
        printf("\nProcessed %d files.\n", count);
    }
}



/* ================= MAIN ================= */

/* 
Usage:
    ./amx --mtx <dir>                       - process all .mtx files in target directory
    ./amx --mtx <dir> --embed 128           - use embed dimension 128
    ./amx --mtx <dir> --embed 128 --bcsr    - use BCSR format
*/
int main(int argc, char** argv) {
    printf("Intel AMX BF16 spMM\n");
    printf("====================\n");
    printf("TILE_MAX_ROWS      = %d\n", TILE_MAX_ROWS);
    printf("TILE_MAX_FP32_COLS = %d\n", TILE_MAX_F32_COLS);
    printf("BCSR block         = %d x %d = %d BF16 elems (one full tile)\n\n", BCSR_BR, BCSR_BC, BCSR_BLOCK_ELEMS);

#ifdef USE_AMD
    printf("SuiteSparse available\n");
#else
    printf("SuiteSparse not compiled in, rebuild with -DUSE_AMD\n");
#endif
    printf("RCM available by default\n\n");

    if (request_amx_perm() != 1) {
        fprintf(stderr, "ERROR: Failed to enable AMX tile state.\n");
        return 1;
    }

    // CLI parsing
    const char* mtx_dir = NULL;
    int embed_dim = 64;
    int use_bcsr = 0;
    ReorderMethod reorder = REORDER_NONE;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--mtx") == 0 && i + 1 < argc) mtx_dir = argv[++i];
        else if (strcmp(argv[i], "--embed") == 0 && i + 1 < argc) embed_dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--bcsr") == 0) use_bcsr = 1;
        else if (strcmp(argv[i], "--reorder") == 0 && i+1 < argc) {
            const char* m = argv[++i];
            if (strcmp(m, "amd") == 0) reorder = REORDER_AMD;
            else if (strcmp(m, "rcm") == 0) reorder = REORDER_RCM;
            else {
                fprintf(stderr, "Unknown reorder method\n");
                return 1;
            }
        }
        else {
            fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            fprintf(stderr, "Usage: %s [--mtx <dir>] [--embed 64|128] [--bcsr] [--reorder amd|rcm]\n", argv[0]);
            return 1;
        }
    }

    if (mtx_dir) {
        run_mtx_directory(mtx_dir, embed_dim, use_bcsr, reorder);
        return 0;
    }
}