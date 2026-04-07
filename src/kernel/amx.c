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
#include <math.h>
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
    REORDER_RCM,
    REORDER_CLUSTER
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



#define BCSR_BR TILE_MAX_ROWS                   // 16 - block height in rows
#define BCSR_BC TILE_MAX_BF16_COLS              // 32 - block width in BF16 cols
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
    uint16_t* values;   // [nblocks * BCSR_BLOCK_ELEMS], row major within each block
} BCSRMatrix;

static BCSRMatrix* bcsr_alloc(int nrows, int ncols, int nblocks) {
    assert(nrows % BCSR_BR == 0 && ncols % BCSR_BC == 0);
    BCSRMatrix* m = (BCSRMatrix*)malloc(sizeof(BCSRMatrix));
    if (!m) {perror("bcsr_alloc struct"); return NULL;}
    m->nrows = nrows;
    m->ncols = ncols;
    m->nblockrows = nrows / BCSR_BR;
    m->nblockcols = ncols / BCSR_BC;
    m->nblocks = nblocks;
    m->blockrowptr = (int*)calloc(m->nblockrows + 1, sizeof(int));
    m->blockcolidx = (int*)malloc(nblocks * sizeof(int));
    if (!m->blockcolidx || !m->blockrowptr) {
        perror("bcsr_alloc index arrays");
        free(m->blockrowptr);
        free(m->blockcolidx);
        free(m);
        return NULL;
    }

    if (posix_memalign((void **)&m->values, 64, (size_t)nblocks * BCSR_BLOCK_ELEMS * sizeof(uint16_t)) != 0) {
        perror("posix_memalign failure");
        free(m->blockrowptr);
        free(m->blockcolidx);
        free(m);
        return NULL;
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

    int total_m_tiles   = (M + TILE_MAX_ROWS - 1) / TILE_MAX_ROWS;
    int report_interval = total_m_tiles / 20;          /* every ~5% */
    if (report_interval < 1000) report_interval = 1000;
    if (report_interval > total_m_tiles) report_interval = total_m_tiles;

    double t_start   = now_sec();
    int    m_tile_idx = 0;

    for (int m0 = 0; m0 < M; m0 += TILE_MAX_ROWS, m_tile_idx++) {

        if (m_tile_idx > 0 && m_tile_idx % report_interval == 0) {
            double elapsed  = now_sec() - t_start;
            double fraction = (double)m_tile_idx / total_m_tiles;
            double eta      = elapsed / fraction - elapsed;
            fprintf(stderr, "    CSR progress: %5.1f%%  elapsed=%.1fs  eta=%.1fs\n",
                    fraction * 100.0, elapsed, eta);
        }

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
            continue;
        }

        for (int n0 = 0; n0 < N; n0 += TILE_MAX_F32_COLS) {
            int n_tile_f32 = (n0 + TILE_MAX_F32_COLS <= N) ? TILE_MAX_F32_COLS : N - n0;

            memset(global_c_buf, 0, sizeof(global_c_buf));
            configure_amx_tiles_bf16(m_tile, n_tile_f32, K_TILE);
            _tile_loadd(TILE_C, global_c_buf, TILE_MAX_F32_COLS * (int)sizeof(float));

            for (int ks = 0; ks < n_kstrips; ks++) {
                if (!kstrip_used[ks]) continue;

                int k0 = ks * K_TILE;
                int k_strip = (k0 + K_TILE <= K) ? K_TILE : A->ncols - k0;
                int k_padded = (k_strip + 1) & ~1;

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

                configure_amx_tiles_bf16(m_tile, n_tile_f32, k_padded);
                _tile_loadd(TILE_C, global_c_buf, TILE_MAX_F32_COLS * (int)sizeof(float));
                _tile_loadd(TILE_A, global_a_buf, K_TILE * (int)sizeof(uint16_t));
                _tile_loadd(TILE_B, global_b_buf, n_tile_f32 * 2 * (int)sizeof(uint16_t));
                _tile_dpbf16ps(TILE_C, TILE_A, TILE_B);
                _tile_stored(TILE_C, global_c_buf, TILE_MAX_F32_COLS * (int)sizeof(float));
            }

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
    fprintf(stderr, "    CSR progress: 100.0%%  elapsed=%.1fs\n", now_sec() - t_start);
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

    // we only want to pre pack B into the cache if it fits within 8 GB, otherwise we can do it on the fly
    int use_cache = (cachebytes <= (size_t)8 * 1024 * 1024 * 1024);
    if (use_cache) {
        if (posix_memalign((void **)&bcachedata, 64, cachebytes) != 0) {
            perror("posix_memalign b_cache");
            use_cache = 0;
        } else {
            memset(bcachedata, 0, cachebytes);
        }
    }
    if (!use_cache) printf(" [b_cache] %.0f MB > 8 GB, packing B on the fly\n", cachebytes / 1.0e6);

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
#include <sys/stat.h>
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
            m->values[(size_t)lo * BCSR_BLOCK_ELEMS + lr * BCSR_BC + lc] = csr->values[p];
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



/*  
    Build a symmetric adjacency CSR from A. (Pattern only, no diagonal)
    both (i, j) and (j, i) are emitted for every off diag nz, duplicates are removed after sorting.
    Used as input to both AMD and RCM, so those algs can operate on an undirected graph 
    regardless of whether A's storage is symmetric.
*/
static int build_sym_adj(const CSRMatrix* A, int **out_ptr, int **out_idx) {
    int n = A->nrows;
 
    int* ptr = (int*)calloc((size_t)(n + 1), sizeof(int));
    if (!ptr) { perror("build_sym_adj ptr"); return -1; }
 
    // Pass 1a: count forward (off-diagonal) edges
    for (int i = 0; i < n; i++)
        for (int p = A->rowptr[i]; p < A->rowptr[i+1]; p++)
            if (A->colidx[p] != i) ptr[i + 1]++;
 
    // Pass 1b: count reverse edges j→i missing from row j
    for (int i = 0; i < n; i++) {
        for (int p = A->rowptr[i]; p < A->rowptr[i+1]; p++) {
            int j = A->colidx[p];
            if (j == i) continue;
            // binary search for i in the sorted colidx of row j
            int lo = A->rowptr[j], hi = A->rowptr[j + 1], found = 0;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if      (A->colidx[mid] == i) { found = 1; break; }
                else if (A->colidx[mid]  < i)   lo = mid + 1;
                else                             hi = mid;
            }
            if (!found) ptr[j + 1]++;
        }
    }
 
    // prefix sum
    for (int i = 0; i < n; i++) ptr[i + 1] += ptr[i];
    int total_edges = ptr[n];
 
    int* idx = (int*)malloc((size_t)total_edges * sizeof(int));
    int* cur = (int*)malloc((size_t)n           * sizeof(int));
    if (!idx || !cur) {
        perror("build_sym_adj idx");
        free(ptr); free(idx); free(cur); return -1;
    }
    memcpy(cur, ptr, (size_t)n * sizeof(int));
 
    // Pass 2a: fill forward edges
    for (int i = 0; i < n; i++)
        for (int p = A->rowptr[i]; p < A->rowptr[i+1]; p++) {
            int j = A->colidx[p];
            if (j != i) idx[cur[i]++] = j;
        }
 
    // Pass 2b: fill missing reverse edges (same binary search)
    for (int i = 0; i < n; i++) {
        for (int p = A->rowptr[i]; p < A->rowptr[i+1]; p++) {
            int j = A->colidx[p];
            if (j == i) continue;
            int lo = A->rowptr[j], hi = A->rowptr[j + 1], found = 0;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if      (A->colidx[mid] == i) { found = 1; break; }
                else if (A->colidx[mid]  < i)   lo = mid + 1;
                else                             hi = mid;
            }
            if (!found) idx[cur[j]++] = i;
        }
    }
 
    free(cur);
    *out_ptr = ptr;
    *out_idx = idx;
    return 0;
}


// George Liu heuristic for finding a good RCM start node
static int pseudo_peripheral_node(int start, const int* adj_ptr, const int* adj_idx, const int* degree, char* visited_scratch, int* queue, int n) {
    // temp visited array local to BFS
    int* level = (int*)calloc((size_t)n, sizeof(int));
    if (!level) return start; // fallback on OOM (scary)

    int cur = start;
    int prev_depth = -1;

    for (;;) {
        // BFS from cur 
        int qhead = 0, qtail = 0;
        queue[qtail++] = cur;
        level[cur] = 1;
 
        int max_depth = 1;
        while (qhead < qtail) {
            int u = queue[qhead++];
            int d = level[u] + 1;
            if (d > max_depth) max_depth = d;
            for (int p = adj_ptr[u]; p < adj_ptr[u+1]; p++) {
                int v = adj_idx[p];
                if (!visited_scratch[v] && !level[v]) {
                    level[v] = d;
                    queue[qtail++] = v;
                }
            }
        }
 
        if (max_depth <= prev_depth) {
            // No improvement — reset level and return previous cur 
            for (int i = 0; i < qtail; i++) level[queue[i]] = 0;
            free(level);
            return cur;
        }
        prev_depth = max_depth;
 
        // Pick lowest-degree node in the last level 
        int best = -1;
        for (int i = 0; i < qtail; i++) {
            int v = queue[i];
            if (level[v] == max_depth)
                if (best < 0 || degree[v] < degree[best]) best = v;
        }
 
        // Reset level markers 
        for (int i = 0; i < qtail; i++) level[queue[i]] = 0;
 
        if (best < 0 || best == cur) { free(level); return cur; }
        cur = best;
    }
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
    int* nbuf = (int*)malloc((size_t)n * sizeof(int)); // holds unvisited neighbours for counting sort
    int max_deg = 0;    // for sorting bucket array
 
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
        if (degree[i] > max_deg) max_deg = degree[i];
    }
 
    // counting sort bucket, indexed by degree value
    int* bucket = (int*)calloc((size_t)(max_deg + 2), sizeof(int));
    int* sorted_nb = (int*)malloc((size_t)n * sizeof(int));
    if (!bucket || !sorted_nb) {
        perror("RCM bucket alloc");
        free(adj_ptr); 
        free(adj_idx);
        free(degree); 
        free(visited);
        free(queue); 
        free(nbuf); 
        free(bucket); 
        free(sorted_nb);
        return -1;
    }
 
 
    int total = 0;      // nodes placed into perm[] so far
    int search = 0;     // cursor for finding next unvisited node
    while (total < n) {
        // advance past already visited nodes
        while (search < n && visited[search]) search++;
        if (search == n) break;
 
        int s = pseudo_peripheral_node(search, adj_ptr, adj_idx, degree, visited, queue, n);
 
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
 
            if (nn == 0) continue;
 
            // counting sort nbuf by degree ascending
            //track local degree range
            int local_max_deg = 0;
            for (int a = 0; a < nn; a++) if (degree[nbuf[a]] > local_max_deg) local_max_deg = degree[nbuf[a]];
 
 
            // count
            for (int a = 0; a < nn; a++) bucket[degree[nbuf[a]]]++;
            // prefix sum
            for (int d = 1; d <= local_max_deg + 1; d++) bucket[d] += bucket[d-1];
            // place in reverse for stability
            for (int a = nn - 1; a >= 0; a--) sorted_nb[--bucket[degree[nbuf[a]]]] = nbuf[a];
            // reset the full range touched by the prefix sum, including bucket[local_max_deg+1]
            memset(bucket, 0, (local_max_deg + 2) * sizeof(int));
 
            for (int a = 0; a < nn; a++) queue[qtail++] = sorted_nb[a];
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
    free(bucket);
    free(sorted_nb);
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


// Hierarchical clustering

// default tuning params, overridable via compute_cluster_order() args
#define CLUSTER_DEFAULT_THRESHOLD 0.3f
#define CLUSTER_MAX_COL_DEGREE 500

typedef struct {
    int row;
    int nnz;
} RowNnz;

// comparator for sorting row indices by decreasing nnz
static int rownnz_cmp_desc(const void* a, const void* b) {
    return ((const RowNnz*)b)->nnz - ((const RowNnz*)a)->nnz;
}


typedef struct {
    int* members;
    int size;
    int min_row;
} Cluster;

// comparator for sorting clusters by minimum row index
static int cluster_cmp(const void* a, const void* b) {
    return ((const Cluster*)a)->min_row - ((const Cluster*)b)->min_row;
}


// fills perm[0..n-1] with a permutation that groups similar rows into consecutive bands of BCSR_BR rows
static int compute_cluster_order(const CSRMatrix* A, int* perm, float sim_threshold, int max_col_degree) {
    int n   = A->nrows;
    int m   = A->ncols;
    int nnz = A->nnz;
 
    if (n != m) {
        fprintf(stderr, "Cluster reorder requires square matrix (%d x %d)\n", n, m);
        return -1;
    }
 
    /* ── 1. Build column inverted index (CSC pattern) ── */
    int* col_ptr = (int*)calloc((size_t)(m + 1), sizeof(int));
    int* col_idx = (int*)malloc((size_t)nnz    * sizeof(int));
    if (!col_ptr || !col_idx) {
        perror("cluster: col index"); free(col_ptr); free(col_idx); return -1;
    }
 
    /* count */
    for (int i = 0; i < n; i++)
        for (int p = A->rowptr[i]; p < A->rowptr[i+1]; p++)
            col_ptr[A->colidx[p] + 1]++;
    /* prefix sum */
    for (int j = 0; j < m; j++) col_ptr[j+1] += col_ptr[j];
    /* fill */
    int* col_cur = (int*)malloc((size_t)m * sizeof(int));
    if (!col_cur) { perror("cluster: col_cur"); free(col_ptr); free(col_idx); return -1; }
    memcpy(col_cur, col_ptr, (size_t)m * sizeof(int));
    for (int i = 0; i < n; i++)
        for (int p = A->rowptr[i]; p < A->rowptr[i+1]; p++)
            col_idx[col_cur[A->colidx[p]]++] = i;
    free(col_cur);
 
    /* ── 2. Per-row nnz sizes ── */
    int* row_size = (int*)malloc((size_t)n * sizeof(int));
    if (!row_size) { perror("cluster: row_size"); free(col_ptr); free(col_idx); return -1; }
    for (int i = 0; i < n; i++)
        row_size[i] = A->rowptr[i+1] - A->rowptr[i];
 
    /* ── 3. Process order: decreasing nnz ── */
    RowNnz* order = (RowNnz*)malloc((size_t)n * sizeof(RowNnz));
    if (!order) { perror("cluster: order"); free(col_ptr); free(col_idx); free(row_size); return -1; }
    for (int i = 0; i < n; i++) { order[i].row = i; order[i].nnz = row_size[i]; }
    qsort(order, (size_t)n, sizeof(RowNnz), rownnz_cmp_desc);
 
    /* ── 4. Cluster assignment ──
     * assigned[i] = 1 once row i has been placed into a cluster */
    char* assigned = (char*)calloc((size_t)n, 1);
    if (!assigned) {
        perror("cluster: assigned");
        free(col_ptr); free(col_idx); free(row_size); free(order); return -1;
    }
 
    /* intersection accumulator: isect[k] = number of shared columns seen so far */
    int* isect = (int*)calloc((size_t)n, sizeof(int));
    /* touched[]: list of rows whose isect[] entry was modified (for fast reset) */
    int* touched  = (int*)malloc((size_t)n * sizeof(int));
    /* cluster member buffer */
    int* members  = (int*)malloc((size_t)BCSR_BR * sizeof(int));
    if (!isect || !touched || !members) {
        perror("cluster: work arrays");
        free(col_ptr); free(col_idx); free(row_size); free(order);
        free(assigned); free(isect); free(touched); free(members);
        return -1;
    }
 
    /* Clusters stored as a flat array of Cluster structs.
     * Upper bound: n clusters (all singletons). */
    Cluster* clusters = (Cluster*)malloc((size_t)n * sizeof(Cluster));
    if (!clusters) {
        perror("cluster: clusters");
        free(col_ptr); free(col_idx); free(row_size); free(order);
        free(assigned); free(isect); free(touched); free(members); return -1;
    }
    int n_clusters = 0;
 
    for (int oi = 0; oi < n; oi++) {
        int seed = order[oi].row;
        if (assigned[seed]) continue;
 
        /* Start new cluster with seed */
        int cl_size = 0;
        members[cl_size++] = seed;
        assigned[seed] = 1;
 
        /* Merged column set: union of columns of all current cluster members.
         * We reuse the isect accumulation pass for this. */
 
        /* Grow cluster up to BCSR_BR */
        while (cl_size < BCSR_BR) {
            /* Accumulate intersection counts from all current cluster members */
            int n_touched = 0;
            for (int ci = 0; ci < cl_size; ci++) {
                int row = members[ci];
                for (int p = A->rowptr[row]; p < A->rowptr[row+1]; p++) {
                    int j = A->colidx[p];
                    /* skip hub columns */
                    if (col_ptr[j+1] - col_ptr[j] > max_col_degree) continue;
                    for (int q = col_ptr[j]; q < col_ptr[j+1]; q++) {
                        int k = col_idx[q];
                        if (assigned[k]) continue;
                        if (isect[k] == 0) touched[n_touched++] = k;
                        isect[k]++;
                    }
                }
            }
 
            /* Compute union size for current cluster (sum of row_sizes, minus
             * duplicates already in the merged set — approximated as the
             * intersection sum divided by a correction factor).
             * For Jaccard we need |cluster_cols ∪ candidate_cols|.
             * cluster_cols size = sum of row_size[members] - over-count from shared cols.
             * We approximate cluster_union_size as the number of distinct columns
             * touched, which we track via n_touched_unique below.
             * Simpler and cheaper: use the pairwise Jaccard between the NEW candidate
             * and the cluster centroid (virtual row = union of all member rows). */
 
            /* cluster_col_count = distinct columns in the merged cluster so far.
             * We recompute it cheaply: it equals the number of unique (row, col)
             * pairs seen. Since isect[k] > 0 for any k that shares a col, the
             * cluster union size equals the sum of all cluster member row_sizes
             * minus the number of duplicate column hits within the cluster itself.
             * For simplicity we track cluster_union_size explicitly. */
            int cluster_union_sz = 0;
            for (int ci = 0; ci < cl_size; ci++) cluster_union_sz += row_size[members[ci]];
            /* subtract intra-cluster duplicates: columns shared by 2+ members
             * were counted multiple times in isect. The exact count is expensive,
             * so we use: cluster_union_sz ≈ n_touched_distinct (conservative). */
            /* Actually n_touched is already the distinct column count contributed
             * by all cluster members combined — it's the union size. */
            /* Wait: n_touched counts distinct *candidate rows* not distinct cols.
             * We need a separate distinct-col count. */
            /* Recount distinct cluster columns via a second pass (only once per
             * grow step, not per candidate): */
            int distinct_cluster_cols = 0;
            {
                /* temporary: mark cluster columns */
                char* col_mark = (char*)calloc((size_t)m, 1);
                if (col_mark) {
                    for (int ci = 0; ci < cl_size; ci++)
                        for (int p = A->rowptr[members[ci]]; p < A->rowptr[members[ci]+1]; p++)
                            col_mark[A->colidx[p]] = 1;
                    for (int j = 0; j < m; j++) if (col_mark[j]) distinct_cluster_cols++;
                    free(col_mark);
                } else {
                    /* OOM fallback: use sum of sizes as upper bound */
                    distinct_cluster_cols = cluster_union_sz;
                }
            }
 
            /* Find the best unassigned candidate by Jaccard */
            int   best_k     = -1;
            float best_jacc  = sim_threshold - 1e-9f;  /* must exceed threshold */
 
            for (int ti = 0; ti < n_touched; ti++) {
                int k = touched[ti];
                int inter = isect[k];
                int union_sz = distinct_cluster_cols + row_size[k] - inter;
                if (union_sz <= 0) continue;
                float jacc = (float)inter / (float)union_sz;
                if (jacc > best_jacc) { best_jacc = jacc; best_k = k; }
            }
 
            /* Reset isect */
            for (int ti = 0; ti < n_touched; ti++) isect[touched[ti]] = 0;
 
            if (best_k < 0) {
                /* No candidate exceeds similarity threshold.
                 * Pad to BCSR_BR with the nearest unassigned rows by index,
                 * scanning outward from the cluster's current min/max bounds. */
                int cl_min = members[0], cl_max = members[0];
                for (int ci = 1; ci < cl_size; ci++) {
                    if (members[ci] < cl_min) cl_min = members[ci];
                    if (members[ci] > cl_max) cl_max = members[ci];
                }
                int lo = cl_min - 1;  /* cursor scanning left  */
                int hi = cl_max + 1;  /* cursor scanning right */
                while (cl_size < BCSR_BR) {
                    /* Advance cursors past already-assigned rows */
                    while (lo >= 0 && assigned[lo]) lo--;
                    while (hi < n  && assigned[hi]) hi++;
                    int lo_dist = (lo >= 0) ? (cl_min - lo) : n + 1;
                    int hi_dist = (hi <  n) ? (hi - cl_max) : n + 1;
                    if (lo_dist > n && hi_dist > n) break; /* no unassigned rows left */
                    int pick = (lo_dist <= hi_dist) ? lo : hi;
                    members[cl_size++] = pick;
                    assigned[pick] = 1;
                    if (pick == lo) lo--; else hi++;
                }
                break;  /* done growing this cluster */
            }
 
            members[cl_size++] = best_k;
            assigned[best_k]   = 1;
        }
 
        /* Store cluster */
        int* cl_mem = (int*)malloc((size_t)cl_size * sizeof(int));
        if (!cl_mem) {
            /* OOM: fall back to placing remaining rows in order */
            for (int i = 0; i < n; i++) if (!assigned[i]) { assigned[i] = 1; }
            break;
        }
        /* Sort members by original row index for locality */
        /* (insertion sort — cl_size ≤ BCSR_BR = 16) */
        memcpy(cl_mem, members, (size_t)cl_size * sizeof(int));
        for (int a = 1; a < cl_size; a++) {
            int key = cl_mem[a], b = a - 1;
            while (b >= 0 && cl_mem[b] > key) { cl_mem[b+1] = cl_mem[b]; b--; }
            cl_mem[b+1] = key;
        }
        clusters[n_clusters].members = cl_mem;
        clusters[n_clusters].size    = cl_size;
        clusters[n_clusters].min_row = cl_mem[0];
        n_clusters++;
    }
 
    free(isect); free(touched); free(members); free(assigned); free(order);
    free(row_size); free(col_ptr); free(col_idx);
 
    /* ── 5. Sort clusters by min_row for spatial locality ── */
    qsort(clusters, (size_t)n_clusters, sizeof(Cluster), cluster_cmp);
 
    /* ── 6. Emit permutation ── */
    int pos = 0;
    for (int ci = 0; ci < n_clusters; ci++) {
        for (int k = 0; k < clusters[ci].size; k++)
            perm[pos++] = clusters[ci].members[k];
        free(clusters[ci].members);
    }
    free(clusters);
 
    /* Sanity: pos should equal n */
    if (pos != n) {
        fprintf(stderr, "cluster: permutation incomplete (%d / %d rows placed)\n", pos, n);
        return -1;
    }
 
    return 0;
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


/* ================= BENCHMARKING HARNESS ================= */

#define N_RUNS 6
#define N_TIMED 5
#define BENCH_SEP    "────────────────────────────────────────────────────────"

// Given original CSR and reorder method, compute permutation, apply to a CSR matrix and produce permuted A and B. Optionally convert to BCSR
static int build_config(const CSRMatrix* A_orig, const DenseBF16* B_orig, int nrows, int ncols, int embed_dim, int use_bcsr, ReorderMethod reorder, float cluster_thresh, CSRMatrix** A_csr_perm_out, BCSRMatrix** A_bcsr_out, DenseBF16** B_amx_out, int** perm_out, int** inv_perm_out, int* B_rows_amx_out, int* C_rows_amx_out, double* t_reorder_out, double* t_permute_out, double* t_bcsr_out) {
    *A_csr_perm_out = NULL;
    *A_bcsr_out = NULL;
    *B_amx_out = NULL;
    *perm_out = NULL;
    *inv_perm_out = NULL;
    *t_reorder_out = 0.0;
    *t_permute_out = 0.0;
    *t_bcsr_out = 0.0;

    // reordering
    int* perm = NULL, *inv_perm = NULL;
    CSRMatrix* A_csr_perm = NULL;

    if (reorder != REORDER_NONE) {
        if (nrows != ncols) {
            printf("    [skip] matrix not square, cannot reorder\n");
            return -1;
        }
        perm = (int*)malloc((size_t)nrows * sizeof(int));
        inv_perm = (int*)malloc((size_t)nrows * sizeof(int));
        if (!perm || !inv_perm) {
            perror("build_config perm"); free(perm); free(inv_perm); return -1;
        }

        double t0 = now_sec();
        int ok = -1;
        if (reorder == REORDER_AMD)     ok = compute_amd_order(A_orig, perm);
        else if (reorder == REORDER_RCM)     ok = compute_rcm_order(A_orig, perm);
        else if (reorder == REORDER_CLUSTER) ok = compute_cluster_order(A_orig, perm, cluster_thresh, CLUSTER_MAX_COL_DEGREE);

        if (ok != 0) {
            printf("    [skip] reorder failed\n");
            free(perm); free(inv_perm); return -1;
        }
        *t_reorder_out = now_sec() - t0;
        printf("    reorder : %.3f s\n", *t_reorder_out);

        for (int i = 0; i < nrows; i++) inv_perm[perm[i]] = i;

        t0 = now_sec();
        A_csr_perm = csr_permute(A_orig, perm, inv_perm);
        if (!A_csr_perm) {
            printf("    [skip] csr_permute failed\n");
            free(perm); free(inv_perm); return -1;
        }
        *t_permute_out = now_sec() - t0;
        printf("    permute : %.3f s\n", *t_permute_out);
    }

    const CSRMatrix* A_csr = A_csr_perm ? A_csr_perm : A_orig;

    // BCSR conversion
    BCSRMatrix* A_bcsr = NULL;
    int B_rows_amx, C_rows_amx;

    if (use_bcsr) {
        double t0 = now_sec();
        A_bcsr = csr_to_bcsr(A_csr);
        if (!A_bcsr) {
            printf("    [skip] csr_to_bcsr failed (out of memory?)\n");
            if (A_csr_perm) csr_free(A_csr_perm);
            free(perm); free(inv_perm); return -1;
        }
        *t_bcsr_out = now_sec() - t0;
        float fill = (float)A_csr->nnz / ((float)A_bcsr->nblocks * BCSR_BLOCK_ELEMS) * 100.f;
        printf("    CSR to BCSR : %.3f s (%d blocks, fill=%.1f%%)\n", *t_bcsr_out, A_bcsr->nblocks, fill);
        B_rows_amx = A_bcsr->ncols;
        C_rows_amx = A_bcsr->nrows;
    } else {
        B_rows_amx = ncols;
        C_rows_amx = nrows;
    }

    // permute B
    DenseBF16* B_amx;
    if (perm != NULL) {
        B_amx = dense_bf16_permute_rows(B_orig, perm, ncols, B_rows_amx);
    } else {
        // no reorder, B already has the correct layout
        if (B_rows_amx > B_orig->rows) {
            B_amx = dense_bf16_alloc(B_rows_amx, embed_dim);
            if (!B_amx) {perror("build_config B_amx pad"); goto fail;}
            for (int r = 0; r < B_orig->rows; r++) {
                memcpy(B_amx->data + (size_t)r * B_amx->stride, B_orig->data + (size_t)r * B_orig->stride, (size_t)embed_dim * sizeof(uint16_t));
            }
        } else {
            // dims match, share data pointer via shallow copy struct
            B_amx = (DenseBF16*)malloc(sizeof(DenseBF16));
            if (!B_amx) {perror("build_config B_amx ref"); goto fail;}
            *B_amx = *B_orig;
        }
    }

    *A_csr_perm_out  = A_csr_perm;
    *A_bcsr_out = A_bcsr;
    *B_amx_out = B_amx;
    *perm_out = perm;
    *inv_perm_out = inv_perm;
    *B_rows_amx_out = B_rows_amx;
    *C_rows_amx_out = C_rows_amx;
    return 0;

fail:
    if (A_bcsr) bcsr_free(A_bcsr);
    if (A_csr_perm) csr_free(A_csr_perm);
    free(perm);
    free(inv_perm);
    return -1;
}


typedef enum { MODE_ALL = 0, MODE_CSR = 1, MODE_BCSR = 2 } BenchMode;
typedef struct { double warmup, mean, stddev; int skipped; } BenchResult;

static BenchResult bench_one_config(const char* label, const CSRMatrix* A_csr, const BCSRMatrix* A_bcsr, const DenseBF16* B_amx, int embed_dim, int nrows, int C_rows_amx, const int* perm, const int* inv_perm) {
    (void)perm; (void)inv_perm; (void)nrows;
    BenchResult res = {0};
    int use_bcsr = (A_bcsr != NULL);

    double times[N_RUNS];
    DenseF32* C_perm = dense_f32_alloc(C_rows_amx, embed_dim);
    if (!C_perm) { fprintf(stderr, "bench: C alloc failed\n"); res.skipped = 1; return res; }

    for (int r = 0; r < N_RUNS; r++) {
        double t0 = now_sec();
        if (use_bcsr) amx_spmm_bcsr(A_bcsr, B_amx, C_perm);
        else amx_spmm_csr (A_csr,  B_amx, C_perm);
        times[r] = now_sec() - t0;
    }
    dense_f32_free(C_perm);

    double sum = 0.0;
    for (int r = 1; r <= N_TIMED; r++) sum += times[r];
    res.mean = sum / N_TIMED;
    double var = 0.0;
    for (int r = 1; r <= N_TIMED; r++) { double d = times[r] - res.mean; var += d*d; }
    res.warmup = times[0];
    res.stddev = (N_TIMED > 1) ? sqrt(var / (N_TIMED - 1)) : 0.0;
    printf(" %-22s  warmup=%5.3fs  mean=%6.3fs  stddev=%5.3fs\n", label, res.warmup, res.mean, res.stddev);
    return res;
}

static void csv_write_header(FILE* csv) {
    fprintf(csv, "matrix,nrows,ncols,nnz,sparsity_pct,embed_dim,config,reorder_s,permute_s,bcsr_convert_s,nblocks,fill_pct,warmup_s,mean_s,stddev_s\n");
}

/* ================= MTX EXTRACTION ================= */

// run program on a single .mtx file
static void run_mtx_file(const char* path, int embed_dim, float cluster_thresh, BenchMode mode, FILE* csv) {
    printf("\n%s\n", BENCH_SEP);
    printf("  File : %s\n", path);

    int nrows, ncols;
    double t0 = now_sec();
    CSRMatrix* A_orig = mtx_read_to_csr(path, &nrows, &ncols);
    if (!A_orig) return;

    double sparsity = 1.0 - (double)A_orig->nnz / ((double)nrows * ncols);
    printf("A : %d x %d  nnz = %d  sparsity = %.2f%%  load = %.3fs\n", nrows, ncols, A_orig->nnz, sparsity * 100, now_sec() - t0);

    const char* matrix_name = strrchr(path, '/');
    matrix_name = matrix_name ? matrix_name + 1 : path;

    random_seed = 0xABCD1234u;
    DenseBF16* B_orig = dense_bf16_alloc(ncols, embed_dim);
    if (!B_orig) { perror("B_orig"); csr_free(A_orig); return; }
    for (int r = 0; r < ncols; r++)
        for (int c = 0; c < embed_dim; c++)
            dense_bf16_set(B_orig, r, c, float_to_bf16(random_float() * 2.0f - 1.0f));
    printf("\n");

    static const struct { const char* label; int use_bcsr; ReorderMethod reorder; } configs[] = {
        { "CSR",            0, REORDER_NONE    },
        { "BCSR",           1, REORDER_NONE    },
        { "BCSR + AMD",     1, REORDER_AMD     },
        { "BCSR + RCM",     1, REORDER_RCM     },
        { "BCSR + Cluster", 1, REORDER_CLUSTER }
    };
    int n_configs = (int)(sizeof(configs) / sizeof(configs[0]));

    for (int ci = 0; ci < n_configs; ci++) {
        if (mode == MODE_CSR  &&  configs[ci].use_bcsr) continue;
        if (mode == MODE_BCSR && !configs[ci].use_bcsr) continue;

        printf("    [%s]\n", configs[ci].label);

        CSRMatrix*  A_csr_perm = NULL;
        BCSRMatrix* A_bcsr     = NULL;
        DenseBF16*  B_amx      = NULL;
        int* perm = NULL, *inv_perm = NULL;
        int  B_rows_amx, C_rows_amx;
        double t_reorder = 0.0, t_permute = 0.0, t_bcsr_conv = 0.0;

        int ok = build_config(A_orig, B_orig, nrows, ncols, embed_dim, configs[ci].use_bcsr, configs[ci].reorder, cluster_thresh, &A_csr_perm, &A_bcsr, &B_amx, &perm, &inv_perm, &B_rows_amx, &C_rows_amx, &t_reorder, &t_permute, &t_bcsr_conv);

        if (ok != 0) {
            if (csv)
                fprintf(csv, "%s,%d,%d,%d,%.4f,%d,%s,,,,,,,,skipped\n", matrix_name, nrows, ncols, A_orig->nnz, sparsity * 100.0, embed_dim, configs[ci].label);
            printf("\n");
            continue;
        }

        int   nblocks  = A_bcsr ? A_bcsr->nblocks : 0;
        float fill_pct = A_bcsr ? (float)A_orig->nnz / ((float)nblocks * BCSR_BLOCK_ELEMS) * 100.f : 0.f;

        const CSRMatrix* A_csr_kernel = A_csr_perm ? A_csr_perm : A_orig;
        BenchResult res = bench_one_config(configs[ci].label, A_bcsr ? NULL : A_csr_kernel, A_bcsr, B_amx, embed_dim, nrows, C_rows_amx, perm, inv_perm);

        if (csv && !res.skipped) {
            fprintf(csv, "%s,%d,%d,%d,%.4f,%d,%s,%.3f,%.3f,%.3f,%d,%.2f,%.6f,%.6f,%.6f\n",
                    matrix_name, nrows, ncols, A_orig->nnz,
                    sparsity * 100.0, embed_dim,
                    configs[ci].label,
                    t_reorder, t_permute, t_bcsr_conv,
                    nblocks, fill_pct,
                    res.warmup, res.mean, res.stddev);
            fflush(csv);
        }

        if (A_csr_perm) csr_free(A_csr_perm);
        bcsr_free(A_bcsr);
        if (B_amx && B_amx->data != B_orig->data) dense_bf16_free(B_amx);
        else free(B_amx);
        free(perm); free(inv_perm);
        printf("\n");
    }
    csr_free(A_orig);
    dense_bf16_free(B_orig);
}

// run program on all .mtx files in a directory
static void run_mtx_directory(const char* dir, int embed_dim, float cluster_thresh, BenchMode mode, FILE* csv) {
    printf("\n========================================\n");
    printf("Source Directory : %s\n", dir);
    printf("embed_dim = %d\n", embed_dim);
    printf("runs = %d (warmup=1, timed=%d)\n", N_RUNS, N_TIMED);
    printf("========================================\n");

    DIR* d = opendir(dir);
    if (!d) { perror(dir); return; }

    struct dirent* ent;
    int count = 0;
    while ((ent = readdir(d)) != NULL) {
        size_t len = strlen(ent->d_name);
        if (len < 4 || strcmp(ent->d_name + len - 4, ".mtx") != 0) continue;
        char fullpath[4096];
        snprintf(fullpath, sizeof(fullpath), "%s/%s", dir, ent->d_name);
        run_mtx_file(fullpath, embed_dim, cluster_thresh, mode, csv);
        count++;
    }
    closedir(d);
    if (count == 0) printf("(no .mtx files found in %s)\n", dir);
    else printf("\nProcessed %d files.\n", count);
}



/* ================= MAIN ================= */

/*
Usage:
    ./amx --mtx <file|dir>                     single file or whole directory
    ./amx --mtx <path> --embed 128             embed dimension (default 64)
    ./amx --mtx <path> --mode csr              CSR only
    ./amx --mtx <path> --mode bcsr             BCSR variants only
    ./amx --mtx <path> --mode all              all configs (default)
    ./amx --mtx <path> --cluster-thresh 0.1    Jaccard threshold
    ./amx --mtx <path> --csv results.csv       write results to CSV
*/
int main(int argc, char** argv) {
    printf("Intel AMX BF16 spMM\n");
    printf("====================\n");
    printf("TILE_MAX_ROWS      = %d\n", TILE_MAX_ROWS);
    printf("TILE_MAX_FP32_COLS = %d\n", TILE_MAX_F32_COLS);
    printf("BCSR block         = %d x %d = %d BF16 elems (one full tile)\n\n", BCSR_BR, BCSR_BC, BCSR_BLOCK_ELEMS);
#ifdef USE_AMD
    printf("AMD available through SuiteSparse\n");
#else
    printf("AMD not available: rebuild with -DUSE_AMD\n");
#endif
    printf("RCM available by default\n");
    printf("Hierarchical Clustering available\n\n");

    if (request_amx_perm() != 1) {
        fprintf(stderr, "ERROR: Failed to enable AMX tile state.\n");
        return 1;
    }

    const char* mtx_path = NULL;
    const char* csv_path = NULL;
    int embed_dim = 64;
    float cluster_thresh = CLUSTER_DEFAULT_THRESHOLD;
    BenchMode mode = MODE_ALL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--mtx") == 0 && i+1 < argc) mtx_path = argv[++i];
        else if (strcmp(argv[i], "--embed") == 0 && i+1 < argc) embed_dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--cluster-thresh") == 0 && i+1 < argc) cluster_thresh = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--csv") == 0 && i+1 < argc) csv_path = argv[++i];
        else if (strcmp(argv[i], "--mode") == 0 && i+1 < argc) {
            const char* m = argv[++i];
            if (strcmp(m, "csr")  == 0) mode = MODE_CSR;
            else if (strcmp(m, "bcsr") == 0) mode = MODE_BCSR;
            else if (strcmp(m, "all")  == 0) mode = MODE_ALL;
            else { fprintf(stderr, "Unknown mode '%s' (choices: csr, bcsr, all)\n", m); return 1; }
        } else {
            fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            fprintf(stderr,
                "Usage: %s --mtx <file|dir> [--embed 64|128]"
                " [--mode csr|bcsr|all] [--cluster-thresh <thresh>]"
                " [--csv <output.csv>]\n", argv[0]);
            return 1;
        }
    }

    if (!mtx_path) { fprintf(stderr, "No --mtx path specified.\n"); return 1; }

    FILE* csv = NULL;
    if (csv_path) {
        csv = fopen(csv_path, "w");
        if (!csv) { perror(csv_path); return 1; }
        csv_write_header(csv);
        printf("CSV output : %s\n\n", csv_path);
    }

    struct stat st;
    if (stat(mtx_path, &st) != 0) { perror(mtx_path); if (csv) fclose(csv); return 1; }

    if (S_ISREG(st.st_mode)) run_mtx_file(mtx_path, embed_dim, cluster_thresh, mode, csv);
    else if (S_ISDIR(st.st_mode)) run_mtx_directory(mtx_path, embed_dim, cluster_thresh, mode, csv);
    else { fprintf(stderr, "--mtx: '%s' is neither a file nor a directory\n", mtx_path); if (csv) fclose(csv); return 1; }

    if (csv) { fclose(csv); printf("\nResults written to %s\n", csv_path); }
    return 0;
}