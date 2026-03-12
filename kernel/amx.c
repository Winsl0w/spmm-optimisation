#include <stdint.h>
#include <stdbool.h>
#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <linux/time.h>


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


/* ===================================================
    CSR MICROKERNEL
 * =================================================== */ 

static void amx_spmm_csr(const CSRMatrix* A, const DenseBF16* B, DenseF32* C) {
    assert(A->ncols == B->rows && A->nrows == C->rows && B->cols == C->cols);

    int M = A->nrows, N = B->cols, K = A->ncols;

    for (int r = 0; r < M; r++) {
        memset(&C->data[r * C->stride], 0, C->cols * sizeof(float));
    }

    for (int m0 = 0; m0 < M; m0 += TILE_MAX_ROWS) {
        int m_tile = (m0 + TILE_MAX_ROWS <= M) ? TILE_MAX_ROWS : M - m0;

        for (int n0 = 0; n0 < N; n0 += TILE_MAX_F32_COLS) {
            int n_tile_f32 = (n0 + TILE_MAX_F32_COLS <= N) ? TILE_MAX_F32_COLS : N - n0;
            
            configure_amx_tiles_bf16(m_tile, n_tile_f32, K_TILE);

            // zero and load C tile
            memset(global_c_buf, 0, sizeof(global_c_buf));
            _tile_loadd(TILE_C, global_c_buf, TILE_MAX_F32_COLS * (int)sizeof(float));

            for (int k0 = 0; k0 < K; k0 += K_TILE) {
                int k_strip = (k0 + K_TILE <= K) ? K_TILE : K - k0;
                int k_strip_padded = (k_strip + 1) & ~1; // round up to even for pairing

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

                pack_b_tile(B, k0, k_strip_padded, n0, n_tile_f32);

                // update tile config for actual k_strip_padded as it could differ from K_TILE on the final strip
                configure_amx_tiles_bf16(m_tile, n_tile_f32, k_strip_padded);
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
    }
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

    // Parse banner
    MM_typecode matcode;
    int ret = mm_read_banner(f, &matcode);
    if (ret != 0) {
        fprintf(stderr, "%s: mm_read_banner failed (code %d)\n", path, ret);
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

    int is_pattern   = mm_is_pattern(matcode);
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
        int    row, col;
        double real_val = 0.0, imag_val = 0.0;
        ret = mm_read_mtx_crd_entry(f, &row, &col, &real_val, &imag_val,
                                    matcode);
        if (ret != 0) {
            fprintf(stderr, "%s: read error at entry %d (code %d)\n",
                    path, e, ret);
            break;
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

    CSRMatrix *m = csr_alloc(nrows, ncols, (int)nnz);

    for (long i = 0; i < nnz; i++)
        m->rowptr[coo[i].row + 1]++;
    for (int r = 0; r < nrows; r++)
        m->rowptr[r+1] += m->rowptr[r];

    int *cursor = (int *)malloc((size_t)nrows * sizeof(int));
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


/* ================= MAIN ================= */

int main() {
    printf("Intel AMX BF16 spMM\n");
    printf("====================\n");
    printf("  sizeof(tilecfg_t)  = %zu  (must be 64)\n", sizeof(__tilecfg));
    printf("  TILE_MAX_ROWS      = %d\n", TILE_MAX_ROWS);
    printf("  TILE_MAX_FP32_COLS = %d\n", TILE_MAX_F32_COLS);
    printf("  BCSR block         = %d x %d = %d BF16 elems (one full tile)\n\n", BCSR_BR, BCSR_BC, BCSR_BLOCK_ELEMS);

    if (amx_request_permission() != 0) {
        fprintf(stderr, "ERROR: Failed to enable AMX tile state.\n");
        return 1;
    }
}