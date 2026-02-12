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
static void amx_tile_config_bf16(__tilecfg *cfg) {
    cfg->palette_id = 1;
    cfg->start_row = 0;

    // tile C (FP32 accumulator)
    cfg->rows[0] = BLOCK_SIZE;
    cfg->colsb[0] = 64;         // 16 * 4 bytes

    // tile A 
    cfg->rows[1] = BLOCK_SIZE;
    cfg->colsb[1] = 32;         // 16 * 2 bytes

    // tile B
    cfg->rows[2] = BLOCK_SIZE;
    cfg->colsb[2] = 32;         // 16 * 2 bytes

    _tile_loadconfig(&cfg);
}



// Initialise the tile config register for int8
static void amx_tile_config_int8(__tilecfg *cfg) {
    cfg->palette_id = 1;
    cfg->start_row = 0;

    // tile C (accumulator)
    cfg->rows[0] = 16;
    cfg->colsb[0] = 16;

    // tile A 
    cfg->rows[1] = 16;
    cfg->colsb[1] = 64;

    // tile B
    cfg->rows[2] = 16;
    cfg->colsb[2] = 64;

    _tile_loadconfig(&cfg);
}


// Request use of AMX from Linux kernel
static bool set_tiledata_use() {
    if (syscall(SYS_arch_prct1, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        printf("\n Failed to enable XFEATURE_XTILEDATA \n\n");
        return false;
    } else {
        printf("\n TILE DATA USE SET - OK \n\n");
        return true;
    }
}

/*
    A: pointer to top left element of submatrix of A
    B: pointer to top left element of submatrix of B
    C: pointer to top left element of submatrix of C
    K: reduction dimension (shared by A and B) for this microkernel needs to be a multiple of 64
*/
// Main int8 microkernel
static void amx_gemm_int8_16x16(const int8_t* restrict A, const int8_t* restrict B, int32_t* restrict C, int K) { 
    // clear the accumulator
    _tile_zero(0);

    //#pragma loop vectorize(enable) unroll(full)
    for (int i = 0; i < K; i += STRIDE) {       // walk through the matrix
        _tile_loadd(1, A + i, STRIDE);          // load A into tmm1, address A+i moves pointer to column i, still points to row 0, rows handled through the STRIDE
        _tile_loadd(2, B + i * STRIDE, STRIDE); // load B into tmm2, move down i rows, stay at column 0
        _tile_dpbssd(0, 1, 2);
    }

    _tile_stored(0, C, STRIDE);                 // writes contents of tmm0 to memory

    _tile_release();                            // release tile config, also releases resources
}

/*================================================*/
/* 
BF16 microkernel
*/
static inline void amx_block_bf16_16x16_accumulate(const uint16_t* restrict A, const uint16_t* restrict B, float* restrict C, int ldc) {
    _tile_load(0, C, ldc*sizeof(float));              // load existing C
    _tile_load(1, A, BLOCK_SIZE*sizeof(uint16_t));    // load A
    _tile_load(2, B, BLOCK_SIZE*sizeof(uint16_t));    // load B

    _tile_dpbf16ps(0, 1, 2);    // C += A * B

    _tile_stored(0, C, ldc * sizeof(float));
}






int main() {
    __tilecfg tile_data = {0};
    int8_t A[MAX];
    int8_t B[MAX];
    int32_t C[MAX/4];
    int rows  = MAX_ROWS;
    int colsb = MAX_COLS;

    // request permission to use AMX
    if (!set_tiledata_use()) {
        exit(-1);
    }

    // load tile config
    amx_tile_config_bf16(&tile_data);
    // amx_tile_config_int8(&tile_data);

    // initialise src matrix buffers with data (untested- currently populated with 2s)
    init_buffer (A, 2);
    print_buffer(A, rows, colsb);

    init_buffer (B, 2);
    print_buffer(B, rows, colsb);

    // initialise destination matrix buffers with data
    init_buffer32(C, 0);

    amx_gemm_int8_16x16(A, B, C, 64);
}