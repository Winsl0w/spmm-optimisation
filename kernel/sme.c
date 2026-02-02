#include <arm_sme.h>
#include <arm_sve.h>


void sme_gemm_f32(const float* A, const float* B, float* C, int lda, int ldb, int ldc, int M, int N, int K) {
    // Enter SME streaming mode
    __arm_sme_enable();

    // Predicates for rows and columns (masking approach)
    svbool_t pm = svwhilelt_b32(0, M);
    svbool_t pn = svwhilelt_b32(0, N);

    // All lanes active (no masking)
    svbool_t p = svptrue_b32();

    // Zero the ZA accumulator
    svzero_za();

    // replace predicates with p, however this can only be done if M == VL, N == VL and all memory is valid and contiguous for those lanes
    for (int k = 0; k < K; k++) {
        // Load A column (M elements)
        svfloat32_t a = svld1(pm, &A[k]);       // &A[k] points to column k of row 0, predicate ensures we only read valid rows

        // Load B row (N elements)
        svfloat32_t b = svld1(pn, &B[k * ldb]); 

        // ZA += a ⊗ b
        svfmopa_za32(pm, pn, a, b);
    }

    // Store ZA → C
    svst1_za32(pm, pn, C, ldc);

    // Exit SME streaming mode
    __arm_sme_disable();
}
