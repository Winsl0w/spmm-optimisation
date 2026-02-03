matmul_opt: // x0: M, x1: K, x2: N, x3: matLeft_mod, x4: matRight, x5: matResult
   stp     x19, x20, [sp, #-48]!
   stp     x21, x22, [sp, #16]
   stp     x23, x24, [sp, #32]
   smstart

   // constants
   cntw    x6                      // SVLs
   mul     x22, x6, x1             // SVLs*K
   mul     x23, x6, x2             // SVLs*N
   add     x25, x23, x2            // SVLs*N + N
   add     x11, x4, x2, lsl #2     // Exit condition for N loop
   mov     x12, #0
   cntb    x6                      // SVLb
   mov     x14, #0
   ptrue   pn10.b                  // Predicate for SME2 VLx2 (a_ptr loads)
   whilelt pn8.s, x12, x0, vlx2    // tiles predicate (M dimension)
   sub     w6, w6, #8              // SVLb-8

 .Loop_M:
   // Extract tile 0/1 and tile 2/3 predicates (M) from vlx2 predicate.
   pext    { p2.s, p3.s }, pn8[0]
   mov     x16, x4                 // b_base
   mov     x9, x5                  // c_base
   whilelt pn9.b, x16, x11, vlx2   // tiles predicate (N dimension)

 .Loop_N:
   mov     x7, x3                  // a_ptr = a_base
   mov     x17, x16                // b_ptr = b_base
   mov     x10, x9                 // c_ptr0 = c_base

   // Extract tile 0/2 and tile 1/3 predicates (N) from vlx2 predicate.
   pext    { p0.b, p1.b }, pn9[0]

   add     x8, x3, x22, lsl #2     // a_base + SVLs*K FP32 elms (bytes)
   addvl   x15, x8, #-1            // Exit condition for K loop
   ld1w    {z1.s}, p2/z, [x7]      // Load 1st vector from a_ptr

   zero    {za}
   ld1w    {z2.s-z3.s}, pn9/z, [x17]  // Load 2 vectors from b_ptr

   fmopa   za0.s, p2/m, p0/m, z1.s, z2.s // ZA0+=1st a_ptr vec OP 1st b_ptr vec
   ld1w    {z5.s}, p3/z, [x7, x22, lsl #2]  // Load 2nd vector from a_ptr
   addvl   x7, x7, #1                       // a_ptr += SVLb (bytes)

 .Loop_K:
   fmopa   za2.s, p3/m, p0/m, z5.s, z2.s // ZA2+=2nd a_ptr vec OP 1st b_ptr vec

   fmopa   za1.s, p2/m, p1/m, z1.s, z3.s // ZA1+=1st a_ptr vec OP 2nd b_ptr vec
   ld1w    {z0.s-z1.s}, pn10/z, [x7]     // Load next 2 vectors from a_ptr

   fmopa   za3.s, p3/m, p1/m, z5.s, z3.s // ZA3+=2nd a_ptr vec OP 2nd b_ptr vec
   ld1w    {z6.s-z7.s}, pn9/z, [x17, x2, lsl #2] // Load next 2 vecs from b_ptr

   fmopa   za0.s, p2/m, p0/m, z0.s, z6.s // ZA0+=1st a_ptr vec OP 1st b_ptr vec
   psel    pn11, pn10, p3.s[w14, 0]      // Select predicate-as-counter
   ld1w    {z4.s-z5.s}, pn11/z, [x7, x22, lsl #2] // Load next 2 vecs from a_ptr

   fmopa   za2.s, p3/m, p0/m, z4.s, z6.s // ZA2+=2nd a_ptr vec OP 1st b_ptr vec
   add     x17, x17, x2, lsl #3          // b_ptr += 2*N FP32 elms (bytes)

   fmopa   za1.s, p2/m, p1/m, z0.s, z7.s // ZA1+=1st a_ptr vec OP 2nd b_ptr vec

   fmopa   za3.s, p3/m, p1/m, z4.s, z7.s // ZA3+=2nd a_ptr vec OP 2nd b_ptr vec
   ld1w    {z2.s-z3.s}, pn9/z, [x17]     // Load next 2 vectors from b_ptr

   fmopa   za0.s, p2/m, p0/m, z1.s, z2.s // ZA0+=1st a_ptr vec OP 1st b_ptr vec
   addvl   x7, x7, #2                    // a_ptr += 2*SVLb (bytes)

   cmp     x7, x15
   b.mi    .Loop_K

   fmopa   za2.s, p3/m, p0/m, z5.s, z2.s // ZA2+=2nd a_ptr vec OP 1st b_ptr vec

   fmopa   za1.s, p2/m, p1/m, z1.s, z3.s // ZA1+=1st a_ptr vec OP 2nd b_ptr vec

   fmopa   za3.s, p3/m, p1/m, z5.s, z3.s // ZA3+=2nd a_ptr vec OP 2nd b_ptr vec
   add     x17, x17, x2, lsl #2          // b_ptr += 2*N FP32 elms (bytes)

   cmp     x7, x8
   b.pl    .Ktail_end

 .Ktail_start:
   ld1w    {z1.s}, p2/z, [x7]
   ld1w    {z2.s-z3.s}, pn9/z, [x17]

   fmopa   za0.s, p2/m, p0/m, z1.s, z2.s
   ld1w    {z5.s}, p3/z, [x7, x22, lsl #2]

   fmopa   za2.s, p3/m, p0/m, z5.s, z2.s

   fmopa   za1.s, p2/m, p1/m, z1.s, z3.s

   fmopa   za3.s, p3/m, p1/m, z5.s, z3.s

 .Ktail_end:
   mov     w13, #0
   psel    pn11, pn9, p2.b[w13, 0]
   psel    pn12, pn9, p3.b[w13, 0]
   // ZA tiles to vecs: z0 = za0h[1], z1 = za1h[1], z2 = za2h[1], z3 = za3h[1]
   mova    { z0.b-z3.b }, za0h.b[w13, 0:3]
   st1w    { z0.s-z1.s }, pn11, [x10]              // Store to c_ptr0
   st1w    { z2.s-z3.s }, pn12, [x10, x23, lsl #2] // Store to c_ptr0+(SVLs*N)
 .Loop_store_ZA:
   psel    pn11, pn9, p2.b[w13, 4]
   psel    pn12, pn9, p3.b[w13, 4]
   mova    { z0.b-z3.b }, za0h.b[w13, 4:7]
   st1w    { z0.s-z1.s }, pn11, [x10, x2, lsl #2]  // Store to c_ptr0+N
   st1w    { z2.s-z3.s }, pn12, [x10, x25, lsl #2] // Store to c_ptr0+(SVLs+1)*N
 
   add     x10, x10, x2, lsl #3    // c_ptr0 += 2*N FP32 elms (bytes)
   add     w13, w13, #8
 
   psel    pn11, pn9, p2.b[w13, 0]
   psel    pn12, pn9, p3.b[w13, 0]
   mova    { z0.b-z3.b }, za0h.b[w13, 0:3]
   st1w    { z0.s-z1.s }, pn11, [x10]               // Store to c_ptr0
   st1w    { z2.s-z3.s }, pn12, [x10, x23, lsl #2]  // Store to c_ptr0+SVLs*N
   cmp     w13, w6
   b.mi    .Loop_store_ZA
 
   psel    pn11, pn9, p2.b[w13, 4]
   psel    pn12, pn9, p3.b[w13, 4]
   mova    { z0.b-z3.b }, za0h.b[w13, 4:7]
   st1w    { z0.s-z1.s }, pn11, [x10, x2, lsl #2]  // Store to c_ptr0+N
   st1w    { z2.s-z3.s }, pn12, [x10, x25, lsl #2] // Store to c_ptr0+(SVLs+1)*N
 
   addvl   x9, x9, #2
   addvl   x16, x16, #2            // b_base += 2*SVLb (bytes)
   whilelt pn9.b, x16, x11, vlx2   // tile predicate (N dimension)
   b.first .Loop_N
 
   add     x3, x3, x22, lsl #3     // a_base += 2*SVLs*K FP32 elms (bytes)
   add     x5, x5, x23, lsl #3     // c_base += 2*SVLs*N FP32 elms (bytes)
   incw    x12, all, mul #2        // M loop counter += 2* SVLs
   whilelt pn8.s, x12, x0, vlx2    // tiles predicate (M dimension)
   b.first .Loop_M
 
   smstop
 
   ldp     x23, x24, [sp, #32]
   ldp     x21, x22, [sp, #16]
   ldp     x19, x20, [sp], #48
 
   ret