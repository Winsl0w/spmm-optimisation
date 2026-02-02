// Rearranges LHS Matrix in a transpose manner so that columns are contiguous in memory
// Taken from arm developer documentation

// x0: M, the number of rows in the matLeft matrix
// x1: K, the number of columns in the matLeft matrix
// x2: The base address of the input matrix, matLeft
// x3: The base address of the output matrix, matLeft_mod

preprocess_l:      // x0: M, x1: K, x2: &matLeft, x3: &matLeft_mod
    smstart
    // constants
    cntw    x4                      // SVLs
    mul     x11, x4, x1             // SVLs*K
    lsl     x14, x1, #1             // 2*K
    add     x15, x14, x1            // 3*K

    mul     x16, x4, x4             // SVLs*SVLs

    mov     x7, #0
    whilelt p0.s, x7, x0            // Tile predicate (M dimension)

.Loop_outer:
    mov     x8, x2                  // matLeft load base address
    mov     x9, x3                  // matLeft_mod store base address
    add     x5,  x2, x1, lsl #2     // Exit condition for inner loop

    add     x10, x9 , x11, lsl #2   // 32b Tile0 store predicate condition
    sub     x13, x10, x16, lsl #2   // 32b Tile1 store predicate condition
    whilelt pn8.b, x8, x5, vlx2     // Tile predicate-as-counter (K dimension)

.Loop_inner:
    mov     x6, x8                  // matLeft

    mov     w12, #0                 // Load_loop counter

.Load_loop:
    psel    pn10, pn8, p0.s[w12, 0]
    psel    pn11, pn8, p0.s[w12, 1]
    psel    pn12, pn8, p0.s[w12, 2]
    psel    pn13, pn8, p0.s[w12, 3]
    ld1w    {z20.s, z28.s}, pn10/z, [x6]                // matLeft
    ld1w    {z21.s, z29.s}, pn11/z, [x6, x1,  lsl #2]   // matLeft + K
    ld1w    {z22.s, z30.s}, pn12/z, [x6, x14, lsl #2]   // matLeft + K*2
    ld1w    {z23.s, z31.s}, pn13/z, [x6, x15, lsl #2]   // matLeft + K*3    
    mova    za0h.s[w12, 0:3], {z20.s-z23.s}
    mova    za1h.s[w12, 0:3], {z28.s-z31.s}

    add     x6, x6, x1, lsl #4      // matLeft+=4*K FP32 elements (bytes)
    add     w12, w12, #4            // Increment counter
    cmp     w12, w4
    b.mi    .Load_loop

    mov     w12, #0                 // Store_loop counter

.Store_loop:
    whilelt pn10.b, x9, x10, vlx4
    whilelt pn11.b, x9, x13, vlx4
    mova    {z0.s-z3.s}, za0v.s[w12, 0:3]
    mova    {z4.s-z7.s}, za1v.s[w12, 0:3]
    st1w    {z0.s-z3.s}, pn10, [x9] // Store 4 col vectors to matLeft_mod
    st1w    {z4.s-z7.s}, pn11, [x9, x16, lsl #2]  // matLeft_mod+SVLs*SVLs
    addvl   x9, x9, #4              // matLeft_mod += 4*SVLb (bytes)
    add     w12, w12, #4            // Increment counter
    cmp     w12, w4
    b.mi    .Store_loop

    add     x9, x9, x16, lsl #2
    addvl   x8, x8, #2              // matLeft+= 2*SVLb (bytes)
    whilelt pn8.b, x8, x5, vlx2
    b.first .Loop_inner

    add     x3, x3, x11, lsl #2     // matLeft_mod+= SVLs*K FP32 elms (bytes)
    add     x2, x2, x11, lsl #2     // matLeft+= SVLs*K FP32 elms (bytes]
    incw    x7

    whilelt p0.s, x7, x0
    b.first .Loop_outer

    smstop

    ret