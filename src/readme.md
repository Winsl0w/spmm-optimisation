This program performs sparse matrix multiplication using the Intel AMX ISA extension. Three reordering algorithms used to compare performance across sparsity patterns, dimensions and structure:
* Reverse Cuthill-McKee (RCM)
* Approximate Minimum Degree (AMD)
* Hierarchical Clustering

### Requirements

* GCC compiler version 22 (may work with earlier versions but this is untested)
* SuiteSparse reordering library (if using) must be in `/src/`, it is up to the user to install and compile SuiteSparse according to their build instructions (only the AMD library and its dependencies is required for operation) available at https://github.com/DrTimothyAldenDavis/SuiteSparse.
* A Linux machine with the Intel AMX CPU extension enabled.

### Build Steps

This project was developed on the Fisherman server, consequently the exact build instructions are slightly different had SuiteSparse been installed to the default directory.<br>

To build the program navigate to `src/kernel/` and execute `gcc -O3 -mamx-bf16 -mamx-tile -march=native -DUSE_AMD -I../SuiteSparse/AMD/Include -I../SuiteSparse/SuiteSparse_config -o amx amx.c ../utility/mmio.o ../SuiteSparse/lib/libamd.a ../SuiteSparse/SuiteSparse_config/build/libsuitesparseconfig.a -lm -fopenmp`.<br>

Other build instructions may work though during testing I was not privileged enough to install SuiteSparse to the default directory and thus are untested.

### Test Steps

Tested on 11 matrices from the SuiteSparse matrix collection. All kernel configurations tested sequentially taking an average of 5 executions after a first execution to account for cache warmup.

* Start the program by executing `./amx --mtx /<path to file or directory>/ --embed 64|128 --mode csr|bcsr` to run a test harness. Full usage description is available in `manual.md`.