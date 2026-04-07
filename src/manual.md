# User manual

Usage: 

./amx --mtx <file|dir>                     single file or whole directory\\
./amx --mtx <path> --embed 128             embed dimension (default 64)\\
./amx --mtx <path> --mode csr              CSR only\\
./amx --mtx <path> --mode bcsr             BCSR variants only\\
./amx --mtx <path> --mode all              all configs (default)\\
./amx --mtx <path> --cluster-thresh 0.1    Jaccard threshold\\
./amx --mtx <path> --csv results.csv       write results to CSV\\