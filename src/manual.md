# User manual

Usage: 

`./amx --mtx <file|dir>`                     single file or whole directory <br>
`./amx --mtx <path> --embed 128`             embed dimension (default 64) <br>
`./amx --mtx <path> --mode csr`              CSR only <br>
`./amx --mtx <path> --mode bcsr`             BCSR variants only <br>
`./amx --mtx <path> --mode all`              all configs (default) <br>
`./amx --mtx <path> --cluster-thresh 0.1`    Jaccard threshold <br>
`./amx --mtx <path> --csv results.csv`       write results to CSV <br>