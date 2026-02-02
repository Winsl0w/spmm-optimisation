# Optimising Sparse Matrix Multiplication with M:N Sparsity on CPUs with Matrix/Vector Extensions
This project focuses on optimising sparse matrix multiplication on CPUs that feature advanced vector and matrix extentions, specifically targeting the M:N sparsity pattern. M:N sparsity, where only a subset of matrix elements are non-zero, is a technique commonly used to reduce the computational and memory demands of deep learning and high-performance computing applications. By exploiting CPU extentions such as ARM's Scalable Vector Extension (SVE) and Scalable Matrix Extension (SME), and Intel's Advanced Matrix Extensions (AMX) this project aims to develop and implement efficient algorithms that leverage hardware capabilities to accelerate sparse matrix operations. 

### Objectives:
- To develop optimised algorithms for sparse matrix multiplication with M:N sparsity, tailored to CPUs with vector/matrix extensions
- To evaluate the performance gains and efficiency improvements of these algorithms on various CPU architectures
