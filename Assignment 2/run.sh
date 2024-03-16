#!/bin/bash
# Compiling the sequential program
gcc -o bucketsortseq bucketsortseq.c
echo "Compiled the sequential version."

# Running the OpenMP program
echo "Running the sequential version..."
./bucketsortseq
echo "Finished running the sequential version."

# Compiling the program with pthreads
gcc -o bucketsort_pthreads bucketsort_pthreads.c -pthread
echo "Compiled the pthreads version."

# Running the pthreads program
echo "Running the pthreads version..."
./bucketsort_pthreads
echo "Finished running the pthreads version."

# Compiling the program with OpenMP
gcc -o bucketsort_openmp bucketsort_openmp.c -fopenmp
echo "Compiled the OpenMP version."

# Running the OpenMP program
echo "Running the OpenMP version..."
./bucketsort_openmp
echo "Finished running the OpenMP version."

