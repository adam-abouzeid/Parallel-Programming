#!/bin/bash

# Compile the programs
mpicc -o MandelbrotStatic MandelbrotStatic.c
mpicc -o MandelbrotDynamic MandelbrotDynamic.c

if [ -z "$1" ]
then
    echo "Please provide the number of processors as an argument."
    exit 1
fi
# Run the programs
mpirun -np $1 ./MandelbrotStatic
mpirun -np $1 ./MandelbrotDynamic



