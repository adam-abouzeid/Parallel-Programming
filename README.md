# README.md

## Prerequisites

These programs are written in C and use the MPI library for parallel processing. To compile and run these programs, you need to have an MPI implementation like OpenMPI or MPICH installed on your system. You can install these using your system's package manager. For example, on Ubuntu, you can use the following commands:

```bash
sudo apt update
sudo apt install mpich
```

## Quick Start

For a quick start, you can use the provided `run.sh` script to compile and run the programs. The script takes one argument, which is the number of processors to use. Here's how you can use it:

1. Give the script execute permissions:

```bash
chmod +x run.sh
```

2. Run the script with the number of processors as an argument:

```bash
./run.sh 4
```

Replace `4` with the number of processors you want to use.

## Manual Compilation and Execution

If you want to manually compile and run the programs, you can follow these steps:

### Compiling the Programs

To compile the programs, you can use the `mpicc` command which is a wrapper around the gcc compiler but also links the MPI libraries. The general format of the command is:

```bash
mpicc -o output_file source_file.c
```

For the specific programs provided, you can use the following commands:

```bash
mpicc -o MandelbrotStatic MandelbrotStatic.c
mpicc -o MandelbrotDynamic MandelbrotDynamic.c
```

### Running the Programs

To run the compiled programs, you can use the `mpirun` command. The general format of the command is:

```bash
mpirun -np num_processes ./output_file
```

For the specific programs provided, you can use the following commands:

```bash
mpirun -np 4 ./MandelbrotStatic
mpirun -np 4 ./MandelbrotDynamic
```

Replace `4` with the number of processes you want to use.
