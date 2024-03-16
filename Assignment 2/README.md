# Parallel Bucket Sort Implementations

This repository contains two implementations of the bucket sort algorithm: one utilizing Threads (pthreads) and the other using OpenMP. The aim is to showcase the efficiency improvements in the sorting algorithm through parallel processing techniques.

## Overview

The project demonstrates the effectiveness of parallel computing in enhancing the performance of the bucket sort algorithm. It includes a sequential version, a pthreads-based parallel version, and an OpenMP-based parallel version.

## Getting Started

These instructions will guide you on how to get the project up and running on your local machine for development and testing purposes.

### Prerequisites

Ensure you have the following installed:

- GCC compiler with pthread and OpenMP support
- A Linux/Unix environment to run the scripts and programs

### Installation

Clone the repository to your local machine using:

```bash
git clone https://github.com/adam-abouzeid/Parallel-Programming.git
cd Assignment/ 2
```

### Usage

The project includes a shell script (`run.sh`) to compile and run the programs. Execute the following commands:

```bash
chmod +x run.sh  # Grant execution permissions
./run.sh         # Compile and run the programs
```

The script compiles the sequential, pthread and OpenMP versions of the bucket sort algorithm and runs them, displaying the execution time.

