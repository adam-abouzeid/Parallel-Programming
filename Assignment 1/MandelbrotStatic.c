/*
@ Author Adam Abou Zeid
@ Date 2/17/2024
@ Description: This program generates a Mandelbrot set and saves it as a PGM file using a static task assignment to processors
@ Idea is to simply divide the image into equal parts and assign each part to a processor (strips)
*/


#include <stdio.h>
#include <time.h>
#include <mpi.h>
#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255

typedef struct complex{
  double real;
  double imag;
} Complex;
//function initialization
int cal_pixel(Complex c);
void save_pgm(const char *filename, int image[HEIGHT][WIDTH]);

int main(){
    double AVG=0;
    int N=10;

    int image[HEIGHT][WIDTH];

    double real_max = 2.0;
    double real_min=-2.0;
    double imag_max=2.0;
    double imag_min=-2.0;
    double scale_real = (real_max-real_min)/WIDTH;
    double scale_imag= (imag_max-imag_min)/HEIGHT;

    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int rows_chunk_size = HEIGHT / world_size; // this gives us how many rows each processor will handle (each row has a height of 1 obviously since 1 pixel)

    int start_row = rank * rows_chunk_size;
    int end_row = start_row + rows_chunk_size;

    int local_image[rows_chunk_size][WIDTH];
    Complex c;
    for(int k=0;k<N;k++){//N trials
    clock_t start_time  = clock();
         for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < WIDTH; j++) {
          //scaling
          c.real = real_min + ((float) j * scale_real);
          c.imag = imag_min + ((float)i * scale_imag);
          // calculate each slot in the local array in parallel on all processors (all local arrays)
          local_image[i-start_row][j] = cal_pixel(c); 
        }
    }
   
   
 
    MPI_Gather(local_image, rows_chunk_size * WIDTH, MPI_INT, image, rows_chunk_size * WIDTH, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        clock_t end_time = clock();
        double total_time = ((double)(end_time - start_time))/CLOCKS_PER_SEC;
        printf("Execution time of trial [%d]: %f seconds\n", k, total_time);
        AVG+=total_time;
    }
     }
     if(rank==0){
         printf("The average execution time of 10 trials is: %f ms\n", AVG/N*1000); //in ms x1000
         save_pgm("mandelbrotstatic.pgm", image);
     }
    
    MPI_Finalize();
    return 0;
}

int cal_pixel(Complex c) {
    
            // z0 = 0
            double z_real = 0;
            double z_imag = 0;

            // z1 = z0^2 + c
            // z2 = z1^2 + c
            // c is the complex coordinates of each pixel in the image
            double z_real2, z_imag2, lengthsq;

            int iter = 0; 
            
            do {
                z_real2 = z_real * z_real;
                z_imag2 = z_imag * z_imag;

                z_imag = 2 * z_real * z_imag + c.imag;
                z_real = z_real2 - z_imag2 + c.real;
                lengthsq =  z_real2 + z_imag2;
                iter++;
            }
            while ((iter < MAX_ITER) && (lengthsq < 4.0));
            // the number of iterations determines the color of the pixel
            return iter;

}


void save_pgm(const char *filename, int image[HEIGHT][WIDTH]) {
    FILE* pgmimg; 
    int temp;
    pgmimg = fopen(filename, "wb");

if (pgmimg == NULL) {
    perror("Error opening file");
    return;
} 
    fprintf(pgmimg, "P2\n"); // Writing Magic Number to the File   
    fprintf(pgmimg, "%d %d\n", WIDTH, HEIGHT);  // Writing Width and Height
    fprintf(pgmimg, "255\n");  // Writing the maximum gray value 
    int count = 0; 
    
    for (int i = 0; i < HEIGHT; i++) { 
        for (int j = 0; j < WIDTH; j++) { 
            temp = image[i][j]; 
            fprintf(pgmimg, "%d ", temp); // Writing the gray values in the 2D array to the file 
        } 
        fprintf(pgmimg, "\n"); 
    } 
    fclose(pgmimg); 
} 
