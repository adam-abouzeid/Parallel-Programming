#include <stdio.h>
#include <mpi.h>
#include <time.h>
#define WIDTH 640
#define HEIGHT 480
#define DATA_TAG 0
#define TERMINATOR_TAG 1
#define RESULT_TAG 3

#define MAX_ITER 255

typedef struct complex{
  double real;
  double imag;
} Complex;
//function initialization
int cal_pixel(Complex c);
void save_pgm(const char *filename, int image[HEIGHT][WIDTH]);

int main(int argc, char const *argv[])
{
    double AVG = 0;
    int N = 10;
    MPI_Init(NULL, NULL);
    int world_size; //number of processors
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int image[HEIGHT][WIDTH];
    double real_max = 2.0;
    double real_min=-2.0;
    double imag_max=2.0;
    double imag_min=-2.0;
    double scale_real = (real_max-real_min)/WIDTH;
    double scale_imag= (imag_max-imag_min)/HEIGHT;
    for(int n=0;n<N;n++){
        clock_t start = clock();
    if(rank==0){ // master
        int count =0;
        int row=0;
        for(int k=1;k<world_size;k++){//send initial row to each process
            MPI_Send(&row, 1, MPI_INT, k, DATA_TAG, MPI_COMM_WORLD);
            count++;
            row++;

        }
        do{
            int recv_row[WIDTH];
            MPI_Status status;
           MPI_Recv(recv_row, WIDTH, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        // memcpy(image[row-1], recv_row, sizeof(recv_row));
        int sender_rank = status.MPI_SOURCE;
            count--;
            if(row<HEIGHT){
                MPI_Send(&row, 1, MPI_INT,sender_rank, DATA_TAG, MPI_COMM_WORLD);
                count++;
                row++;
            }else{
                MPI_Send(&row, 1, MPI_INT, sender_rank, TERMINATOR_TAG, MPI_COMM_WORLD);
            }

            int row_worked_on = status.MPI_TAG;
            //copy array values into the image 2d array
            for(int j=0;j<WIDTH;j++){
                image[row_worked_on][j] = recv_row[j];
            }
        } while(count>0);
        clock_t end = clock();
        double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        printf("The execution time of trial %d is: %f ms\n", n, time_spent*1000); //in ms x1000
        AVG += time_spent;

       
    }else{
        int color[WIDTH];
        int row;
        MPI_Status status;
        int tag;
        MPI_Recv(&row, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        tag = status.MPI_TAG;
        Complex c;
        while(tag==DATA_TAG){
            c.imag = imag_min + ((float)row * scale_imag);

            for(int j=0;j<WIDTH;j++){
                c.real = real_min + ((float)j * scale_real);
                color[j] = cal_pixel(c);
            }
            MPI_Send(color, WIDTH, MPI_INT, 0, row, MPI_COMM_WORLD);
            MPI_Recv(&row, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            tag = status.MPI_TAG;
        }
}   }
    if(rank==0){
            printf("The average execution time of 10 trials is: %f ms\n", AVG/N*1000); //in ms x1000
    save_pgm("mandelbrotdynamic.pgm", image);
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
