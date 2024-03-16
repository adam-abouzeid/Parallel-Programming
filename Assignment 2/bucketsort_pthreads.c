
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#define NUM_THREADS 4 //since I have 4 cores
#define NUM_BUCKETS 10
#define NUM_ELEMENTS 1000

//function  declarations
void merge(int arr[], int left, int mid, int right);
void merge_sort(int arr[], int left, int right);
void merge_buckets(int buckets[][NUM_ELEMENTS], int arr[]);
void bucket_sort(int arr[]);
void * runner(void * args);

//since we cannot pass more than 1 argument to the pthread_Create so we pass a struct with data 
typedef struct ThreadArgs{
    int (*arr)[NUM_ELEMENTS];
    int left;
    int right;
} thread_args;


int main() {

    double time_taken;
    double AVG;
    int trials =10;
    clock_t start, end;
    //repeat for 10 trials and take average
    srand(time(NULL));
    for(int trial=0;trial<trials;trial++){
        
        int numbers_array[NUM_ELEMENTS];
        
        start = clock();
        // Seed the random number generator
        
        for (int i = 0; i < NUM_ELEMENTS; i++) {
            numbers_array[i] = rand() % 100; //between 0 and 99
        }

        bucket_sort(numbers_array);
       
        end = clock();

        time_taken = ((double) (end - start)) / CLOCKS_PER_SEC;
        AVG+=time_taken;
    }
    printf("Average Time taken over 10 trials: %f seconds\n", AVG/trials);

    return 0;
}

void merge(int arr[], int left, int mid, int right) {
    //code from cp3 class
    int l1 = mid - left + 1;
    int l2 = right - mid;
    int L[l1], R[l2];

    for (int i = 0; i < l1; i++)
        L[i] = arr[left + i];
    for (int j = 0; j < l2; j++)
        R[j] = arr[mid + 1 + j];

    int i = 0;
    int j = 0;
    int k = left;
    while (i < l1 && j < l2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < l1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < l2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void merge_sort(int arr[], int left, int right) {
    //code form cp3 class
    if (left < right) {
        int mid = left + (right - left) / 2;

        merge_sort(arr, left, mid);
        merge_sort(arr, mid + 1, right);

        merge(arr, left, mid, right);
    }
}

void merge_buckets(int buckets[][NUM_ELEMENTS], int arr[]) {
    
    int id = 0;
    for (int i = 0; i < NUM_BUCKETS; i++) {
        for (int j = 0; j < NUM_ELEMENTS; j++) {
            if (buckets[i][j] != -1) {
                arr[id++] = buckets[i][j];
            }
        }
    }
}

void bucket_sort(int arr[]) {
    
    int buckets[NUM_BUCKETS][NUM_ELEMENTS];
    pthread_t threads[NUM_THREADS];
    thread_args t_args[NUM_THREADS];
    int buckets_for_thread = NUM_BUCKETS / NUM_THREADS;
    // Bucket Initializing
    for (int i = 0; i < NUM_BUCKETS; i++) {
        for (int j = 0; j < NUM_ELEMENTS; j++) {
            buckets[i][j] = -1; //smallest number (-1) outside the range of generated numbers [0-99]
        }
    }

   
       for (int i = 0; i < NUM_ELEMENTS; i++) {
        int element = arr[i];
        int bucket_id = element / (100 / NUM_BUCKETS); // Get the correct bucket

        
        for (int j = 0; j < NUM_ELEMENTS; j++) {
            if (buckets[bucket_id][j] == -1) {
                buckets[bucket_id][j] = element;
                break; // break after inserting the element
            }
        }
    }

    for(int i=0;i<NUM_THREADS;i++){
        t_args[i].arr = buckets;
        t_args[i].left = i * buckets_for_thread;
        t_args[i].right = (i + 1) * buckets_for_thread;

        //for last thread, handle the remaining buckets
        if (i == NUM_THREADS - 1) {
            t_args[i].right = NUM_BUCKETS;
        }
        pthread_create(&threads[i], NULL, runner,(void*) &t_args[i]);
    }
     for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL); // wait for all threads to finish
    }
    // for (int i = 0; i < NUM_BUCKETS; i++) { // non empty bcukets are sorted
    //     if (buckets[i][0] != -1) { // check  bucket is not empty
    //         merge_sort(buckets[i], 0, NUM_ELEMENTS - 1);
    //     }
    // }

    // Merge sorted buckets into the original array
    merge_buckets(buckets, arr);
}


void * runner(void * args){
    thread_args * t_args = (thread_args *) args;
    //multiple buckets may be handled by 1 thread
    for (int i = t_args->left; i < t_args->right; i++) {
        if (t_args->arr[i][0] != -1) {
            //only sort non-empty buckets
            merge_sort(t_args->arr[i], 0, NUM_ELEMENTS - 1);
        }
    }
    pthread_exit(0);
}