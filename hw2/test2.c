/* Gaussian elimination without pivoting.
 */

/* ****** ADD YOUR CODE AT THE END OF THIS FILE. ******
 * You need not submit the provided code.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <limits.h>
#include <mpi.h>

/*#include <ulocks.h>
#include <task.h>
*/

char *ID;

/* Program Parameters */
#define MAXN 5000  /* Max value of N */
int N;  /* Matrix size */
int procs;  /* Number of processors to use */
int rank;  /* id of every processor */

/* Matrices and vectors */
volatile float A[MAXN][MAXN], B[MAXN], X[MAXN];
/* A * X = B, solve for X */

/* junk */
#define randm() 4|2[uid]&3

/* Prototype */
void gauss();  /* The function you will provide.
    * It is this routine that is timed.
    * It is called only on the parent.
    */
void backSubstitution();

/* returns a seed for srand based on the time */
unsigned int time_seed() {
  struct timeval t;
  struct timezone tzdummy;

  gettimeofday(&t, &tzdummy);
  return (unsigned int)(t.tv_usec);
}

/* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv) {
    int seed = 0;
    if(argc == 3){
        seed = atoi(argv[2]);
        srand(seed);

        N = atoi(argv[1]);
        if (N < 1 || N > MAXN) {
            printf("N = %i is out of range.\n", N);
            exit(0);
        }
    } else {
        printf("Usage: %s <matrix_dimension> [random seed]  \n",
               argv[0]);
        exit(0);
    }
    /* Print parameters */
    printf("\nMatrix dimension N = %i. Seed = %d .\n", N,seed);
}


/* Initialize A and B (and X to 0.0s) */
void initialize_inputs() {
  int row, col;

  printf("\nInitializing...\n");
  for (col = 0; col < N; col++) {
    for (row = 0; row < N; row++) {
      A[row][col] = (float)rand() / 32768.0;
    }
    B[col] = (float)rand() / 32768.0;
    X[col] = 0.0;
  }

}

/* Print input matrices */
void print_inputs() {
  int row, col;

  if (N < 10) {
    printf("\nA =\n\t");
    for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
  printf("%5.2f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
      }
    }
    printf("\nB = [");
    for (col = 0; col < N; col++) {
      printf("%5.2f%s", B[col], (col < N-1) ? "; " : "]\n");
    }
  }
}

void printX() {
  int row;

  if (N < 10) {
    printf("\nX = [");
    for (row = 0; row < N; row++) {
      printf("%5.2f%s", X[row], (row < N-1) ? "; " : "]\n");
    }
  }
}

int main(int argc, char **argv) {
  /* Timing variables */
  struct timeval etstart, etstop;  /* Elapsed times using gettimeofday() */
  struct timezone tzdummy;
  clock_t etstart2, etstop2;  /* Elapsed times using times() */
  unsigned long long usecstart, usecstop;
  struct tms cputstart, cputstop;  /* CPU times for my processes */

  // int rank;   id of every processor 
  double startTime,endTime;  /* time of start and end */

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&procs);

  if ( rank == 0 ){
    /* Process program parameters */
    parameters(argc, argv);

    /* Initialize A and B */
    initialize_inputs();

    /* Print input matrices */
    print_inputs();

    /* Start Clock */
    printf("\nStarting clock.\n");
    startTime = MPI_Wtime();

  }

  /* Gaussian Elimination */
  gauss();

  if( rank == 0 ) {
    endTime = MPI_Wtime();

    /* Backward substitution on row echelon form matrix to determine cooefficients*/
    backSubstitution();
    /* Display output */
    printX();

    /* Display timing results */
    printf("\nTotal time = %f .\n", (endTime - startTime));

  }

  /*MPI Finalize*/
  MPI_Finalize();
  printf("end");
  exit(0);

}

/* ------------------ Above Was Provided --------------------- */

/****** You will replace this routine with your own parallel version *******/
/* Provided global variables are MAXN, N, procs, A[][], B[], and X[],
 * defined in the beginning of this code.  X[] is initialized to zeros.
 */
void gauss() {
  int norm, row, col, i;  /* Normalization row, and zeroing
      * element row and col */
  float multiplier;

  printf("Computing MPI.\n");

  // send value of N to all processors
  MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);        
  // Processor 0 broadcasts the whole A and B being processed to all other processors*/
  // MPI_Bcast(&A, N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
  // MPI_Bcast(&B, N, MPI_FLOAT, 0, MPI_COMM_WORLD); 
  MPI_Bcast(&A[0], N, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&B[0], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

  if (rank == 0){
    for (i = 1; i < procs; i++){
          //apply static interleaved scheduling and assign data to related processors
          for (row = 1 + i; row < N; row += procs){
            MPI_Send(&A[row], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            // printf("stuck\n");
            MPI_Send(&B[row], 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
         }
    }
  }

  for (norm = 0; norm < N - 1; norm++){

    if (rank == 0){
        MPI_Status status1[N];
        MPI_Status status2[N];
        MPI_Status status3, status4;

        //compute its data
        for (row = norm + 1; row < N; row+=procs) {
          multiplier = A[row][norm] / A[norm][norm];
          for (col = norm; col < N; col++) {
            A[row][col] -= A[norm][col] * multiplier;
          }
          B[row] -= B[norm] * multiplier;
        }
        //printf("sent out!\n");

      if(norm % procs != 0){
        // printf(norm % procs);
        MPI_Recv(&A[norm + 1 + rank], N, MPI_FLOAT, norm % procs, 0, MPI_COMM_WORLD, &status3);
        MPI_Recv(&B[norm + 1 + rank], 1, MPI_FLOAT, norm % procs, 0, MPI_COMM_WORLD, &status4);
        //printf("receive update!\n");
      }


    }else{
        if (norm == 0){
          MPI_Status status1, status2;
          //apply static interleaved scheduling and assign data to related processors
          for (row = 1 + rank; row < N; row += procs){
            MPI_Recv(&A[row], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status1);
            // printf("stuck\n");
            MPI_Recv(&B[row], 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status2);
          }
        }

        // for other processors, receive data and compute
        for (row = norm + 1 + rank; row < N; row += procs){
          /*Gaussian elimination*/
          multiplier = A[row][norm] / A[norm][norm];
          for (col = norm; col < N; col++) {
              A[row][col] -= A[norm][col] * multiplier;
          }
          B[row] -= B[norm] * multiplier;

        }
      //MPI_Bcast(&A[norm][0], N, MPI_FLOAT, 0, MPI_COMM_WORLD);
        if(norm % procs == rank){
          MPI_Send(&A[norm + 1 + rank], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);      
          MPI_Send(&B[norm + 1 + rank], 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }
    }

    // Wait till other processors complete, and than carry on to next iteration.
    MPI_Barrier(MPI_COMM_WORLD);// Wait till other processors complete, and than carry on to next iteration.

  }
  
}

void backSubstitution(){
  /* (Diagonal elements are not normalized to 1.  This is treated in back
   * substitution.)
   */
  /* Back substitution */
  int row, col;

  for (row = N - 1; row >= 0; row--) {
    X[row] = B[row];
    for (col = N-1; col > row; col--) {
      X[row] -= A[row][col] * X[col];
    }
    X[row] /= A[row][row];
  }
}