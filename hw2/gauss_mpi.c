/* stuff I might need to change often: */
#define MATRIX_SIZE 12
#define RAND_SEED 99
#define MIN_CHUNK_SIZE 3
#define ROOT 0 /* root proc should always be 0 for this assignment */

/* Gaussian elimination without pivoting - MPI implementation.
* WRITE DESCRIPTION HERE
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <limits.h>
#include <assert.h>

/* stuff for MPI */
#include <mpi.h>
MPI_Status status;

/* variables and functions I need */
int procID;
int input_nprocs;
void gauss();
void backSubstitute();

/* program variables / parameters */
#define MAXN 5000 /* Max value of N */
int N;  /* Matrix size */
int nprocs;  /* Number of processors to use */
char *ID;

/* matrices and vectors */
volatile float A[MATRIX_SIZE][MATRIX_SIZE], B[MATRIX_SIZE], X[MATRIX_SIZE];

/* initialize program parameters */
void parameters(int argc, char **argv) {
  srand(RAND_SEED); /* initialize random seed */
  N = MATRIX_SIZE; /* initialize N to matrix size */
  printf("Random seed = %i\n", RAND_SEED);
  printf("\nMatrix dimension N = %i.\n", N);
  printf("Number of processors = %i.\n", nprocs);
}

/* initialize A, B, and X to 0.0s */
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

void print_X() {
  int row;
  // if(N < 10) {
  printf("\nX = [");
  for(row = 0; row < N; row++) {
    printf("%5.2f%s", X[row], (row < N-1) ? "; " : "]\n");
  }
  // }
}


int main(int argc, char **argv) {

  double start_time, end_time; /* timing variables */

  MPI_Init(&argc, &argv); /* initialize MPI */
  MPI_Comm_rank(MPI_COMM_WORLD, &procID); /* identify proc # */
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs); /* get nprocs */

  if(procID == ROOT) {
    /* initial setup */
    parameters(argc, argv); /* process program parameters */
    initialize_inputs(); /* Initialize A, B, and X */
    print_inputs(); /* Print input matrices */
    start_time = MPI_Wtime(); /*start timer*/
    printf("\nstarting clock...\n");
  }

  // <<< 1 >>> send N, A, B to all other procs so everything's synced to start:
  MPI_Bcast(&N, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
  // printf("proc%d: N = %d\n", procID, N);
  // MPI_Barrier(MPI_COMM_WORLD);
  // MPI_Bcast(&A, N*N, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
  // int i;
  // for(i = 0; i < nprocs; ++i){
  //   if(procID == i) {
  //     printf("\nproc %d: A =\n\t", procID);
  //     int R, C;
  //     for(R = 0; R < N; R++) {
  //       for(C = 0; C < N; C++) {
  //         printf("%5.2f%s", A[R][C], (C < N-1) ? ", " : ";\n\t");
  //       }
  //     }
  //   }
  //   MPI_Barrier(MPI_COMM_WORLD);
  // }
  // MPI_Barrier(MPI_COMM_WORLD);
  // MPI_Bcast(&B, N, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
  // printf("\nproc %d: B =\n\t", procID);
  // int C;
  // for(C = 0; C < N; C++) {
  //   printf("%5.2f%s", B[C], (C < N-1) ? "; " : "]\n");
  // }
  MPI_Barrier(MPI_COMM_WORLD); /* broadcast is non-blocking, so sync once all data has been broadcasted */


  // TODO <<< 2 >>> calculate chunk size, send chunks to procs for calculation:
  int norm;
  for(norm = 0; norm < N - 1; norm++) {

    // // DEBUG #################
    // if(procID != ROOT) {
    //   int row, col;
    //   for (col = 0; col < N; col++) {
    //     for (row = 0; row < N; row++) {
    //       A[row][col] = 0.0;
    //     }
    //   }
    // }

    int chunk_size = (int)(N-norm-1)/nprocs;
    /* if chunk size is large enough, split the work among procs */
    if(chunk_size > MIN_CHUNK_SIZE) {
      /* first, broadcast normalization row to all procs */
      // float allones[N];
      // int i;
      // for(i = 0; i < N; ++i) {
      //   A[norm][i] = 1.0;
      // }
      // MPI_Bcast(&(allones), N, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
      MPI_Bcast((void *)&A[norm][0], N, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
      MPI_Bcast((void *)&B[norm],    1, MPI_FLOAT, ROOT, MPI_COMM_WORLD);

      if(procID == ROOT) {

        /* send appropriate portion of arrays to other procs from 0 */
        int p;
        for(p = 1; p < nprocs; ++p) {
          int send_start = (p-1)*chunk_size + norm + 1;
          printf("sending A starting at %d and length %d\n", send_start, N*chunk_size);
          fflush(stdout); // Will now print everything in the stdout buffer
          MPI_Send((void *)&A[send_start][0], N*chunk_size, MPI_FLOAT, p, 99*p, MPI_COMM_WORLD);
          MPI_Send((void *)&B[send_start],      chunk_size, MPI_FLOAT, p, 999*p, MPI_COMM_WORLD);
        }

        /* calculate "lefttover" matrix rows with root */
        int start_index = (nprocs-1) * chunk_size + norm + 1;
        int end_index = N;
        gauss(norm, start_index, end_index);

        /* resync - receive data back from other procs */
        for(p = 1; p < nprocs; ++p){
          int start_index = (p-1)*chunk_size + norm + 1;
          MPI_Recv((void *)&A[start_index][0], N*chunk_size, MPI_FLOAT, p, 88*p, MPI_COMM_WORLD, &status);
          MPI_Recv((void *)&B[start_index],      chunk_size, MPI_FLOAT, p, 888*p, MPI_COMM_WORLD, &status);
        }

      }

      /* else receive appropriate portion of arrays from root */
      else {
        int start_index = (procID-1)*chunk_size + norm + 1;
        int end_index = start_index + chunk_size;
        MPI_Recv((void *)&A[start_index][0], N*chunk_size, MPI_FLOAT, ROOT, 99*procID, MPI_COMM_WORLD, &status);
        MPI_Recv((void *)&B[start_index],      chunk_size, MPI_FLOAT, ROOT, 999*procID, MPI_COMM_WORLD, &status);

        /* calculate appropriate portion of matrix as allocated from root */
        gauss(norm, start_index, end_index);

        /* send data back to root proc for syncing */
        MPI_Send((void *)&A[start_index][0], N*chunk_size, MPI_FLOAT, ROOT, 88*procID, MPI_COMM_WORLD);
        MPI_Send((void *)&B[start_index],      chunk_size, MPI_FLOAT, ROOT, 888*procID, MPI_COMM_WORLD);

      }

      // DEBUG:
      MPI_Barrier(MPI_COMM_WORLD);
      int i;
      for(i = 0; i < nprocs; ++i) {
        if(procID == i) {
          printf("\nproc %d: A =\n\t", procID);
          int R, C;
          for(R = 0; R < N; R++) {
            for(C = 0; C < N; C++) {
              printf("%5.2f%s", A[R][C], (C < N-1) ? ", " : ";\n\t");
            }
          }
        }
        fflush(stdout); // Will now print everything in the stdout buffer
        MPI_Barrier(MPI_COMM_WORLD);
      }
    }
    else {
      /* if chunk size is too small, just have root do the rest */
      if(procID == ROOT) {
        printf("chunk size small, root will take it from here\n");
        gauss(norm, norm+1, N);
      }
    }

    // DEBUG ######################################
    // if(norm > 5) { break; }
  } /* END OUTERMOST FOR LOOP (norm) */



  // int rowbreaks[nprocs]; // all members implicitly initialized to zero
  // for(int norm = 0; norm < N - 1; norm++) {
  //   // printf("\nnorm: %d\r\n", norm);
  //   int rowidx = norm+1;
  //   rowbreaks[0] = rowidx; // first member will always be norm+1
  //   for(int k = 1; k < nprocs; ++k) {
  //     // determine desired chunk size
  //     int desired = (int)(pow(N-norm-1, 2)/(2*nprocs));
  //     if(desired <= MIN_CHUNK_SIZE) {
  //       // if desired chunk size is too small, skip sending any messages and just have root calculate the rest
  //       // printf("exiting, chunk size is too small\n");
  //       break;
  //     }
  //     // else (if large enough), send chunks out for processing
  //     else {
  //       int chunk = 0;
  //       while(chunk < desired) {
  //         chunk += N-rowidx;
  //         // printf("it: %d\trow: %d\tchunk: %d [%d]\r\n", k, rowidx, chunk, desired);
  //         rowidx++;
  //         if(rowidx >= N) {
  //           printf("ERROR: CALCULATING CHUNKS ENDED IN OVERRUN CONDITION\r\n");
  //           assert(0);
  //         }
  //       }
  //       rowbreaks[k] = rowidx;
  //       // printf("rowbreak %d is %d with area %d [%d]\r\n", k, rowbreaks[k], chunk, desired);
  //     }
  //     // printf("ROW INDICES:\r\n");
  //     // for(int q = 0; q < nprocs; ++q) {
  //     //   printf("%d %d\n", q, rowbreaks[q]);
  //     // }
  //   } /* END CHUNKING OPERATIONS */
  //
  //   // DO SENDING HERE
  //   // THEN CALCULATE root's PORTION HERE
  //
  // } /* END 1st FOR LOOP */

  // TODO <<< 3 >>> do own portion of the calculations:
  // gauss();

  // TODO <<< 4 >>> do own portion of the calculations:

  // }
  // else {
  //   // wait for incoming message, do calcs once it's here
  // }

  // if(procID == ROOT)
  //   backSubstitute();

  // JUST PRINTING OUT MY NEWLY GAUSSIAN ELIMINATED MATRIX
  // printf("\nANSWER =\n\t");
  // for(int R = 0; R < N; R++) {
  //   for(int C = 0; C < N; C++) {
  //     printf("%5.2f%s", A[R][C], (C < N-1) ? ", " : ";\n\t");
  //   }
  // }


  if(procID == ROOT) {

    /* do back substitution and finish */
    backSubstitute();

    /* Stop Clock */
    end_time = MPI_Wtime(); /*stop timer*/

    /* print out final A */
    printf("\nFINAL A:\n\t");
    int R, C;
    for(R = 0; R < N; R++) {
      for(C = 0; C < N; C++) {
        printf("%5.2f%s", A[R][C], (C < N-1) ? ", " : ";\n\t");
      }
    }

    /* Display output */
    print_X();

    /* Display timing results */
    printf("DONE\n");
    printf("Elapsed time: %f seconds\n", end_time - start_time);
  }

  MPI_Finalize();
  return 0;
}


/* Gaussian elimination */
void gauss(int norm, int start_index, int end_index) {
  float multiplier;
  int row, col;
  for (row = start_index; row < end_index; row++) {
    multiplier = A[row][norm] / A[norm][norm];
    for (col = norm; col < N; col++) {
      A[row][col] -= A[norm][col] * multiplier;
    }
    B[row] -= B[norm] * multiplier;
  }
} /* END OF FUNCTION gauss() */


void backSubstitute() {
  /* back substitution - no parallelization needed per instructions */
  int row, col;
  for(row = N-1; row >= 0; row--) {
    X[row] = B[row];
    for(col = N-1; col > row; col--)
    { X[row] -= A[row][col] * X[col]; }
    X[row] /= A[row][row];
  }
} /* END OF FUNCTION backSubstitute() */
