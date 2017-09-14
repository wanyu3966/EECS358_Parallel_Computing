#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <limits.h>
#include <mpi.h>

#define MAXN 10  /* Max value of N */
int N = 9;  /* Matrix size */
int procs;  /* Number of processors to use */

/* Matrices and vectors */
float A[MAXN][MAXN], B[MAXN], X[MAXN];

void initialize_inputs() {
  int row, col;

  printf("\nInitializing...\n");
  for (col = 0; col < N; col++) {
    for (row = 0; row < N; row++) {
      A[row][col] = (float)rand() / 1132768.0;
    }
    B[col] = (float)rand() / 1132768.0;
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

  if (N < 10) {
    printf("\nX = [");
    for (row = 0; row < N; row++) {
      printf("%5.2f%s", X[row], (row < N-1) ? "; " : "]\n");
    }
  }
}


int main(int argc, char **argv) {
  int pid;
  int nprocs;
  int norm, row, col;  /* Normalization row, and zeroing
                        * element row and col */
  float multiplier=0;
  int k;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if(pid == 0){
  	  	/* Initialize A and B */
      	initialize_inputs();

		/* Print input matrices */
		print_inputs();
		printf("matrix initialized.\n");
  }

  MPI_Bcast(&A, MAXN * MAXN, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(B, MAXN, MPI_FLOAT, 0, MPI_COMM_WORLD);



  for(norm=0; norm<N; norm++){
   	MPI_Bcast(&A[norm][norm], MAXN - norm, MPI_FLOAT, norm % nprocs, MPI_COMM_WORLD);
        MPI_Bcast(&B[norm], 1, MPI_FLOAT, norm % nprocs, MPI_COMM_WORLD);
	for(row=norm+1; row<N; row++){
		if(row % nprocs == pid){
		    multiplier = A[row][norm]/A[norm][norm];
			for(col=norm; col<N; col++){
				A[row][col] -= multiplier * A[norm][col];
			}
			B[row] -= multiplier * B[norm];
	        }
	}
	

  }  
  
  if(pid == 0){
  /* Back substitution */
 	 for (row = N - 1; row >= 0; row--) {
    		X[row] = B[row];
    		for (col = N-1; col > row; col--) {
      			X[row] -= A[row][col] * X[col];
    		}
    		X[row] /= A[row][row];
 	 }
 
  /* Display output */
  printf("displaying X");
  print_X();
  }
  MPI_Finalize();
  return 0; 
}

