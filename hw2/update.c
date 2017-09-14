/*
  Usage: ./gauss SIZE SEED

  Algorithm: The program basically runs in following steps:

  1) Processor 0 sends rows to other processors(One to all scatter). Rows are distributed in terms of interleaved scheduling, which means processor 1 will receive 
  Row[1], Row[1+numprocs], Row[1+2*numprocs]... After that, each processor has the rows that it will use in the future. (Distribution Process.)
  
  2) Through each 'norm' loop, the processor that hold the 'norm' row, which means (myid == norm%numprocs), 
  will be responsible for broadcasting Row[norm] to all other processors. Meanwhile, this processor will also compute the rows in its range.
  
  3) Upon receiving the norm row, other processors will begin the computation process. 

  4) A barrier here is kept until all processors complete their work, then next iteration will continue.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <limits.h>
#include <pthread.h>
#include "mpi.h"
/*#include <ulocks.h>
#include <task.h>
*/

char *ID;

/* Program Parameters */
#define MAXN 5000  /* Max value of N */
#define SendTag0 0
#define SendTag1 1
#define SendTag2 2
#define SendTag3 3
#define SendBackTag 4
int N;  /* Matrix size */
int numprocs=1;  /* Number of processors to use */
int myid;
int count=0;
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

/* returns a seed for srand based on the time */
unsigned int time_seed() {
  struct timeval t;
  struct timezone tzdummy;

  gettimeofday(&t, &tzdummy);
  return (unsigned int)(t.tv_usec);
}

/* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv) {
  int submit = 0;  /* = 1 if submission parameters should be used */
  int seed = 0;  /* Random seed */
    if (argc == 3) {
      seed = atoi(argv[2]);
      srand(seed);
       N = atoi(argv[1]);
    if (N < 1 || N > MAXN) {
      printf("N = %i is out of range.\n", N);
      exit(0);
    }
      printf("Random seed = %i\n", seed);
    }
    else {
      printf("Usage: %s <matrix_dimension> <num_procs> [random seed]\n",
	     argv[0]);
      printf("       %s submit\n", argv[0]);
      exit(0);
    }
  
    //  }
  /* Interpret command-line args */   
    // numprocs = atoi(argv[2]);
    // if (numprocs < 1) {
    //   printf("Warning: Invalid number of processors = %i.  Using 1.\n", numprocs);
    //   numprocs = 1;
    // }

  /* Print parameters */
  printf("\nMatrix dimension N = %i.\n", N);
  printf("Number of processors = %i.\n", numprocs);
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
  /* Timing variables */
  // struct timeval etstart, etstop;  /* Elapsed times using gettimeofday() */
  // struct timezone tzdummy;
  // clock_t etstart2, etstop2;   Elapsed times using times() 
  // unsigned long long usecstart, usecstop;
  // struct tms cputstart, cputstop;  /* CPU times for my processes */

  //MPI varaibles
  double startwtime = 0.0, endwtime;
  int namelen;
  char processor_name[MPI_MAX_PROCESSOR_NAME];


  //MPI Initialize
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  MPI_Get_processor_name(processor_name,&namelen);

  // ID = argv[argc-1];
  // argc--;
  if(myid==0){
  /* Process program parameters */
    // printf("myid\n");
  parameters(argc, argv);

  /* Initialize A and B */
  initialize_inputs();

  /* Print input matrices */
  print_inputs();
  startwtime=MPI_Wtime();
}
  /* Start Clock */
  // printf("\nStarting clock.\n");
  // gettimeofday(&etstart, &tzdummy);
  // etstart2 = times(&cputstart);

  /* Gaussian Elimination */
  gauss();

int norm, row=0, col;
  /* Back substitution */
  for (row = N - 1; row >= 0; row--) {
    X[row] = B[row];
    for (col = N-1; col > row; col--) {
      X[row] -= A[row][col] * X[col];
    }
    X[row] /= A[row][row];
  }
  /* Stop Clock */
  // gettimeofday(&etstop, &tzdummy);
  // etstop2 = times(&cputstop);
  // printf("Stopped clock.\n");
  // usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
  // usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

  /* Display output */
  

   // Display timing results 
  // printf("\nElapsed time = %g ms.\n",
	 // (float)(usecstop - usecstart)/(float)1000);

  if(myid==0){
    // print_inputs();
    print_X();
    endwtime=MPI_Wtime();
    printf("time =%f\n", endwtime-startwtime);
  }
  MPI_Finalize();
  return 0;

}

/* ------------------ Above Was Provided --------------------- */

/****** You will replace this routine with your own parallel version *******/
/* Provided global variables are MAXN, N, procs, A[][], B[], and X[],
 * defined in the beginning of this code.  X[] is initialized to zeros.
 */
void gauss() {
  int norm, row=0, col;  /* Normalization row, and zeroing
			* element row and col */
  float multiplier;
  int i;
  MPI_Bcast(&N,1,MPI_INT,0,MPI_COMM_WORLD);
  /* Gaussian elimination */
  for (norm = 0; norm < N ; norm++) {
    //sending rows
    if(norm==0){
      if(myid==0){
          for(i=1;i<numprocs;i++){
            for(row=i;row<N;row+=numprocs){
            MPI_Send(&A[row],N,MPI_FLOAT,i,SendTag0,MPI_COMM_WORLD);
            MPI_Send(&B[row],1,MPI_FLOAT,i,SendTag1,MPI_COMM_WORLD);
          }
        }
      }
      else{
        for(row=myid;row<N;row+=numprocs){
          MPI_Status Areceiver,Breceiver;
          MPI_Recv(&A[row],N,MPI_FLOAT,0,SendTag0,MPI_COMM_WORLD,&Areceiver);      
          // printf("id%dvalue%f\n",myid,A[row][0] );
          MPI_Recv(&B[row],1,MPI_FLOAT,0,SendTag1,MPI_COMM_WORLD,&Breceiver);
        }
      }
    }

    //calculating
    if(myid==norm%numprocs){
      for(i=0;i<numprocs;i++){
      if(i!=myid){
      // printf("myid%drow%dcount%d\n", myid,row,count); 
      MPI_Send(&A[norm],N,MPI_FLOAT,i,SendTag2,MPI_COMM_WORLD);
      MPI_Send(&B[norm],1,MPI_FLOAT,i,SendTag3,MPI_COMM_WORLD);
      }
      }
      for(row=norm+numprocs;row<N;row+=numprocs){
        // printf("row:%dnorm:%drownorm%fnormnorm%f\n",row,norm,A[row][norm],A[norm][norm] );
          multiplier=A[row][norm]/A[norm][norm];
          for(col=norm;col<N;col++){
            A[row][col]-=A[norm][col]*multiplier;
          }
          B[row]-=B[norm]*multiplier;
        }
    }
    else{
        MPI_Status ReceiveAStatus,ReceiveBStatus;
        // printf("%d\n", count++);
        MPI_Recv(&A[norm],N,MPI_FLOAT,norm%numprocs,SendTag2,MPI_COMM_WORLD,&ReceiveAStatus);
        MPI_Recv(&B[norm],1,MPI_FLOAT,norm%numprocs,SendTag3,MPI_COMM_WORLD,&ReceiveBStatus);
        for(row=myid;row<N;row+=numprocs){
          if(row>norm){
          // printf("myid:%drow:%dnorm:%drownorm%fnormnorm%f\n",myid,row,norm,A[row][norm],A[norm][norm] );
          multiplier=A[row][norm]/A[norm][norm];
          // printf("row%dnorm%dvalue%f\n",row,norm,A[row][norm] );
          for(col=norm;col<N;col++){
            A[row][col]-=A[norm][col]*multiplier;
          }
          B[row]-=B[norm]*multiplier;
        }
      }
    }

      MPI_Barrier(MPI_COMM_WORLD);
    }



}
 