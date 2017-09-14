/* Algorithm: 
    Our basic idea is based on Slide 5.9 and pi2.c. The algorithm runs based on the outer row - norm, and the inner two loops run parallelly. Through each outer 
    iteration, each row below the base row(norm) will modify the value by a multiplier = A[row][norm]/A[norm][norm]. Since each row is independent, the threads 
    could run parallelly. 

    In controlling the running size of each iteration of threads, we use a dynamically changed variable, 'chunk', to modify the real-time size of threads. The 
    varaible is computed by chunk = (N - global_i + 1) / (2 * procs) + 1, same as Slide 5.9. We use a global lock global_i to control the position of rows, 
    as well as the chunk size. 

*/

/* This is used for part(a). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <limits.h>
#include <pthread.h>


/*#include <ulocks.h>
 * #include <task.h>
 */

char *ID;

/* Program Parameters */
#define MAXN 5000       /* Max value of N */
int N;              /* Matrix size */
int procs;          /* Number of processors to use */


int chunk = 4;          /* Used for dynamic scheduling */


/* Matrices and vectors */
volatile float A[MAXN][MAXN], B[MAXN], X[MAXN];
/* A * X = B, solve for X */

/* junk */
#define randm() 4 | 2[uid] & 3

pthread_mutex_t global_i_lock;

void BackSubstitution();


int global_i  = 0; /*global index*/
int global_norm = 0;
int min( int a, int b );


int max( int a, int b );


void *calculator( void *param );


/* Prototype */
void gauss();  /* The function you will provide.
                * It is this routine that is timed.
                * It is called only on the parent.
                */


int min( int a, int b )
{
  if ( a < b )
    return(a);
  else
    return(b);
}


int max( int a, int b )
{
  if ( a < b )
    return(b);
  else
    return(a);
}


struct ParamToSend
{
  int ThreadID;
  int Row; /* The row sent to threads. */
};

/* returns a seed for srand based on the time */
unsigned int time_seed()
{
  struct timeval  t;
  struct timezone tzdummy;

  gettimeofday( &t, &tzdummy );
  return( (unsigned int) (t.tv_usec) );
}


/* Set the program parameters from the command-line arguments */
void parameters( int argc, char **argv )
{
  int submit  = 0;            /* = 1 if submission parameters should be used */
  int seed  = 0;            /* Random seed */
  char  uid[L_cuserid + 2];     /*User name */

  /* Read command-line arguments */
  /*  if (argc != 3) { */
  if ( argc == 1 && !strcmp( argv[1], "submit" ) )
  {
    /* Use submission parameters */
    submit  = 1;
    N = 4;
    procs = 2;
    printf( "\nSubmission run for \"%s\".\n", cuserid( uid ) );
    /*uid = ID;*/
    strcpy( uid, ID );
    srand( randm() );
  }else {
    if ( argc == 3 )
    {
      seed = atoi( argv[3] );
      srand( seed );
      printf( "Random seed = %i\n", seed );
    }else {
      printf( "Usage: %s <matrix_dimension> <num_procs> [random seed]\n",
        argv[0] );
      printf( "       %s submit\n", argv[0] );
      exit( 0 );
    }
  }
  /*  } */
  /* Interpret command-line args */
  if ( !submit )
  {
    N = atoi( argv[1] ); /* matrix size */
    if ( N < 1 || N > MAXN )
    {
      printf( "N = %i is out of range.\n", N );
      exit( 0 );
    }
    procs = atoi( argv[2] );
    if ( procs < 1 )
    {
      printf( "Warning: Invalid number of processors = %i.  Using 1.\n", procs );
      procs = 1;
    }
  }

  /* Print parameters */
  printf( "\nMatrix dimension N = %i.\n", N );
  printf( "Number of processors = %i.\n", procs );
}


/* Initialize A and B (and X to 0.0s) */
void initialize_inputs()
{
  int row, col;

  printf( "\nInitializing...\n" );
  for ( col = 0; col < N; col++ )
  {
    for ( row = 0; row < N; row++ )
    {
      A[row][col] = (float) rand() / 32768.0;
    }
    B[col]  = (float) rand() / 32768.0;
    X[col]  = 0.0;
  }
}


/* Print input matrices */
void print_inputs()
{
  int row, col;

  if ( N < 10 )
  {
    printf( "\nA =\n\t" );
    for ( row = 0; row < N; row++ )
    {
      for ( col = 0; col < N; col++ )
      {
        printf( "%5.2f%s", A[row][col], (col < N - 1) ? ", " : ";\n\t" );
      }
    }
    printf( "\nB = [" );
    for ( col = 0; col < N; col++ )
    {
      printf( "%5.2f%s", B[col], (col < N - 1) ? "; " : "]\n" );
    }
  }
}


void print_X()
{
  int row;

  if ( N < 10 )
  {
    printf( "\nX = [" );
    for ( row = 0; row < N; row++ )
    {
      printf( "%5.2f%s", X[row], (row < N - 1) ? "; " : "]\n" );
    }
  }
}


int main( int argc, char **argv )
{
  /* Timing variables */
  struct timeval    etstart, etstop;        /* Elapsed times using gettimeofday() */
  struct timezone   tzdummy;
  clock_t     etstart2, etstop2;      /* Elapsed times using times() */
  unsigned long long  usecstart, usecstop;
  struct tms    cputstart, cputstop;    /* CPU times for my processes */

  pthread_mutex_init( &global_i_lock, NULL );
  ID = argv[argc - 1];
  argc--;

  /* Process program parameters */
  parameters( argc, argv );

  /* Initialize A and B */
  initialize_inputs();

  /* Print input matrices */
  print_inputs();

  /* Start Clock */
  printf( "\nStarting clock.\n" );
  gettimeofday( &etstart, &tzdummy );
  etstart2 = times( &cputstart );

  /* Gaussian Elimination */
  gauss();

  /* Stop Clock */
  gettimeofday( &etstop, &tzdummy );
  etstop2 = times( &cputstop );
  printf( "Stopped clock.\n" );
  usecstart = (unsigned long long) etstart.tv_sec * 1000000 + etstart.tv_usec;
  usecstop  = (unsigned long long) etstop.tv_sec * 1000000 + etstop.tv_usec;

  /* Display output */
  print_X();

  /* Display timing results */
  printf( "\nElapsed time = %g ms.\n",
    (float) (usecstop - usecstart) / (float) 1000 );
}


/* Rewrite */
void gauss()
{
  /* norm for elements position */
  int   row, col;
  long    norm;
  float   multiplier;
  pthread_t pthread[procs];
  /* struct ParamToSend *paramtosend=(struct ParamToSend *)malloc(procs*sizeof(struct ParamToSend)); */
  /* ParamToSend paramtosend[procs]; */
  /* Thread ID */
  int i;

  for ( norm = 0; norm < N - 1; norm++ )
  {
    int t = 0;
    global_i = norm + 1;
    for ( i = 0; i < procs; i++ )
    {
      pthread_create( &pthread[i], NULL, &calculator, (void *) norm );
    }
    for ( i = 0; i < procs; i++ )
    {
      pthread_join( pthread[i], NULL );
    }
  }


  /* Back substitution */
  for ( row = N - 1; row >= 0; row-- )
  {
    X[row] = B[row];
    for ( col = N - 1; col > row; col-- )
    {
      X[row] -= A[row][col] * X[col];
    }
    X[row] /= A[row][row];
  }
}


void *calculator( void *norm_p )
{
  int row, col;
  long  norm = (long) norm_p;
  float multiplier;
  int local_i = 0;

  while ( local_i < N )
  {
    pthread_mutex_lock( &global_i_lock );
    chunk = (N - global_i + 1) / (2 * procs) + 1;
    local_i   = global_i;
    global_i  += chunk;
    pthread_mutex_unlock( &global_i_lock );
    for ( row = local_i; row < min( local_i + chunk, N ); row++ )
    {
      multiplier = A[row][norm] / A[norm][norm];
      for ( col = norm; col < N; col++ )
      {
        A[row][col] -= A[norm][col] * multiplier;
      }
      B[row] -= B[norm] * multiplier;
    }
  }
}


/*
 * void *calculator( void *param )
 * {
 *   // ParamToSend *paramtosend = (ParamToSend *) param;
 *   ParamToSend *paramtosend = (ParamToSend *) param;
 *   int   norm, row, col;
 *   float   multiplier;
 *   norm = paramtosend->Row;
 *   int index = paramtosend->ThreadID;
 *   for ( row = norm + index; row < N; row+=procs )
 *   {
 *     multiplier = A[row][norm] / A[norm][norm];
 *     for ( col = norm; col < N; col++ )
 *     {
 *       A[row][col] -= A[norm][col] * multiplier;
 *     }
 *     B[row] -= B[norm] * multiplier;
 *   }
 *   pthread_exit(NULL);
 * }
 */