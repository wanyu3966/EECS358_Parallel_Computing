#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define CUTX    0
#define CUTY    1
#define NUM_POINTS  524288

unsigned int  X_axis[NUM_POINTS];
unsigned int  Y_axis[NUM_POINTS];
int   myid, numprocs;

void quicksort( unsigned int *x, unsigned int *y, int low, int high, int cutdir );


/* void split(unsigned int a[],int low, int high); */

void swap( unsigned int *x, int a, int b )
{
  unsigned int temp;
  temp  = x[a];
  x[a]  = x[b];
  x[b]  = temp;
}


void find_quadrants( num_quadrants )
int num_quadrants;


{
  /* sort all points in terms of X_axis and Y_axis (in consistency with coordinates) */
  /* YOU NEED TO FILL IN HERE */
  int quadrants, quastart, quaend, i, j, k, count;
  int numpoints;                                              /* number of points in each quadrant */
  int cutdir = 0, middle = 0;
  int minx, miny, maxx, maxy;
  int cut[num_quadrants - 1], symbol[num_quadrants - 1];      /* cut number=num_quadrants-1 */


  if ( myid == 0 )
  {
/*
 *   minx=X_axis[0];
 * maxx=X_axis[0];
 * miny=Y_axis[0];
 * maxy=Y_axis[0];
 * for(i=0;i<NUM_POINTS;i++){
 *   if(X_axis[i]>maxx)
 *     maxx=X_axis[i];
 *   if(X_axis[i]<minx)
 *     minx=X_axis[i];
 *   if(Y_axis[i]>maxy)
 *     maxy=Y_axis[i];
 *   if(Y_axis[i]<miny)
 *     miny=Y_axis[i];
 * }
 */
    quicksort( X_axis, Y_axis, 0, NUM_POINTS - 1, CUTX );
    maxx  = X_axis[NUM_POINTS - 1];
    minx  = X_axis[0];
    quicksort( X_axis, Y_axis, 0, NUM_POINTS - 1, CUTY );
    maxy  = Y_axis[NUM_POINTS - 1];
    miny  = Y_axis[0];

    quadrants = 1;
    count   = 0;
    //initialize the cut positions
    while ( quadrants < num_quadrants )
    {
      if ( cutdir == CUTX )
      {
        for ( i = 0; i < quadrants; i++ )
        {
          numpoints = NUM_POINTS / quadrants;
          quastart  = numpoints * i;
          quaend    = numpoints * (i + 1) - 1;
          quicksort( X_axis, Y_axis, quastart, quaend, cutdir );
          /*
           * if(quadrants==1){
           *   maxx=X_axis[NUM_POINTS-1];
           *   minx=X_axis[0];
           * }
           * printf("sortfinish\n");
           * for(j=0;j<10;j++)
           *   printf("X%dY%d\n",X_axis[j],Y_axis[j] );
           */
          middle      = X_axis[numpoints / 2 + 1 + i * numpoints];
          cut[count + i]    = middle;       /* cut position */
          symbol[count + i] = cutdir;       /* cut operation type */
        }
        cutdir = CUTY;
      }else  {
        for ( i = 0; i < quadrants; i++ )
        {
          numpoints = NUM_POINTS / quadrants;
          quastart  = numpoints * i;
          quaend    = numpoints * (i + 1) - 1;
          quicksort( X_axis, Y_axis, quastart, quaend, cutdir );
          middle      = Y_axis[numpoints / 2 + 1 + i * numpoints];
          cut[count + i]    = middle;
          symbol[count + i] = cutdir;
        }
        cutdir = CUTX;
      }
      count   += quadrants;
      quadrants *= 2;
    }


    unsigned int top[num_quadrants], right[num_quadrants], left[num_quadrants], bottom[num_quadrants];

    top[0]    = maxy;
    bottom[0] = miny;
    left[0]   = minx;
    right[0]  = maxx;
    quadrants = 1;
    int index = 0;
/* repeat the cutting process */
    while ( quadrants < num_quadrants )
    {
      for ( i = 0; i < quadrants; i++ )
      {
        top[i + quadrants] = top[i];
        // printf( "%d\n", top[i] );
        bottom[i + quadrants] = bottom[i];
        left[i + quadrants] = left[i];
        right[i + quadrants]  = right[i];
      }
      i=0;
      while ( i < 2 * quadrants )
      {
        int index1=i/2+quadrants;
        if ( symbol[index] == CUTX )
        {

          top[i]    = top[index1];
          bottom[i] = bottom[index1];
          left[i]   = left[index1];
          right[i]  = cut[index];

          top[i + 1]  = top[i];
          bottom[i + 1] = bottom[i];
          left[i + 1] = cut[index];
          right[i + 1]  = right[i];
          index++;
        }else  {
          top[i]    = cut[index];
          bottom[i] = bottom[index1];
          left[i]   = left[index1];
          right[i]  = right[index1];

          top[i + 1]  = top[i];
          bottom[i + 1] = cut[index];
          left[i + 1] = left[i];
          right[i + 1]  = right[i];
          index++;
        }
        i+=2;
      }
      quadrants *= 2;
    }

    //print quadrants
    for ( i = 0; i < num_quadrants; i++ )
    {
      printf( "\nPoints %d : ", i );
      printf( " (%d,%d)  (%d,%d)  (%d,%d)  (%d,%d) ", left[i], bottom[i] , right[i], bottom[i], left[i], top[i], right[i], top[i] );
    }
  }

  //compute costf
  double  distance = 0, global_distance;
  int x, y, p1, p2;
  numpoints = NUM_POINTS / num_quadrants;
  MPI_Bcast( &X_axis, NUM_POINTS, MPI_INT, 0, MPI_COMM_WORLD );
  MPI_Bcast( &Y_axis, NUM_POINTS, MPI_INT, 0, MPI_COMM_WORLD );
  for ( i = myid; i < num_quadrants; i += numprocs )
  {
    for ( x = 0; x < numpoints; x++ )
    {
      for ( y = x + 1; y < numpoints; y++ )
      {
        p1    = numpoints * i + x;
        p2    = numpoints * i + y;
        distance  += sqrt( (X_axis[p1] - X_axis[p2]) * (X_axis[p1] - X_axis[p2]) + (Y_axis[p1] - Y_axis[p2]) * (Y_axis[p1] - Y_axis[p2]) );
      }
    }
  }
  //reduce result
  MPI_Reduce( &distance, &global_distance, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
  if ( myid == 0 )
  {
    printf( "total cost = %lf\n", global_distance );
  }
}

//sort the array
void quicksort( unsigned int *x, unsigned int *y, int low, int high, int cutdir )
{
  int mid;
  /* printf("sort\n" ); */
  int   i = low, j = high;
  unsigned int  key, temp, tempx, tempy;
  if ( low >= high )
    return;
  if ( cutdir == CUTX )
  {
    key = x[low];
    temp  = y[low];
    while ( i < j )
    {
      while ( i < j && key <= x[j] )
      {
        j--;
      }
      swap( x, i, j );
      swap( y, i, j );
      while ( i < j && key >= x[i] )
      {
        i++;
      }
      swap( x, i, j );
      swap( y, i, j );
    }
    mid = i;
  }
  if ( cutdir == CUTY )
  {
    key = y[low];
    temp  = x[low];
    while ( i < j )
    {
      while ( i < j && key <= y[j] )
      {
        j--;
      }
      swap( y, i, j );
      swap( x, i, j );
      while ( i < j && key >= y[i] )
      {
        i++;
      }
      swap( y, i, j );
      swap( x, i, j );
    }
    mid = i;
  }
  /*
   * x[i]=key;
   * y[i]=temp;
   */
  quicksort( x, y, low, mid - 1, cutdir );
  quicksort( x, y, mid + 1, high, cutdir );
}


int main( argc, argv )
int argc;


char *argv[];
{
  int num_quadrants;

  int namelen;
  char  processor_name[MPI_MAX_PROCESSOR_NAME];
  double  startwtime = 0.0, endwtime;
  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &numprocs );
  MPI_Comm_rank( MPI_COMM_WORLD, &myid );
  MPI_Get_processor_name( processor_name, &namelen );

  if ( argc != 2 )
  {
    fprintf( stderr, "Usage: recursive_bisection <#of quadrants>\n" );
    MPI_Finalize();
    exit( 0 );
  }

  fprintf( stderr, "Process %d on %s\n", myid, processor_name );

  num_quadrants = atoi( argv[1] );

  if ( myid == 0 )
    fprintf( stdout, "Extracting %d quadrants with %d processors \n", num_quadrants, numprocs );

  if ( myid == 0 )
  {
    int i;

    srand( 10000 );

    for ( i = 0; i < NUM_POINTS; i++ )
      X_axis[i] = (unsigned int) rand();

    for ( i = 0; i < NUM_POINTS; i++ )
      Y_axis[i] = (unsigned int) rand();
    startwtime = MPI_Wtime();
  }

  MPI_Bcast( &X_axis, NUM_POINTS, MPI_INT, 0, MPI_COMM_WORLD );
  MPI_Bcast( &Y_axis, NUM_POINTS, MPI_INT, 0, MPI_COMM_WORLD );

  find_quadrants( num_quadrants );

  if ( myid == 0 )
  {
    endwtime = MPI_Wtime();
    printf( "time =%f\n", endwtime - startwtime );
  }
  MPI_Finalize();
  return(0);
}


