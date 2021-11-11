//mpicc mpi_gather-test.c -o mpi_gather-test -lm
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#define MSG_LENGTH 10
#define MAXTASKS 32
#define ITERATIONS 2e09
int main (int argc, char *argv[])
{
    int i, tasks, iam, root=0, namelen;
    double buff2recv[MAXTASKS],piTotal=0.0;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &tasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &iam); 
    int nIters = (ITERATIONS / tasks);
    int initIteration = nIters * iam;
    int endIteration = initIteration + nIters -1 ; 
    for ( i = initIteration ; i <= endIteration ; i+=2){
    	piTotal = piTotal + (double)(4.0/ ((i*2)+1));
	piTotal = piTotal - (double)(4.0/ (((i+1)*2)+1));
    }
    double test [2];
    test[0] = iam;
    test[1] = iam;
    MPI_Gather((void *)test, 2, MPI_DOUBLE, buff2recv, 2, MPI_DOUBLE, root, MPI_COMM_WORLD);
    
    piTotal = 0.0;
    if (iam == 0) {
        for(i = 0; i < 2*tasks; i++){
            printf(" %.7f \n", buff2recv[i]);
	    piTotal = piTotal + buff2recv[i];
	}
        printf("PI TOTAL : %f \n" , piTotal);
    }
    MPI_Finalize();
}
