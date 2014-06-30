#include "headers.h"


int main(int argc, char** argv)
{
   PetscInitialize(&argc,&argv,"options",NULL); 
   
	// set the number of sources and points per mpi rank
	PetscTruth flg;
	PetscInt n_pos;  
	pC( PetscOptionsGetInt(0, "-numsrc", &n_pos, &flg) );
	if(flg!=true)n_pos=1000;	
	
	int dim = 3;
	npos = n_pos;
	
	double *pos = new double[(npos + nsphere)*dim];
	double *rad = new double[npos];
	double *force = new double[npos*dim];	
	double *shell = pos + (dim * npos);
	
	double *rpy = new double[npos*dim];	
	double *temp_rpy = new double[npos*dim];
	double *A;


//*** for brownian motion
	double* standardNormalZ = new double[npos * dim];
	double* standardNormalZ1;
	
	double *force_serial;
	
	if(CHECKCODE){
		force_serial = new double[npos*dim];
		standardNormalZ1 = new double[npos*dim];
		A = new double[9*npos*npos];
	}
	
	
	
//***************** Variables for Local Interactions********************	
	//! TODO : make boxdim variable.
	double L = 2 * shell_radius;
	int boxdim = 10;
	double cutoff2 = 4 * shell_particle_radius * shell_particle_radius;
	int maxnumpairs = 5000000;
	int *pairs = new int[2 * maxnumpairs];
	int *finalPairs = new int[2 * maxnumpairs];	
	double *distances2 = new double[maxnumpairs];
	int numpairs_p;
//**********************************************************************


/*
//*****************Lanczos ka kachda************************************
	lanczos_t *lanczos, *lanczos1;
	lanczos = new lanczos_t;
	lanczos1 = new lanczos_t;
	int maxiters = 200;
//**********************************************************************	
*/

	
	setPosRad(pos, rad);
	savePos(pos, rad, 0);
	getShell(shell);
	
	for(int tstep=0; tstep<TMAX; tstep++){

	    interactions(npos+nsphere, pos, L, boxdim, cutoff2, distances2, pairs, maxnumpairs, &numpairs_p);
    	interactionsFilter(&numpairs_p, pairs, finalPairs, rad, pos);
    	
    	getNorm((100000000+(rand()%99999999)), standardNormalZ);
		
    	computeForce(force, pos, rad, finalPairs, numpairs_p);

		if(CHECKCODE){
			
			computeForceSerial(force_serial, pos, rad, shell);
			double error1 = relError(force_serial, force, npos, 3);
			printf("Relative Error in computeForce %lf\n", error1);
			
			double error2 = maxError(force_serial, force, npos, 3);
			printf("Max Error in computeForce %lf\n", error2); 
			
			for(int i=0;i<3*npos;i++){
				standardNormalZ1[i] = standardNormalZ[i];
			}
			
			createDiag(A, rad);
			mobilityMatrix(A, pos, rad);
//			create_lanczos (&lanczos1, 1, maxiters, npos*3);
//			compute_lanczos(lanczos1, 1e-4, 1, standardNormalZ1, 3*npos,
//					SERIAL, force, lanczos_out, pos, rad, numpairs_p, finalPairs, A);
//			
			multiplyMatrix(A, force_serial, 1);
					
		}
		
		
		computeRPY(npos, pos, force, rad, rpy, temp_rpy);
		postCorrectionAll(npos, pos, rad, numpairs_p, pairs, force, rpy);
		
//		create_lanczos (&lanczos, 1, maxiters, npos*3);
//		compute_lanczos(lanczos, 1e-4, 1, standardNormalZ, 3*npos,
//					FMM, force, lanczos_out, pos, rad, numpairs_p, finalPairs, A);
//					
		
		if(CHECKCODE){
			
			printf("relative Error : %lf\n", relError(standardNormalZ, standardNormalZ1, npos, 3));
		}
		
		
	//	updatePos(pos, rpy, standardNormalZ);
	//	savePos(pos, rad, tstep+1);
		printf("%d time steps done\n", tstep+1);
	}
	
	PetscFinalize();
	gettimeofday(&endTime, NULL);
	totaltime = (endTime.tv_sec-startTime.tv_sec)*1000000 + endTime.tv_usec-startTime.tv_usec;
	printf("Total time computing %d time steps (%d particles) : %ld msec\n", tmax, npos, totaltime/1000);
	return 0;
}
																																							 
