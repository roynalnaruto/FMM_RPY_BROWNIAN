#include "headers.h"

void getBasic(){
	//printf("enter getBasic\n");
	printf("Enter number of particles (npos) {100, 300, 600, 1000, 2500, 3000, 5000}\n");
	scanf("%d", &npos);

	if(npos<100)
		shell_radius = 6;
	
	else{	
		switch(npos){
			case 100:
			shell_radius = 6;
			break;

			case 300:
			shell_radius = 9;
			break;

			case 600:
			shell_radius = 11;
			break;

			case 1000:
			shell_radius = 14;
			break;

			case 2500:
			shell_radius = 18;
			break;

			case 3000:
			shell_radius = 18;
			break;


			case 5000:
			shell_radius = 24;
			break;
			
			case 10000:
			shell_radius = 24;
			break;
			
			case 20000:
			shell_radius = 24;
			break;
			
			case 50000:
			shell_radius = 24;
			break;

			
			default:
			shell_radius = 24;
			//printf("Invalid number of particles...exiting ...\n");
			//exit(0);
		}
	}

	nsphere = 900;
	if(shell_radius>20)
		nsphere = 1444;
	//printf("exit getBasic\n");
}

int main(){

	int i, j;

	time_t t;
	srand((unsigned) time(&t));
	
	struct timeval startTime, endTime;
	long long time1, time2;

	getBasic();
    printf("%d, %d, %d \n", npos, nsphere, shell_radius);

	double *pos, *shell, *rad;
	pos = (double *)malloc(sizeof(double)*(npos+nsphere)*3);
	rad = (double *)malloc(sizeof(double)*npos);
	shell = pos + 3*npos;

	
	setPosRad(pos, rad);
    getShell(shell);

	
    double L = 2*shell_radius;
	int boxdim = shell_radius;
	double cutoff2 = (2 * shell_particle_radius) * (2 * shell_particle_radius);
	


	int maxnumpairs = 5000000;
	int *pairs = (int *)malloc(sizeof(int)*2*maxnumpairs);
	int *finalPairs = (int *)malloc(sizeof(int)*2*maxnumpairs);
    double *distances2 = (double *)malloc(sizeof(double)*maxnumpairs);
    int numpairs_p;

	gettimeofday(&startTime, NULL); 
    interactions(npos+nsphere, pos, L, boxdim, cutoff2, distances2, pairs, maxnumpairs, &numpairs_p);
    //printPairs(pairs, numpairs_p);
    //printf("Number of pairs : %d\n", numpairs_p);
    
    interactionsFilter(&numpairs_p, pairs, finalPairs, rad, pos);
    //printPairs(finalPairs, numpairs_p);
    printf("Final number of pairs (after filtering) : %d\n", numpairs_p);

    double *f;
    f = (double *)malloc(sizeof(double)*3*npos);
    complex *f1, *f2, *f3, *rpy, *lanczos_out;
    f1 = (complex *)malloc(sizeof(complex)*npos);
    f2 = (complex *)malloc(sizeof(complex)*npos);
    f3 = (complex *)malloc(sizeof(complex)*npos);
    rpy = (complex *)malloc(sizeof(complex)*(npos*3));
    lanczos_out = (complex *)malloc(sizeof(complex)*(npos*3));

    computeForce(f, f1, f2, f3, pos, rad, finalPairs, numpairs_p);


	if(CHECKCODE){		
		double *f_serial;
		f_serial = (double *)malloc(sizeof(double)*3*npos);
		computeForceSerial(f_serial, pos, rad, shell);
		//printVectors(f, npos, 3);
		//printVectors(f_serial, npos, 3);	
		double error = relError(f_serial, f, npos, 3);
		printf("Relative Error in computeForce %lf\n", error);
		error = maxError(f_serial, f, npos, 3);
		printf("Max Error in computeForce %lf\n", error); 
    }
    
    
    char dir[100] = "outputs/";
    printf("Calling computeRPY : \n");
    computerpy_(&npos, pos, rad, f1, f2, f3, rpy,dir);
    printf("Calling postCorrection : \n");
    postCorrection(npos, pos, rad, numpairs_p, finalPairs, f1, f2, f3,rpy);
    gettimeofday(&endTime, NULL); 
    time1 = (endTime.tv_sec-startTime.tv_sec)*1000000 + endTime.tv_usec-startTime.tv_usec;    
	printf("PostCorrection Done \n");
	printf("Total Time taken : %ld \n", time1);
	
	
	double *standardNormalZ = (double *)malloc(sizeof(double)*npos*3);
	//obtain standardNormalZ from a standard Normal Distribution ~ N(0, I)
	//
	getNorm((1000000+(rand()%999999)), standardNormalZ);
	//
	
	
	lanczos_t *lanczos = malloc(sizeof(lanczos));
	int maxiters = 100;
	create_lanczos (&lanczos, 1, maxiters, npos*3);
	compute_lanczos(lanczos, 0.01, 1, standardNormalZ, 3*npos,
				FMM, f1, f2, f3, lanczos_out, pos, rad, numpairs_p, finalPairs);
	printVectors(lanczos->v, npos, 3);

	


	//
	// Mobility Matrix
	//
	
	if(CHECKCODE){
		double *A;
		A = (double *)malloc(sizeof(double)*3*3*npos*npos);
		createDiag(A, rad);
		mobilityMatrix(A, pos, rad);
		multiplyMatrix(A, f);
		double error = relErrorRealComplex(A, rpy, npos, 3);	
		//printVectorsComplex(rpy, npos, 3);
		//printVectors(A, npos, 3);
		//printVectors(pos, npos, 3);
		printf("Relative Error in Mf and rpy is : %lf\n", error);
		error = maxErrorRealComplex(A, rpy, npos, 3);
		printf("Max Error in Mf and rpy is : %lf\n", error);
	}
	  
	return 0;
}
