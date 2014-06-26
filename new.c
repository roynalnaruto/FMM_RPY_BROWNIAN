#include "headers.h"

void getBasic(){
	//printf("enter getBasic\n");
	printf("Enter number of particles (npos) {100, 300, 600, 1000, 2500, 3000, 5000} : ");
	scanf("%d", &npos);

	if(npos<100)
		shell_radius = 6;
	
	else{	
		switch(npos){
			case 100:
			shell_radius = 6;
			nsphere = 625;
			//shell_radius = 11;
			break;

			case 300:
			shell_radius = 14;
			nsphere = 529;
			break;

			case 600:
			shell_radius = 11;
			nsphere = 900;
			break;

			case 1000:
			shell_radius = 14;
			nsphere = 900;
			break;

			case 2500:
			shell_radius = 18;
			nsphere = 900;
			break;

			case 3000:
			shell_radius = 18;
			nsphere = 900;
			break;

			case 5000:
			shell_radius = 24;
			nsphere = 1444;
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
			//shell_radius = 24;
			printf("Invalid number of particles...exiting ...\n");
			exit(0);
		}
	}

	//nsphere = 900;
	if(shell_radius>50)
		nsphere = 1444;

	//printf("exit getBasic\n");
}

/*
void getBasic(){
	npos = 1000000;
	shell_radius = 1275;
	//nsphere = 196002;
	printf("Number of particles : %d\n", npos);
	printf("Shell radius : %d\n", shell_radius);
	//printf("Number of particles on the shell : %d\n", nsphere);
	//printf("Radius of particles on shell : %lf\n", shell_particle_radius);
}
*/

int main(){

	int i, j, tstep;

	time_t t;
	srand((unsigned) time(&t));
	
	struct timeval startTime, endTime;
	long long time1, time2, totaltime;

	getBasic();
    //printf("No. of Particles : %d\nNo. of Particles on shell : %d\nShell Radius : %d \n", npos, nsphere, shell_radius);

    /////START TOTAL_TIME
    gettimeofday(&startTime, NULL);

	double *pos, *shell, *rad;
	pos = (double *)malloc(sizeof(double)*(npos+nsphere)*3);
	rad = (double *)malloc(sizeof(double)*npos);
	shell = pos + 3*npos;

	setPosRad(pos, rad);
	savePos(pos, rad, 0);
    getShell(shell);

    //check pos & shell
    //printVectors(pos, npos+nsphere, 3);

    double L = 2*shell_radius;
	int boxdim = shell_radius;
	double cutoff2 = (2 * shell_particle_radius) * (2 * shell_particle_radius);
	
	int maxnumpairs = 5000000;
	int *pairs = (int *)malloc(sizeof(int)*2*maxnumpairs);
	int *finalPairs = (int *)malloc(sizeof(int)*2*maxnumpairs);
    double *distances2 = (double *)malloc(sizeof(double)*maxnumpairs);
    int numpairs_p;

    double *f;
    f = (double *)malloc(sizeof(double)*3*npos);
    complex *f1, *f2, *f3, *rpy, *lanczos_out;
    f1 = (complex *)malloc(sizeof(complex)*npos);
    f2 = (complex *)malloc(sizeof(complex)*npos);
    f3 = (complex *)malloc(sizeof(complex)*npos);
    rpy = (complex *)malloc(sizeof(complex)*(npos*3));
    lanczos_out = (complex *)malloc(sizeof(complex)*(npos*3));

    double *f_serial;
	f_serial = (double *)malloc(sizeof(double)*3*npos);
	double error1, error2;
	double *standardNormalZ1, *standardNormalZ;
	standardNormalZ = (double *)malloc(sizeof(double)*npos*3);
	standardNormalZ1 = (double *)malloc(sizeof(double)*npos*3);

	lanczos_t *lanczos, *lanczos1;
	lanczos = malloc(sizeof(lanczos));

	int maxiters = 200;
	double *A;
	lanczos1 = malloc(sizeof(lanczos));
	A = (double *)malloc(sizeof(double)*3*3*npos*npos);

    for(tstep = 0; tstep<tmax; tstep++){


    	interactions(npos+nsphere, pos, L, boxdim, cutoff2, distances2, pairs, maxnumpairs, &numpairs_p);
    	interactionsFilter(&numpairs_p, pairs, finalPairs, rad, pos);
    	//printf("Final number of pairs (after filtering) : %d\n", numpairs_p);

    	computeForce(f, f1, f2, f3, pos, rad, finalPairs, numpairs_p);

		if(CHECKCODE){		
			
			computeForceSerial(f_serial, pos, rad, shell);
			
			error1 = relError(f_serial, f, npos, 3);
			//printf("Relative Error in computeForce %lf\n", error1);
			error2 = maxError(f_serial, f, npos, 3);
			//printf("Max Error in computeForce %lf\n", error2); 
    	}
    
    	char dir[100] = "outputs/";
    	//printf("Calling computeRPY : \n");
    	//gettimeofday(&startTime, NULL); 
    	computerpy_(&npos, pos, rad, f1, f2, f3, rpy, dir);
    	//gettimeofday(&endTime, NULL); 
    	//time1 = (endTime.tv_sec-startTime.tv_sec)*1000000 + endTime.tv_usec-startTime.tv_usec;    
		//printf("Total Time taken for RPY with FMM: %ld \n", time1/1000);
		
    	//printf("Calling postCorrection : \n");
		//gettimeofday(&startTime, NULL); 
    	postCorrection(npos, pos, rad, numpairs_p, finalPairs, f1, f2, f3, rpy);
    	//gettimeofday(&endTime, NULL); 
    	//time1 = (endTime.tv_sec-startTime.tv_sec)*1000000 + endTime.tv_usec-startTime.tv_usec;
		//printf("Total Time taken for postCorrection: %ld \n", time1/1000);
		
		
	
		getNorm((100000000+(rand()%99999999)), standardNormalZ);
		//printVectors(standardNormalZ, 3*npos, 1);
		
		if(CHECKCODE){
			
			for(i=0;i<npos*3;i++)
				standardNormalZ1[i] = standardNormalZ[i];
		}
	
		//
		
		if(CHECKCODE){
				
			create_lanczos (&lanczos1, 1, maxiters, npos*3);
			
			//double *Az;
			//Az = (double *)malloc(sizeof(double)*3*npos);
			createDiag(A, rad);
			mobilityMatrix(A, pos, rad);

			//printVectorsToFile(A, npos*3);

			compute_lanczos(lanczos1, 1e-4, 1, standardNormalZ1, 3*npos,
					SERIAL, f1, f2, f3, lanczos_out, pos, rad, numpairs_p, finalPairs, A);
		}
			
		//printf("Adding Random Brownian motion ... \n");
		create_lanczos (&lanczos, 1, maxiters, npos*3);
		compute_lanczos(lanczos, 1e-4, 1, standardNormalZ, 3*npos,
					FMM, f1, f2, f3, lanczos_out, pos, rad, numpairs_p, finalPairs, A);
	
		/////////END TOTAL_TIME
		//gettimeofday(&endTime, NULL);
		//totaltime = (endTime.tv_sec-startTime.tv_sec)*1000000 + endTime.tv_usec-startTime.tv_usec;
		//printf("Total time computing 1 time step (%d particles) : %ld msec\n", npos, totaltime/1000);


		if(CHECKCODE){
			printf("relative Error : %lf\n", relError(standardNormalZ, standardNormalZ1, npos, 3));
		}

		updatePos(pos, rpy, standardNormalZ);
		savePos(pos, rad, tstep+1);

		printf("%d time steps done\n", tstep+1);

	}

	gettimeofday(&endTime, NULL);
	totaltime = (endTime.tv_sec-startTime.tv_sec)*1000000 + endTime.tv_usec-startTime.tv_usec;
	printf("Total time computing %d time steps (%d particles) : %ld msec\n", tmax, npos, totaltime/1000);
		
	return 0;
}