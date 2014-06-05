#include "headers.h"

void getBasic(){
	//printf("enter getBasic\n");
	printf("Enter number of particles (npos) {100, 300, 600, 1000, 2500, 3000, 5000}\n");
	scanf("%d", &npos);

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
	int maxnumpairs = 200000;
	int *pairs = (int *)malloc(sizeof(int)*2*maxnumpairs);
	int *finalPairs = (int *)malloc(sizeof(int)*2*maxnumpairs);
    double *distances2 = (double *)malloc(sizeof(double)*maxnumpairs);
    int numpairs_p;

    interactions(npos+nsphere, pos, L, boxdim, cutoff2, distances2, pairs, maxnumpairs, &numpairs_p);
    printPairs(pairs, numpairs_p);
    printf("Number of pairs : %d\n", numpairs_p);
    
    interactionsFilter(&numpairs_p, pairs, finalPairs, rad, pos);
    printPairs(finalPairs, numpairs_p);
    printf("Final number of pairs (after filtering) : %d\n", numpairs_p);

    double *f;
    f = (double *)malloc(sizeof(double)*3*npos);
    complex *f1, *f2, *f3;
    f1 = (complex *)malloc(sizeof(complex)*npos);
    f2 = (complex *)malloc(sizeof(complex)*npos);
    f3 = (complex *)malloc(sizeof(complex)*npos);
    computeForce(f, f1, f2, f3, pos, rad, finalPairs, numpairs_p);

    //printf("%lf\n", f[]);

	return 0;
}