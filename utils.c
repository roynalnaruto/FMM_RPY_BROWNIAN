#include "headers.h"

inline int min(int a, int b){
    return ((a>b)? b : a);
}

inline int max(int a, int b){
    return ((a>b)? a : b);
}

inline double mod(double x){
	return ((x>0)? x : -x) ;
}


void printPairs(int *array, int pair){

    int i;
    printf("\n");
    for(i=0; i<pair; i+=2){
        printf("%d, %d\n", array[i], array[i+1]);
    }
    printf("\n");
}

void print3DVectors(double *array, int size, int dim){
    int i,j;
    for(i=0;i<size;i++){
        for(j=0;j<dim;j++)
            printf("%lf, ", array[3*i + j]);
        printf("\n");
    }
}

void setPosRad(double *pos, double *rad){
    //printf("enter setPosRad\n");
    double temp_r, temp_theta, temp_phi;
    int i, j;
    for(i=0; i<npos; i++){

        //r, phi and theta
        temp_r = (double)rand()/(double)(RAND_MAX/1.0);
        temp_r = pow(temp_r, 1.0/3.0)*(shell_radius - 1.0);
        temp_theta = -1.0 + (float)rand()/(float)(RAND_MAX/2.0);
        temp_theta = acos(temp_theta);
        temp_phi = (float)rand()/(float)(RAND_MAX/(2.0*pie));

        rad[i] = 0.3 + (double)rand()/(double)(RAND_MAX/0.7);
        //spherical co-ordinates to cartesian
        pos[0+(i*3)] = temp_r*sin(temp_theta)*cos(temp_phi) + shell_radius;
        pos[1+(i*3)] = temp_r*sin(temp_theta)*sin(temp_phi) + shell_radius;
        pos[2+(i*3)] = temp_r*cos(temp_theta) + shell_radius;

    }

    //printf("exit setPosRad\n");
}

void getShell(double *shell){
    //printf("enter getShell\n");
    char input_index[100];
    int i, j;
    FILE *input;
    sprintf(input_index, "/home/rohit/final/data-for-outer-shell/data%d_%d.csv", nsphere, shell_radius);
    input = fopen(input_index, "r");
    for(i=0; i<nsphere; i++){
        for(j=0; j<3; j++){
            fscanf(input, "%lf, ", &shell[3*i + j]);
            shell[3*i + j] += shell_radius;
        }
        fscanf(input, "\n");
    }
    fclose(input);
    //printf("exit getShell\n");
}

double relError(double *V1, double *V2, int size, int dimension){
	
	int i;
	long double total = 0;
	long double error = 0;
	for(i=0;i<size*dimension;i++){
		total += mod(V1[i]);
		error += mod(V1[i] - V2[i]);
	}
	return (error/total);	
	
	
}


