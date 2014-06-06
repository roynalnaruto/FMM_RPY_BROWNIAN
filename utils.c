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






void printVectors(double *array, int size, int dimension){
    int i,j;
    printf("\n");
    for(i=0;i<size;i++){
        for(j=0;j<dimension;j++)
            printf("%lf\t", array[3*i + j]);
        printf("\n");
    }
    printf("\n");
}



void printVectorsComplex(complex *array, int size, int dimension){
    int i,j;
    printf("\n");
    for(i=0;i<size;i++){
        for(j=0;j<dimension;j++)
            printf("(%lf, %lf)\t", array[3*i + j].dr, array[3*i + j].di);
        printf("\n");
    }
    printf("\n");
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
    sprintf(input_index, "./data-for-outer-shell/data%d_%d.csv", nsphere, shell_radius);
    input = fopen(input_index, "r");
    if(input == NULL){
		printf("File for getShell could not be opened\n");
		exit(0);
	}
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

double relErrorRealComplex(double *V1, complex *V2, int size, int dimension){
    int i;
    long double total = 0;
    long double error = 0;
    for(i=0; i<size*dimension; i++){
        total += mod(V1[i]);
        error += mod(V1[i] - V2[i].dr);
    }
    return (error/total);
}



double maxError(double *V1, double *V2, int size, int dimension){
	
	int i;
	long double maxError = 0;
	long double error;
	int index=-1;
	for(i=0;i<size*dimension;i++){
		error = mod(V1[i] - V2[i])/mod(V1[i]);
		if(V1[i] == 0) continue;
		index =  (maxError>error)? index : i; 
		maxError = (maxError>error)? maxError : error;
	}
	
	printf("Index is %d \n", index);
	return maxError;	
}



double maxErrorRealComplex(double *V1, complex *V2, int size, int dimension){

	int i;
	long double maxError = 0;
	long double error;
	int index = -1;
	for(i=0;i<size*dimension;i++){
		error = mod(V1[i] - V2[i].dr)/mod(V1[i]);
		if(V1[i] == 0) continue;
		index =  (maxError>error)? index : i; 	
		maxError = (maxError>error)? maxError : error; 
	}
	printf("Index is %d,V1[i]: %lf, V2[i]: %lf\n", index, V1[index], V2[index].dr);
	return maxError;	
}











void createDiag(double *A, double *rad){
    //printf("enter createDiag\n");

    int i, j;
    for(i=0; i<(3*(npos)); i++){
        for(j=0; j<(3*(npos)); j++){
            if(i==j){
                A[i+3*npos*j] = 1.0/rad[i/3];
                //A_vec[(3*(npos) + 1)*i] = d[i];
            }
            else{
                A[i+3*npos*j] = 0.0;
                //A_vec[3*(npos)*i + j] = 0.0;
            }
        }   
    }

    //printf("exit createDiag\n");
}


void mobilityMatrix(double *A, double *pos, double *rad){
    //printf("enter mobilityMatrix\n");

    //variables
    double a1, a2;
    int k, m, n, index1, index2, i, j;
    double s, a_mean, term1, term2;
    double pi[3], pj[3];
    double e[3], r[3];
    double ee[3][3], A12[3][3];

    double eye3[3][3];
    eye3[0][0] = eye3[1][1] = eye3[2][2] = 1.0;
    eye3[0][1] = eye3[0][2] = eye3[1][0] = eye3[1][2] = eye3[2][0] = eye3[2][1] = 0.0;

    //find mobility matrix
    for(i=0; i<npos; i++){
        for(j=i+1; j<npos; j++){

            //create pi and pj and compute r
            for(k=0; k<3; k++){
                pi[k] = pos[3*i+k];
                pj[k] = pos[3*j+k];
                r[k] = pi[k] - pj[k];
            }

            //euclidean distance
            s = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);

            //calculate e
            for(k=0; k<3; k++){
                e[k] = r[k]/s;
            }

            //calculate ee
            for(m=0; m<3; m++){
                for(n=0; n<3; n++){
                    ee[m][n] = e[n]*e[m];
                }
            }

            //get radii
            a1 = rad[i];
            a2 = rad[j];

            //compute A12
            if(s >= (a1+a2)){
                for(m=0; m<3; m++){
                    for(n=0; n<3; n++){
                        A12[m][n] = (1.0/s)*((3.0/4.0)*(eye3[m][n]+ee[m][n]) + (1.0/4.0)*(a1*a1 + a2*a2)*(eye3[m][n]-3.0*ee[m][n])/(s*s));
                    }
                }
            }
            else{
				//! change to cube root
                a_mean = sqrt((1.0/2.0)*(a1*a1 + a2*a2));
                term1 = (1.0/a_mean) - (9.0*s)/(32.0*a_mean*a_mean);
                term2 = (3.0*s)/(32.0*a_mean*a_mean);
                for(m=0; m<3; m++){
                    for(n=0; n<3; n++){
                        A12[m][n] = term1*eye3[m][n] + term2*ee[m][n];
                    }
                }
            }

            //indices and update
            for(m=0; m<3; m++){
                index1 = 3*(i+1)-(3-m);
                for(n=0; n<3; n++){
                    index2 = 3*(j+1)-(3-n);
                    A[index1 + 3*npos*index2] += A12[m][n];
                    //A_vec[3*npos*index1 + index2] = A[index1][index2];
                    A[index2 + 3*npos*index1] += A12[n][m];
                    //A_vec[3*npos*index2 + index1] = A[index2][index1];
                }
            }

        }
    }

    //printf("exit mobilityMatrix\n");
}


void multiplyMatrix(double *A, double *f){
    //printf("enter multiplyMatrix\n");

    int i, j;
    double temp_value[3*npos];
    for(i=0; i<3*(npos); i++){
        temp_value[i] = 0.0;
        for(j = 0; j<3*(npos); j++){
            temp_value[i] += A[i+3*npos*j]*f[j];
        }
    }

    for(i=0; i<3*npos; i++){
        A[i] = temp_value[i];
    }
    
    //printf("exit multiplyMatrix\n");
}
