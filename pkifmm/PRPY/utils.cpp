#include "headers.h"


inline double mod(double x){
	return ((x>0)? x : -x) ;
}




/**
 * 
 * Error Computing Functions
 * 
 */ 
 
///Computes Relative Error b/w 2 vectors. 
double relError(double *V1, double *V2, int size, int dimension){
	
	long double total = 0;
	long double error = 0;
	for(int i=0;i<size*dimension;i++){
		total += mod(V1[i]);
		error += mod(V1[i] - V2[i]);
	}
	return (error/total);	
}


///Computes the maximum difference in any dimension b/w 2 vectors 
double maxError(double *V1, double *V2, int size, int dimension){
	
	long double maxError = 0;
	long double error;
	int index=-1; //max. difference is at this index
	for(int i=0;i<size*dimension;i++){
		if(V1[i] == 0) continue;
		error = mod(V1[i] - V2[i])/mod(V1[i]);
		index =  (maxError>error)? index : i; 
		maxError = (maxError>error)? maxError : error;
	}
	return maxError;	
}





/** 
 *	Some Printing Routines
 */ 
 
///Prints elements of a one dimensional array  in pairs.
void printPairs(int *array, int pair, ostream & file_buffer){

    for(int i=0; i<pair; i+=2){
		file_buffer<< array[i]<< ", "<< array[i+1] <<endl;
    }
}

///Prints elements of a multi dimensional array(but stored as a single dimensional array).
void printVectors(double *array, int size, int dimension, ostream & file_buffer){
    
    for(int i=0;i<size;i++){
        for(int j=0;j<dimension;j++)
			file_buffer<< array[dimension*i + j]<< "\t";
        file_buffer<<"\n";
    }
}






/**
 * Randomly(and uniformly) generates positions within a sphere(called shell henceforth) 
 * Also gives a certain radii to those points
 */ 
void setPosRad(int npos, double *pos, double *rad, double shell_radius, double shell_particle_radius){
    
    double r, theta, phi;   
    double center[3] = {shell_radius, shell_radius, shell_radius};
    for(int i=0; i<npos; i++){
		
		r = drand48();
        r = pow(r, 1.0/3.0) * (shell_radius - 0.1);
        theta = -1.0 + (2* drand48());
        theta = acos(theta);
        phi = drand48() * 2.0 * pie;

        //spherical co-ordinates to cartesian
        pos[0+(i*3)] = center[0] + r*sin(theta)*cos(phi);
        pos[1+(i*3)] = center[1] + r*sin(theta)*sin(phi);
        pos[2+(i*3)] = center[2] + r*cos(theta);
                
        //point generated, now give it a certain radius.
        ///!TODO: SCALE RADIUS APPROPRIATELY
        rad[i] = 0.2 * (shell_particle_radius) * drand48(); // USED FOR VARIABLE RADII
        //rad[i] = shell_particle_radius * 0.2;                                           // USED FOR CONST RADII        
    }
}


/**
 * Gets points on the surface of the shell.
 * The points are generated like a grid on the shell.
 * 
 */ 
void getShell(int *nsphere, double *shell){
	int factor;
	int node;
	//factor determines the number of points generated on the shell. 
	//The greater the factor, larger the number of points generated. 
	factor = 4;
	*nsphere = sphere_icos_point_num ( factor );
	
	///!TODO : Change this, infi memory leak going on here. !!CHANGE
	double *shell1 = sphere_icos1_points (factor, *nsphere);
	for(int i=0;i<3*(*nsphere);i++)
		shell[i] = shell1[i];
}




/**
 * Generates a diagonal matrix. 
 * Used only for debugging purposes.
 */
void createDiag(int npos, double *A, double *rad){
 
    for(int i=0; i<(3*npos); i++) 
		for(int j=0; j<(3*npos); j++) 
			A[j+3*npos*i] = 0.0;
	
    
    for(int i=0; i<(3*npos); i++) 
		    A[i+3*npos*i] = 1.0/rad[i/3];
            
}


/**
 * Generates the (3N * 3N) mobility matrix for the RPY tensor for different particle radii. 
 * Used only for debugging purposes.
 */ 

void mobilityMatrix(int npos, double *A, double *pos, double *rad){

    double a1, a2;
    int k, index1, index2;
    double s, a_mean, term1, term2;
    double pi[3], pj[3];
    double e[3], r[3];
    double ee[3][3], A12[3][3];

    double eye3[3][3];
    eye3[0][0] = eye3[1][1] = eye3[2][2] = 1.0;
    eye3[0][1] = eye3[0][2] = eye3[1][0] = eye3[1][2] = eye3[2][0] = eye3[2][1] = 0.0;

    //find mobility matrix
    for(int i=0; i<npos; i++){
        for(int j=i+1; j<npos; j++){

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
            for(int m=0; m<3; m++){
                for(int n=0; n<3; n++){
                    ee[m][n] = e[n]*e[m];
                }
            }

            //get radii
            a1 = rad[i];
            a2 = rad[j];

            //compute A12
            if((s >= (a1+a2)) || false){
                for(int m=0; m<3; m++){
                    for(int n=0; n<3; n++){
                        A12[m][n] = (1.0/s)*((3.0/4.0)*(eye3[m][n]+ee[m][n]) +
									(1.0/4.0)*(a1*a1 + a2*a2)*(eye3[m][n]-3.0*ee[m][n])/(s*s));
					}
                }
            }
            
            else{
				///! TODO : change to cube root
                a_mean = sqrt((1.0/2.0)*(a1*a1 + a2*a2));
                term1 = (1.0/a_mean) - (9.0*s)/(32.0*a_mean*a_mean);
                term2 = (3.0*s)/(32.0*a_mean*a_mean);
                for(int m=0; m<3; m++){
                    for(int n=0; n<3; n++){
                        A12[m][n] = term1*eye3[m][n] + term2*ee[m][n];
                    }
                }
            }

            //indices and update
            for(int m=0; m<3; m++){
                index1 = 3*(i+1)-(3-m);
                for(int n=0; n<3; n++){
                    index2 = 3*(j+1)-(3-n);
                    A[index1 + 3*npos*index2] += A12[m][n];
                    A[index2 + 3*npos*index1] += A12[n][m];
                }
            }

        }
    }
}




/**
 * A is a N*N matrix. f is a N length vector
 * Multiplies Af and stores it in the first few entries of A.
 * 
 */ 
void multiplyMatrix(int npos, double *A, double *f){
    
    double sum;
    for(int i=0; i<3*(npos); i++){
        sum = 0.0;
        for(int j = 0; j<3*(npos); j++){
            sum += A[i+3*npos*j]*f[j];
        }
        ///! CAREFUL HERE. CHANGING A. TODO
        A[i] = sum;
	}
}



/**
 * A is a N*N matrix. z is a N length vector
 * Multiplies Az and stores it in Az. 
 */ 
void multiplyMatrix_AZ(int npos, double *A, double *Az, double *z){
    for(int i=0; i<3*(npos); i++){
        Az[i] = 0.0;
        for(int j = 0; j<3*(npos); j++){
            Az[i] += A[i+3*npos*j]*z[j];
        }
    }
}


/**
 * Generates a standard normal distribution of dimension 3*npos.
 */ 
void getNorm(int npos, int seed, double *z){
	for(int i=0; i<3*npos; i++){
		z[i] = r8_normal_01(seed);
	}
}


/**
 *Update Position for the next time step. 
 */
void updatePos(int npos, double *pos, double *rpy, double *z){
    int i;
    for(i=0; i<3*npos; i++){
        pos[i] += rpy[i]*dt + sqrt(2*dt)*z[i];
    }
}



/**
 * Save position in a file whose name is indexed by <index>
 */ 
void savePos(int npos, double *pos, double *rad, int index){
    
    FILE *file;
    char filename[50];
    sprintf(filename, "./output_data/pos.csv.%04d", index);
    file = fopen(filename, "w");
    
    if(file == NULL){
		printf("File:%s in savePos could not be opened\n", filename);
		exit(0);
	}
	
	for(int i=0; i<npos; i++){
        for(int j=0; j<3; j++)
            fprintf(file, "%f, ", pos[3*i + j]);
        fprintf(file, "%f\n", rad[i]);
    }
    fclose(file);
}


