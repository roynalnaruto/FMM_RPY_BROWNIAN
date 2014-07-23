#ifndef _HEADERS_H
#define _HEADERS_H

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <cstring>
#include <string>
# include <iomanip>
# include <ctime>
# include <complex>



using namespace std;

#include "normal.hpp"
//#include "lanczos.h"
#include "sphere_grid.hpp"


#include "fmm3d_mpi.hpp"
#include "manage_petsc_events.hpp"
#include "sys/sys.h"
#include "parUtils.h"



#define constantC1 0.75
#define constantC2 0.50

///!TODO CHANGE ALL THIS TO CAPS
#define kparticle 65.0
#define kshell 150.0
#define dt 0.002
#define pie 3.14159
///---------------------------

#define NUM_BOX_NEIGHBORS 13

#define CHECKCODE 0
#define TMAX 1

//Petsc functions.
extern PetscInt  procLclNum(Vec pos); //{ PetscInt tmp; VecGetLocalSize(pos, &tmp); return tmp/3  /*dim*/; }
extern PetscInt  procGlbNum(Vec pos); //{ PetscInt tmp; VecGetSize(     pos, &tmp); return tmp/3  /*dim*/; }




/*
 * Error Computing Functions 
 */ 
double relError(double *V1, double *V2, int size, int dimension);
double maxError(double *V1, double *V2, int size, int dimension);



/* 
 *	Some Printing Routines
 */ 
void printPairs(int *array, int pair, ostream & file_buffer);
void printVectors(double *array, int size, int dimension, ostream & file_buffer);








void postCorrectionAll(int npos, double *srcpos, double *radii, int numpairs_p, int *pairs, 
						double *forceVec, double *output);
void postCorrection(int index1, int index2, double *srcPosition, double *forceVector, double *radii, double *output);
int part1rpy(int numsrc, double *srcposition, double *forceVec, double *radii, double* &output);
int part2rpy(int numsrc, double *srcposition, double *forceVec, double* &output);
void computeRpy(int nsrc, double *srcPos, double *forceVec, double *radii, double *output, double *temp_output);




void setPosRad(int npos, double *pos, double *rad);
void getShell(double *shell);
void createDiag(int npos, double *A, double *rad);
void mobilityMatrix(int npos, double *A, double *pos, double *rad);
void multiplyMatrix(int npos, double *A, double *f);
void multiplyMatrix_AZ(int npos, double *A, double *Az, double *z);
void getNorm(int npos, int seed, double *z);
void updatePos(int npos, double *pos, double *rpy, double *z);
void savePos(int npos, double *pos, double *rad, int index);



//!Interactive functions
void computeForce(int npos, double *f, double *pos, double *rad, int *pairs, int numpairs);
void computeForceSerial(int npos, double *f, double *pos, double *rad, double *shell);
void interactionsFilter(int npos, int *numpairs_p, int *pairs, int *finalPairs, double *rad, double *pos);
int interactions(int numpos, double *pos, double L, int boxdim, double cutoff2,
							double *distances2, int *pairs, int maxnumpairs, int *numpairs_p);


struct box{
    int head;
};



//Variables
extern double shell_radius;
extern double shell_particle_radius;
extern int nsphere;
extern int npos;


#endif
