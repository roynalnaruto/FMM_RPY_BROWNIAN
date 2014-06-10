#ifndef _HEADERS_H
#define _HEADERS_H



#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include "normal.h"
#include "fmm.h"
#include "lanczos.h"

#define kparticle 65.0
#define kshell 150.0
#define dt 0.002
#define t_end 1
#define shell_particle_radius 1.0
#define pie 3.14159
#define NUM_BOX_NEIGHBORS 13
#define CHECKCODE 0

inline int min(int a, int b);
inline int max(int a, int b);
inline double mod(double x);

void printPairs(int *array, int pair);
void printVectors(double *array, int size, int dimension);
void printVectorsComplex(complex *array, int size, int dimension);


void setPosRad(double *pos, double *rad);
void getShell(double *shell);

int interactions(int npos, double *pos, double L, int boxdim, double cutoff2,
					double *distances2, int *pairs, int maxnumpairs, int *numpairs_p);
					
void interactionsFilter(int *numpairs_p, int *pairs, int *finalPairs, double *rad, double *pos);

double relError(double *V1, double *V2, int size, int dimension);
double relErrorRealComplex(double *V1, complex *V2, int size, int dimension);
double maxError(double *V1, double *V2, int size, int dimension);
double maxErrorRealComplex(double *V1, complex *V2, int size, int dimension);




void computeForceSerial(double *f, double *pos, double *rad, double *shell);
void multiplyMatrix(double *A, double *f);
void mobilityMatrix(double *A, double *pos, double *rad);
void createDiag(double *A, double *rad);

int postCorrection(int npos, double *pos, double *rad, int numpairs_p, int *pairs,
					complex *f1, complex *f2, complex *f3, complex *rpy);



struct box{
    int head;
};

int shell_radius;
int nsphere;
int npos;


#endif
