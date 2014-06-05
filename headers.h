#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include "normal.h"
#include "fmm.h"

#define kparticle 65.0
#define kshell 150.0
#define dt 0.002
#define t_end 1
#define shell_particle_radius 1.0
#define pie 3.14159
#define NUM_BOX_NEIGHBORS 13

inline int min(int a, int b);
inline int max(int a, int b);

void printPairs(int *array, int pair);
void setPosRad(double *pos, double *rad);
void getShell(double *shell);
void print3DVectors(double *array, int size, int dim);
int interactions(int npos, double *pos, double L, int boxdim, double cutoff2, double *distances2, int *pairs, int maxnumpairs, int *numpairs_p);
void interactionsFilter(int *numpairs_p, int *pairs, int *finalPairs, double *rad, double *pos);

struct box{
    int head;
};

int shell_radius;
int nsphere;
int npos;