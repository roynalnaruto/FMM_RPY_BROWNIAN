#ifndef __LANCZOS_H__
#define __LANCZOS_H__

#include "headers.h"

typedef struct _lanczos_t
{
    int maxiters;
    int maxnrhs;
    int nm;   
    double *band0;
    double *band1;
    double *d;
    double *h2;
    double *v;    
    int ldv;
    double *b;
    double *b2;
    
    double *h;
    double *R;
    double *tau;
    double memory;
} lanczos_t;

#define DPRINTF printf
#define ALIGNED_MALLOC(size)  _mm_malloc(size, ALIGNLEN)
#define ALIGNED_FREE(addr) _mm_free(addr)
#define PAD_LEN(N,size)     (((N+(ALIGNLEN/size)-1)/(ALIGNLEN/size))* (ALIGNLEN/size))
#define ALIGNLEN 64

enum {FMM, SERIAL};


void create_lanczos (lanczos_t **lanczos, int maxnrhs, int maxiters, int nm);

void destroy_lanczos (lanczos_t *lanczos);

void compute_lanczos (lanczos_t *lanczos, double tol,
                      int nrhs, double *z, int ldz,
                      int mobtype,
                      complex *fmmin1, complex *fmmin2, complex *fmmin3,
                      complex *fmmout, 
                      double *sourcepos, double *radii,
                      int numpairs_p, int *pairs, double *A,...);
                      


#endif /* __LANCZOS_H__ */
