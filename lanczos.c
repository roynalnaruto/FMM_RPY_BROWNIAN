#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <sys/time.h>
#include <mkl.h>
#include <math.h>
#include <stdarg.h>
#include <assert.h>
#include "lanczos.h"
#include "fmm.h"



void create_lanczos (lanczos_t **_lanczos, int maxnrhs, int maxiters, int nm)
{
    lanczos_t *lanczos;
    int ldr;
    int ldh;

    maxnrhs = (maxnrhs > nm/3 ? nm/3 : maxnrhs);    
    lanczos = (lanczos_t *)malloc (sizeof(lanczos_t));
    assert (lanczos != NULL);

    lanczos->memory = 0.0;
    lanczos->nm = nm;
    lanczos->maxnrhs = maxnrhs;
    lanczos->maxiters = maxiters;
    lanczos->ldv = PAD_LEN (nm, sizeof(double));
    ldh = PAD_LEN (maxiters * maxnrhs, sizeof(double));
    ldr = PAD_LEN (maxnrhs, sizeof(double));
  
    lanczos->h2 = (double *)ALIGNED_MALLOC (sizeof(double) * 
        maxnrhs * maxiters * ldh);
    lanczos->band0 = (double *)ALIGNED_MALLOC (sizeof(double) *
        (maxnrhs + 1) * ldh);
    lanczos->band1 = (double *)ALIGNED_MALLOC (sizeof(double) *
        (maxnrhs + 1) * ldh);
    lanczos->v = (double *)ALIGNED_MALLOC (sizeof(double) * maxnrhs * 
        (maxiters + 1) * lanczos->ldv);
    lanczos->b = (double *)ALIGNED_MALLOC (sizeof(double) * 
        maxnrhs * lanczos->ldv);
    lanczos->b2 = (double *)ALIGNED_MALLOC (sizeof(double) * 
        maxnrhs * lanczos->ldv);
    assert (NULL != lanczos->band0);
    assert (NULL != lanczos->band1);
    assert (NULL != lanczos->h2);
    assert (NULL != lanczos->v);
    assert (NULL != lanczos->b);
    assert (NULL != lanczos->b2);
    lanczos->memory += (1.0 * maxnrhs * maxiters * ldh + 
                        (maxnrhs + 1.0) * ldh +
                        (maxiters + 1.0) * lanczos->ldv + 
                        1.0 * maxnrhs * lanczos->ldv) * sizeof(double);
    if (maxnrhs > 1)
    {
        lanczos->d = (double *)
            ALIGNED_MALLOC (sizeof(double) * (maxnrhs + 1) * ldh);  
        lanczos->h = (double *)
            ALIGNED_MALLOC (sizeof(double) * maxnrhs * ldr);
        lanczos->tau = (double *)
            ALIGNED_MALLOC (sizeof(double) * maxnrhs);
        lanczos->R = (double *)ALIGNED_MALLOC (sizeof(double) * maxnrhs * ldr);
        assert (lanczos->R != NULL &&
                lanczos->tau != NULL &&
                lanczos->h != NULL &&
                lanczos->d);
        lanczos->memory += ((maxnrhs + 1.0) * ldh +
                            2.0 * maxnrhs * ldr +
                            1.0 * maxnrhs) * sizeof(double);
    }
    lanczos->memory;
    
    *_lanczos = lanczos;
}


void destroy_lanczos (lanczos_t *lanczos)
{
    ALIGNED_FREE (lanczos->v);
    ALIGNED_FREE (lanczos->b);
    ALIGNED_FREE (lanczos->b2);
    ALIGNED_FREE (lanczos->band0);
    ALIGNED_FREE (lanczos->band1);
    ALIGNED_FREE (lanczos->h2);

    if (lanczos->maxnrhs > 1)
    {
        ALIGNED_FREE (lanczos->h);
        ALIGNED_FREE (lanczos->d);
        ALIGNED_FREE (lanczos->tau);
        ALIGNED_FREE (lanczos->R);
    }
    
    free (lanczos);
}

/*
 *  Use fmmin1, fmmin2, fmmin3 and fmmout as temporaries, no need to allocate space.
 * 
 */
void compute_lanczos (lanczos_t *lanczos, double tol,
                      int nrhs, double *z, int ldz,
                      int mobtype,
                      complex *fmmin1, complex *fmmin2, complex *fmmin3,
                      complex *fmmout, 
                      double *sourcepos, double *radii,
                      int numpairs_p, int *pairs,...)
{
    struct timeval tv1;
    struct timeval tv2;
    double timepass;
    DPRINTF ("  Computing lanczos nrhs = %d mobtype = %d...\n", nrhs, mobtype);
    gettimeofday (&tv1, NULL);
    
    int i;
    int j;
    int k;
    int maxits;
    double *w;
    double *v;
    double *d0;
    double * d1;    
    double *e0;
    double * e1;
    double * h2;
    double *b;
    double *b2;
    double *tmp;
    double normz;
    double normw;
    int nm;
    int ldh;
    double normy;
    int converged;
    int row;
    int col;
    int row2;
    int col2;
    int kd;
    double *W;
    double *V;
    double *D;
    double *H2;
    double *band0;
    double *band1;
    double *H;
    double *tau;
    double *R;
    int ldr;
    va_list ap;
    double *pos;
    double *mob;
    double flops;
    int _nrhs;
    int nn;
    double *zz;
    int ldv;
    int ldm;
    double beta;
    double *_pos;
    
   int nposfmm;
 
   va_start (ap, mobtype);    

/*
    if (mobtype == EWALD)
    {
        mob = (double *)(va_arg (ap, double *));
        ldm = (int)(va_arg (ap, int));
    }
    else if (mobtype == SPME)
    {
        spme = (spme_t *)(va_arg (ap, void *));
        pos = (double *)(va_arg (ap, double *));
        spmat = (spmat_t *)(va_arg (ap, void *));     
    }
*/ 
   if(mobtype == FMM)
   {
	   nposfmm = lanczos->nm/3;
	   //printf("nm is %d *******************\n", nm);
   }
   else if(mobtype == SERIAL){
	   nposfmm = lanczos->nm/3;
   }
   
    va_end (ap);    
    maxits = lanczos->maxiters;
    nm = lanczos->nm;
    ldv = lanczos->ldv;
    flops = 0.0;
    _pos = pos;
    
    for (nn = 0; nn < nrhs; nn += lanczos->maxnrhs)
    {
        _nrhs = (nn + lanczos->maxnrhs > nrhs ? 
                     nrhs - nn: lanczos->maxnrhs);
        zz = &(z[ldz * nn]);
        b = lanczos->b;
        b2 = lanczos->b2;
        DPRINTF ("    nrhs = %d\n", _nrhs);        
        if (_nrhs == 1)
        {
            ldh = PAD_LEN (maxits, sizeof(double));
            v =  lanczos->v;
            d0 = &(lanczos->band0[0 * ldh]);
            e0 = &(lanczos->band0[1 * ldh]);
            d1 =  &(lanczos->band1[0 * ldh]);
            e1 =  &(lanczos->band1[1 * ldh]);
            h2 = lanczos->h2;

            // v(:,1) = z/norm(z)
            normz = cblas_dnrm2 (nm, zz, 1);
            cblas_dcopy (nm, zz, 1, &(v[0]), 1);
            cblas_dscal (nm, 1.0/normz, &(v[0]), 1);
            converged = 0;
            for (i = 0; i < maxits; i++)
            {
                // w = mat * v(i);           
                w = &(v[(i + 1) * ldv]);      
                
/*                
                if (mobtype == EWALD)
                {
                    cblas_dsymv (CblasColMajor, CblasLower,
                                 nm, 1.0, mob, ldm,
                                 &(v[i * ldv]), 1, 0.0, w, 1);
                }
                else if (mobtype == SPME)
                {
                    compute_spme (spme, _pos, 1,
                                  1.0, &(v[i * ldv]), ldv,
                                  0.0, w, ldv, 1);
                    spmv_3x3 (spmat, 1, 1.0, &(v[i * ldv]), ldv,
                              1.0, w, ldv, 1);
                }
                else 
*/ 
                if(mobtype == FMM)
                {
					//Copy input to temporaries.
					int tempi;
					double *tempin =  &(v[i * ldv]);
					for(tempi = 0; tempi<nposfmm; tempi++){
						fmmin1[tempi].dr = tempin[3*tempi];
						fmmin2[tempi].dr = tempin[3*tempi+1];
						fmmin3[tempi].dr = tempin[3*tempi+2];
						fmmin1[tempi].di = 0;
						fmmin2[tempi].di = 0;
						fmmin3[tempi].di = 0;
					}
					char *dir;
					computerpy_(&nposfmm, sourcepos, radii, fmmin1, fmmin2, fmmin3, fmmout, dir);
					postCorrection(nposfmm, sourcepos, radii, numpairs_p, pairs, fmmin1, fmmin2, fmmin3, fmmout);
					//Copy output back from temporary.
					for(tempi = 0; tempi<nposfmm*3; tempi++){
						w[tempi] = fmmout[tempi].dr;
					} 
				}

                // w = w - h(i-1, i) * v(i-1)                
                if (i > 0)
                {
                    cblas_daxpy (nm, -e0[i - 1],
                                 &(v[(i - 1) * ldv]), 1, w, 1);
                }

                // h(i, i) = <w, v[i]>
                d0[i] = cblas_ddot (nm, w, 1, &(v[i * ldv]), 1);

                // w = w - h(i, i)*v(i)
                cblas_daxpy (nm, -d0[i],
                             &(v[i * ldv]), 1, w, 1);
                normw = cblas_dnrm2 (nm, w, 1);
                e0[i] = normw;

                // v(i + 1) = w = w/normw;
                cblas_dscal (nm, 1.0/normw, w, 1);

                // h2 = sqrtm(h)      
                cblas_dcopy (i + 1, d0, 1, d1, 1);
                cblas_dcopy (i, e0, 1, e1, 1);
                LAPACKE_dstev (LAPACK_ROW_MAJOR, 'V', i + 1, d1, e1, h2, ldh);
                #pragma vector aligned
                for (j = 0; j < i + 1; j++)
                {
                    e1[j] = h2[j] * sqrt (d1[j]);
                }
                cblas_dgemv (CblasRowMajor, CblasNoTrans,
                             i + 1, i + 1, 1.0, h2, ldh, e1, 1, 0.0, d1, 1);
            
                // y = v(:,1:i)*sqrth(:,1)*norm(z);      
                cblas_dgemv (CblasColMajor, CblasNoTrans,
                             nm, i + 1, normz, v, ldv, d1, 1, 0.0, b, 1);

                // check convergence
                if (i > 0)
                {
                    cblas_daxpy (nm, -1.0, b, 1, b2, 1);
                    normy = cblas_dnrm2 (nm, b2, 1);
                    if (normy/normz < tol)
                    {         
                        converged = 1;
                        break;
                    }
                }
                tmp = b2;
                b2 = b;
                b = tmp;
                _pos = NULL;
            }
        }

        if (converged == 1)
        {
            DPRINTF ("    converge in %d iterations\n", i + 1);
            for (j = 0; j < _nrhs; j++)
            {
                cblas_dcopy (nm, &(b[j * ldv]), 1, &(zz[j * ldz]), 1);
            }
        }
        else
        {
            DPRINTF ("    failed to converge after %d iterations\n", i + 1);
            for (j = 0; j < _nrhs; j++)
            {
                cblas_dcopy (nm, &(b2[j * ldv]), 1, &(zz[j * ldz]), 1);
            }
        }
        flops += 2.0 * (i + 1) * nm * nm * _nrhs;
    }
    
    gettimeofday (&tv2, NULL);
    timersub (&tv2, &tv1, &tv1);
    timepass = tv1.tv_sec + tv1.tv_usec/1e6;
  
/*  
    if (mobtype == EWALD)
    {
        
        DPRINTF (3, "    takes %.3le secs %.3le GFlops\n",
            timepass, flops/timepass/1e9);
    }
    else if (mobtype == SPME)
    {
        DPRINTF (3, "    takes %.3le secs\n",
            timepass);
    }
    else
    */
      if(mobtype == FMM)
    {
		DPRINTF ("    takes %.3le secs\n",
            timepass);
	}
}
